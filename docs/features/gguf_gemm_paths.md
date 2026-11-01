# GGUF GEMM paths and device-memory layout

## Scope

How a GGUF linear layer picks its kernel for a given batch size, how merged layers
whose shards have different quant types are stored, and the option to keep the input
embedding table in host memory. Measured on an RTX 3090 serving Qwen3.6-27B-UD-Q4_K_XL.

Not in scope: MoE GEMMs (`fused_moe.py`), the Triton fallbacks, weight-name mapping.

## Data and control flow

`GGUFLinearMethod.apply(layer, x)` (quantization/linear.py):

1. If `layer.gguf_shard_groups` exists (mixed-type merged layer, see below): one
   `fused_mul_mat_gguf` call per group on `layer.qweight_group<i>`, results
   concatenated along the output dimension.
2. Else the merged/plain weight goes to `fused_mul_mat_gguf(x, qweight, type)`, a
   custom op (opaque to torch.compile) that dispatches on `M = x.shape[0]`:
   - unquantized types: `x @ W.T`;
   - `M <= _mmvq_max_rows(...)`: `ggml_mul_mat_vec_a8` (MMVQ);
   - `M >= VLLM_GGUF_MMQ_SAFE` (when > 0): dequantize to `x.dtype`, then cuBLAS;
   - otherwise `ggml_mul_mat_a8` (MMQ).

`_mmvq_max_rows`:

| VLLM_GGUF_MMA_MMQ | VLLM_GGUF_MMVQ_MULTI | rows on MMVQ |
|---|---|---|
| off | any | legacy heuristic: 2 (N > 5120) or 6; 8/16 for i-quants |
| on (sm_75+, MMQ type) | off | 1 |
| on | on | 3 |

C++ side (csrc/gguf/gguf_kernel.cu):

- `ggml_mul_mat_vec_a8`: quantizes activations to q8_1 (rows padded to 512). With
  `VLLM_GGUF_MMVQ_MULTI=1` and 1-8 rows of a supported type, runs the vendored
  llama.cpp multi-column MMVQ (`mmvq_ggml.cu` -> `llamacpp/mmvq.cu`, fp32 out, then
  converted); otherwise the legacy one-row-per-pass kernel (`mmvq.cuh`).
- `ggml_mul_mat_a8`: with `VLLM_GGUF_MMA_MMQ=1`, the vendored llama.cpp tensor-core
  MMQ with stream-k (`mmq_mma.cu`); otherwise the legacy dp4a MMQ (`mmq.cuh`).
- `ggml_dequantize`: full dequantization for the cuBLAS path.

Outputs of `ggml_dequantize`, MMVQ (both kernels) and the MMA path are allocated
uninitialised: every element is written by the kernel.

### Mixed-type merged layers

`GGUFLinearMethod.process_weights_after_loading` first builds the padded merged
weight (`_create_padded_weight_param`). If the shards' types differ,
`_split_mixed_shard_groups` then stores each run of consecutive same-typed shards
(in output order: q, k, v for attention qkv; 0..3 for Qwen3.5's in_proj_qkvz) as its
own contiguous parameter `qweight_group<i>` (a clone, so no view keeps the padded
data alive) and releases the padded data. `layer.gguf_shard_groups` is a tuple of
`(param_name, ggml_type)`. Subclasses whose apply() reads the padded weight set
`split_mixed_shards = False` (`DiffusionGGUFLinearMethod`).

### Host-resident input embedding

With `VLLM_GGUF_EMBED_ON_HOST=1`, `GGUFEmbeddingMethod.process_weights_after_loading`
copies a `VocabParallelEmbedding`'s quantized table into pinned host memory
(`layer._gguf_host_qweight`) and points `qweight` at a UVA device view of it. The
lookup (`index_select` + dequantize of the batch's rows) runs unchanged on the GPU,
reading only those rows over PCIe. `ParallelLMHead` reads its whole weight every
step and is never moved.

## Files

| file | role |
|---|---|
| quantization/linear.py | `_mmvq_max_rows`, `_fused_mul_mat_gguf` dispatch, `GGUFLinearMethod` (shard groups) |
| quantization/vocal_embeds.py | `GGUFEmbeddingMethod`, `_move_qweight_to_host` |
| quantization/diffusion_config.py | `DiffusionGGUFLinearMethod` (opts out of shard groups) |
| csrc/gguf/gguf_kernel.cu | torch ops; env gates `mma_mmq_enabled`, `mmvq_multi_enabled` |
| csrc/gguf/mmq_mma.{h,cu}, mmq_mma_quantize.cu, mmq_mma_shim.cu | tensor-core MMQ launcher, activation quantizer, ggml runtime shim |
| csrc/gguf/mmvq_ggml.{h,cu} | multi-column MMVQ launcher (`gguf_mmvq::supported`, `launch`) and stubs for ggml symbols only mmvq.cu's unused entry points reference |
| csrc/gguf/llamacpp/ | llama.cpp sources vendored verbatim (version in LLAMA_CPP_VERSION.txt) |
| tests/test_mixed_shards.py | shard groups match the padded-slice path; dispatch thresholds |
| tests/test_mmvq_multi.py | multi-column MMVQ vs reference (run with VLLM_GGUF_MMVQ_MULTI=1) |
| tests/test_embed_on_host.py | host-resident lookups are bit-identical |

## Invariants and constraints

- The env gates are read once per process (C++ statics); the Python side reads the
  same variables with the same rule (set and not starting with "0"), so the two
  agree on which kernel serves a batch.
- Every kernel whose output is allocated with `new_empty` writes all `M x N` outputs.
- `qweight_group<i>` tensors own their storage; the padded merged weight is empty
  after the split.
- MMQ (both kernels) quantizes activations to int8 per 32 values; its error against
  an fp32 reference is ~1.2e-2 relative for Q4_K/Q5_K and ~5.6e-3 for Q6_K/Q8_0, versus
  ~5.6e-3 for MMVQ and ~2.4e-3 for dequantize+cuBLAS. Raising `VLLM_GGUF_MMQ_SAFE`
  moves batches between those numerics classes.
- Every symbol in the vendored llama.cpp sources must resolve inside the extension:
  an undefined one makes the extension fail to dlopen and the plugin silently falls
  back to Triton. Check `nm -D --undefined-only` and an import after adding sources.

## Measurements (RTX 3090, Qwen3.6-27B blk.0 weights, time per call)

| weight | M | legacy MMVQ | multi MMVQ | dp4a MMQ | MMA MMQ | dequant+cuBLAS |
|---|---|---|---|---|---|---|
| ffn_gate Q4_K 17408x5120 | 1 | 72 us | 64 | 114 | 95 | - |
| | 3 | - | 83 | - | 96 | - |
| | 8 | 512 | 192 | 209 | 98 | - |
| | 512 | - | - | - | 983 | 1262 |
| | 4096 | - | - | - | 7859 | 5779 |
| ffn_down Q6_K 5120x17408 | 1 | 112 | 94 | 310 | 135 | - |
| | 8 | 863 | 172 | 341 | 141 | - |
