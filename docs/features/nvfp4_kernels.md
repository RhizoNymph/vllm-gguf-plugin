# ggml NVFP4 kernels

## Scope

Which kernel computes a GEMM over ggml NVFP4 (type 40) weights, and the
CUDA vector kernels that serve decode: the dense matrix-vector kernel
(`ggml_mul_mat_vec_a8`) and the per-(token, expert) MoE kernel
(`ggml_moe_a8_vec`), both built on llama.cpp's `vec_dot_nvfp4_q8_1`. NVFP4
is what Inkling 2 Mini GGUFs store the experts of layers 0-19 in (44 of 48
layer visits); see `inkling2.md` for the model.

Not in scope: a CUDA dequantize or MMQ (prefill) kernel for NVFP4. Prefill
stays on the Triton kernels (grouped fused MoE; dequantize + cuBLAS for
dense layers). The vendored llama.cpp MMA MMQ (`mmq_mma.cu`, opt-in via
`VLLM_GGUF_MMA_MMQ`) has NVFP4 tile loaders, but it is dense-only, so wiring
it would not reach the MoE experts, which are the only NVFP4 tensors in
published files; the plugin's own DP4A MMQ/MoE kernels (`mmq.cuh`,
`moe.cuh`) predate NVFP4.

## Format

A block is 64 values in 36 bytes: four UE4M3 scales (one per 16 values),
then 32 bytes of E2M1 codes; sub-block `s` holds values `16s..16s+7` in the
low nibbles of `qs[8s..8s+7]` and `16s+8..16s+15` in the high nibbles. A
value is `e2m1(code) * ue4m3(scale)`; UE4M3 codes 0x00 and 0x7F decode to 0
and bit 7 is ignored (ggml's CPU decoder). The torch reference is
`quantization/nvfp4.py` (`dequantize_nvfp4`).

A tensor may carry an F32 `.scale` (per expert for routed experts). It is not
part of the blocks: `_fused_moe_gguf` multiplies it into each expert's GEMM
output (gate/up rows before the activation, the down projection through the
routing weight), around whichever kernel runs.

## Data and control flow

### Dense linears (`quantization/linear.py`)

`_fused_mul_mat_gguf`: NVFP4 is in `MMVQ_QUANT_TYPES`, so up to `mmvq_safe`
rows (6, or 2 for weights over 5120 rows) call `ops.ggml_mul_mat_vec_a8`;
larger batches fall through to `_dequant_gemm` (NVFP4 is in `DEQUANT_TYPES`,
not `MMQ_QUANT_TYPES`): `ops.ggml_dequantize` (Triton) then `x @ W.T`.

### MoE (`quantization/fused_moe.py`)

`_fused_moe_gguf` asks `select_moe_kernel(weight_type, weight_type2,
num_tokens)` for a `MoEKernel`:

| Kernel | When | NVFP4 runs on |
|---|---|---|
| `GROUPED` | `num_tokens > GROUPED_MOE_MIN_TOKENS` (64) and both types share a grouped backend: both in `MMQ_QUANT_TYPES` (CUDA MMQ) or both in `FP4_QUANT_TYPES` (Triton) | `ops.ggml_moe_a8` -> Triton `ggml_moe_a8_triton`, tokens aligned with `ops.ggml_moe_get_block_size` (Triton's `BLOCK_M`) |
| `VECTOR` | otherwise, both types in `MOE_VEC_QUANT_TYPES` | `ops.ggml_moe_a8_vec` -> CUDA `moe_vec_q<…, block_nvfp4, …>` |
| `PER_TOKEN` | anything else (dense / unknown types) | n/a |

A grouped launch aligns the tokens once with one block size for both GEMMs,
which is why a pair such as NVFP4 gate/up with a Q5_0 down projection stays on
`VECTOR` at any token count.

### Op dispatch (`ops.py`)

`CUDA_OP_QUANT_TYPES` maps each `_C_gguf` op to the ggml types its CUDA
kernel serves; `_cuda_kernel_available(op, type)` also requires the
extension to be loaded and enabled (`VLLM_GGUF_USE_CUDA`, default on).
Otherwise the op runs its Triton kernel. NVFP4 is in the sets of
`ggml_mul_mat_vec_a8` and `ggml_moe_a8_vec` only.

### CUDA (`csrc/gguf/`)

1. `quantize_row_q8_1_cuda` quantizes the activations to q8_1 (int8 per 32
   values, scale and sum as `half2`), rows padded to 512 values.
2. `mul_mat_vec_q` / `moe_vec_q` (one warp per output row; for MoE, grid z
   = token x top-k slot, the expert read from `topk_ids`) iterate the row's
   NVFP4 blocks with `qk = 64`, `qi = QI_NVFP4 = 8`, `vdr = 4`: two threads
   per block, each covering two 16-value sub-blocks. A CUDA block holds
   `GGML_CUDA_MMV_Y = 4` such warps (four rows), for every quant type: with
   one-warp blocks an sm_86 SM's 16-block limit caps occupancy at 16 of 48
   warps, too few loads in flight to approach bandwidth. Four rows per
   block cut the Inkling 2 Mini NVFP4 MoE call from 39 to 32 µs and lift
   decode 5-6% on Inkling and Qwen3.6-35B-A3B; results are bit-identical
   (each row is still reduced by one warp).
3. `vec_dot_nvfp4_q8_1` maps 8 codes at a time to doubled E2M1 int8 values
   with `__byte_perm` (`get_int2_from_table_16`, table `kvalues_mxfp4`),
   takes four `__dp4a` against the matching half of a q8_1 block, and scales
   by `nvfp4_ue4m3_to_fp32_half(d[s]) * q8_1.d`.
4. `gguf_kernel.cu` dispatches type 40 in `ggml_mul_mat_vec_a8` and
   `ggml_moe_a8_vec`; an unknown type now raises instead of returning zeros.

## Files

| File | Role / key exports |
|---|---|
| `vllm_gguf_plugin/csrc/gguf/ggml-common.h` | `block_nvfp4`, `QK_NVFP4`, `QK_NVFP4_SUB`, `QR_NVFP4`, `QI_NVFP4`, `kvalues_mxfp4` (copied from `llamacpp/ggml-common.h`) |
| `vllm_gguf_plugin/csrc/gguf/vecdotq.cuh` | `VDR_NVFP4_Q8_1_MMVQ`, `get_int2_from_table_16`, `nvfp4_ue4m3_to_fp32_half`, `vec_dot_nvfp4_q8_1` (from `llamacpp/vecdotq.cuh` / `common.cuh`) |
| `vllm_gguf_plugin/csrc/gguf/mmvq.cuh` | `mul_mat_vec_nvfp4_q8_1_cuda` |
| `vllm_gguf_plugin/csrc/gguf/moe_vec.cuh` | `moe_vec_nvfp4_q8_1_cuda` |
| `vllm_gguf_plugin/csrc/gguf/gguf_kernel.cu` | type 40 cases in `ggml_mul_mat_vec_a8`, `ggml_moe_a8_vec` |
| `vllm_gguf_plugin/ops.py` | `CUDA_OP_QUANT_TYPES`, `_cuda_kernel_available` |
| `vllm_gguf_plugin/quantization/utils.py` | `FP4_QUANT_TYPES`, `MMVQ_QUANT_TYPES` (now with NVFP4), `MOE_VEC_QUANT_TYPES` |
| `vllm_gguf_plugin/quantization/fused_moe.py` | `MoEKernel`, `GROUPED_MOE_MIN_TOKENS`, `select_moe_kernel`, `_fused_moe_gguf` |
| `vllm_gguf_plugin/quantization/nvfp4.py` | torch reference decoder |
| `vllm_gguf_plugin/triton/{dequantize,fused_moe}/fp4_quant/nvfp4.py` | Triton dequantize and grouped MoE (prefill) |
| `tests/test_nvfp4_cuda.py` | GPU: CUDA MMVQ / MoE vs `dequantize_nvfp4` with q8_1-rounded activations (tight) and unrounded (loose), special scale codes, vs the Triton kernel, `_fused_moe_gguf` with expert scales, SITU and top-k 8 on both sides of the 64-token switch |
| `tests/test_nvfp4_dispatch.py` | CPU: per-op CUDA type sets, `select_moe_kernel`, which op `_fused_moe_gguf` and `_fused_mul_mat_gguf` call |
| `tests/test_nvfp4_cuda_tables.py` | CPU: the header copies match the vendored header and the torch decoder |

## Invariants

* The CUDA vector kernels decode NVFP4 exactly as `dequantize_nvfp4` (and
  ggml's CPU decoder) for every scale code; their only rounding is the q8_1
  activation quantization shared with every other MMVQ type.
* `ggml_moe_get_block_size` and `ggml_moe_a8` agree on the backend for a
  type, so grouped token alignment always matches the kernel that consumes
  it; a grouped launch never mixes a Triton type with a CUDA MMQ type.
* Per-expert output scales are applied outside the kernels, identically on
  every `MoEKernel` path.
* `moe_vec_q` (every type, NVFP4 included) leaves the output rows of slots
  with expert id < 0 (vLLM's padding mask) at zero and never reads `W[-1]`.
* NVFP4 needs a `_C_gguf` built from this tree: an older extension has no
  type-40 case and returned zeros for it (the switches now raise on unknown
  types).
* Row length must be a multiple of 64 (whole blocks), as for every ggml
  type; the MoE vector kernel's grid z (`tokens * top_k`) stays far below
  CUDA's 65535 limit because NVFP4 switches to the grouped path above 64
  tokens.
