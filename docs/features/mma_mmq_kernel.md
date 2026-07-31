# MMA MMQ kernel

Routing quantized prefill GEMMs at llama.cpp's tensor-core MMQ kernels instead of
the plugin's own DP4A-era copy.

## Scope

In scope:

- The 2-D quantized GEMM behind `ggml_mul_mat_a8` — every non-MoE linear layer
  during prefill.
- Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, on NVIDIA Turing
  (sm_75) and newer.

Not in scope:

- MoE (`ggml_moe_a8`), MMVQ (`ggml_mul_mat_vec_a8`, the decode path), and
  dequantization all keep their existing kernels.
- IQ* / MXFP4 / NVFP4 types. They have upstream MMQ kernels but no template
  instance is compiled here, and `gguf_mma::supported()` rejects them.
- Pre-Turing NVIDIA and all AMD hardware, which have no MMA path and fall back
  to the existing kernels.

## Why

`csrc/gguf/mmq.cuh` was copied from llama.cpp `b2899` (April 2024). Two things
about that vintage matter:

1. Its Q4_K output tile is **4 tokens deep** on NVIDIA. The kernel therefore
   re-reads the entire weight matrix `M/4` times, and measures ~5.7 TFLOP/s flat
   from M=32 to M=8192 on an RTX 3090 — 8% of peak, and batch-independent, so
   larger batches cannot help.
2. Upstream paired that narrow tile with a guard (`MMQ_MAX_BATCH_SIZE 32`) that
   sent larger batches to cuBLAS. The tile is only correct *because* of the
   guard. The port took the tile and not the guard.

Upstream has since rewritten MMQ around int8 tensor-core MMA
(`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`) with a tile up to 128 tokens
wide, and deleted the cuBLAS fallback on Turing+ because it is no longer needed.

Measured on an RTX 3090, Qwen3.6-27B-UD-Q4_K_XL, M=2048:

| tensor | legacy DP4A | this kernel | speedup |
|---|---|---|---|
| `ffn_down` 5120x17408 Q4_K | 64.13 ms | **4.94 ms** | **13.0x** |
| `ffn_gate` 17408x5120 Q4_K | 63.83 ms | **4.88 ms** | **13.1x** |
| `ffn_down` 5120x17408 Q5_K | 83.66 ms | **5.17 ms** | **16.2x** |
| `ffn_down` 5120x17408 Q6_K | 85.33 ms | **5.61 ms** | **15.2x** |
| `ssm_out` 5120x6144 Q8_0 | 24.28 ms | **1.78 ms** | **13.7x** |

The MMA kernel is faster at every M tested (1 -> 2048), so unlike a
dequantize-then-cuBLAS approach it needs no batch-size threshold.

## Files

| file | role |
|---|---|
| `csrc/gguf/llamacpp/` | llama.cpp sources, vendored verbatim. Version in `LLAMA_CPP_VERSION.txt`. Do not edit — re-vendoring must stay a copy, not a merge. |
| `csrc/gguf/llamacpp/template-instances/mmq-instance-*.cu` | One translation unit per quant type. Compiling all types in one TU is prohibitively slow. |
| `csrc/gguf/mmq_mma.h` | Caller-facing interface. Exposes only plain integers and pointers. |
| `csrc/gguf/mmq_mma.cu` | Builds `mmq_args`, selects the type, owns the dtype conversions. |
| `csrc/gguf/mmq_mma_shim.cu` | The handful of ggml runtime functions the kernels reference: `ggml_cuda_info`, `ggml_cuda_get_device`, `ggml_blck_size`, `ggml_cuda_error`, `ggml_abort`, and a memory pool for stream-k. |
| `csrc/gguf/gguf_kernel.cu` | `ggml_mul_mat_a8` checks the env var and `gguf_mma::supported()`, else falls through to the existing path. |

Key exports: `gguf_mma::supported`, `gguf_mma::quantized_activation_bytes`,
`gguf_mma::launch`, `gguf_mma::to_f32`, `gguf_mma::from_f32`.

## Control flow

```
ggml_mul_mat_a8(W, X, type, row)
  └─ VLLM_GGUF_MMA_MMQ set, and gguf_mma::supported(type, device)?
       ├─ no  → existing quantize_row_q8_1 + mul_mat_q*_q8_1_cuda (unchanged)
       └─ yes → to_f32(X)                     [staging, see below]
                quantize_mmq_q8_1_cuda(...)   [upstream q8_1 MMQ layout]
                mul_mat_q_case<type>(...)     [tensor-core MMA, stream-k]
                from_f32(→ Y)
```

## Invariants and constraints

- **The two `ggml-common.h` copies must never meet in one translation unit.**
  `llamacpp/ggml-common.h` and the plugin's own are different vintages defining
  the same block structs. This is why `mmq_mma.cu` is a separate TU and why
  `mmq_mma.h` exposes no ggml types.
- **`use_stream_k` must be true on Turing+.** Without it the tiling path
  degrades as M grows: measured 60.5 TFLOP/s at M=128 falling to 5.1 at M=2048,
  i.e. worse than the kernel it replaces. It is not an optional tuning knob.
- **The extension must be import-checked, not just build-checked.** Missing ggml
  symbols still *link* (shared objects tolerate undefined symbols) and fail at
  `dlopen`, after which `ops.ggml_mul_mat_a8` silently falls back to Triton.
  That failure mode is quiet enough to produce plausible-looking benchmark
  numbers. `nm -u` should report no undefined `ggml_*` symbols.
- **torch's `-D__CUDA_NO_HALF_CONVERSIONS__` family must be undefined** for this
  extension; the vendored kernels are written against stock CUDA semantics.
- Adding a quant type means adding a `DECL_MMQ_CASE` instance file *and* an
  entry in `kSupported`. Omitting the instance fails at link rather than
  silently degrading.
- The kernels are fp32-in/fp32-out, so activations and results stage through
  float buffers (`M*K` and `M*N`). They cost little time (~0.5 ms of 70 at
  M=2048) but real memory. Templating upstream's `quantize_mmq_q8_1` and
  `mmq_write_back_*` on the element type would remove both.

## Known issues

- `tests/test_kernels.py::test_mmq[12-Q4_K-dtype0-1024-2048]` fails with the
  kernel enabled: 1 element in 2,097,152. Aggregate accuracy is unchanged
  (relative Frobenius 1.29e-2 vs the legacy kernel's 1.30e-2), but the
  worst-case element is ~9% larger (1.125 vs 1.031), which crosses the test's
  `atol=1.0` where the reference value is too small for `rtol` to absorb.
  Consistent with stream-k's split-K fixup adding a rounding step. The default
  is off partly for this reason.
