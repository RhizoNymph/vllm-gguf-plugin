# W8A8 prefill GEMM (`VLLM_GGUF_W8A8`)

## Scope

A group-scaled int8 x int8 tensor-core GEMM for prefill-sized GGUF linears
(`M >= VLLM_GGUF_MMQ_SAFE`), replacing dequantise-to-bf16 + cuBLAS. On sm_86 the int8
MMA has about twice the bf16 tensor-core rate (cuBLAS: ~135 TFLOP/s bf16 vs ~240 TOPS
int8 on an RTX 3090), and prefill GEMMs at M ~ 2048 are compute bound.

In scope: the env gate and routing, the three CUDA stages (weight re-quantisation,
activation quantisation, GEMM), their torch ops, numerics.

Not in scope: decode-sized GEMMs (MMVQ / MMQ, see gguf_gemm_paths.md), MoE, weight
types other than Q4_K / Q5_K / Q6_K / Q8_0, non-CUDA devices.

## Numerics

Both operands are symmetric int8 with one fp32 scale (amax / 127) per row and group
of `G` consecutive K values (G = 128 or 256):

    Y[m, n] = sum_g  sa[g, m] * sb[g, n] * sum_{k in g} Aq[m, k] * Bq[n, k]

- `Bq, sb`: the GGUF weight decoded to fp32 in registers and re-quantised per call.
  The GGUF value grid is not the int8 grid, so this is a second rounding of the
  weight (exact only for Q8_0 when its 32-value blocks happen to have amax at +-127 d).
- `Aq, sa`: activations quantised per token and group.
- Integer partials are exact inside a group; the fold into fp32 happens once per
  group. Output rounded to the activation dtype (bf16 / fp16).

Error against an fp32 reference (fp32 activations x exactly dequantised weight) is the
quadrature sum of the two roundings, about 0.5-0.7e-2 each for gaussian data; measured
values for real activations are in the table below. For comparison: dequant+cuBLAS
~2.4e-3, MMVQ ~5.6e-3, MMQ ~1.2e-2 (Q4_K/Q5_K) / ~5.6e-3 (Q6_K/Q8_0).

## Type selection (`VLLM_GGUF_W8A8_TYPES`)

Default `Q4_K,Q5_K`. Re-quantising Q6_K / Q8_0 adds a second weight rounding that MMQ
(exact weights) does not, and with all four types the end-to-end greedy replies
diverged at larger margins than the noise floor. Q4_K + Q5_K carry 79% of
Qwen3.6-27B-UD-Q4_K_XL's GEMM FLOPs (Q4_K 66%, Q6_K 15%, Q5_K 13%, Q8_0 6%).

## Data and control flow

`_fused_mul_mat_gguf` (quantization/linear.py), for a GEMM that would take the
dequantize+cuBLAS branch (`_prefer_dequant_gemm`): if `_use_w8a8` (W8A8 on, type in `_W8A8_TYPES`, CUDA,
bf16/fp16 contiguous activations, extension present, `ops.w8a8_supported(type, N, K,
G)`), call `ops.ggml_mul_mat_w8a8(qweight, x, type, N, G)`; otherwise dequantize +
cuBLAS as before. All other branches are unchanged.

`ggml_mul_mat_w8a8` (csrc/gguf/w8a8/op.cu) allocates (torch caching allocator) and runs,
on the current stream:

1. `requantize_weight` (quantize.cu): one thread per 32-value chunk of a weight row.
   Decodes the chunk from its GGUF block into fp32 registers (layouts as ggml's
   `dequantize_row_*`), reduces amax over the G / 32 lanes of its group with warp
   shuffles, writes 32 int8 and (one lane) the scale. Output `bq [N, K] int8`,
   `sb [K / G, N] fp32`.
2. `quantize_activations` (quantize.cu): one thread per 8 values (one 16-byte load),
   G / 8 lanes per group. Output `aq [M, K] int8`, `sa [K / G, lda] fp32` with
   `lda = act_scale_ld(M)` (M rounded up to 4, so 16-byte scale copies are aligned).
3. `gemm` (gemm.cu): CTA tile 128 x 128 x 128, 8 warps of 64 x 32, 3-stage cp.async
   pipeline (99 KB, the sm_86 per-block maximum), 128-byte shared-memory rows with an
   XOR chunk swizzle read by ldmatrix, fragments for the next 32-wide k step loaded
   while the current one's `mma.sync.m16n8k32.s8` issue. Per output each thread holds an int32 group partial and an
   fp32 accumulator. The first MMA of each group is seeded with `0x4B400000`
   (1.5 * 2^23 as fp32 bits): integer addition onto that pattern yields the fp32 value
   1.5 * 2^23 + p for |p| < 2^22, so the fold is FADD + FMUL + FFMA without an
   int-to-float conversion (quarter rate on sm_86). Each stage also copies the group's
   128 activation and 128 weight scales. Tiles are ordered M-fastest so CTAs running
   together share weight tiles in L2. Rows past M are zero-filled on load and not
   stored.

Peak extra device memory per call: `N*K + M*K` bytes plus scales, half of the bf16
weight the dequant path materialises.

## Files

| file | role / key exports |
|---|---|
| quantization/linear.py | `_parse_w8a8_group`, `GGUFW8A8ConfigError`, `_W8A8_GROUP`, `_use_w8a8`, routing in `_fused_mul_mat_gguf` |
| ops.py | `w8a8_available`, `w8a8_supported` (cached), `ggml_mul_mat_w8a8`, stage ops `w8a8_quantize_act`, `w8a8_requant_weight`, `w8a8_gemm`; fake impl for torch.compile |
| csrc/gguf/w8a8/w8a8.h | `gguf_w8a8::Group`, `DType`, `WeightType`, tile constants, `shape_supported`, `act_scale_ld`, stage launchers |
| csrc/gguf/w8a8/quantize.cu | `quantize_activations`, `requantize_weight` (GGUF decoders for Q4_K, Q5_K, Q6_K, Q8_0) |
| csrc/gguf/w8a8/gemm.cu | `gemm`, `shape_supported` |
| csrc/gguf/w8a8/op.cu | torch stable-ABI ops `ggml_mul_mat_w8a8`, `w8a8_supported`, `w8a8_quantize_act`, `w8a8_requant_weight`, `w8a8_gemm` |
| csrc/torch_bindings.cpp | op schemas and registrations |
| tests/test_w8a8.py | decoders vs gguf-py, quantisers vs torch, GEMM vs exact fp64 grouped product (incl. M tails, extreme partials), fused op error and stage composition |
| tests/test_w8a8_dispatch.py | env parsing, routing and fallbacks |

## Invariants and constraints

- `VLLM_GGUF_W8A8` is `0`/unset (off), `1` (group 128), `128` or `256`;
  `VLLM_GGUF_W8A8_TYPES` a comma list of Q4_K, Q5_K, Q6_K, Q8_0 (default Q4_K,Q5_K).
  Anything else raises `GGUFW8A8ConfigError` at import. Both read once per process.
- W8A8 only ever replaces the dequantize+cuBLAS branch, so it never applies below
  `VLLM_GGUF_MMQ_SAFE` rows (decode, CUDA-graph-captured batches below that size).
- Supported shapes: `N % 128 == 0`, `K % 256 == 0`, `K % G == 0` (G is at least one
  128-wide k-tile); anything else falls
  back to dequantize+cuBLAS.
- `G * 127 * 127 < 2^22` (G <= 256) keeps every group partial inside the exact range of
  the accumulator seeding trick; `test_gemm_extreme_partials_stay_exact` checks the
  bound.
- Every output element is written (rows >= M are never stored, every column tile is
  full), so outputs are allocated uninitialised.
- Q6_K and Q8_0 blocks (210 / 34 bytes) are only 2-byte aligned; their decoders use
  16-bit loads. Q4_K / Q5_K blocks are 16-byte multiples and use 128-bit loads.

## Measurements (RTX 3090, Qwen3.6-27B-UD-Q4_K_XL)

Hardware ceilings at these shapes (M = 2048): cuBLAS bf16 129-138 TFLOP/s, cublasLt
int8 (`torch._int_mm`) 230-258 TOPS; dequant+bf16 cuBLAS (the replaced path) 114-122.

Per call, M = 2048, G = 128 (G = 256 in brackets), gaussian activations:

| weight | dequant+bf16 | W8A8 GEMM only | W8A8 fused | speedup | requant | act quant |
|---|---|---|---|---|---|---|
| ffn_gate Q4_K 17408x5120 | 119.6 TFLOP/s | 180.2 TOPS (189.9) | 160.4 (169.8) | 1.34x (1.42x) | 206 us | 40 us |
| ffn_down Q6_K 5120x17408 | 122.7 | 184.8 (195.3) | 157.1 (167.3) | 1.28x (1.36x) | 200 us | 130 us |
| attn_qkv Q6_K 10240x5120 | 120.3 | 178.1 (187.6) | 155.8 (165.2) | 1.29x (1.37x) | 119 us | 40 us |
| attn_gate Q5_K 6144x5120 | 116.9 | 172.7 (183.0) | 146.8 (155.9) | 1.26x (1.33x) | 79 us | 40 us |

Relative error vs fp32 on real prefill activations (one 2,048-token chunk of a
labeller request, 71 GEMMs over layers 0-3, 15, 31, 32, 47, 48, 62, 63), median / max:

| type | dequant+bf16 | MMQ (decode path) | W8A8 G=128 | W8A8 G=256 |
|---|---|---|---|---|
| Q4_K | 2.2e-3 / 2.7e-3 | 1.7e-2 / 6.1e-2 | 1.4e-2 / 5.4e-2 | 1.7e-2 / 7.5e-2 |
| Q5_K | 1.9e-3 / 2.6e-3 | 1.6e-2 / 4.7e-2 | 1.05e-2 / 4.1e-2 | 1.4e-2 / 5.6e-2 |
| Q6_K | 2.1e-3 / 2.5e-3 | 8.8e-3 / 3.5e-2 | 1.6e-2 / 7.2e-2 | 2.1e-2 / 1.0e-1 |
| Q8_0 | 2.0e-3 / 2.2e-3 | 7.4e-3 / 1.1e-2 | 1.25e-2 / 2.1e-2 | 1.6e-2 / 2.7e-2 |

Real activations have amax / rms up to ~110 per token (massive-activation channels),
which is what lifts the int8 error above the gaussian 0.9e-2.

End to end (labeller bench, 48 poles, 1 neuron per request, deployed env plus the
variables below, `--max-num-seqs 10`):

| setting | poles/s @8 | @16 | parse | greedy vs deployed (24 fixed requests) |
|---|---|---|---|---|
| deployed (re-measured) | 0.98 | 1.01 | 1.00 | floor: 4/24 identical c1 vs c8, divergence margins <= 0.375 nats |
| `VLLM_GGUF_W8A8=128`, all four types | 1.15 | 1.15 | 1.00 | 4-5/24 identical; 2-4 divergences at 0.5-0.875 nats |
| `VLLM_GGUF_W8A8=128` (Q4_K, Q5_K) | 1.13 | 1.11 | 1.00 | 2-3/24 identical; median margin 0.125; 2 divergences at 0.5-0.75 nats |
