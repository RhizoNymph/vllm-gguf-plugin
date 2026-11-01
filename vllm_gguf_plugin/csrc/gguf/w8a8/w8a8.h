// Group-scaled int8 x int8 tensor-core GEMM for prefill-sized GGUF linears.
//
//   Y[M, N] = sum_g  sa[g, m] * sb[g, n] * sum_{k in g} Aq[m, k] * Bq[n, k]
//
// A (activations) and B (the GGUF weight, re-quantised on every call) are
// symmetric int8 with one fp32 scale per (row, G consecutive K values). Integer
// partials accumulate exactly inside a group on the int8 MMA and are folded
// into fp32 once per group, so the tensor cores run at the int8 rate (about
// twice bf16 on sm_86) while each group keeps its own scale.
//
// Free of torch and ggml types, so it can be used from any translation unit.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace gguf_w8a8 {

// K values sharing one scale. The int32 partial of one group must stay below
// 2^22 for the accumulator-seeding trick in gemm.cu (G * 127 * 127 < 2^22), so
// 256 is the largest group the GEMM accepts; the smallest is one k-tile (128).
enum class Group : int { G128 = 128, G256 = 256 };

// Element type of activations and output.
enum class DType : int { F16 = 1, BF16 = 2 };

// GGUF weight types the re-quantiser decodes (ggml type ids).
enum class WeightType : int { Q8_0 = 8, Q4_K = 12, Q5_K = 13, Q6_K = 14 };

// GEMM tile: N must be a multiple of kTileN, K of kTileK and of the group.
constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileK = 128;

// ggml values per block of a weight type (32 for Q8_0, 256 for the K-quants).
int block_values(WeightType type);

// True when (type, N, K, group) is a shape this path computes.
bool shape_supported(WeightType type, int64_t N, int64_t K, Group group);

// Leading dimension of the activation scale matrix sa[K / G][lda]: M rounded
// up to 4, so every 16-byte row chunk the GEMM copies is aligned.
inline int64_t act_scale_ld(int64_t M) { return (M + 3) / 4 * 4; }

// x[M, K] (dtype) -> aq[M, K] int8, sa[K / G][act_scale_ld(M)] fp32.
void quantize_activations(const void* x, DType dtype, int8_t* aq, float* sa,
                          int64_t M, int64_t K, Group group,
                          cudaStream_t stream);

// GGUF weight w (N rows of K values in `type` blocks) -> bq[N, K] int8,
// sb[K / G][N] fp32.
void requantize_weight(const void* w, WeightType type, int8_t* bq, float* sb,
                       int64_t N, int64_t K, Group group, cudaStream_t stream);

// y[M, N] (dtype) = the group-scaled product above.
void gemm(const int8_t* aq, const float* sa, const int8_t* bq, const float* sb,
          void* y, DType dtype, int64_t M, int64_t N, int64_t K, Group group,
          cudaStream_t stream);

}  // namespace gguf_w8a8
