// Activation quantiser and GGUF -> int8 weight re-quantiser for the W8A8 GEMM.
//
// Both write symmetric int8 with one fp32 scale (amax / 127) per (row, group),
// the scales stored group-major ([K / G][rows]) so a GEMM tile's scales for one
// group are contiguous.

#include "w8a8.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <type_traits>

namespace gguf_w8a8 {

namespace {

__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

// amax over `lanes` consecutive lanes (a power of two, aligned within the warp).
template <int lanes>
__device__ __forceinline__ float group_amax(float v) {
#pragma unroll
  for (int off = lanes / 2; off > 0; off /= 2) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  }
  return v;
}

__device__ __forceinline__ int8_t quant(float v, float inv) {
  const int q = __float2int_rn(v * inv);
  return (int8_t)max(-127, min(127, q));
}

// ---------------------------------------------------------------- activations
// One thread per 8 consecutive values (one 16-byte load); G / 8 lanes per group.
template <typename T, int G>
__global__ void quantize_act_kernel(const T* __restrict__ x,
                                    int8_t* __restrict__ aq,
                                    float* __restrict__ sa, int64_t M,
                                    int64_t K, int64_t lda) {
  constexpr int lanes = G / 8;
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t per_row = K / 8;
  // Full warps only: M * K / 8 is a multiple of 32 because K % 256 == 0.
  if (i >= M * per_row) return;
  const int64_t m = i / per_row;
  const int64_t k8 = i % per_row;

  const uint4 raw = *reinterpret_cast<const uint4*>(x + m * K + k8 * 8);
  const T* v = reinterpret_cast<const T*>(&raw);
  float f[8];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    f[j] = to_float(v[j]);
    amax = fmaxf(amax, fabsf(f[j]));
  }
  amax = group_amax<lanes>(amax);
  const float inv = amax > 0.f ? 127.f / amax : 0.f;
  uint2 out;
  int8_t* o = reinterpret_cast<int8_t*>(&out);
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = quant(f[j], inv);
  *reinterpret_cast<uint2*>(aq + m * K + k8 * 8) = out;
  if (k8 % lanes == 0) sa[(k8 / lanes) * lda + m] = amax / 127.f;
}

// --------------------------------------------------------------- GGUF decode
// Each decoder writes the 32 dequantised values of chunk `c` (32 consecutive K
// values) of one block. Layouts follow ggml's dequantize_row_* (ggml-quants.c).

__device__ __forceinline__ float half_at(const uint8_t* p) {
  return __half2float(*reinterpret_cast<const __half*>(p));
}

// Unaligned-safe: Q6_K (210 B) and Q8_0 (34 B) blocks are only 2-byte aligned.
__device__ __forceinline__ void load32_u16(const uint8_t* p, uint8_t* dst) {
  const uint16_t* s = reinterpret_cast<const uint16_t*>(p);
  uint16_t* d = reinterpret_cast<uint16_t*>(dst);
#pragma unroll
  for (int j = 0; j < 16; ++j) d[j] = s[j];
}

__device__ __forceinline__ void load32_u128(const uint8_t* p, uint8_t* dst) {
  const uint4* s = reinterpret_cast<const uint4*>(p);
  uint4* d = reinterpret_cast<uint4*>(dst);
  d[0] = s[0];
  d[1] = s[1];
}

// ggml get_scale_min_k4 over the 12 packed scale bytes of a Q4_K / Q5_K block.
__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q,
                                             int& sc, int& mn) {
  if (j < 4) {
    sc = q[j] & 63;
    mn = q[j + 4] & 63;
  } else {
    sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// Q4_K, 144 B: half d, half dmin, u8 scales[12], u8 qs[128].
__device__ __forceinline__ void decode_q4_k(const uint8_t* b, int c, float* w) {
  uint8_t sc12[12];
  const uint32_t* s32 = reinterpret_cast<const uint32_t*>(b + 4);
  *reinterpret_cast<uint32_t*>(sc12 + 0) = s32[0];
  *reinterpret_cast<uint32_t*>(sc12 + 4) = s32[1];
  *reinterpret_cast<uint32_t*>(sc12 + 8) = s32[2];
  int sc, mn;
  scale_min_k4(c, sc12, sc, mn);
  const float d = half_at(b) * sc;
  const float m = half_at(b + 2) * mn;
  uint8_t qs[32];
  load32_u128(b + 16 + 32 * (c / 2), qs);
  const int shift = 4 * (c & 1);
#pragma unroll
  for (int l = 0; l < 32; ++l) w[l] = d * ((qs[l] >> shift) & 0xF) - m;
}

// Q5_K, 176 B: half d, half dmin, u8 scales[12], u8 qh[32], u8 qs[128].
__device__ __forceinline__ void decode_q5_k(const uint8_t* b, int c, float* w) {
  uint8_t sc12[12];
  const uint32_t* s32 = reinterpret_cast<const uint32_t*>(b + 4);
  *reinterpret_cast<uint32_t*>(sc12 + 0) = s32[0];
  *reinterpret_cast<uint32_t*>(sc12 + 4) = s32[1];
  *reinterpret_cast<uint32_t*>(sc12 + 8) = s32[2];
  int sc, mn;
  scale_min_k4(c, sc12, sc, mn);
  const float d = half_at(b) * sc;
  const float m = half_at(b + 2) * mn;
  uint8_t qh[32], qs[32];
  load32_u128(b + 16, qh);
  load32_u128(b + 48 + 32 * (c / 2), qs);
  const int shift = 4 * (c & 1);
#pragma unroll
  for (int l = 0; l < 32; ++l) {
    const int q = ((qs[l] >> shift) & 0xF) | (((qh[l] >> c) & 1) << 4);
    w[l] = d * q - m;
  }
}

// Q6_K, 210 B: u8 ql[128], u8 qh[64], i8 scales[16], half d.
__device__ __forceinline__ void decode_q6_k(const uint8_t* b, int c, float* w) {
  const int h = c / 4;
  const int qd = c % 4;
  uint8_t ql[32], qh[32];
  load32_u16(b + h * 64 + (qd & 1) * 32, ql);
  load32_u16(b + 128 + h * 32, qh);
  const int8_t* scales = reinterpret_cast<const int8_t*>(b + 192 + h * 8 + qd * 2);
  const float d = half_at(b + 208);
  const float d0 = d * scales[0];
  const float d1 = d * scales[1];
  const int lshift = 4 * (qd >> 1);
  const int hshift = 2 * qd;
#pragma unroll
  for (int l = 0; l < 32; ++l) {
    const int q = (((ql[l] >> lshift) & 0xF) | (((qh[l] >> hshift) & 3) << 4)) - 32;
    w[l] = (l < 16 ? d0 : d1) * q;
  }
}

// Q8_0, 34 B: half d, i8 qs[32].
__device__ __forceinline__ void decode_q8_0(const uint8_t* b, float* w) {
  uint8_t qs[32];
  load32_u16(b + 2, qs);
  const float d = half_at(b);
#pragma unroll
  for (int l = 0; l < 32; ++l) w[l] = d * (int8_t)qs[l];
}

template <WeightType type>
struct BlockInfo;
template <>
struct BlockInfo<WeightType::Q4_K> {
  static constexpr int values = 256, bytes = 144;
};
template <>
struct BlockInfo<WeightType::Q5_K> {
  static constexpr int values = 256, bytes = 176;
};
template <>
struct BlockInfo<WeightType::Q6_K> {
  static constexpr int values = 256, bytes = 210;
};
template <>
struct BlockInfo<WeightType::Q8_0> {
  static constexpr int values = 32, bytes = 34;
};

// One thread per 32-value chunk; G / 32 lanes per group.
template <WeightType type, int G>
__global__ void requant_weight_kernel(const uint8_t* __restrict__ w,
                                      int8_t* __restrict__ bq,
                                      float* __restrict__ sb, int64_t N,
                                      int64_t K) {
  using info = BlockInfo<type>;
  constexpr int lanes = G / 32;
  constexpr int chunks_per_block = info::values / 32;
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t per_row = K / 32;
  if (i >= N * per_row) return;  // N * K / 32 is a multiple of 32 (K % 256 == 0)
  const int64_t n = i / per_row;
  const int64_t chunk = i % per_row;
  const int64_t blocks_per_row = K / info::values;
  const uint8_t* blk =
      w + (n * blocks_per_row + chunk / chunks_per_block) * info::bytes;
  const int c = (int)(chunk % chunks_per_block);

  float v[32];
  if constexpr (type == WeightType::Q4_K) decode_q4_k(blk, c, v);
  if constexpr (type == WeightType::Q5_K) decode_q5_k(blk, c, v);
  if constexpr (type == WeightType::Q6_K) decode_q6_k(blk, c, v);
  if constexpr (type == WeightType::Q8_0) decode_q8_0(blk, v);

  float amax = 0.f;
#pragma unroll
  for (int l = 0; l < 32; ++l) amax = fmaxf(amax, fabsf(v[l]));
  amax = group_amax<lanes>(amax);
  const float inv = amax > 0.f ? 127.f / amax : 0.f;
  uint4 out[2];
  int8_t* o = reinterpret_cast<int8_t*>(out);
#pragma unroll
  for (int l = 0; l < 32; ++l) o[l] = quant(v[l], inv);
  uint4* dst = reinterpret_cast<uint4*>(bq + n * K + chunk * 32);
  dst[0] = out[0];
  dst[1] = out[1];
  if (chunk % lanes == 0) sb[(chunk / lanes) * N + n] = amax / 127.f;
}

template <typename F>
void with_group(Group group, F&& f) {
  switch (group) {
    case Group::G128: f(std::integral_constant<int, 128>{}); break;
    case Group::G256: f(std::integral_constant<int, 256>{}); break;
  }
}

unsigned blocks_for(int64_t threads, int block) {
  return (unsigned)((threads + block - 1) / block);
}

}  // namespace

int block_values(WeightType type) {
  return type == WeightType::Q8_0 ? 32 : 256;
}

void quantize_activations(const void* x, DType dtype, int8_t* aq, float* sa,
                          int64_t M, int64_t K, Group group,
                          cudaStream_t stream) {
  constexpr int block = 256;
  const int64_t threads = M * K / 8;
  const int64_t lda = act_scale_ld(M);
  with_group(group, [&](auto g) {
    constexpr int G = decltype(g)::value;
    if (dtype == DType::BF16) {
      quantize_act_kernel<__nv_bfloat16, G>
          <<<blocks_for(threads, block), block, 0, stream>>>(
              (const __nv_bfloat16*)x, aq, sa, M, K, lda);
    } else {
      quantize_act_kernel<__half, G>
          <<<blocks_for(threads, block), block, 0, stream>>>(
              (const __half*)x, aq, sa, M, K, lda);
    }
  });
}

void requantize_weight(const void* w, WeightType type, int8_t* bq, float* sb,
                       int64_t N, int64_t K, Group group, cudaStream_t stream) {
  constexpr int block = 256;
  const int64_t threads = N * K / 32;
  const auto* wb = (const uint8_t*)w;
  with_group(group, [&](auto g) {
    constexpr int G = decltype(g)::value;
    const unsigned grid = blocks_for(threads, block);
    switch (type) {
      case WeightType::Q4_K:
        requant_weight_kernel<WeightType::Q4_K, G>
            <<<grid, block, 0, stream>>>(wb, bq, sb, N, K);
        break;
      case WeightType::Q5_K:
        requant_weight_kernel<WeightType::Q5_K, G>
            <<<grid, block, 0, stream>>>(wb, bq, sb, N, K);
        break;
      case WeightType::Q6_K:
        requant_weight_kernel<WeightType::Q6_K, G>
            <<<grid, block, 0, stream>>>(wb, bq, sb, N, K);
        break;
      case WeightType::Q8_0:
        requant_weight_kernel<WeightType::Q8_0, G>
            <<<grid, block, 0, stream>>>(wb, bq, sb, N, K);
        break;
    }
  });
}

}  // namespace gguf_w8a8
