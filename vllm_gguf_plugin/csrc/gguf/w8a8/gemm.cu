// Group-scaled int8 tensor-core GEMM (sm_80+ mma.sync m16n8k32 s8).
//
// CTA tile 128 x 128 x 128, 8 warps (2 along M x 4 along N, 64 x 32 each),
// 3-stage cp.async pipeline (99 KB of shared memory, the sm_86 maximum). A and B
// are both K-major; shared-memory rows are 128 bytes, their eight 16-byte chunks
// XOR-swizzled by row so ldmatrix is conflict-free. Fragments for the next
// 32-wide k step are loaded while the current step's MMAs issue.
//
// Per group of G K values each thread keeps an int32 partial per output; the
// first MMA of a group is seeded with 0x4B400000, the bit pattern of 1.5 * 2^23.
// Integer addition onto that pattern then *is* the fp32 value 1.5 * 2^23 + p for
// |p| < 2^22, so folding the partial into the fp32 accumulator costs one FADD,
// one FMUL (sa * sb) and one FFMA -- no int -> float conversion, which runs at a
// quarter of the FP32 rate on sm_86.

#include "w8a8.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <type_traits>

namespace gguf_w8a8 {

namespace {

constexpr int BM = kTileM, BN = kTileN, BK = kTileK;
constexpr int STAGES = 3;
constexpr int THREADS = 256;
constexpr int WARP_M = 64, WARP_N = 32;
constexpr int MI = WARP_M / 16;  // m16 tiles per warp
constexpr int NI = WARP_N / 8;   // n8 tiles per warp
constexpr int A_BYTES = BM * BK, B_BYTES = BN * BK;
constexpr int SA_BYTES = BM * 4, SB_BYTES = BN * 4;
constexpr int STAGE_BYTES = A_BYTES + B_BYTES + SA_BYTES + SB_BYTES;
constexpr int SMEM_BYTES = STAGES * STAGE_BYTES;
constexpr int MAGIC_I = 0x4B400000;
constexpr float MAGIC_F = 12582912.0f;

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

// 128-byte rows of eight 16-byte chunks; XOR with the low row bits spreads the
// 8 rows one ldmatrix phase reads over all 8 bank groups.
__device__ __forceinline__ int swz(int row, int chunk) {
  return row * BK + ((chunk ^ (row & 7)) << 4);
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src,
                                           int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
               "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int n>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr));
}

__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4],
                                       uint32_t b0, uint32_t b1,
                                       const int (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

__device__ __forceinline__ void store2(__nv_bfloat16* p, float a, float b) {
  *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b);
}
__device__ __forceinline__ void store2(__half* p, float a, float b) {
  *reinterpret_cast<__half2*>(p) = __floats2half2_rn(a, b);
}

struct Args {
  const int8_t* A;   // [M, K]
  const float* sa;   // [K / G, lda]
  const int8_t* B;   // [N, K]
  const float* sb;   // [K / G, N]
  void* C;           // [M, N]
  int M, N, K, lda, tiles_m;
};

// Copies k-tile `kt` of this CTA's A, B and the scales of the group holding it.
template <int G>
__device__ __forceinline__ void load_stage(const Args& p, uint8_t* stage,
                                           int m0, int n0, int kt, int tid) {
  const uint32_t sA = smem_addr(stage);
  const uint32_t sB = sA + A_BYTES;
  const uint32_t sSa = sB + B_BYTES;
  const uint32_t sSb = sSa + SA_BYTES;
  const int64_t k0 = (int64_t)kt * BK;
  constexpr int CHUNKS = BK / 16;
#pragma unroll
  for (int r = 0; r < BM * CHUNKS / THREADS; ++r) {
    const int i = tid + r * THREADS;  // 16-byte chunks of one operand tile
    const int row = i / CHUNKS, ch = i % CHUNKS;
    const int gm = m0 + row;
    const bool ok = gm < p.M;
    const int8_t* srcA = p.A + (ok ? (int64_t)gm * p.K + k0 + ch * 16 : 0);
    cp_async16(sA + swz(row, ch), srcA, ok ? 16 : 0);
    const int8_t* srcB = p.B + (int64_t)(n0 + row) * p.K + k0 + ch * 16;
    cp_async16(sB + swz(row, ch), srcB, 16);
  }
  const int64_t g = k0 / G;
  if (tid < BM / 4) {
    const int r0 = m0 + tid * 4;
    const int valid = max(0, min(4, p.M - r0));
    const float* src = p.sa + g * p.lda + (valid ? r0 : 0);
    cp_async16(sSa + tid * 16, src, valid * 4);
  } else if (tid < BM / 4 + BN / 4) {
    const int c = tid - BM / 4;
    cp_async16(sSb + c * 16, p.sb + g * p.N + n0 + c * 4, 16);
  }
}

template <int G, typename OutT>
__global__ void __launch_bounds__(THREADS, 1) gemm_kernel(const Args p) {
  static_assert(G % BK == 0, "group must be a whole number of k-tiles");
  constexpr int TILES_PER_GROUP = G / BK;
  extern __shared__ __align__(128) uint8_t smem[];

  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int warp_m = warp / 4, warp_n = warp % 4;
  const int tile = blockIdx.x;
  const int m0 = (tile % p.tiles_m) * BM;
  const int n0 = (tile / p.tiles_m) * BN;
  const int KT = p.K / BK;

  float acc[MI][NI][4];
  int part[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;
  const int magic[4] = {MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I};

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) load_stage<G>(p, smem + s * STAGE_BYTES, m0, n0, s, tid);
    cp_async_commit();
  }

  // Per-lane ldmatrix row / chunk offsets (see the fragment layouts of
  // mma.m16n8k32: A rows g / g+8, k halves 0-15 / 16-31; B likewise along n).
  const int a_row = warp_m * WARP_M + (lane & 7) + ((lane >> 3) & 1) * 8;
  const int a_ch = lane >> 4;
  const int b_row = warp_n * WARP_N + (lane & 7) + (lane >> 4) * 8;
  const int b_ch = (lane >> 3) & 1;

  for (int kg = 0; kg < KT; kg += TILES_PER_GROUP) {
#pragma unroll
    for (int j = 0; j < TILES_PER_GROUP; ++j) {
      const int kt = kg + j;
      cp_async_wait<STAGES - 2>();
      __syncthreads();
      {
        const int pf = kt + STAGES - 1;
        if (pf < KT)
          load_stage<G>(p, smem + (pf % STAGES) * STAGE_BYTES, m0, n0, pf, tid);
        cp_async_commit();
      }
      const uint8_t* stage = smem + (kt % STAGES) * STAGE_BYTES;
      const uint32_t sA = smem_addr(stage);
      const uint32_t sB = sA + A_BYTES;
      constexpr int KSTEPS = BK / 32;
      uint32_t af[2][MI][4];
      uint32_t bf[2][NI / 2][4];
      auto load_frags = [&](int buf, int ks) {
#pragma unroll
        for (int i = 0; i < MI; ++i)
          ldsm_x4(af[buf][i], sA + swz(a_row + i * 16, ks * 2 + a_ch));
#pragma unroll
        for (int i = 0; i < NI / 2; ++i)
          ldsm_x4(bf[buf][i], sB + swz(b_row + i * 16, ks * 2 + b_ch));
      };
      load_frags(0, 0);
#pragma unroll
      for (int ks = 0; ks < KSTEPS; ++ks) {
        const int cur = ks & 1;
        if (ks + 1 < KSTEPS) load_frags(cur ^ 1, ks + 1);
#pragma unroll
        for (int i = 0; i < MI; ++i)
#pragma unroll
          for (int n = 0; n < NI; ++n) {
            const uint32_t b0 = bf[cur][n / 2][(n & 1) * 2];
            const uint32_t b1 = bf[cur][n / 2][(n & 1) * 2 + 1];
            if (j == 0 && ks == 0)
              mma_s8(part[i][n], af[cur][i], b0, b1, magic);
            else
              mma_s8(part[i][n], af[cur][i], b0, b1, part[i][n]);
          }
      }
      if (j == TILES_PER_GROUP - 1) {
        // Fold this group's partials: acc += sa[row] * sb[col] * partial.
        const float* sSa = reinterpret_cast<const float*>(stage + A_BYTES + B_BYTES);
        const float* sSb = sSa + BM;
        const int g = lane >> 2, t = lane & 3;
        float ra[MI][2], cb[NI][2];
#pragma unroll
        for (int i = 0; i < MI; ++i) {
          ra[i][0] = sSa[warp_m * WARP_M + i * 16 + g];
          ra[i][1] = sSa[warp_m * WARP_M + i * 16 + g + 8];
        }
#pragma unroll
        for (int n = 0; n < NI; ++n) {
          const float2 v =
              *reinterpret_cast<const float2*>(sSb + warp_n * WARP_N + n * 8 + 2 * t);
          cb[n][0] = v.x;
          cb[n][1] = v.y;
        }
#pragma unroll
        for (int i = 0; i < MI; ++i)
#pragma unroll
          for (int n = 0; n < NI; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const float v = __int_as_float(part[i][n][e]) - MAGIC_F;
              acc[i][n][e] = fmaf(ra[i][e >> 1] * cb[n][e & 1], v, acc[i][n][e]);
            }
      }
    }
  }
  cp_async_wait<0>();

  OutT* C = reinterpret_cast<OutT*>(p.C);
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = m0 + warp_m * WARP_M + i * 16 + g + h * 8;
      if (row >= p.M) continue;
#pragma unroll
      for (int n = 0; n < NI; ++n) {
        const int col = n0 + warp_n * WARP_N + n * 8 + 2 * t;
        store2(C + (int64_t)row * p.N + col, acc[i][n][h * 2], acc[i][n][h * 2 + 1]);
      }
    }
}

template <int G, typename OutT>
void launch(const Args& a, cudaStream_t stream) {
  // Per device and cheap; set on every call rather than tracking devices.
  cudaFuncSetAttribute(gemm_kernel<G, OutT>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
  const int tiles = a.tiles_m * (a.N / BN);
  gemm_kernel<G, OutT><<<tiles, THREADS, SMEM_BYTES, stream>>>(a);
}

}  // namespace

bool shape_supported(WeightType type, int64_t N, int64_t K, Group group) {
  const int64_t G = static_cast<int64_t>(group);
  return N > 0 && N % BN == 0 && K > 0 && K % BK == 0 && K % G == 0 &&
         K % block_values(type) == 0 && K % 256 == 0 && N < (1 << 30) &&
         K < (1 << 30);
}

void gemm(const int8_t* aq, const float* sa, const int8_t* bq, const float* sb,
          void* y, DType dtype, int64_t M, int64_t N, int64_t K, Group group,
          cudaStream_t stream) {
  if (M == 0) return;
  Args a{aq, sa, bq, sb, y, (int)M, (int)N, (int)K, (int)act_scale_ld(M),
         (int)((M + BM - 1) / BM)};
  auto run = [&](auto g) {
    constexpr int G = decltype(g)::value;
    if (dtype == DType::BF16)
      launch<G, __nv_bfloat16>(a, stream);
    else
      launch<G, __half>(a, stream);
  };
  switch (group) {
    case Group::G128: run(std::integral_constant<int, 128>{}); break;
    case Group::G256: run(std::integral_constant<int, 256>{}); break;
  }
}

}  // namespace gguf_w8a8
