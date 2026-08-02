// Activation quantization for the MMA MMQ path, templated on the input dtype.
//
// Derived from llamacpp/quantize.cu `quantize_mmq_q8_1` (see
// llamacpp/LLAMA_CPP_VERSION.txt). Upstream's version takes `const float *` and
// loads via `float4`, so feeding it bf16/fp16 activations meant staging the
// whole M x K matrix through an fp32 buffer first -- 142 MB for the largest FFN
// at M=2048, which is most of what this path was supposed to save over
// dequantize-then-cuBLAS.
//
// This lives OUTSIDE llamacpp/ deliberately: that directory is verbatim so
// re-vendoring stays a copy rather than a merge. Only the load is changed; the
// q8_1 block layout, the scale/sum reductions and the write-back are upstream's,
// and the layout constants come from the vendored headers rather than being
// duplicated here, so a layout change upstream is a compile error rather than
// silent corruption.
//
// RE-VENDOR CHECKLIST: if `quantize_mmq_q8_1` or `block_q8_1_mmq` changes
// upstream, re-derive this kernel. `mmq_mma.cu` static_asserts what it can, but
// the kernel body itself has to be eyeballed.

#include "llamacpp/common.cuh"
#include "llamacpp/mmq.cuh"
#include "llamacpp/quantize.cuh"

#include "mmq_mma.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace gguf_mma {
namespace {

// Load four consecutive elements as floats. For 16-bit types this is one 8-byte
// vector load, matching the coalescing upstream gets from float4 on fp32.
template <typename src_t>
__device__ __forceinline__ float4 load4(const src_t * p);

template <>
__device__ __forceinline__ float4 load4<float>(const float * p) {
    return *((const float4 *) p);
}

template <>
__device__ __forceinline__ float4 load4<__half>(const __half * p) {
    const __half2 * h2 = (const __half2 *) p;
    const float2 a = __half22float2(h2[0]);
    const float2 b = __half22float2(h2[1]);
    return make_float4(a.x, a.y, b.x, b.y);
}

template <>
__device__ __forceinline__ float4 load4<__nv_bfloat16>(const __nv_bfloat16 * p) {
    const __nv_bfloat162 * b2 = (const __nv_bfloat162 *) p;
    const float2 a = __bfloat1622float2(b2[0]);
    const float2 b = __bfloat1622float2(b2[1]);
    return make_float4(a.x, a.y, b.x, b.y);
}

// Body follows llamacpp/quantize.cu; only the operand load differs. `ids` and
// the channel/sample dimensions are dropped -- this path is 2-D and non-MoE.
template <mmq_q8_1_ds_layout ds_layout, typename src_t>
__global__ void quantize_mmq_q8_1_typed(
        const src_t * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t ne0, const int ne1) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t) blockDim.x*blockIdx.y + threadIdx.x)*4;
    if (i0 >= ne0) {
        return;
    }

    const int64_t i01 = blockIdx.x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib = (i0 / (4*QK8_1))*ne1 + blockIdx.x;
    const int64_t iqs = i0 % (4*QK8_1);

    const float4 xi = i0 < ne00 ? load4<src_t>(x + i01*s01 + i0)
                                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }
        y[ib].d2s6[2 + iqs/16] = sum;
        if (iqs % 64 != 0) {
            return;
        }
        y[ib].d2s6[iqs/64] = 1.0f / d_inv;
        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;
    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32] = d;
    }
}

template <typename src_t>
void launch_typed(const src_t * x, void * vy, ggml_type type_src0, int64_t K,
                  int64_t k_padded, int64_t M, cudaStream_t stream) {
    GGML_ASSERT(K % 4 == 0);
    GGML_ASSERT(k_padded % (4*QK8_1) == 0);

    const int64_t block_num_y =
        (k_padded + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(M, block_num_y, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);

    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1_typed<MMQ_Q8_1_DS_LAYOUT_D4, src_t>
                <<<num_blocks, block_size, 0, stream>>>(x, vy, K, K, k_padded, M);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1_typed<MMQ_Q8_1_DS_LAYOUT_DS4, src_t>
                <<<num_blocks, block_size, 0, stream>>>(x, vy, K, K, k_padded, M);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1_typed<MMQ_Q8_1_DS_LAYOUT_D2S6, src_t>
                <<<num_blocks, block_size, 0, stream>>>(x, vy, K, K, k_padded, M);
            break;
        default:
            GGML_ABORT("unsupported q8_1 ds layout");
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void quantize_activations(const void * x, DType dtype, void * q8, int ggml_type_id,
                          int64_t M, int64_t K, cudaStream_t stream) {
    const ggml_type type = (ggml_type) ggml_type_id;
    const int64_t k_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
    switch (dtype) {
        case DType::F32:
            launch_typed<float>((const float *) x, q8, type, K, k_padded, M, stream);
            break;
        case DType::F16:
            launch_typed<__half>((const __half *) x, q8, type, K, k_padded, M, stream);
            break;
        case DType::BF16:
            launch_typed<__nv_bfloat16>((const __nv_bfloat16 *) x, q8, type, K,
                                        k_padded, M, stream);
            break;
    }
}

}  // namespace gguf_mma
