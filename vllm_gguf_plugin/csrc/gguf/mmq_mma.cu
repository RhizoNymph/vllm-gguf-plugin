// Launcher for the vendored llama.cpp MMA (tensor-core) MMQ kernels.
//
// Why this exists: the plugin's own mmq.cuh was copied from llama.cpp b2899 and
// is DP4A-only with a 4-token-deep output tile, so it re-reads the whole weight
// matrix M/4 times and pins at ~5.7 TFLOP/s regardless of batch size. Upstream
// has since rewritten MMQ around int8 tensor-core MMA with a tile up to 128
// tokens wide. This routes prefill-shaped GEMMs at that kernel instead.
//
// Only the 2-D, non-MoE case is handled -- that is what ggml_mul_mat_a8 does.
// MoE keeps its existing path.

#include "llamacpp/common.cuh"
#include "llamacpp/mmq.cuh"
#include "llamacpp/quantize.cuh"

#include "mmq_mma.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace gguf_mma {

namespace {

// ggml type ids the plugin can route here. Mirrors the DECL_MMQ_CASE template
// instances actually compiled in template-instances/ -- adding a type means
// adding an instance file, or the link fails rather than silently degrading.
constexpr ggml_type kSupported[] = {
    GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
};

template <typename dst_t>
__global__ void convert_from_f32(const float * __restrict__ src,
                                 dst_t * __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = (dst_t) src[i];
    }
}

dim3 grid_for(int64_t n, int block) {
    return dim3((unsigned) ((n + block - 1) / block), 1, 1);
}

// The switch mirrors llama.cpp's ggml_cuda_mul_mat_q_switch_type, restricted to
// the instantiated types. `ctx` is never dereferenced: use_stream_k is false, and
// launch_mul_mat_q only touches ctx.pool() inside the stream-k branch.
void dispatch_type(ggml_backend_cuda_context & ctx, const mmq_args & args,
                   cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q4_0: mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream); break;
        case GGML_TYPE_Q4_1: mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream); break;
        case GGML_TYPE_Q5_0: mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream); break;
        case GGML_TYPE_Q5_1: mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream); break;
        case GGML_TYPE_Q8_0: mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream); break;
        case GGML_TYPE_Q2_K: mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream); break;
        case GGML_TYPE_Q3_K: mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream); break;
        case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream); break;
        case GGML_TYPE_Q5_K: mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream); break;
        case GGML_TYPE_Q6_K: mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream); break;
        default: GGML_ABORT("unsupported ggml type for MMA MMQ: %d", (int) args.type_x);
    }
}

}  // namespace

bool supported(int ggml_type_id, int device) {
    const int cc = ggml_cuda_info().devices[device].cc;
    // The whole point is the MMA path; without it the vendored kernel falls back
    // to its own DP4A variant, which is no better than what the plugin already
    // has and is not worth the extra buffers.
    if (!turing_mma_available(cc)) {
        return false;
    }
    for (ggml_type t : kSupported) {
        if ((int) t == ggml_type_id) {
            return true;
        }
    }
    return false;
}

size_t quantized_activation_bytes(int64_t M, int64_t K, int device) {
    const int cc = ggml_cuda_info().devices[device].cc;
    const int64_t k_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
    // Same expression as ggml_cuda_mul_mat_q. The trailing term is slack the
    // kernel may write past the last full tile.
    return (size_t) (M * k_padded) * sizeof(block_q8_1) / QK8_1 +
           (size_t) get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
}

void launch(const void * w, int ggml_type_id, const void * x, DType dtype,
            float * y_f32, void * q8, int64_t M, int64_t N, int64_t K,
            cudaStream_t stream) {
    const ggml_type type = (ggml_type) ggml_type_id;
    const int64_t k_padded = GGML_PAD(K, MATRIX_ROW_PADDING);

    // Reads bf16/fp16 activations directly (mmq_mma_quantize.cu) rather than
    // going through an M*K fp32 staging buffer.
    quantize_activations(x, dtype, q8, ggml_type_id, M, K, stream);

    // Rows of the weight are ggml blocks, so the row stride is in blocks.
    const int64_t stride_row_x = K / ggml_blck_size(type);
    const int64_t stride_col_y = M * k_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));

    const mmq_args args = {
        (const char *) w, type, (const int *) q8, /*ids_dst=*/nullptr,
        /*expert_bounds=*/nullptr, y_f32,
        /*ncols_x=*/K, /*nrows_x=*/N, /*ncols_dst=*/M,
        /*stride_row_x=*/stride_row_x, /*ncols_y=*/M, /*nrows_dst=*/N,
        /*nchannels_x=*/1, /*nchannels_y=*/1,
        /*stride_channel_x=*/N * stride_row_x, /*stride_channel_y=*/stride_col_y,
        /*stride_channel_dst=*/M * N,
        /*nsamples_x=*/1, /*nsamples_y=*/1,
        /*stride_sample_x=*/N * stride_row_x, /*stride_sample_y=*/stride_col_y,
        /*stride_sample_dst=*/M * N,
        // Matches llama.cpp's own condition. Without stream-k the plain tiling
        // path collapses as M grows -- measured 60.5 TFLOP/s at M=128 falling to
        // 5.1 at M=2048, i.e. slower than the DP4A kernel it replaces. It is not
        // an optional tuning knob on this hardware.
        /*use_stream_k=*/turing_mma_available(
            ggml_cuda_info().devices[ggml_cuda_get_device()].cc),
        /*ncols_max=*/M};

    // launch_mul_mat_q takes ggml_backend_cuda_context& and, with stream-k on,
    // really does use ctx.pool() for the fixup scratch (see the single-slot pool
    // in mmq_mma_shim.cu). Heap-allocated and intentionally never freed:
    // ~ggml_backend_cuda_context is declared in common.cuh but defined in
    // ggml-cuda.cu, which we do not vendor, so calling it would not link.
    static ggml_backend_cuda_context * const ctx =
        new ggml_backend_cuda_context(ggml_cuda_get_device());
    dispatch_type(*ctx, args, stream);
    CUDA_CHECK(cudaGetLastError());
}

void from_f32(const float * src, void * dst, DType dst_dtype, int64_t n,
              cudaStream_t stream) {
    constexpr int block = 256;
    switch (dst_dtype) {
        case DType::F32:
            CUDA_CHECK(cudaMemcpyAsync(dst, src, n * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
            break;
        case DType::F16:
            convert_from_f32<<<grid_for(n, block), block, 0, stream>>>(
                src, (__half *) dst, n);
            break;
        case DType::BF16:
            convert_from_f32<<<grid_for(n, block), block, 0, stream>>>(
                src, (__nv_bfloat16 *) dst, n);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace gguf_mma
