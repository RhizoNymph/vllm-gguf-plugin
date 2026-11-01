// Launcher for the vendored llama.cpp MMVQ (multi-column quantized GEMV) kernels.
//
// Why this exists: the plugin's own mmvq.cuh (llama.cpp b2899) handles one
// activation row per pass, so a decode batch of M rows reads every weight M
// times. Upstream's kernel reads each weight block once for up to
// MMVQ_MAX_BATCH_SIZE (8) activation rows, which is what llama.cpp itself uses
// for small batches.
//
// `llamacpp/mmvq.cu` is included verbatim so its static launch templates are in
// this translation unit; only `mul_mat_vec_q_switch_type` is called, for the
// plain 2-D case. Its ggml-tensor entry points are compiled but never called;
// the ggml symbols only they reference are stubbed below.

#include "llamacpp/mmvq.cu"

#include "mmvq_ggml.h"

// Symbols referenced only by mmvq.cu's ggml-tensor entry points
// (ggml_cuda_mul_mat_vec_q, ggml_cuda_op_mul_mat_vec_q), which the plugin never
// calls. They must still resolve or the extension fails to dlopen and the plugin
// silently falls back to Triton; each aborts if ever reached.
size_t ggml_nbytes(const struct ggml_tensor *) { GGML_ABORT("ggml_nbytes: not available in the plugin"); }
size_t ggml_type_size(enum ggml_type) { GGML_ABORT("ggml_type_size: not available in the plugin"); }
bool ggml_is_contiguously_allocated(const struct ggml_tensor *) {
    GGML_ABORT("ggml_is_contiguously_allocated: not available in the plugin");
}
bool ggml_are_same_stride(const struct ggml_tensor *, const struct ggml_tensor *) {
    GGML_ABORT("ggml_are_same_stride: not available in the plugin");
}
size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t, const struct ggml_tensor *) {
    GGML_ABORT("ggml_backend_buffer_get_alloc_size: not available in the plugin");
}
enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t) {
    GGML_ABORT("ggml_backend_buffer_get_usage: not available in the plugin");
}
void ggml_cuda_set_device(int device) { CUDA_CHECK(cudaSetDevice(device)); }

namespace gguf_mmvq {

bool supported(int ggml_type_id, int64_t ncols_dst) {
    if (ncols_dst < 1 || ncols_dst > MMVQ_MAX_BATCH_SIZE) {
        return false;
    }
    switch ((ggml_type) ggml_type_id) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0: case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

void launch(const void * w, int ggml_type_id, const void * q8, float * y,
            int64_t M, int64_t N, int64_t K, cudaStream_t stream) {
    const ggml_type type = (ggml_type) ggml_type_id;
    // Activations are q8_1 blocks, each row padded to MATRIX_ROW_PADDING values
    // (the plugin's quantize_row_q8_1_cuda pads to 512, the same constant).
    const int64_t k_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
    const int stride_row_x = (int) (K / ggml_blck_size(type));
    const int stride_col_y = (int) (k_padded / QK8_1);
    const ggml_cuda_mm_fusion_args_device fusion{};
    mul_mat_vec_q_switch_type(
        w, type, q8, /*ids=*/nullptr, fusion, y,
        /*ncols_x=*/(int) K, /*nrows_x=*/(int) N, /*ncols_dst=*/(int) M,
        stride_row_x, stride_col_y, /*stride_col_dst=*/(int) N,
        /*nchannels_x=*/1, /*nchannels_y=*/1, /*nchannels_dst=*/1,
        /*stride_channel_x=*/1, /*stride_channel_y=*/1, /*stride_channel_dst=*/1,
        /*nsamples_x=*/1, /*nsamples_dst=*/1,
        /*stride_sample_x=*/1, /*stride_sample_y=*/1, /*stride_sample_dst=*/1,
        /*ids_stride=*/0, stream);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace gguf_mmvq
