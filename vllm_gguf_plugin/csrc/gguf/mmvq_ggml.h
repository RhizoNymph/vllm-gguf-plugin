// Interface to the vendored llama.cpp MMVQ kernels (mmvq_ggml.cu).
//
// Free of ggml types for the same reason as mmq_mma.h: the plugin's own
// ggml-common.h and llamacpp/ggml-common.h must never meet in one TU.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace gguf_mmvq {

// True when the vendored kernel serves this ggml type at this batch size
// (1..MMVQ_MAX_BATCH_SIZE rows).
bool supported(int ggml_type_id, int64_t ncols_dst);

// y_f32[M, N] = q8[M, K] * dequant(w)[N, K]^T. `q8` holds M rows of q8_1 blocks,
// each row padded to 512 values (the layout of the plugin's quantize_row_q8_1_cuda).
void launch(const void * w, int ggml_type_id, const void * q8, float * y_f32,
            int64_t M, int64_t N, int64_t K, cudaStream_t stream);

}  // namespace gguf_mmvq
