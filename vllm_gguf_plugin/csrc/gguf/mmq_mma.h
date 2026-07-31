// Interface to the vendored llama.cpp MMA (tensor-core) MMQ kernels.
//
// Deliberately free of any ggml type: `llamacpp/ggml-common.h` and the plugin's
// own `ggml-common.h` are different vintages of the same header and define the
// same block structs, so they must never meet in one translation unit. Callers
// see plain integers here; everything ggml stays inside mmq_mma.cu.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace gguf_mma {

// Element type of the activations/output as seen by the caller.
enum class DType : int { F32 = 0, F16 = 1, BF16 = 2 };

// True when the vendored MMA kernels can serve this ggml type on this device.
// False for types we did not instantiate, or hardware without the MMA path,
// in which case the caller must keep using the legacy DP4A kernels.
bool supported(int ggml_type_id, int device);

// Bytes of scratch the q8_1 activation buffer needs for this shape.
size_t quantized_activation_bytes(int64_t M, int64_t K, int device);

// C[M, N] = A[M, K] * dequant(W)[N, K]^T, with A quantized to q8_1 internally.
//
// `w` is the raw ggml-quantized weight, N rows of K values.
// `x_f32` must be M*K floats, `y_f32` M*N floats, `q8` at least
// quantized_activation_bytes(). All are caller-owned device memory.
void launch(const void * w, int ggml_type_id, const float * x_f32, float * y_f32,
            void * q8, int64_t M, int64_t N, int64_t K, cudaStream_t stream);

// Elementwise dtype conversions, so callers need not own a conversion kernel.
void to_f32(const void * src, DType src_dtype, float * dst, int64_t n,
            cudaStream_t stream);
void from_f32(const float * src, void * dst, DType dst_dtype, int64_t n,
              cudaStream_t stream);

}  // namespace gguf_mma
