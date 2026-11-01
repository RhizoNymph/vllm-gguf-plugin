// torch (stable ABI) entry points for the W8A8 prefill GEMM.
//
//   ggml_mul_mat_w8a8   the whole path: requantise W, quantise X, GEMM.
//   w8a8_quantize_act / w8a8_requant_weight / w8a8_gemm
//                       the three stages with caller-provided outputs, for
//                       tests and per-stage timing.

#include "w8a8.h"

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <optional>

using torch::headeronly::ScalarType;
using torch::stable::Tensor;
using torch::stable::accelerator::DeviceGuard;

namespace {

cudaStream_t current_stream(int32_t device_index) {
  void* raw = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw));
  return static_cast<cudaStream_t>(raw);
}

gguf_w8a8::Group to_group(int64_t g) {
  STD_TORCH_CHECK(g == 128 || g == 256,
                  "w8a8: group must be 128 or 256, got ", g);
  return static_cast<gguf_w8a8::Group>(g);
}

gguf_w8a8::WeightType to_weight_type(int64_t t) {
  STD_TORCH_CHECK(t == 8 || t == 12 || t == 13 || t == 14,
                  "w8a8: unsupported GGUF weight type ", t,
                  " (Q8_0, Q4_K, Q5_K, Q6_K only)");
  return static_cast<gguf_w8a8::WeightType>(t);
}

gguf_w8a8::DType to_dtype(ScalarType s) {
  STD_TORCH_CHECK(s == ScalarType::BFloat16 || s == ScalarType::Half,
                  "w8a8: activations must be bf16 or fp16");
  return s == ScalarType::BFloat16 ? gguf_w8a8::DType::BF16
                                   : gguf_w8a8::DType::F16;
}

void check_cuda_contig(const Tensor& t, const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "w8a8: ", name,
                  " must be a contiguous CUDA tensor");
}

void check_launch() {
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "w8a8: kernel launch failed: ",
                  cudaGetErrorString(err));
}

}  // namespace

bool w8a8_supported(int64_t type, int64_t row, int64_t col, int64_t group) {
  if (!(type == 8 || type == 12 || type == 13 || type == 14)) return false;
  if (!(group == 128 || group == 256)) return false;
  return gguf_w8a8::shape_supported(static_cast<gguf_w8a8::WeightType>(type), row,
                                    col, static_cast<gguf_w8a8::Group>(group));
}

Tensor ggml_mul_mat_w8a8(Tensor W, Tensor X, int64_t type, int64_t row,
                         int64_t group) {
  check_cuda_contig(W, "W");
  check_cuda_contig(X, "X");
  const auto g = to_group(group);
  const auto wt = to_weight_type(type);
  const auto dt = to_dtype(X.scalar_type());
  const int64_t M = X.sizes()[0], K = X.sizes()[1], N = row;
  STD_TORCH_CHECK(gguf_w8a8::shape_supported(wt, N, K, g),
                  "w8a8: unsupported shape N=", N, " K=", K, " group=", group);
  const int32_t dev = X.get_device_index();
  const DeviceGuard guard(dev);
  cudaStream_t stream = current_stream(dev);

  Tensor Y = torch::stable::new_empty(X, {M, N}, X.scalar_type());
  if (M == 0) return Y;
  Tensor bq = torch::stable::new_empty(X, {N, K}, ScalarType::Char);
  Tensor sb = torch::stable::new_empty(X, {K / group, N}, ScalarType::Float);
  Tensor aq = torch::stable::new_empty(X, {M, K}, ScalarType::Char);
  Tensor sa = torch::stable::new_empty(
      X, {K / group, gguf_w8a8::act_scale_ld(M)}, ScalarType::Float);

  gguf_w8a8::requantize_weight(W.data_ptr(), wt, (int8_t*)bq.data_ptr(),
                               (float*)sb.data_ptr(), N, K, g, stream);
  gguf_w8a8::quantize_activations(X.data_ptr(), dt, (int8_t*)aq.data_ptr(),
                                  (float*)sa.data_ptr(), M, K, g, stream);
  gguf_w8a8::gemm((const int8_t*)aq.data_ptr(), (const float*)sa.data_ptr(),
                  (const int8_t*)bq.data_ptr(), (const float*)sb.data_ptr(),
                  Y.data_ptr(), dt, M, N, K, g, stream);
  check_launch();
  return Y;
}

void w8a8_quantize_act(Tensor X, Tensor aq, Tensor sa, int64_t group) {
  check_cuda_contig(X, "X");
  const auto g = to_group(group);
  const int64_t M = X.sizes()[0], K = X.sizes()[1];
  STD_TORCH_CHECK(K % 256 == 0, "w8a8: K must be a multiple of 256");
  STD_TORCH_CHECK(aq.numel() == M * K && aq.scalar_type() == ScalarType::Char,
                  "w8a8: aq must be int8 [M, K]");
  STD_TORCH_CHECK(sa.numel() == K / group * gguf_w8a8::act_scale_ld(M) &&
                      sa.scalar_type() == ScalarType::Float,
                  "w8a8: sa must be fp32 [K / G, act_scale_ld(M)]");
  const int32_t dev = X.get_device_index();
  const DeviceGuard guard(dev);
  gguf_w8a8::quantize_activations(X.data_ptr(), to_dtype(X.scalar_type()),
                                  (int8_t*)aq.data_ptr(), (float*)sa.data_ptr(),
                                  M, K, g, current_stream(dev));
  check_launch();
}

void w8a8_requant_weight(Tensor W, Tensor bq, Tensor sb, int64_t type,
                         int64_t group) {
  check_cuda_contig(W, "W");
  const auto g = to_group(group);
  const auto wt = to_weight_type(type);
  const int64_t N = bq.sizes()[0], K = bq.sizes()[1];
  STD_TORCH_CHECK(gguf_w8a8::shape_supported(wt, N, K, g),
                  "w8a8: unsupported shape N=", N, " K=", K);
  STD_TORCH_CHECK(bq.scalar_type() == ScalarType::Char &&
                      sb.numel() == K / group * N &&
                      sb.scalar_type() == ScalarType::Float,
                  "w8a8: bq must be int8 [N, K], sb fp32 [K / G, N]");
  const int32_t dev = W.get_device_index();
  const DeviceGuard guard(dev);
  gguf_w8a8::requantize_weight(W.data_ptr(), wt, (int8_t*)bq.data_ptr(),
                               (float*)sb.data_ptr(), N, K, g,
                               current_stream(dev));
  check_launch();
}

void w8a8_gemm(Tensor aq, Tensor sa, Tensor bq, Tensor sb, Tensor Y,
               int64_t group) {
  const auto g = to_group(group);
  const int64_t M = aq.sizes()[0], K = aq.sizes()[1], N = bq.sizes()[0];
  STD_TORCH_CHECK(bq.sizes()[1] == K && Y.sizes()[0] == M && Y.sizes()[1] == N,
                  "w8a8: shape mismatch");
  STD_TORCH_CHECK(N % gguf_w8a8::kTileN == 0 && K % group == 0 &&
                      K % gguf_w8a8::kTileK == 0,
                  "w8a8: unsupported shape");
  const int32_t dev = aq.get_device_index();
  const DeviceGuard guard(dev);
  gguf_w8a8::gemm((const int8_t*)aq.data_ptr(), (const float*)sa.data_ptr(),
                  (const int8_t*)bq.data_ptr(), (const float*)sb.data_ptr(),
                  Y.data_ptr(), to_dtype(Y.scalar_type()), M, N, K, g,
                  current_stream(dev));
  check_launch();
}
