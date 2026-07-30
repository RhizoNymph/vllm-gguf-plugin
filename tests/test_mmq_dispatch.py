# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch tests for the `_MMQ_SAFE` prefill threshold in `_fused_mul_mat_gguf`.

Above `_MMQ_SAFE` rows, quantized GEMMs should take the dequantize-then-cuBLAS
path rather than the fused MMQ kernel (see the comment on `_MMQ_SAFE` for the
measurements motivating this). Below it, and for decode-shaped batches, the
existing MMVQ/MMQ routing must be unchanged.

These assert on *which kernel is selected*, so they patch the op entry points
with counters rather than measuring numerics -- correctness of each individual
kernel is already covered by `test_kernels.py`.
"""

import pytest
import torch
from gguf import GGMLQuantizationType

from vllm_gguf_plugin.quantization import linear as gguf_linear

from .utils import get_gguf_sample_tensors, seed_everything

HIDDEN_SIZE = 256
QUANT_TYPE = GGMLQuantizationType.Q4_K


@pytest.fixture
def counting_ops(monkeypatch):
    """Replace the three GEMM entry points with counters that still compute."""
    calls = {"mmvq": 0, "mmq": 0, "dequantize": 0}
    real = {
        "mmvq": gguf_linear.ops.ggml_mul_mat_vec_a8,
        "mmq": gguf_linear.ops.ggml_mul_mat_a8,
        "dequantize": gguf_linear.ops.ggml_dequantize,
    }

    def wrap(name):
        def inner(*args, **kwargs):
            calls[name] += 1
            return real[name](*args, **kwargs)

        return inner

    monkeypatch.setattr(gguf_linear.ops, "ggml_mul_mat_vec_a8", wrap("mmvq"))
    monkeypatch.setattr(gguf_linear.ops, "ggml_mul_mat_a8", wrap("mmq"))
    monkeypatch.setattr(gguf_linear.ops, "ggml_dequantize", wrap("dequantize"))
    return calls


def _qweight():
    tensors = get_gguf_sample_tensors(HIDDEN_SIZE, QUANT_TYPE)
    return torch.tensor(tensors[0].data, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("num_tokens", [32, 83, 2048])
@torch.inference_mode()
def test_large_m_uses_dequant_gemm(counting_ops, monkeypatch, num_tokens):
    """At or above the threshold, prefer dequantize+cuBLAS over MMQ."""
    seed_everything(0)
    monkeypatch.setattr(gguf_linear, "_MMQ_SAFE", 32)
    qweight = _qweight()
    x = torch.rand((num_tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")

    gguf_linear._fused_mul_mat_gguf(x, qweight, QUANT_TYPE)

    assert counting_ops["dequantize"] == 1
    assert counting_ops["mmq"] == 0
    assert counting_ops["mmvq"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("num_tokens", [7, 16, 31])
@torch.inference_mode()
def test_mid_m_still_uses_mmq(counting_ops, monkeypatch, num_tokens):
    """Between mmvq_safe and the threshold, MMQ is still selected."""
    seed_everything(0)
    monkeypatch.setattr(gguf_linear, "_MMQ_SAFE", 32)
    qweight = _qweight()
    x = torch.rand((num_tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")

    gguf_linear._fused_mul_mat_gguf(x, qweight, QUANT_TYPE)

    assert counting_ops["mmq"] == 1
    assert counting_ops["dequantize"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_decode_shape_still_uses_mmvq(counting_ops, monkeypatch):
    """Decode-shaped batches keep the MMVQ vector kernel."""
    seed_everything(0)
    monkeypatch.setattr(gguf_linear, "_MMQ_SAFE", 32)
    qweight = _qweight()
    # hidden 256 -> qweight.shape[0] <= 5120 -> mmvq_safe = 6
    x = torch.rand((2, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")

    gguf_linear._fused_mul_mat_gguf(x, qweight, QUANT_TYPE)

    assert counting_ops["mmvq"] == 1
    assert counting_ops["dequantize"] == 0
    assert counting_ops["mmq"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_threshold_zero_restores_legacy_dispatch(counting_ops, monkeypatch):
    """VLLM_GGUF_MMQ_SAFE=0 must reproduce the pre-change routing exactly."""
    seed_everything(0)
    monkeypatch.setattr(gguf_linear, "_MMQ_SAFE", 0)
    qweight = _qweight()
    x = torch.rand((2048, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")

    gguf_linear._fused_mul_mat_gguf(x, qweight, QUANT_TYPE)

    assert counting_ops["mmq"] == 1
    assert counting_ops["dequantize"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_dequant_path_matches_mmq_within_tolerance(monkeypatch):
    """Both paths must agree; the dequant path is the more accurate of the two."""
    seed_everything(0)
    qweight = _qweight()
    x = torch.rand((128, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")

    monkeypatch.setattr(gguf_linear, "_MMQ_SAFE", 0)
    y_mmq = gguf_linear._fused_mul_mat_gguf(x, qweight, QUANT_TYPE).float()
    monkeypatch.setattr(gguf_linear, "_MMQ_SAFE", 32)
    y_deq = gguf_linear._fused_mul_mat_gguf(x, qweight, QUANT_TYPE).float()

    assert y_mmq.shape == y_deq.shape
    denom = y_mmq.norm().clamp_min(1e-6)
    assert float((y_mmq - y_deq).norm() / denom) < 0.05
