# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch tests for the `_MMQ_SAFE` prefill threshold in `_fused_mul_mat_gguf`.

At or above `_MMQ_SAFE` rows, quantized GEMMs should take the
dequantize-then-cuBLAS path rather than the fused MMQ kernel (see the comment on
`_MMQ_SAFE` for the measurements motivating this). Below it, and for
decode-shaped batches, the existing MMVQ/MMQ routing must be unchanged.

These assert on *which kernel is selected*, so they swap the op entry points for
counting wrappers -- correctness of each individual kernel is already covered by
`test_kernels.py`.
"""

import contextlib

import pytest
import torch
from gguf import GGMLQuantizationType

from vllm_gguf_plugin.quantization import linear as gguf_linear

from .utils import get_gguf_sample_tensors, seed_everything

HIDDEN_SIZE = 256
QUANT_TYPE = GGMLQuantizationType.Q4_K
OP_NAMES = {
    "mmvq": "ggml_mul_mat_vec_a8",
    "mmq": "ggml_mul_mat_a8",
    "dequantize": "ggml_dequantize",
}


@contextlib.contextmanager
def counted_ops():
    """Swap the GEMM entry points for counting wrappers that still compute."""
    calls = dict.fromkeys(OP_NAMES, 0)
    original = {k: getattr(gguf_linear.ops, v) for k, v in OP_NAMES.items()}

    def wrap(key):
        real = original[key]

        def inner(*args, **kwargs):
            calls[key] += 1
            return real(*args, **kwargs)

        return inner

    for key, attr in OP_NAMES.items():
        setattr(gguf_linear.ops, attr, wrap(key))
    try:
        yield calls
    finally:
        for key, attr in OP_NAMES.items():
            setattr(gguf_linear.ops, attr, original[key])


@contextlib.contextmanager
def mmq_safe(threshold):
    """Temporarily override the dispatch threshold."""
    previous = gguf_linear._MMQ_SAFE
    gguf_linear._MMQ_SAFE = threshold
    try:
        yield
    finally:
        gguf_linear._MMQ_SAFE = previous


def _qweight():
    tensors = get_gguf_sample_tensors(HIDDEN_SIZE, QUANT_TYPE)
    return torch.tensor(tensors[0].data, device="cuda")


def _activations(num_tokens):
    return torch.rand((num_tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("num_tokens", [32, 83, 2048])
@torch.inference_mode()
def test_large_m_uses_dequant_gemm(num_tokens):
    """At or above the threshold, prefer dequantize+cuBLAS over MMQ."""
    seed_everything(0)
    qweight = _qweight()
    x = _activations(num_tokens)

    with mmq_safe(32), counted_ops() as calls:
        gguf_linear._fused_mul_mat_gguf(x, qweight, QUANT_TYPE)

    assert calls["dequantize"] == 1
    assert calls["mmq"] == 0
    assert calls["mmvq"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("num_tokens", [7, 16, 31])
@torch.inference_mode()
def test_mid_m_still_uses_mmq(num_tokens):
    """Between mmvq_safe and the threshold, MMQ is still selected."""
    seed_everything(0)
    qweight = _qweight()
    x = _activations(num_tokens)

    with mmq_safe(32), counted_ops() as calls:
        gguf_linear._fused_mul_mat_gguf(x, qweight, QUANT_TYPE)

    assert calls["mmq"] == 1
    assert calls["dequantize"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_decode_shape_still_uses_mmvq():
    """Decode-shaped batches keep the MMVQ vector kernel."""
    seed_everything(0)
    qweight = _qweight()
    # hidden 256 -> qweight.shape[0] <= 5120 -> mmvq_safe = 6
    x = _activations(2)

    with mmq_safe(32), counted_ops() as calls:
        gguf_linear._fused_mul_mat_gguf(x, qweight, QUANT_TYPE)

    assert calls["mmvq"] == 1
    assert calls["dequantize"] == 0
    assert calls["mmq"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_threshold_zero_is_the_default_and_keeps_mmq():
    """The shipped default (0) must reproduce the pre-change routing exactly."""
    assert gguf_linear._MMQ_SAFE == 0, "threshold should ship disabled"
    seed_everything(0)
    qweight = _qweight()
    x = _activations(2048)

    with counted_ops() as calls:
        gguf_linear._fused_mul_mat_gguf(x, qweight, QUANT_TYPE)

    assert calls["mmq"] == 1
    assert calls["dequantize"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_dequant_path_matches_mmq_within_tolerance():
    """Both paths must agree; the dequant path is the more accurate of the two."""
    seed_everything(0)
    qweight = _qweight()
    x = _activations(128)

    with mmq_safe(0):
        y_mmq = gguf_linear._fused_mul_mat_gguf(x, qweight, QUANT_TYPE).float()
    with mmq_safe(32):
        y_deq = gguf_linear._fused_mul_mat_gguf(x, qweight, QUANT_TYPE).float()

    assert y_mmq.shape == y_deq.shape
    denom = y_mmq.norm().clamp_min(1e-6)
    assert float((y_mmq - y_deq).norm() / denom) < 0.05
