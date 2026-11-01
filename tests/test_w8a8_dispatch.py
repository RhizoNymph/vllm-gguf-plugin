# SPDX-License-Identifier: Apache-2.0
"""Routing of prefill GEMMs to the W8A8 kernel (VLLM_GGUF_W8A8).

With W8A8 on, exactly the GEMMs that would take dequantize+cuBLAS (M >= _MMQ_SAFE)
and that the kernel supports go to ggml_mul_mat_w8a8; everything else is routed as
before. Parsing of the variable is strict.
"""

import contextlib

import pytest
import torch
from gguf import GGMLQuantizationType as QT

from vllm_gguf_plugin.quantization import linear as gguf_linear

from .test_w8a8 import random_blocks

OP_NAMES = {
    "w8a8": "ggml_mul_mat_w8a8",
    "dequantize": "ggml_dequantize",
    "mmq": "ggml_mul_mat_a8",
    "mmvq": "ggml_mul_mat_vec_a8",
}


@pytest.mark.parametrize(
    "raw,expected",
    [("", 0), ("0", 0), (" 0 ", 0), ("1", 128), ("128", 128), ("256", 256)],
)
def test_parse_group(raw, expected):
    assert gguf_linear._parse_w8a8_group(raw) == expected


@pytest.mark.parametrize("raw", ["32", "64", "512", "yes", "-1", "128.0", "true"])
def test_parse_group_rejects(raw):
    with pytest.raises(gguf_linear.GGUFW8A8ConfigError):
        gguf_linear._parse_w8a8_group(raw)


def test_parse_types_default_and_explicit():
    q4k, q5k, q6k, q8 = (int(t) for t in (QT.Q4_K, QT.Q5_K, QT.Q6_K, QT.Q8_0))
    assert gguf_linear._parse_w8a8_types("") == {q4k, q5k}
    assert gguf_linear._parse_w8a8_types("q6_k, Q8_0") == {q6k, q8}
    assert gguf_linear._parse_w8a8_types("Q4_K,Q5_K,Q6_K,Q8_0") == {q4k, q5k, q6k, q8}


@pytest.mark.parametrize("raw", ["Q4_0", "Q4_K,IQ4_XS", " , "])
def test_parse_types_rejects(raw):
    with pytest.raises(gguf_linear.GGUFW8A8ConfigError):
        gguf_linear._parse_w8a8_types(raw)


@contextlib.contextmanager
def routing(group, mmq_safe, types=(QT.Q4_K, QT.Q5_K, QT.Q6_K, QT.Q8_0)):
    calls = dict.fromkeys(OP_NAMES, 0)
    original = {k: getattr(gguf_linear.ops, v) for k, v in OP_NAMES.items()}

    def wrap(key):
        real = original[key]

        def inner(*args, **kwargs):
            calls[key] += 1
            return real(*args, **kwargs)

        return inner

    previous = gguf_linear._W8A8_GROUP, gguf_linear._MMQ_SAFE, gguf_linear._W8A8_TYPES
    gguf_linear._W8A8_GROUP, gguf_linear._MMQ_SAFE = group, mmq_safe
    gguf_linear._W8A8_TYPES = frozenset(int(t) for t in types)
    for key, attr in OP_NAMES.items():
        setattr(gguf_linear.ops, attr, wrap(key))
    try:
        yield calls
    finally:
        gguf_linear._W8A8_GROUP, gguf_linear._MMQ_SAFE, gguf_linear._W8A8_TYPES = previous
        for key, attr in OP_NAMES.items():
            setattr(gguf_linear.ops, attr, original[key])


def _weight(n, k, qtype=QT.Q4_K):
    return torch.from_numpy(random_blocks(qtype, n, k)).cuda()


def _route(qweight, m, k, qtype=QT.Q4_K):
    x = (torch.randn(m, k, device="cuda") * 0.5).to(torch.bfloat16)
    return gguf_linear._fused_mul_mat_gguf(x, qweight, int(qtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("m", [32, 300, 2048])
@torch.inference_mode()
def test_prefill_goes_to_w8a8(m):
    w = _weight(256, 1024)
    with routing(128, 32) as calls:
        y = _route(w, m, 1024)
    assert calls["w8a8"] == 1 and calls["dequantize"] == 0
    assert y.shape == (m, 256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_below_threshold_is_unchanged():
    w = _weight(256, 1024)
    with routing(128, 32) as calls:
        _route(w, 16, 1024)
    assert calls["w8a8"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_off_keeps_dequant_path():
    w = _weight(256, 1024)
    with routing(0, 32) as calls:
        _route(w, 512, 1024)
    assert calls["w8a8"] == 0 and calls["dequantize"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_unsupported_shape_falls_back_to_dequant():
    w = _weight(200, 1024)  # N % 128 != 0
    with routing(128, 32) as calls:
        _route(w, 512, 1024)
    assert calls["w8a8"] == 0 and calls["dequantize"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_unsupported_type_falls_back_to_dequant():
    w = _weight(256, 1024, QT.Q4_0)
    with routing(128, 32) as calls:
        _route(w, 512, 1024, QT.Q4_0)
    assert calls["w8a8"] == 0 and calls["dequantize"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_type_outside_enabled_set_keeps_dequant_path():
    w = _weight(256, 1024, QT.Q6_K)
    with routing(128, 32, types=(QT.Q4_K, QT.Q5_K)) as calls:
        _route(w, 512, 1024, QT.Q6_K)
    assert calls["w8a8"] == 0 and calls["dequantize"] == 1
    with routing(128, 32, types=(QT.Q6_K,)) as calls:
        _route(w, 512, 1024, QT.Q6_K)
    assert calls["w8a8"] == 1
