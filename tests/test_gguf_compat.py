# SPDX-License-Identifier: Apache-2.0
"""ggml tensor types newer than the installed gguf-py, and Q2_0 -> Q4_0.

Every released gguf-py (<= 0.19.0) predates ggml's Q1_0/Q2_0/F8_E4M3
types, and ``GGUFReader`` raises on the first tensor of an unknown type.
unsloth's Qwen3.8-Flash-Next UD-Q2_K_XL and UD-IQ3_XXS store some
``ffn_down_exps`` as Q2_0, so without the shim those files cannot even be
opened. The plugin has no Q2_0 kernels; Q2_0 is transcoded to Q4_0, which
represents every Q2_0 value exactly.
"""

import numpy as np
import pytest
import torch
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFReader, GGUFWriter
from gguf.quants import dequantize

from vllm_gguf_plugin.gguf_compat import (
    Q2_0,
    ensure_new_ggml_types,
    transcode_q2_0_to_q4_0,
)
from vllm_gguf_plugin.weight_utils import gguf_quant_weights_iterator_multi

pytestmark = [pytest.mark.cpu]


def _q2_0_reference(blocks: np.ndarray) -> np.ndarray:
    """llama.cpp's dequantize_row_q2_0, one element at a time."""
    blocks = blocks.reshape(-1, 18)
    out = np.empty((blocks.shape[0], 64), dtype=np.float32)
    for i, block in enumerate(blocks):
        d = np.frombuffer(block[:2].tobytes(), dtype=np.float16)[0].astype(np.float32)
        for j in range(64):
            q = (int(block[2 + j // 4]) >> ((j % 4) * 2)) & 0x03
            out[i, j] = (q - 1) * d
    return out.reshape(-1)


def _random_q2_0(rows: int, cols: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nb = cols // 64
    d = rng.standard_normal((rows, nb, 1)).astype(np.float16).view(np.uint8)
    qs = rng.integers(0, 256, size=(rows, nb, 16), dtype=np.uint8)
    return np.concatenate([d, qs], axis=-1).reshape(rows, nb * 18)


def test_new_types_are_registered():
    ensure_new_ggml_types()
    assert GGMLQuantizationType(42) is Q2_0
    assert GGMLQuantizationType["Q2_0"] is Q2_0
    assert GGMLQuantizationType.Q2_0 is Q2_0
    assert Q2_0.name == "Q2_0"
    assert int(Q2_0) == 42
    assert GGML_QUANT_SIZES[Q2_0] == (64, 18)
    assert GGMLQuantizationType(41).name == "Q1_0"
    assert GGML_QUANT_SIZES[GGMLQuantizationType(41)] == (128, 18)
    assert GGMLQuantizationType(43).name == "F8_E4M3"


def test_registration_is_idempotent():
    ensure_new_ggml_types()
    ensure_new_ggml_types()
    assert list(GGMLQuantizationType).count(Q2_0) == 1


def test_transcode_is_exact():
    rows, cols = 3, 256
    q2 = _random_q2_0(rows, cols)
    q4 = transcode_q2_0_to_q4_0(q2)
    assert q4.dtype == np.uint8
    assert q4.shape == (rows, cols // 32 * 18)
    expected = _q2_0_reference(q2).reshape(rows, cols)
    got = dequantize(q4, GGMLQuantizationType.Q4_0).reshape(rows, cols)
    np.testing.assert_array_equal(got, expected)


def test_transcode_keeps_leading_dims():
    """Stacked expert tensors are [experts, rows, bytes]."""
    q2 = _random_q2_0(4 * 2, 128).reshape(4, 2, -1)
    q4 = transcode_q2_0_to_q4_0(q2)
    assert q4.shape == (4, 2, 128 // 32 * 18)


def test_transcode_rejects_partial_blocks():
    with pytest.raises(ValueError, match="multiple of 18"):
        transcode_q2_0_to_q4_0(np.zeros((2, 17), dtype=np.uint8))


def _write_q2_0_gguf(path, data: np.ndarray) -> None:
    writer = GGUFWriter(str(path), "llama")
    writer.add_tensor("blk.0.ffn_down.weight", data, raw_dtype=Q2_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_reader_opens_q2_0_files(tmp_path):
    ensure_new_ggml_types()
    rows, cols = 2, 128
    data = _random_q2_0(rows, cols)
    path = tmp_path / "q2.gguf"
    _write_q2_0_gguf(path, data)

    tensor = GGUFReader(str(path)).tensors[0]
    assert tensor.tensor_type is Q2_0
    np.testing.assert_array_equal(np.asarray(tensor.data), data)


def test_iterator_serves_q2_0_as_q4_0(tmp_path):
    rows, cols = 2, 128
    data = _random_q2_0(rows, cols)
    path = tmp_path / "q2.gguf"
    _write_q2_0_gguf(path, data)

    weights = dict(
        gguf_quant_weights_iterator_multi(
            [str(path)], {"blk.0.ffn_down.weight": "model.down.weight"}
        )
    )
    assert int(weights["model.down.weight_type"]) == GGMLQuantizationType.Q4_0
    packed = weights["model.down.weight"]
    assert packed.dtype == torch.uint8
    got = dequantize(packed.numpy(), GGMLQuantizationType.Q4_0).reshape(rows, cols)
    np.testing.assert_array_equal(got, _q2_0_reference(data).reshape(rows, cols))
