# SPDX-License-Identifier: Apache-2.0
"""MXFP4 -> Q5_0, bit-exact.

unsloth's Inkling-2-Mini UD-Q2_K_XL stores one ``ffn_down_exps`` as ggml
MXFP4, which no GGUF plugin kernel reads. An MXFP4 block is 32 E2M1 values
with one E8M0 scale, so every value is ``k * 2**(e - 128)`` with ``k`` in
-12..12. Q5_0 stores ``(q - 16) * d`` with ``q`` in 0..31 and an fp16 ``d``:
``q = k + 16`` and ``d = 2**(e - 128)`` hold the same values whenever ``d``
is an fp16 power of two.
"""

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
from gguf.quants import dequantize

from vllm_gguf_plugin.gguf_compat import (
    MXFP4ScaleOutOfRange,
    transcode_mxfp4_to_q5_0,
)
from vllm_gguf_plugin.weight_utils import gguf_quant_weights_iterator_multi

pytestmark = [pytest.mark.cpu]

MXFP4 = GGMLQuantizationType.MXFP4
Q5_0 = GGMLQuantizationType.Q5_0
E2M1_DOUBLED = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def _mxfp4_reference(blocks: np.ndarray) -> np.ndarray:
    """ggml's dequantize_row_mxfp4, one element at a time."""
    blocks = blocks.reshape(-1, 17)
    out = np.empty((blocks.shape[0], 32), dtype=np.float32)
    for i, block in enumerate(blocks):
        d = np.float32(2.0) ** (int(block[0]) - 128)
        for j in range(16):
            out[i, j] = E2M1_DOUBLED[block[1 + j] & 0x0F] * d
            out[i, j + 16] = E2M1_DOUBLED[block[1 + j] >> 4] * d
    return out.reshape(-1)


def _random_mxfp4(rows: int, cols: int, seed: int = 0, lo=104, hi=144):
    rng = np.random.default_rng(seed)
    nb = cols // 32
    e = rng.integers(lo, hi, size=(rows, nb, 1), dtype=np.uint8)
    qs = rng.integers(0, 256, size=(rows, nb, 16), dtype=np.uint8)
    return np.concatenate([e, qs], axis=-1).reshape(rows, nb * 17)


def test_reference_matches_gguf_py():
    data = _random_mxfp4(3, 64)
    assert np.array_equal(
        _mxfp4_reference(data), dequantize(data, MXFP4).reshape(-1)
    )


@pytest.mark.parametrize("shape", [(1, 32), (4, 256), (2, 3, 128)])
def test_transcode_is_exact(shape):
    *lead, cols = shape
    rows = int(np.prod(lead))
    data = _random_mxfp4(rows, cols).reshape(*lead, cols // 32 * 17)
    q5 = transcode_mxfp4_to_q5_0(data)
    assert q5.shape == (*lead, cols // 32 * 22)
    assert q5.dtype == np.uint8
    want = dequantize(data.reshape(rows, -1), MXFP4)
    got = dequantize(q5.reshape(rows, -1), Q5_0)
    assert np.array_equal(got, want)


def test_every_code_and_the_fp16_range_edges():
    # One block per (scale, code) pair at both ends of the fp16 range.
    blocks = []
    for e in (104, 105, 127, 128, 142, 143):
        for code in range(16):
            blocks.append([e] + [code | (code << 4)] * 16)
    data = np.array(blocks, dtype=np.uint8)
    got = dequantize(transcode_mxfp4_to_q5_0(data), Q5_0)
    assert np.array_equal(got, dequantize(data, MXFP4))


@pytest.mark.parametrize("e", [0, 103, 144, 255])
def test_scale_outside_fp16_is_rejected(e):
    data = _random_mxfp4(1, 64)
    data[0, 17] = e
    with pytest.raises(MXFP4ScaleOutOfRange) as info:
        transcode_mxfp4_to_q5_0(data)
    assert info.value.exponents == (e,)


def test_rejects_partial_blocks():
    with pytest.raises(ValueError, match="17-byte"):
        transcode_mxfp4_to_q5_0(np.zeros((1, 20), dtype=np.uint8))


def test_iterator_serves_mxfp4_as_q5_0(tmp_path):
    data = _random_mxfp4(4, 64, seed=3)
    path = tmp_path / "mxfp4.gguf"
    writer = GGUFWriter(str(path), "llama")
    writer.add_tensor("blk.0.ffn_down.weight", data, raw_dtype=MXFP4)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    assert GGUFReader(path).tensors[0].tensor_type == MXFP4

    tensors = dict(gguf_quant_weights_iterator_multi([str(path)], None))
    assert int(tensors["blk.0.ffn_down.weight_type"]) == int(Q5_0)
    weight = tensors["blk.0.ffn_down.weight"]
    assert weight.dtype == torch.uint8
    assert np.array_equal(
        dequantize(weight.numpy(), Q5_0), dequantize(data, MXFP4)
    )
