# SPDX-License-Identifier: Apache-2.0
"""ggml NVFP4: the plugin's decoder and Triton kernels.

unsloth's Inkling-2-Mini GGUFs keep 20 of the 24 expert layers in ggml's
NVFP4 (64-value blocks: four UE4M3 scales, one per 16 values, and 32 bytes
of E2M1 codes) plus a per-expert F32 ``.scale``. ``nvfp4_rows.json`` holds
real rows of UD-IQ4_XS ``blk.0.ffn_gate_up_exps`` beside the same logical
rows of the official checkpoint's NVFP4 ``w13_weight``, so the decoder is
checked against Thinking Machines' own encoding.

The Triton tests run on the CPU under ``TRITON_INTERPRET=1`` (or on CUDA).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import dequantize

from vllm_gguf_plugin.quantization.nvfp4 import NVFP4_BLOCK_BYTES, dequantize_nvfp4

pytestmark = [pytest.mark.cpu]

NVFP4 = GGMLQuantizationType.NVFP4
FIXTURES = Path(__file__).parent / "fixtures" / "inkling2"
E2M1 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float32,
)

_TRITON = os.environ.get("TRITON_INTERPRET") == "1" or torch.cuda.is_available()
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
needs_triton = pytest.mark.skipif(
    not _TRITON, reason="needs CUDA or TRITON_INTERPRET=1"
)


def _e4m3(codes: np.ndarray) -> np.ndarray:
    codes = codes.astype(np.int32)
    e, m = (codes & 0x7F) >> 3, (codes & 7).astype(np.float32)
    v = np.where(e == 0, np.ldexp(m, -9), np.ldexp(1 + m / 8, e - 7))
    return np.where(codes & 0x80, -v, v).astype(np.float32)


def _random_nvfp4(rows: int, cols: int, seed: int = 0) -> np.ndarray:
    """Random NVFP4 rows with every scale a finite, nonzero UE4M3 code."""
    rng = np.random.default_rng(seed)
    nb = cols // 64
    d = rng.integers(1, 0x7F, size=(rows, nb, 4), dtype=np.uint8)
    qs = rng.integers(0, 256, size=(rows, nb, 32), dtype=np.uint8)
    return np.concatenate([d, qs], axis=-1).reshape(rows, nb * NVFP4_BLOCK_BYTES)


def test_block_geometry():
    assert GGML_QUANT_SIZES[NVFP4] == (64, NVFP4_BLOCK_BYTES) == (64, 36)
    assert int(NVFP4) == 40


@pytest.mark.parametrize("shape", [(1, 64), (5, 256), (2, 3, 128)])
def test_decoder_matches_gguf_py(shape):
    *lead, cols = shape
    rows = int(np.prod(lead))
    data = _random_nvfp4(rows, cols, seed=rows).reshape(*lead, -1)
    got = dequantize_nvfp4(torch.from_numpy(data))
    assert got.shape == (*lead, cols)
    want = dequantize(data.reshape(rows, -1), NVFP4).reshape(*lead, cols)
    assert torch.equal(got, torch.from_numpy(want))


def test_zero_and_nan_scale_codes_decode_to_zero():
    # ggml treats UE4M3 0x00 and 0x7F (NaN) as a zero scale.
    data = _random_nvfp4(1, 64)
    data[0, 0] = 0x00
    data[0, 1] = 0x7F
    got = dequantize_nvfp4(torch.from_numpy(data))[0]
    assert torch.all(got[:32] == 0)
    assert torch.equal(got, torch.from_numpy(dequantize(data, NVFP4)[0]))


def _official_row(row) -> np.ndarray:
    packed = np.frombuffer(bytes.fromhex(row["hf_packed_e2m1"]), np.uint8)
    values = np.empty(packed.size * 2, np.float32)
    values[0::2] = E2M1[packed & 15]
    values[1::2] = E2M1[packed >> 4]
    scale = _e4m3(np.frombuffer(bytes.fromhex(row["hf_scale_e4m3"]), np.uint8))
    return (values.reshape(-1, 16) * scale[:, None]).reshape(-1)


def test_real_rows_decode_to_the_official_nvfp4_values():
    fixture = json.loads((FIXTURES / "nvfp4_rows.json").read_text())
    for row in fixture["rows"]:
        gguf_row = np.frombuffer(bytes.fromhex(row["gguf_nvfp4"]), np.uint8)
        got = dequantize_nvfp4(torch.from_numpy(gguf_row.copy())).numpy()
        assert np.array_equal(got, _official_row(row)), row["gguf_row"]


def test_expert_scale_is_amax_over_the_denominator():
    fixture = json.loads((FIXTURES / "nvfp4_rows.json").read_text())
    amax = np.array(fixture["hf_amax"], np.float32)
    den = np.float32(fixture["hf_outer_scale_denominator"][0])
    assert np.array_equal(np.array(fixture["gguf_scale"], np.float32), amax / den)


# ------------------------------------------------------------------ Triton


@needs_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_triton_dequantize(dtype):
    from vllm_gguf_plugin.triton.dequantize.interface import ggml_dequantize_triton

    data = _random_nvfp4(6, 192, seed=7)
    w = torch.from_numpy(data).to(_DEVICE)
    got = ggml_dequantize_triton(w, int(NVFP4), 6, 192, dtype)
    want = dequantize_nvfp4(torch.from_numpy(data)).to(dtype)
    assert got.dtype == dtype
    assert torch.equal(got.cpu(), want)


def _align(topk_ids: torch.Tensor, block_m: int, num_experts: int):
    """Pure-torch moe_align_block_size: token slots grouped by expert, each
    group padded to block_m with -1 (an invalid slot)."""
    flat = topk_ids.reshape(-1)
    sorted_ids, expert_ids = [], []
    for e in range(num_experts):
        slots = (flat == e).nonzero().reshape(-1).tolist()
        while slots:
            chunk, slots = slots[:block_m], slots[block_m:]
            sorted_ids += chunk + [-1] * (block_m - len(chunk))
            expert_ids.append(e)
    return (
        torch.tensor(sorted_ids, dtype=torch.int32),
        torch.tensor(expert_ids, dtype=torch.int32),
        torch.tensor([len(sorted_ids)], dtype=torch.int32),
    )


@needs_triton
@pytest.mark.parametrize("tokens,top_k", [(1, 2), (5, 3)])
def test_triton_moe_matches_dense_experts(tokens, top_k):
    from vllm_gguf_plugin.triton.fused_moe.interface import ggml_moe_a8_triton
    from vllm_gguf_plugin.triton.fused_moe.utils import get_triton_moe_block_m

    experts, rows, cols = 4, 40, 128
    data = np.stack([_random_nvfp4(rows, cols, seed=s) for s in range(experts)])
    w = torch.from_numpy(data)
    dense = dequantize_nvfp4(w)  # [E, rows, cols]
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(tokens, cols, generator=g) / 16).to(torch.float32)
    topk_ids = torch.stack(
        [torch.randperm(experts, generator=g)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    block_m = get_triton_moe_block_m(int(NVFP4))
    sorted_ids, expert_ids, num_post = _align(topk_ids, block_m, experts)
    out = ggml_moe_a8_triton(
        x.to(_DEVICE),
        w.to(_DEVICE),
        sorted_ids.to(_DEVICE),
        expert_ids.to(_DEVICE),
        num_post.to(_DEVICE),
        int(NVFP4),
        rows,
        top_k,
        tokens,
    ).cpu()
    want = torch.stack(
        [x[t] @ dense[topk_ids[t, k]].T for t in range(tokens) for k in range(top_k)]
    )
    assert out.shape == (tokens * top_k, rows)
    torch.testing.assert_close(out, want, rtol=1e-5, atol=1e-4)


def test_every_published_ggml_type_is_served():
    """Every tensor type in every unsloth Inkling-2-Mini file has a plugin
    kernel, or is transcoded on load to one that has (Q2_0 -> Q4_0,
    MXFP4 -> Q5_0)."""
    from vllm_gguf_plugin.gguf_compat import Q2_0
    from vllm_gguf_plugin.quantization.utils import (
        DEQUANT_TYPES,
        MOE_VEC_QUANT_TYPES,
        UNQUANTIZED_TYPES,
    )

    transcoded = {
        Q2_0: GGMLQuantizationType.Q4_0,
        GGMLQuantizationType.MXFP4: GGMLQuantizationType.Q5_0,
    }
    survey = json.loads((FIXTURES / "quant_types.json").read_text())
    assert len(survey) == 15  # 10 quants, 3 mmproj, 2 DSpark drafts
    for file, types in survey.items():
        for name in types:
            qtype = GGMLQuantizationType[name]
            served = transcoded.get(qtype, qtype)
            assert served in UNQUANTIZED_TYPES | DEQUANT_TYPES, (file, name)
            if served not in UNQUANTIZED_TYPES:
                assert served in MOE_VEC_QUANT_TYPES, (file, name)
