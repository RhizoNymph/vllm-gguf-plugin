# SPDX-License-Identifier: Apache-2.0
"""ggml NVFP4 on the CUDA vector kernels (dense MMVQ and per-token MoE).

The CUDA kernels quantize activations to q8_1 (int8 per 32 values, the scale
stored as half), then take integer dot products against the E2M1 codes and
scale by the UE4M3 sub-block scales, as llama.cpp's ``vec_dot_nvfp4_q8_1``
does. ``_q8_1_roundtrip`` reproduces that activation rounding in torch, so
the kernels are checked tightly against ``dequantize_nvfp4`` + a float64
matmul of the rounded activations, and loosely against the unrounded
activations and against the Triton kernel.

Needs CUDA and a built ``vllm_gguf_plugin._C_gguf``.
"""

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType

from vllm_gguf_plugin import ops
from vllm_gguf_plugin.quantization.nvfp4 import NVFP4_BLOCK_BYTES, dequantize_nvfp4

NVFP4 = int(GGMLQuantizationType.NVFP4)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and ops._CUDA_ENABLED),
    reason="needs CUDA and the _C_gguf extension",
)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _random_nvfp4(
    lead: tuple[int, ...], cols: int, seed: int, special_scales: bool = False
) -> torch.Tensor:
    """Random NVFP4 rows ``[*lead, cols / 64 * 36]``. Scales are finite,
    nonzero UE4M3 codes in a range that keeps products moderate; with
    ``special_scales`` some sub-blocks use 0x00, 0x7F (both decode to 0) and
    codes with bit 7 set (ignored by ggml's decoder)."""
    rng = np.random.default_rng(seed)
    nb = cols // 64
    d = rng.integers(0x28, 0x40, size=(*lead, nb, 4), dtype=np.uint8)
    if special_scales:
        pick = rng.integers(0, 6, size=d.shape)
        d = np.where(pick == 0, np.uint8(0x00), d)
        d = np.where(pick == 1, np.uint8(0x7F), d)
        d = np.where(pick == 2, d | np.uint8(0x80), d)
    qs = rng.integers(0, 256, size=(*lead, nb, 32), dtype=np.uint8)
    data = np.concatenate([d, qs], axis=-1).reshape(*lead, nb * NVFP4_BLOCK_BYTES)
    return torch.from_numpy(data)


def _q8_1_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """``x`` as the CUDA kernels see it after ``quantize_q8_1``: per 32
    values ``d = amax / 127``, ``q = roundf(x / d)``, and ``q * half(d)``."""
    xf = x.float()
    blocks = xf.reshape(*xf.shape[:-1], -1, 32)
    d = blocks.abs().amax(dim=-1, keepdim=True) / 127
    safe = torch.where(d == 0, torch.ones_like(d), d)
    q = blocks / safe
    q = torch.sign(q) * torch.floor(q.abs() + 0.5)  # roundf: half away from 0
    q = torch.where(d == 0, torch.zeros_like(q), q)
    return (q.double() * d.half().double()).reshape(xf.shape)


def _assert_close_rel(got: torch.Tensor, want: torch.Tensor, rel: float) -> None:
    """Max error relative to the largest reference magnitude."""
    got, want = got.double().cpu(), want.double().cpu()
    scale = want.abs().max().clamp_min(1e-6)
    err = (got - want).abs().max() / scale
    assert err <= rel, f"max error {err.item():.3e} of max |want| > {rel}"


def _dense(w: torch.Tensor) -> torch.Tensor:
    return dequantize_nvfp4(w.cpu()).double()


# ----------------------------------------------------------------- dense


def test_nvfp4_routes_to_the_cuda_vector_kernels():
    assert ops._cuda_kernel_available("ggml_mul_mat_vec_a8", NVFP4)
    assert ops._cuda_kernel_available("ggml_moe_a8_vec", NVFP4)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows,cols", [(1, 64), (37, 1152), (2304, 2304)])
@pytest.mark.parametrize("vecs", [1, 3])
@torch.inference_mode()
def test_mmvq_matches_dequantized_matmul(dtype, rows, cols, vecs):
    w = _random_nvfp4((rows,), cols, seed=rows + cols)
    g = torch.Generator().manual_seed(vecs)
    x = (torch.randn(vecs, cols, generator=g) / 4).to(dtype)

    got = ops.ggml_mul_mat_vec_a8(w.cuda(), x.cuda(), NVFP4, rows)

    assert got.shape == (vecs, rows) and got.dtype == dtype
    dense = _dense(w)
    exact = _q8_1_roundtrip(x) @ dense.T
    # fp32 accumulation and the output cast are the only differences.
    out_eps = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1e-2}
    _assert_close_rel(got, exact, out_eps[dtype])
    _assert_close_rel(got, x.double() @ dense.T, 3e-2)


@torch.inference_mode()
def test_mmvq_decodes_special_scale_codes_like_ggml():
    rows, cols = 64, 512
    w = _random_nvfp4((rows,), cols, seed=11, special_scales=True)
    x = torch.randn(2, cols, generator=torch.Generator().manual_seed(3))

    got = ops.ggml_mul_mat_vec_a8(w.cuda(), x.cuda(), NVFP4, rows)

    _assert_close_rel(got, _q8_1_roundtrip(x) @ _dense(w).T, 1e-5)


@torch.inference_mode()
def test_mmvq_zero_scales_give_zero_rows():
    w = _random_nvfp4((8,), 128, seed=5)
    w[:4, :4] = 0x00  # first block of rows 0-3: zero scales
    w[:4, 36:40] = 0x7F  # second block: NaN code, decoded as zero
    x = torch.randn(1, 128, generator=torch.Generator().manual_seed(0))

    got = ops.ggml_mul_mat_vec_a8(w.cuda(), x.cuda(), NVFP4, 8).cpu()

    assert torch.all(got[0, :4] == 0)
    assert torch.all(got[0, 4:] != 0)


# ------------------------------------------------------------------- MoE


def _moe_problem(tokens, top_k, experts, rows, cols, seed):
    w = _random_nvfp4((experts, rows), cols, seed=seed)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, cols, generator=g) / 4
    topk_ids = torch.stack(
        [torch.randperm(experts, generator=g)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    return w, x, topk_ids


def _moe_reference(x, dense, topk_ids, top_k):
    xq = _q8_1_roundtrip(x)
    ids = topk_ids.reshape(-1).long()
    return torch.stack([xq[i // top_k] @ dense[ids[i]].T for i in range(ids.numel())])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tokens,top_k", [(1, 1), (1, 8), (5, 3), (64, 8)])
@torch.inference_mode()
def test_moe_vec_matches_dequantized_experts(dtype, tokens, top_k):
    experts, rows, cols = 16, 96, 1152
    w, x, topk_ids = _moe_problem(tokens, top_k, experts, rows, cols, seed=tokens)
    x = x.to(dtype)

    got = ops.ggml_moe_a8_vec(
        x.cuda(), w.cuda(), topk_ids.cuda(), top_k, NVFP4, rows, tokens
    )

    assert got.shape == (tokens * top_k, rows) and got.dtype == dtype
    want = _moe_reference(x, _dense(w), topk_ids, top_k)
    out_eps = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1e-2}
    _assert_close_rel(got, want, out_eps[dtype])


@pytest.mark.parametrize("tokens,top_k", [(1, 8), (7, 4)])
@torch.inference_mode()
def test_moe_vec_matches_the_triton_kernel(tokens, top_k):
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    from vllm_gguf_plugin.triton.fused_moe.interface import ggml_moe_a8_triton
    from vllm_gguf_plugin.triton.fused_moe.utils import get_triton_moe_block_m

    experts, rows, cols = 8, 2304, 2304
    w, x, topk_ids = _moe_problem(tokens, top_k, experts, rows, cols, seed=21)
    w, x, topk_ids = w.cuda(), x.cuda().to(torch.bfloat16), topk_ids.cuda()

    cuda = ops.ggml_moe_a8_vec(x, w, topk_ids, top_k, NVFP4, rows, tokens)
    sorted_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids, get_triton_moe_block_m(NVFP4), experts
    )
    triton = ggml_moe_a8_triton(
        x, w, sorted_ids, expert_ids, num_post, NVFP4, rows, top_k, tokens
    )

    # Triton multiplies unrounded activations; the CUDA path rounds them to
    # q8_1, so the two agree to the activation quantization error.
    _assert_close_rel(cuda, triton, 3e-2)


def _situ(gate_up: torch.Tensor, gate_cap: float, up_cap: float) -> torch.Tensor:
    g, u = gate_up.chunk(2, dim=-1)
    return (
        gate_cap
        * torch.tanh(g / gate_cap)
        * torch.sigmoid(g)
        * (up_cap * torch.tanh(u / up_cap))
    )


def _fused_reference(x, d13, d2, topk_ids, topk_weights, s13, s2, act):
    out = torch.zeros(x.shape[0], d2.shape[1], dtype=torch.float64)
    for t in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[t, k])
            h = act((x[t].double() @ d13[e].T) * float(s13[e]))
            out[t] += float(topk_weights[t, k]) * float(s2[e]) * (h @ d2[e].T)
    return out


@pytest.mark.parametrize("tokens", [1, 4, 64, 80])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_fused_moe_with_expert_scales(tokens, dtype):
    """``_fused_moe_gguf`` over NVFP4 experts with per-expert output scales
    and SITU, top-k 8: the CUDA vector path up to 64 tokens, the grouped
    Triton path above."""
    from vllm_gguf_plugin.quantization.fused_moe import _fused_moe_gguf

    experts, hidden, inter, top_k = 16, 256, 128, 8
    gate_cap, up_cap = 4.0, 25.0
    w13 = _random_nvfp4((experts, 2 * inter), hidden, seed=1)
    w2 = _random_nvfp4((experts, hidden), inter, seed=2)
    g = torch.Generator().manual_seed(tokens)
    x = torch.randn(tokens, hidden, generator=g) / 4
    topk_ids = torch.stack(
        [torch.randperm(experts, generator=g)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    topk_weights = torch.rand(tokens, top_k, generator=g)
    s13 = torch.rand(experts, generator=g) * 0.5 + 0.25
    s2 = torch.rand(experts, generator=g) * 0.5 + 0.25

    got = _fused_moe_gguf(
        x.to(dtype).cuda(),
        w13.cuda(),
        w2.cuda(),
        topk_weights.to(dtype).cuda(),
        topk_ids.cuda(),
        NVFP4,
        NVFP4,
        "situ",
        s13.cuda(),
        s2.cuda(),
        gate_cap,
        up_cap,
    )

    want = _fused_reference(
        x.to(dtype),
        _dense(w13),
        _dense(w2),
        topk_ids,
        topk_weights.to(dtype),
        s13,
        s2,
        lambda h: _situ(h, gate_cap, up_cap),
    )
    assert got.shape == (tokens, hidden)
    # Two quantized GEMMs, a bf16/fp16 intermediate and the activation.
    _assert_close_rel(got, want, 5e-2)


@torch.inference_mode()
def test_fused_moe_scales_change_the_cuda_output():
    """The scales are applied around the CUDA kernels, not dropped."""
    from vllm_gguf_plugin.quantization.fused_moe import _fused_moe_gguf

    experts, hidden, inter, top_k, tokens = 4, 128, 64, 2, 2
    w13 = _random_nvfp4((experts, 2 * inter), hidden, seed=7).cuda()
    w2 = _random_nvfp4((experts, hidden), inter, seed=8).cuda()
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) / 4
    ids = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
    weights = torch.ones(tokens, top_k, device="cuda", dtype=torch.bfloat16)
    ones = torch.ones(experts, device="cuda")
    s2 = torch.tensor([0.5, 2.0, 0.25, 4.0], device="cuda")

    def run(s13, s2_):
        return _fused_moe_gguf(
            x, w13, w2, weights, ids, NVFP4, NVFP4, "silu", s13, s2_
        ).float()

    plain = run(ones, ones)
    # The down scale is an exact factor per expert (powers of two); check it
    # through top-1 routing so each output row has one expert.
    single = torch.tensor([[1], [3]], device="cuda", dtype=torch.int32)
    w1 = torch.ones(tokens, 1, device="cuda", dtype=torch.bfloat16)
    a = _fused_moe_gguf(x, w13, w2, w1, single, NVFP4, NVFP4, "silu", ones, ones)
    b = _fused_moe_gguf(x, w13, w2, w1, single, NVFP4, NVFP4, "silu", ones, s2)
    torch.testing.assert_close(
        b.float(), a.float() * s2[single.long()].float(), rtol=1e-2, atol=1e-3
    )
    assert not torch.allclose(plain, run(ones * 2, ones), rtol=1e-2)
