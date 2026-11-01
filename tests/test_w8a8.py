# SPDX-License-Identifier: Apache-2.0
"""The group-scaled W8A8 prefill GEMM (csrc/gguf/w8a8, VLLM_GGUF_W8A8).

Each stage is checked against an independent reference:
- the GGUF -> int8 re-quantiser against gguf-py's dequantize followed by a torch
  symmetric per-group quantisation (random but valid blocks of every supported type);
- the activation quantiser against the same torch quantisation;
- the GEMM against an exact fp64 evaluation of the group-scaled product, including
  row counts that are not tile multiples;
- the fused op against an fp32 reference, at the error the int8 rounding predicts.
"""

import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType as QT

from vllm_gguf_plugin import ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

TYPES = [QT.Q4_K, QT.Q5_K, QT.Q6_K, QT.Q8_0]
GROUPS = [128, 256]


def random_blocks(qtype: QT, rows: int, cols: int, seed: int = 0) -> np.ndarray:
    """Random bytes shaped like a GGUF weight, with every fp16 scale finite and small."""
    rng = np.random.default_rng(seed)
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    nb = cols // block
    raw = rng.integers(0, 256, size=(rows, nb, size), dtype=np.uint8)

    def put_half(offset: int, lo: float, hi: float) -> None:
        vals = rng.uniform(lo, hi, size=(rows, nb)).astype(np.float16)
        raw[:, :, offset : offset + 2] = vals.view(np.uint8).reshape(rows, nb, 2)

    if qtype in (QT.Q4_K, QT.Q5_K):
        put_half(0, 1e-3, 4e-3)  # d
        put_half(2, 0.0, 2e-3)  # dmin
    elif qtype == QT.Q6_K:
        put_half(208, 1e-3, 4e-3)
    elif qtype == QT.Q8_0:
        put_half(0, 1e-3, 4e-3)
    return raw.reshape(rows, nb * size)


def fake_quant(x: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 per (row, group): (int8 values, scales [rows, K / group])."""
    xs = x.float().view(x.shape[0], -1, group)
    amax = xs.abs().amax(-1)
    inv = torch.where(amax > 0, 127.0 / amax, torch.zeros_like(amax))
    q = torch.round(xs * inv.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q.view_as(x), amax / 127.0


def quantize_act(x: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    ld = (m + 3) // 4 * 4
    aq = torch.empty(m, k, dtype=torch.int8, device=x.device)
    sa = torch.empty(k // group, ld, dtype=torch.float32, device=x.device)
    ops.w8a8_quantize_act(x, aq, sa, group)
    return aq, sa


def requant_weight(w: torch.Tensor, qtype: QT, n: int, k: int, group: int):
    bq = torch.empty(n, k, dtype=torch.int8, device=w.device)
    sb = torch.empty(k // group, n, dtype=torch.float32, device=w.device)
    ops.w8a8_requant_weight(w, bq, sb, int(qtype), group)
    return bq, sb


def grouped_reference(aq, sa, bq, sb, group: int) -> torch.Tensor:
    """sum_g sa[g, m] sb[g, n] (Aq_g @ Bq_g^T), exactly, in fp64."""
    m, k = aq.shape
    a = aq.double().view(m, k // group, group) * sa[:, :m].T.double().unsqueeze(-1)
    b = bq.double().view(bq.shape[0], k // group, group) * sb.T.double().unsqueeze(-1)
    return a.view(m, k) @ b.view(bq.shape[0], k).T


def assert_int8_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Equal up to rounding ties: values may differ by one step, and rarely."""
    diff = (actual.int() - expected.int()).abs()
    assert diff.max().item() <= 1
    assert (diff > 0).float().mean().item() < 1e-3


@pytest.mark.parametrize("qtype", TYPES)
@pytest.mark.parametrize("group", GROUPS)
@torch.inference_mode()
def test_requant_weight_matches_gguf_dequantize(qtype, group):
    n, k = 256, 1024
    raw = random_blocks(qtype, n, k, seed=int(qtype) + group)
    ref_w = torch.from_numpy(gguf.quants.dequantize(raw, qtype).astype(np.float32))
    assert ref_w.shape == (n, k)
    q_ref, s_ref = fake_quant(ref_w, group)

    bq, sb = requant_weight(torch.from_numpy(raw).cuda(), qtype, n, k, group)
    torch.testing.assert_close(sb.T.cpu(), s_ref, rtol=1e-5, atol=0)
    assert_int8_close(bq.cpu(), q_ref)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("m", [1, 7, 128, 300])
@torch.inference_mode()
def test_quantize_act_matches_reference(dtype, group, m):
    torch.manual_seed(m)
    x = torch.randn(m, 1024, device="cuda").to(dtype)
    x[0, :group] = 0  # an all-zero group must give scale 0 and zeros, not NaN
    aq, sa = quantize_act(x, group)
    q_ref, s_ref = fake_quant(x, group)
    torch.testing.assert_close(sa[:, :m].T, s_ref, rtol=1e-6, atol=0)
    assert_int8_close(aq, q_ref)
    assert sa[0, 0].item() == 0 and (aq[0, :group] == 0).all()


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("m", [1, 5, 128, 129, 250, 512])
@pytest.mark.parametrize("n,k", [(128, 256), (384, 1024), (256, 2304)])
@torch.inference_mode()
def test_gemm_matches_exact_grouped_product(group, m, n, k):
    if k % group:
        pytest.skip("K must be a multiple of the group")
    torch.manual_seed(m * 7 + n)
    aq = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
    bq = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
    ld = (m + 3) // 4 * 4
    sa = torch.rand(k // group, ld, device="cuda") * 1e-2
    sb = torch.rand(k // group, n, device="cuda") * 1e-2
    y = torch.empty(m, n, dtype=torch.float32, device="cuda").to(torch.bfloat16)
    ops.w8a8_gemm(aq, sa, bq, sb, y, group)
    ref = grouped_reference(aq, sa, bq, sb, group)
    # Only the final bf16 rounding separates the kernel from the exact value.
    torch.testing.assert_close(y.double(), ref, rtol=8e-3, atol=1e-6)


@pytest.mark.parametrize("group", GROUPS)
@torch.inference_mode()
def test_gemm_extreme_partials_stay_exact(group):
    """All +/-127 operands: the largest int32 partial a group can produce."""
    m, n, k = 128, 128, 1024
    aq = torch.full((m, k), 127, dtype=torch.int8, device="cuda")
    bq = torch.full((n, k), -127, dtype=torch.int8, device="cuda")
    bq[1::2] = 127
    sa = torch.ones(k // group, m, device="cuda")
    sb = torch.ones(k // group, n, device="cuda")
    y = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    ops.w8a8_gemm(aq, sa, bq, sb, y, group)
    expected = torch.full((n,), -127.0 * 127 * k, device="cuda")
    expected[1::2] *= -1
    # Exact in fp32; compare after the same bf16 rounding.
    torch.testing.assert_close(y.float(), expected.bfloat16().float().expand(m, n))


@pytest.mark.parametrize("qtype", TYPES)
@pytest.mark.parametrize("group", [128, 256])
@pytest.mark.parametrize("m", [33, 512, 1000])
@torch.inference_mode()
def test_fused_op_error_vs_fp32(qtype, group, m):
    n, k = 512, 2048
    raw = torch.from_numpy(random_blocks(qtype, n, k, seed=m)).cuda()
    x = (torch.randn(m, k, device="cuda") * 0.5).to(torch.bfloat16)
    w32 = ops.ggml_dequantize(raw, int(qtype), n, k, torch.float32)
    ref = x.float() @ w32.T
    y = ops.ggml_mul_mat_w8a8(raw, x, int(qtype), n, group)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16
    rel = ((y.float() - ref).norm() / ref.norm()).item()
    # Two independent int8 roundings at ~0.5-0.7e-2 each (see quant_err studies).
    assert rel < 1.3e-2, rel
    # And the fused op is exactly the three stages composed.
    aq, sa = quantize_act(x, group)
    bq, sb = requant_weight(raw, qtype, n, k, group)
    y2 = torch.empty_like(y)
    ops.w8a8_gemm(aq, sa, bq, sb, y2, group)
    torch.testing.assert_close(y, y2, rtol=0, atol=0)


def test_supported_shapes():
    assert ops.w8a8_supported(int(QT.Q4_K), 17408, 5120, 128)
    assert ops.w8a8_supported(int(QT.Q8_0), 5120, 6144, 256)
    assert not ops.w8a8_supported(int(QT.Q4_K), 17400, 5120, 128)  # N % 128
    assert not ops.w8a8_supported(int(QT.Q4_0), 1024, 1024, 128)  # type
    assert not ops.w8a8_supported(int(QT.Q4_K), 1024, 1024, 64)  # group
