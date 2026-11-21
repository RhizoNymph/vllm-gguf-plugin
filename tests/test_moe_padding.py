"""Padded tokens in the GGUF MoE.

vLLM's routers mark padded tokens (CUDA graph capture, DP padding) with
expert id -1 and routing weight 0. The GGUF MoE kernels index experts by id,
so ``_fused_moe_gguf`` must not hand them a negative id.
"""

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType, quantize

from vllm_gguf_plugin.quantization.fused_moe import (
    _fused_moe_gguf,
    _mask_padded_slots,
)


def test_mask_padded_slots_points_padding_at_expert_zero_with_no_weight():
    ids = torch.tensor([[3, 1], [-1, -1], [2, -1]], dtype=torch.int32)
    weights = torch.tensor([[0.5, 0.25], [0.7, 0.3], [0.9, 0.1]])

    masked_weights, safe_ids = _mask_padded_slots(weights, ids)

    assert safe_ids.dtype == torch.int32
    assert safe_ids.tolist() == [[3, 1], [0, 0], [2, 0]]
    assert masked_weights.tolist() == pytest.approx(
        [[0.5, 0.25], [0.0, 0.0], [0.9, 0.0]]
    )
    # The inputs are left untouched.
    assert ids[1].tolist() == [-1, -1]
    assert weights[1].tolist() == pytest.approx([0.7, 0.3])


def _q8_0_experts(num_experts: int, rows: int, cols: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((num_experts, rows, cols), dtype=np.float32) * 0.05
    packed = quantize(w, GGMLQuantizationType.Q8_0)
    return torch.from_numpy(np.ascontiguousarray(packed)).cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("num_tokens", [8, 128])  # MMVQ (<= 64) and MMQ paths
@torch.inference_mode()
def test_padded_tokens_contribute_nothing_and_do_not_fault(num_tokens: int):
    torch.manual_seed(0)
    hidden, inter, experts, top_k = 512, 256, 16, 4
    w13 = _q8_0_experts(experts, 2 * inter, hidden, seed=1)
    w2 = _q8_0_experts(experts, hidden, inter, seed=2)
    q8 = GGMLQuantizationType.Q8_0

    x = torch.rand(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    weights = torch.rand(num_tokens, top_k, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(
        0, experts, (num_tokens, top_k), dtype=torch.int32, device="cuda"
    )
    padded = torch.zeros(num_tokens, dtype=torch.bool, device="cuda")
    padded[num_tokens // 2 :] = True
    ids[padded] = -1
    weights[padded] = 0

    out = _fused_moe_gguf(x, w13, w2, weights, ids, q8, q8, "silu")
    torch.cuda.synchronize()

    assert torch.isfinite(out).all()
    assert (out[padded] == 0).all()
    valid = ~padded
    ref = _fused_moe_gguf(
        x[valid].contiguous(),
        w13,
        w2,
        weights[valid].contiguous(),
        ids[valid].contiguous(),
        q8,
        q8,
        "silu",
    )
    torch.testing.assert_close(out[valid], ref, atol=1e-2, rtol=1e-2)
