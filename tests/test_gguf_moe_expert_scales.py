# SPDX-License-Identifier: Apache-2.0
"""GGUF MoE: per-expert output scales, SITU, and NVFP4 experts.

Inkling 2's NVFP4 experts carry a per-expert F32 scale that multiplies each
expert's GEMM output (llama.cpp's ``build_lora_mm_id``: before the
activation for ``gate_up``, after the down projection for ``down``). Its
experts use vLLM's SITU activation, which needs both betas from the layer's
``FusedMoEConfig``.

The kernels are replaced by torch references (gguf-py dequantization), so
this runs on the CPU and checks what ``_fused_moe_gguf`` and
``GGUFMoEMethod`` do around them.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType
from gguf.quants import dequantize
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_gguf_plugin import quantization as gguf_quantization
from vllm_gguf_plugin.quantization import fused_moe as gguf_fused_moe
from vllm_gguf_plugin.quantization.utils import MOE_VEC_QUANT_TYPES

pytestmark = [pytest.mark.cpu]

T = GGMLQuantizationType
E, TOP_K, H, INTER = 4, 2, 64, 64
GATE_CAP, UP_CAP = 4.0, 25.0


def _random_nvfp4(experts, rows, cols, seed):
    rng = np.random.default_rng(seed)
    nb = cols // 64
    d = rng.integers(0x20, 0x40, size=(experts, rows, nb, 4), dtype=np.uint8)
    qs = rng.integers(0, 256, size=(experts, rows, nb, 32), dtype=np.uint8)
    return torch.from_numpy(np.concatenate([d, qs], -1).reshape(experts, rows, -1))


def _dense(w, qtype):
    return torch.stack(
        [torch.from_numpy(dequantize(w[e].numpy(), qtype)) for e in range(w.shape[0])]
    )


def _situ(gate_up, gate_cap, up_cap):
    g, u = gate_up.float().chunk(2, dim=-1)
    return (
        gate_cap
        * torch.tanh(g / gate_cap)
        * torch.sigmoid(g)
        * (up_cap * torch.tanh(u / up_cap))
    )


@pytest.fixture
def reference_kernels(monkeypatch):
    """Swap the CUDA/Triton entry points for torch references."""
    calls = []

    def moe_a8_vec(x, w, topk_ids, top_k, qtype, row, tokens):
        calls.append(qtype)
        dense = _dense(w.cpu(), T(qtype))
        ids = topk_ids.reshape(-1)
        rows = [x[i // top_k].float() @ dense[ids[i]].T for i in range(ids.numel())]
        return torch.stack(rows).to(x.dtype)

    def moe_sum(inp, out):
        out.copy_(inp.sum(dim=1))

    def activation(act, out, inp, *, activation_config=None, **_):
        assert act is MoEActivation.SITU
        assert activation_config is not None
        out.copy_(
            _situ(
                inp,
                activation_config.activation_situ_beta,
                activation_config.activation_situ_linear_beta,
            )
        )

    monkeypatch.setattr(gguf_fused_moe.ops, "ggml_moe_a8_vec", moe_a8_vec)
    monkeypatch.setattr(gguf_fused_moe.ops, "moe_sum", moe_sum)
    monkeypatch.setattr(gguf_fused_moe, "apply_moe_activation", activation)
    return calls


def _problem(tokens=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    w13 = _random_nvfp4(E, 2 * INTER, H, seed)
    w2 = _random_nvfp4(E, H, INTER, seed + 1)
    x = torch.randn(tokens, H, generator=g) / 8
    topk_ids = torch.stack(
        [torch.randperm(E, generator=g)[:TOP_K] for _ in range(tokens)]
    ).to(torch.int32)
    topk_weights = torch.rand(tokens, TOP_K, generator=g)
    s13 = torch.rand(E, generator=g) * 1e-2 + 1e-3
    s2 = torch.rand(E, generator=g) * 1e-2 + 1e-3
    return x, w13, w2, topk_ids, topk_weights, s13, s2


def _expected(x, w13, w2, topk_ids, topk_weights, s13, s2):
    d13, d2 = _dense(w13, T.NVFP4), _dense(w2, T.NVFP4)
    out = torch.zeros(x.shape[0], H)
    for t in range(x.shape[0]):
        for k in range(TOP_K):
            e = int(topk_ids[t, k])
            h = _situ((x[t] @ d13[e].T) * s13[e], GATE_CAP, UP_CAP)
            out[t] += topk_weights[t, k] * s2[e] * (h @ d2[e].T)
    return out


def test_nvfp4_takes_the_vectorized_moe_path():
    assert T.NVFP4 in MOE_VEC_QUANT_TYPES


def test_scaled_nvfp4_experts(reference_kernels):
    x, w13, w2, ids, weights, s13, s2 = _problem()
    out = gguf_fused_moe._fused_moe_gguf(
        x,
        w13,
        w2,
        weights,
        ids,
        int(T.NVFP4),
        int(T.NVFP4),
        "situ",
        s13,
        s2,
        GATE_CAP,
        UP_CAP,
    )
    assert reference_kernels == [int(T.NVFP4), int(T.NVFP4)]
    torch.testing.assert_close(out, _expected(x, w13, w2, ids, weights, s13, s2))


def test_without_scales_the_output_is_unscaled(reference_kernels):
    x, w13, w2, ids, weights, _, _ = _problem(seed=4)
    ones = torch.ones(E)
    out = gguf_fused_moe._fused_moe_gguf(
        x,
        w13,
        w2,
        weights,
        ids,
        int(T.NVFP4),
        int(T.NVFP4),
        "situ",
        None,
        None,
        GATE_CAP,
        UP_CAP,
    )
    torch.testing.assert_close(out, _expected(x, w13, w2, ids, weights, ones, ones))


# ---------------------------------------------------------- GGUFMoEMethod


def _method(expert_scales: bool):
    config = gguf_quantization.GGUFConfig()
    config.moe_expert_scales = expert_scales
    moe = SimpleNamespace(
        activation_situ_beta=GATE_CAP, activation_situ_linear_beta=UP_CAP
    )
    method = gguf_fused_moe.GGUFMoEMethod.__new__(gguf_fused_moe.GGUFMoEMethod)
    method.quant_config = config
    method.moe = moe
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        num_experts=E,
        hidden_size=H,
        intermediate_size_per_partition=INTER,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    return method, layer


def test_expert_scale_parameters_exist_only_when_asked():
    _, plain = _method(expert_scales=False)
    assert not hasattr(plain, "w13_expert_scale")
    _, layer = _method(expert_scales=True)
    for name in ("w13_expert_scale", "w2_expert_scale"):
        param = getattr(layer, name)
        assert param.dtype == torch.float32
        assert torch.equal(param.data, torch.ones(E))


def test_expert_scale_loader_writes_one_expert():
    _, layer = _method(expert_scales=True)
    param = layer.w13_expert_scale
    param.weight_loader(
        param,
        torch.tensor([0.25]),
        "model.layers.0.mlp.experts.2.gate_proj.expert_scale",
        shard_id="w1",
        expert_id=2,
    )
    assert param.data.tolist() == [1.0, 1.0, 0.25, 1.0]


def test_apply_threads_scales_and_betas(monkeypatch):
    method, layer = _method(expert_scales=True)
    seen = {}

    def fake_op(x, w1, w2, tw, ti, t1, t2, act, s13, s2, beta, linear_beta):
        seen.update(s13=s13, s2=s2, beta=beta, linear_beta=linear_beta, act=act)
        return torch.zeros_like(x)

    monkeypatch.setattr(gguf_quantization, "fused_moe_gguf", fake_op)
    layer.apply_router_weight_on_input = False
    layer.activation = MoEActivation.SITU
    layer.moe_config = method.moe
    layer.w13_weight_type.weight_type = int(T.NVFP4)
    layer.w2_weight_type.weight_type = int(T.NVFP4)
    x = torch.zeros(2, H)
    method.apply(layer, x, torch.ones(2, TOP_K), torch.zeros(2, TOP_K), None, None)
    assert seen["s13"] is layer.w13_expert_scale
    assert seen["s2"] is layer.w2_expert_scale
    assert (seen["beta"], seen["linear_beta"], seen["act"]) == (
        GATE_CAP,
        UP_CAP,
        "situ",
    )
