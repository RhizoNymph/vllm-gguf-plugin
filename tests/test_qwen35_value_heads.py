# SPDX-License-Identifier: Apache-2.0

"""Re-indexing of Qwen3.5 gated-delta-net value heads.

ggml strides value heads as ``r * num_k_heads + g``; HF groups them under
their key head as ``g * ratio + r``. The orderings coincide only when
num_v_heads == num_k_heads, so a 1:1 model cannot detect a mistake here.

Unquantized tensors are re-indexed on load. Quantized weights are left in
ggml order and the *activations* are permuted to match at matmul time, via
the layout registered on the linear layer -- that avoids permuting bytes
inside a packed row, which would split a quantisation block whenever a head
is not a whole number of blocks (Q5_K packs 256 elements, so a 128-element
head is half a block).
"""

import torch

from vllm_gguf_plugin.quantization.layout import GGUFHeadTilingLayout
from vllm_gguf_plugin.weights_adapter.qwen3_5 import (
    Qwen35GGUFAdapter,
    _gdn_value_head_layout,
)


class _Cfg:
    """Minimal stand-in for the HF text config the adapter reads."""

    model_type = "qwen3_5_text"

    def __init__(self, nk=4, nv=12, hv=2, hk=2):
        self.linear_num_key_heads = nk
        self.linear_num_value_heads = nv
        self.linear_value_head_dim = hv
        self.linear_key_head_dim = hk

    def get_text_config(self):
        return self


class _ModelConfig:
    def __init__(self, cfg):
        self.hf_config = cfg


def _ggml_head_order(nk, ratio):
    """Value-head index sequence as ggml stores it."""
    return [r * nk + g for g in range(nk) for r in range(ratio)]


def _restore(cfg, name, weight):
    layout = _gdn_value_head_layout(cfg)
    return Qwen35GGUFAdapter()._restore_gdn_weight(name, weight, cfg, layout)


def test_value_heads_regrouped_under_key_head():
    nk, ratio = 4, 3
    nv = nk * ratio
    cfg = _Cfg(nk=nk, nv=nv, hv=1)
    # One row per value head, tagged with its ggml index.
    src = torch.arange(nv, dtype=torch.float32).reshape(nv, 1)
    out = _restore(cfg, "model.layers.0.linear_attn.in_proj_b.weight", src)
    assert out.flatten().tolist() == [float(i) for i in _ggml_head_order(nk, ratio)]


def test_ratio_one_registers_no_layout():
    """A 1:1 model must be left untouched - the orderings are identical."""
    assert _gdn_value_head_layout(_Cfg(nk=4, nv=4, hv=1)) is None


def test_ratio_one_declares_no_linear_layouts():
    cfg = _Cfg(nk=4, nv=4, hv=1)
    name_map = {"blk.0.ssm_out.weight": "model.layers.0.linear_attn.out_proj.weight"}
    layouts = Qwen35GGUFAdapter().get_linear_layouts(
        None, _ModelConfig(cfg), name_map
    )
    assert layouts == {}


def test_out_proj_gets_a_layout_so_activations_are_reordered():
    """Grouped value heads index out_proj's *input* axis. The weight may be
    packed, so the reordering has to happen on the activations instead."""
    nk, ratio, hv = 4, 3, 128
    cfg = _Cfg(nk=nk, nv=nk * ratio, hv=hv)
    name_map = {"blk.0.ssm_out.weight": "model.layers.0.linear_attn.out_proj.weight"}
    layouts = Qwen35GGUFAdapter().get_linear_layouts(
        None, _ModelConfig(cfg), name_map
    )
    assert layouts == {
        "model.layers.0.linear_attn.out_proj": GGUFHeadTilingLayout(
            heads_per_group=ratio, head_dim=hv
        )
    }


def test_packed_out_proj_is_left_in_ggml_order():
    """Quantised tensors arrive renamed to ``.qweight``. Permuting bytes
    inside a packed row could split a quantisation block, so the weight must
    be handed through untouched - the layout fixes up the input instead."""
    nk, ratio, hv = 4, 3, 128
    cfg = _Cfg(nk=nk, nv=nk * ratio, hv=hv)
    src = torch.zeros((8, nk * ratio * 88), dtype=torch.uint8)
    assert _restore(cfg, "model.layers.0.linear_attn.out_proj.qweight", src) is None


def test_qweight_type_scalars_are_left_alone():
    cfg = _Cfg(nk=4, nv=12, hv=1)
    src = torch.tensor([13], dtype=torch.uint8)
    out = _restore(cfg, "model.layers.0.linear_attn.in_proj_z.qweight_type", src)
    assert out is None


def test_activation_reorder_matches_weight_reorder():
    """The load-bearing invariant of the packed path: permuting the input
    against a ggml-ordered weight must equal leaving the input alone and
    re-indexing the weight. If these disagree, quantised and unquantised
    models silently produce different results."""
    nk, ratio, hv = 4, 3, 2
    nv = nk * ratio
    layout = GGUFHeadTilingLayout(heads_per_group=ratio, head_dim=hv)

    x_hf = torch.randn(5, nv * hv)
    w_ggml = torch.randn(7, nv * hv)
    w_hf = layout.weight_to_vllm(w_ggml, dim=1)

    torch.testing.assert_close(x_hf @ w_hf.T, layout.input_to_gguf(x_hf) @ w_ggml.T)


def test_only_the_value_block_of_qkv_moves():
    """q and k are indexed by key heads and must stay in place."""
    nk, ratio, hk, hv = 4, 3, 2, 2
    nv = nk * ratio
    cfg = _Cfg(nk=nk, nv=nv, hv=hv, hk=hk)
    qk_rows = 2 * nk * hk
    qk = torch.full((qk_rows, 1), -1.0)
    v = torch.arange(nv * hv, dtype=torch.float32).reshape(nv * hv, 1)
    out = _restore(
        cfg, "model.layers.0.linear_attn.in_proj_qkv.weight", torch.cat([qk, v])
    )
    assert torch.equal(out[:qk_rows], qk)
    assert not torch.equal(out[qk_rows:], v)


def test_a_log_is_unexponentiated_and_reindexed():
    """ggml stores the decay already exponentiated as A = -exp(A_log); vLLM's
    kernels exponentiate A_log themselves, so passing it straight through
    applies exp twice."""
    nk, ratio = 4, 3
    nv = nk * ratio
    cfg = _Cfg(nk=nk, nv=nv, hv=1)
    a_log = torch.arange(nv, dtype=torch.float32).reshape(nv, 1)
    out = _restore(cfg, "model.layers.0.linear_attn.A_log", -torch.exp(a_log))
    expected = a_log[_ggml_head_order(nk, ratio)]
    torch.testing.assert_close(out, expected)
