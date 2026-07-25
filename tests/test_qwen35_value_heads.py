# SPDX-License-Identifier: Apache-2.0

"""Re-indexing of Qwen3.5 gated-delta-net value heads.

ggml strides value heads as ``r * num_k_heads + g``; HF groups them under
their key head as ``g * ratio + r``. The orderings coincide only when
num_v_heads == num_k_heads, so a 1:1 model cannot detect a mistake here.
"""

import pytest
import torch

from vllm_gguf_plugin.weights_adapter.default import GGUFWeightsAdapter


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


def _adapter(**kw):
    return GGUFWeightsAdapter(_Cfg(**kw))


def _ggml_head_order(nk, ratio):
    """Value-head index sequence as ggml stores it."""
    return [r * nk + g for g in range(nk) for r in range(ratio)]


def test_value_heads_regrouped_under_key_head():
    nk, ratio = 4, 3
    nv = nk * ratio
    ad = _adapter(nk=nk, nv=nv, hv=1)
    # One row per value head, tagged with its ggml index.
    src = torch.arange(nv, dtype=torch.float32).reshape(nv, 1)
    out = ad.transform_weight("model.layers.0.linear_attn.in_proj_b.weight", src)
    assert out.flatten().tolist() == [float(i) for i in _ggml_head_order(nk, ratio)]


def test_ratio_one_is_a_noop():
    """A 1:1 model must be left untouched - the orderings are identical."""
    ad = _adapter(nk=4, nv=4, hv=1)
    src = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    out = ad.transform_weight("model.layers.0.linear_attn.in_proj_b.weight", src)
    assert out.flatten().tolist() == [0.0, 1.0, 2.0, 3.0]


def test_quantised_tensors_are_reordered_despite_rename():
    """The weight iterator renames ``.weight`` to ``.qweight`` for quantised
    tensors. Matching only ``.weight`` silently skips exactly the projections
    that carry most of the model."""
    nk, ratio = 4, 3
    nv = nk * ratio
    ad = _adapter(nk=nk, nv=nv, hv=1)
    src = torch.arange(nv, dtype=torch.uint8).reshape(nv, 1)
    out = ad.transform_weight("model.layers.0.linear_attn.in_proj_z.qweight", src)
    assert out.flatten().tolist() == _ggml_head_order(nk, ratio)


def test_qweight_type_scalars_are_left_alone():
    ad = _adapter(nk=4, nv=12, hv=1)
    src = torch.tensor([13], dtype=torch.uint8)
    out = ad.transform_weight(
        "model.layers.0.linear_attn.in_proj_z.qweight_type", src
    )
    assert torch.equal(out, src)


def test_only_the_value_block_of_qkv_moves():
    """q and k are indexed by key heads and must stay in place."""
    nk, ratio, hk, hv = 4, 3, 2, 2
    nv = nk * ratio
    ad = _adapter(nk=nk, nv=nv, hv=hv, hk=hk)
    key_dim = nk * hk
    qk = torch.full((2 * key_dim, 1), -1.0)
    v = torch.arange(nv * hv, dtype=torch.float32).reshape(nv * hv, 1)
    out = ad.transform_weight(
        "model.layers.0.linear_attn.in_proj_qkv.weight", torch.cat([qk, v])
    )
    assert torch.equal(out[: 2 * key_dim], qk)
    assert not torch.equal(out[2 * key_dim :], v)


def test_half_block_head_raises_rather_than_corrupting():
    """out_proj re-indexes inside a packed row, so a head that is half a
    quantisation block cannot be moved without splitting the block."""
    nk, ratio, hv = 4, 3, 128
    nv = nk * ratio
    ad = _adapter(nk=nk, nv=nv, hv=hv)
    name = "model.layers.0.linear_attn.out_proj.weight"
    ad._weight_type_map = {name: "Q5_K"}  # 256-element blocks
    src = torch.zeros((8, nv * 88), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="whole Q5_K blocks"):
        ad.transform_weight(name, src)


def test_block_aligned_head_is_permitted():
    nk, ratio, hv = 4, 3, 128
    nv = nk * ratio
    ad = _adapter(nk=nk, nv=nv, hv=hv)
    name = "model.layers.0.linear_attn.out_proj.weight"
    ad._weight_type_map = {name: "Q8_0"}  # 32-element blocks
    src = torch.zeros((8, nv * 136), dtype=torch.uint8)
    ad.transform_weight(name, src)  # must not raise
