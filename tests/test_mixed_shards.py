# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed-type merged linears and the MMA-aware MMVQ threshold.

`GGUFLinearMethod._split_mixed_shard_groups` stores a merged layer whose shards have
different quant types as one contiguous weight per run of same-typed shards; apply()
must give the same result as the padded-slice path it replaces, without the per-call
weight copy. `_mmvq_max_rows` keeps single-row GEMMs on MMVQ and sends the rest of the
decode range to MMQ when the tensor-core MMQ kernel is enabled.
"""

import contextlib

import pytest
import torch
from gguf import GGMLQuantizationType

from vllm_gguf_plugin.quantization import linear as gguf_linear
from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod

from .utils import get_gguf_sample_tensors, seed_everything

HIDDEN_SIZE = 256
Q4K = GGMLQuantizationType.Q4_K
Q6K = GGMLQuantizationType.Q6_K


def _weight(quant_type, rows=None):
    data = torch.tensor(get_gguf_sample_tensors(HIDDEN_SIZE, quant_type)[0].data, device="cuda")
    return data if rows is None else data[:rows].contiguous()


class _TypeParam:
    def __init__(self, shard_weight_type, weight_type):
        self.shard_weight_type = shard_weight_type
        self.weight_type = weight_type


def _merged_layer(shards):
    """A layer as _create_padded_weight_param leaves it: `shards` is [(id, type, data)]."""
    width = max(d.size(1) for _, _, d in shards)
    rows = sum(d.size(0) for _, _, d in shards)
    padded = torch.zeros((rows, width), dtype=torch.uint8, device="cuda")
    offsets, cur = {}, 0
    for sid, _, d in shards:
        padded[cur : cur + d.size(0), : d.size(1)] = d
        offsets[sid] = (cur, cur + d.size(0), d.size(1))
        cur += d.size(0)
    layer = torch.nn.Module()
    # A plain Parameter with the attributes apply() reads (GGUFWeightParameter needs a
    # tensor-parallel group).
    param = torch.nn.Parameter(padded, requires_grad=False)
    param.shard_id = [sid for sid, _, _ in shards]
    param.shard_id_map = {sid: i for i, (sid, _, _) in enumerate(shards)}
    param.shard_offset_map = offsets
    layer.register_parameter("qweight", param)
    layer.qweight_type = _TypeParam({sid: int(t) for sid, t, _ in shards}, int(shards[0][1]))
    return layer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("num_tokens", [1, 4, 64])
@torch.inference_mode()
def test_split_groups_match_padded_slices(num_tokens):
    seed_everything(0)
    q, k, v = _weight(Q4K, 64), _weight(Q4K, 32), _weight(Q6K, 32)
    method = GGUFLinearMethod(quant_config=None)
    x = torch.rand((num_tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")

    reference = method.apply(_merged_layer([("q", Q4K, q), ("k", Q4K, k), ("v", Q6K, v)]), x)

    layer = _merged_layer([("q", Q4K, q), ("k", Q4K, k), ("v", Q6K, v)])
    method._split_mixed_shard_groups(layer)
    # q and k share a type: one group; v its own.
    assert [t for _, t in layer.gguf_shard_groups] == [int(Q4K), int(Q6K)]
    assert layer.qweight_group0.shape[0] == 96 and layer.qweight_group1.shape[0] == 32
    assert layer.qweight.numel() == 0
    # Each group owns exactly its bytes: no view keeps the padded merged weight alive.
    for name, _ in layer.gguf_shard_groups:
        group = getattr(layer, name)
        assert group.untyped_storage().nbytes() == group.numel() * group.element_size()
    torch.testing.assert_close(method.apply(layer, x), reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_single_type_merged_layer_is_left_alone():
    method = GGUFLinearMethod(quant_config=None)
    layer = _merged_layer([(0, Q4K, _weight(Q4K, 32)), (1, Q4K, _weight(Q4K, 32))])
    method._split_mixed_shard_groups(layer)
    assert not hasattr(layer, "gguf_shard_groups")
    assert layer.qweight.shape[0] == 64


@contextlib.contextmanager
def _mma(enabled):
    previous = gguf_linear._MMA_MMQ
    gguf_linear._MMA_MMQ = enabled
    try:
        yield
    finally:
        gguf_linear._MMA_MMQ = previous


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mmvq_threshold_with_mma_is_one_row():
    w = _weight(Q4K)
    x = torch.empty((4, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
    with _mma(True):
        assert gguf_linear._mmvq_max_rows(x, w, int(Q4K)) == 1
    with _mma(False):
        assert gguf_linear._mmvq_max_rows(x, w, int(Q4K)) == 6
