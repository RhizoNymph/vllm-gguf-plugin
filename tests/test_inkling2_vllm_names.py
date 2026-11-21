# SPDX-License-Identifier: Apache-2.0
"""The adapter's weight stream against vLLM's Inkling2 loading tables.

Runs the whole UD-IQ4_XS + mmproj tensor list (tiny placeholder values of
the right rank) through ``inkling2_weights`` and then through the name
handling of ``Inkling2ForConditionalGeneration.load_weights`` in the vLLM
fork (``_renames``, ``_split_renames``, ``_split_shards`` and the
``QUANT_METHOD`` expert regex of ``experts_quant.py``). Checks the contract
in ``notes/inkling2-expert-interface.md`` without building the model.
"""

import json
from collections import Counter
from pathlib import Path

import pytest
import torch
from gguf import GGMLQuantizationType

from vllm_gguf_plugin.weight_utils import weight_type_name
from vllm_gguf_plugin.weights_adapter.inkling2 import (
    Conversion,
    Inkling2GGUFMetadata,
    build_load_plan,
    inkling2_weights,
)

pytestmark = [pytest.mark.cpu]

model_module = pytest.importorskip("vllm.models.inkling2.nvidia.model")
experts_quant = pytest.importorskip("vllm.models.inkling2.nvidia.experts_quant")
Model = model_module.Inkling2ForConditionalGeneration

FIXTURES = Path(__file__).parent / "fixtures" / "inkling2"
T = GGMLQuantizationType
DENSE = {T.F32, T.F16, T.BF16}
NUM_EXPERTS = 128


def _json(name):
    return json.loads((FIXTURES / name).read_text())


def _placeholder(target, shape):
    if target.conversion in (Conversion.EXPERTS_GATE_UP, Conversion.EXPERTS_DOWN):
        return torch.zeros(NUM_EXPERTS, 2, 1)
    if target.conversion is Conversion.PATCH_EMBED:
        return torch.zeros(2, 3, 1, 1)
    if target.conversion in (
        Conversion.EXPERT_SCALE_GATE_UP,
        Conversion.EXPERT_SCALE_DOWN,
    ):
        return torch.ones(NUM_EXPERTS)
    return torch.zeros((1,) * len(shape))


@pytest.fixture(scope="module")
def stream():
    tensors = {
        n: (T[t], s)
        for f in ("ud_iq4_xs", "mmproj_bf16")
        for n, t, s in _json(f"{f}.tensors.json")
    }
    meta = Inkling2GGUFMetadata.from_kv(_json("ud_iq4_xs.metadata.json"))
    plan = build_load_plan(
        {n: t for n, (t, _) in tensors.items()}, meta, num_experts=NUM_EXPERTS
    )
    raw = []
    for gguf_name, target in plan.targets.items():
        qtype, shape = tensors[gguf_name]
        if qtype not in DENSE:
            raw.append((weight_type_name(target.hf_name), torch.tensor(int(qtype))))
        raw.append((target.hf_name, _placeholder(target, shape)))
    return list(inkling2_weights(raw, plan))


def _vllm_name(name):
    """``load_weights``' renames for the QUANT_METHOD backend with a quant
    config; returns (name, shard_id, is_expert)."""
    for old, new in Model._renames:
        name = name.replace(old, new)
    for old, new in Model._split_renames:
        name = name.replace(old, new)
    m = experts_quant._PER_EXPERT.match(name)
    if m is not None:
        fused, shard = experts_quant._SHARDS[m["proj"]]
        return f"{m['base']}.{fused}_{m['suffix']}", shard, True
    for old, (new, shard) in Model._split_shards.items():
        if old in name:
            return name.replace(old, new), shard, False
    return name, None, False


def test_experts_reach_routed_experts_parameters(stream):
    loaded = Counter()
    for name, _ in stream:
        if ".mlp.experts." not in name:
            continue
        param, shard, is_expert = _vllm_name(name)
        assert is_expert, name
        loaded[(param, shard)] += 1
    for layer in range(24):
        base = f"model.layers.{layer}.mlp.experts."
        for suffix in ("weight", "weight_type"):
            assert loaded[(base + f"w13_{suffix}", "w1")] == NUM_EXPERTS
            assert loaded[(base + f"w13_{suffix}", "w3")] == NUM_EXPERTS
            assert loaded[(base + f"w2_{suffix}", "w2")] == NUM_EXPERTS
        assert loaded[(base + "w13_expert_scale", "w1")] == NUM_EXPERTS
        assert loaded[(base + "w2_expert_scale", "w2")] == NUM_EXPERTS
    assert len(loaded) == 24 * 8


def test_every_q_proj_shard_and_type_arrives(stream):
    shards = Counter()
    for name, _ in stream:
        param, shard, _ = _vllm_name(name)
        if param.endswith(".attn.q_proj.weight") or param.endswith(
            ".attn.q_proj.weight_type"
        ):
            shards[param] += 1
            assert shard in (0, 1, 2, 3)
    for layer in range(24):
        for suffix in ("weight", "weight_type"):
            assert shards[f"model.layers.{layer}.attn.q_proj.{suffix}"] == 4


def test_no_name_outside_the_checkpoint(stream):
    hf = set(_json("hf_checkpoint_shapes.json"))
    allowed = hf | {weight_type_name(n) for n in hf if "weight" in n}
    for name, _ in stream:
        if ".mlp.experts." in name:
            continue
        assert name in allowed, name


def test_dense_tensors_keep_their_float_dtype(stream):
    for name, tensor in stream:
        if name.endswith(("weight_type", "expert_scale")):
            continue
        if ".mlp.experts." in name:
            continue
        assert tensor.dtype.is_floating_point, name
