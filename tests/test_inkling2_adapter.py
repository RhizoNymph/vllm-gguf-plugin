# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini (ggml ``inkling2``) GGUF -> HF ``Inkling2Loop`` names.

The fixtures under ``fixtures/inkling2`` are header-only extracts of the
unsloth files at revision ea0c79b2 (UD-IQ4_XS, UD-Q4_K_XL, mmproj-BF16 and
the DSpark draft Q8_0): metadata, plus every tensor's name, ggml type and
logical shape in HF (row-major) order. ``hf_checkpoint_shapes.json`` lists
every tensor of ``thinkingmachines/Inkling-2-Mini`` @ ed40e7da
(``model.safetensors`` and, prefixed ``mtp.``, ``mtp.safetensors``) with its
stored dtype and shape. ``config.json`` is that repo's config.
"""

import copy
import json
import re
from pathlib import Path

import pytest
import torch
from gguf import GGMLQuantizationType
from vllm.models.inkling2.configs import Inkling2LoopMMConfig

from vllm_gguf_plugin.weights_adapter import get_adapter_architecture
from vllm_gguf_plugin.weights_adapter.inkling2 import (
    Conversion,
    Inkling2ConfigMismatch,
    Inkling2GGUFAdapter,
    Inkling2GGUFMetadata,
    UnmappedTensorsError,
    build_load_plan,
    dspark_name_map,
    inkling2_targets,
    inkling2_weights,
    validate_inkling2_config,
    validate_inkling2_dspark_config,
    validate_inkling2_vision_config,
)

pytestmark = [pytest.mark.cpu]

FIXTURES = Path(__file__).parent / "fixtures" / "inkling2"
QUANTS = ["ud_iq4_xs", "ud_q4_k_xl"]
T = GGMLQuantizationType


def _json(name):
    return json.loads((FIXTURES / name).read_text())


def _tensors(name):
    return {n: (T[t], tuple(s)) for n, t, s in _json(f"{name}.tensors.json")}


def _metadata(quant="ud_iq4_xs"):
    return Inkling2GGUFMetadata.from_kv(_json(f"{quant}.metadata.json"))


def _config(**text_overrides):
    raw = copy.deepcopy(_json("config.json"))
    raw["text_config"].update(text_overrides)
    return Inkling2LoopMMConfig(**raw)


def _backbone_and_vision(quant):
    return {**_tensors(quant), **_tensors("mmproj_bf16")}


HF = {k: (v[0], tuple(v[1])) for k, v in _json("hf_checkpoint_shapes.json").items()}
HF_MAIN = {k: v for k, v in HF.items() if not k.startswith("mtp.")}
HF_MTP = {k.removeprefix("mtp."): v for k, v in HF.items() if k.startswith("mtp.")}

# Official companions folded into GGUF tensors: the NVFP4/MXFP8 block
# scales live inside the ggml blocks, amax / denominator in ``.scale``.
_FOLDED = re.compile(
    r"\.mlp\.experts\.w(13|2)_weight_(scale|amax|outer_scale_denominator)$"
)
# Plugin names for the per-expert NVFP4 scales (amax / denominator).
_SYNTHESIZED = re.compile(r"\.mlp\.experts\.w(13|2)_expert_scale$")


def _hf_logical_shape(name):
    dtype, shape = HF_MAIN[name]
    if name.endswith((".w13_weight", ".w2_weight")) and dtype == "U8":
        # NVFP4: two E2M1 codes per byte.
        return (*shape[:-1], shape[-1] * 2)
    return shape


def _converted_shape(conversion, shape):
    match conversion:
        case Conversion.UNSQUEEZE_CONV:
            return (shape[0], 1, shape[1])
        case Conversion.STATIC_CONV | Conversion.TRANSPOSE:
            return (shape[1], shape[0])
        case Conversion.PATCH_EMBED:
            return (shape[0], shape[1] * shape[2] * shape[3])
        case _:
            return shape


# ---------------------------------------------------------------- metadata


def test_metadata_describes_the_loop():
    meta = _metadata()
    assert meta.num_layer_visits == 48
    assert meta.num_layers == 24
    loop = meta.loop
    assert (loop.core_start, loop.core_end, loop.num_passes, loop.gate_rank) == (
        6,
        18,
        3,
        64,
    )


def test_mainline_layout_is_rejected_with_a_pointer_to_the_root_files():
    kv = _json("ud_iq4_xs.metadata.json") | {"general.architecture": "inkling2_loop"}
    with pytest.raises(NotImplementedError, match="Mainline_Support"):
        Inkling2GGUFMetadata.from_kv(kv)


def test_other_architectures_are_rejected():
    kv = _json("ud_iq4_xs.metadata.json") | {"general.architecture": "llama"}
    with pytest.raises(ValueError, match="Not an inkling2 GGUF"):
        Inkling2GGUFMetadata.from_kv(kv)


def test_adapter_selects_the_loop_architecture():
    assert get_adapter_architecture(_config()) == "Inkling2LoopForConditionalGeneration"
    assert Inkling2GGUFAdapter.matches(_config())


# ------------------------------------------------------------------- names


@pytest.mark.parametrize("quant", QUANTS)
def test_every_tensor_is_mapped(quant):
    tensors = _backbone_and_vision(quant)
    targets = inkling2_targets(tensors, _metadata(quant))
    assert set(targets) == set(tensors)
    assert len(targets) == 734 + 422


@pytest.mark.parametrize("quant", QUANTS)
def test_targets_are_distinct_hf_checkpoint_names(quant):
    targets = inkling2_targets(_backbone_and_vision(quant), _metadata(quant))
    hf_names = [t.hf_name for t in targets.values()]
    assert len(hf_names) == len(set(hf_names))
    missing = {n for n in hf_names if n not in HF_MAIN and not _SYNTHESIZED.search(n)}
    assert not missing


@pytest.mark.parametrize("quant", QUANTS)
def test_every_hf_checkpoint_tensor_comes_from_the_gguf(quant):
    targets = inkling2_targets(_backbone_and_vision(quant), _metadata(quant))
    covered = {t.hf_name for t in targets.values()}
    uncovered = {n for n in HF_MAIN if n not in covered and not _FOLDED.search(n)}
    assert not uncovered


@pytest.mark.parametrize("quant", QUANTS)
def test_converted_shapes_match_the_hf_checkpoint(quant):
    tensors = _backbone_and_vision(quant)
    targets = inkling2_targets(tensors, _metadata(quant))
    for gguf_name, target in targets.items():
        _, shape = tensors[gguf_name]
        if target.conversion in (
            Conversion.EXPERT_SCALE_GATE_UP,
            Conversion.EXPERT_SCALE_DOWN,
        ):
            assert shape == (128,), gguf_name
            amax = target.hf_name.replace("_expert_scale", "_weight_amax")
            assert HF_MAIN[amax] == ("F32", (128,))
            continue
        want = _hf_logical_shape(target.hf_name)
        assert _converted_shape(target.conversion, shape) == want, gguf_name


def test_nvfp4_scales_only_on_the_nvfp4_layers():
    tensors = _tensors("ud_q4_k_xl")
    targets = inkling2_targets(tensors, _metadata("ud_q4_k_xl"))
    scaled = {
        int(n.split(".")[1])
        for n, t in targets.items()
        if t.conversion is Conversion.EXPERT_SCALE_GATE_UP
    }
    nvfp4 = {
        int(n.split(".")[1])
        for n, (ty, _) in tensors.items()
        if n.endswith("ffn_gate_up_exps.weight") and ty is T.NVFP4
    }
    official_nvfp4 = {
        int(n.split(".")[3])
        for n, (dtype, _) in HF_MAIN.items()
        if n.endswith(".w13_weight") and dtype == "U8"
    }
    assert scaled == nvfp4 == official_nvfp4 == set(range(20))


def test_attention_input_queries_shift_by_one_layer():
    targets = inkling2_targets(_tensors("ud_iq4_xs"), _metadata())
    L = "model.llm."
    assert targets["blk.1.attn_res_query.weight"].hf_name == (
        L + "attn_res.attn_queries.0"
    )
    assert targets["blk.23.attn_res_query.weight"].hf_name == (
        L + "attn_res.attn_queries.22"
    )
    assert targets["blk.7.ffn_res_query.weight"].hf_name == (
        L + "attn_res.mlp_queries.7"
    )
    assert "blk.0.attn_res_query.weight" not in _tensors("ud_iq4_xs")
    with pytest.raises(UnmappedTensorsError):
        inkling2_targets(["blk.0.attn_res_query.weight"], _metadata())


def test_short_convs_on_local_and_global_layers():
    targets = inkling2_targets(_tensors("ud_iq4_xs"), _metadata())
    L = "model.llm.layers."
    local = targets["blk.4.attn_k_conv.weight"]
    assert local.hf_name == L + "4.attn.k_sconv.weight"
    assert local.conversion is Conversion.UNSQUEEZE_CONV
    for layer in (5, 11, 17, 23):
        for kv in "kv":
            target = targets[f"blk.{layer}.attn_{kv}_conv.weight"]
            assert target.hf_name == f"{L}{layer}.attn.{kv}_sconv.static_w"
            assert target.conversion is Conversion.STATIC_CONV
    assert targets["blk.0.attn_post_conv.weight"].hf_name == (L + "0.attn_sconv.weight")
    assert targets["blk.0.ffn_post_conv.weight"].hf_name == (L + "0.mlp_sconv.weight")


def test_loop_tensors():
    targets = inkling2_targets(_tensors("ud_iq4_xs"), _metadata())
    L = "model.llm."
    assert targets["loop_res_query.weight"].hf_name == (
        L + "middle_cycle_cross_pass_queries"
    )
    assert targets["loop_kv_res_query.weight"].hf_name == (
        L + "middle_cycle_cross_pass_kv_queries"
    )
    assert targets["blk.6.loop_attn_down.weight"].hf_name == (
        L + "layers.6.controllers.attn.down_weight"
    )
    assert targets["blk.17.loop_mlp_forget.weight"].hf_name == (
        L + "layers.17.controllers.mlp.forget_weight"
    )
    loop_layers = {
        int(n.split(".")[1]) for n in targets if n.startswith("blk.") and ".loop_" in n
    }
    assert loop_layers == set(range(6, 18))


@pytest.mark.parametrize(
    "name",
    [
        "blk.5.loop_attn_down.weight",  # outside the loop core
        "blk.24.attn_q.weight",  # past the physical layers
        "blk.0.attn_qkv.weight",
        "output_hc_norm.weight",
        "v.blk.0.attn_qkv.weight",
        "mm.0.weight",
    ],
)
def test_unknown_tensors_raise(name):
    names = [*_tensors("ud_iq4_xs"), name]
    with pytest.raises(UnmappedTensorsError) as info:
        inkling2_targets(names, _metadata())
    assert info.value.names == (name,)


def test_vision_names():
    targets = inkling2_targets(_tensors("mmproj_bf16"), _metadata())
    V = "model.visual."
    assert targets["mm.1.weight"].hf_name == V + "proj_in.weight"
    assert targets["mm.2.weight"].hf_name == V + "proj_out.weight"
    assert targets["v.blk.3.ls2.weight"].hf_name == (V + "backbone.layers.3.ls_2.gamma")
    assert targets["v.patch_embd.weight"].conversion is Conversion.PATCH_EMBED


# ------------------------------------------------------------------ DSpark


def test_every_dspark_tensor_is_mapped_onto_mtp_safetensors():
    tensors = _tensors("dspark_q8_0")
    names = dspark_name_map(tensors, num_layers=3)
    assert set(names) == set(tensors)
    assert sorted(names.values()) == sorted(HF_MTP)
    for gguf_name, hf_name in names.items():
        assert tensors[gguf_name][1] == HF_MTP[hf_name][1], gguf_name


def test_dspark_unknown_tensor_raises():
    names = [*_tensors("dspark_q8_0"), "blk.3.attn_q.weight"]
    with pytest.raises(UnmappedTensorsError):
        dspark_name_map(names, num_layers=3)


# ------------------------------------------------------------------ config


@pytest.mark.parametrize("quant", QUANTS)
def test_repo_config_matches_the_gguf(quant):
    validate_inkling2_config(_metadata(quant), _config().text_config)


@pytest.mark.parametrize(
    ("override", "key"),
    [
        ({"hidden_size": 2048}, "embedding_length"),
        ({"num_hidden_layers": 23}, "block_count"),
        ({"local_layer_ids": [0, 1, 2, 3, 4]}, "sliding_window_pattern"),
        ({"route_scale": 4.0}, "expert_weights_scale"),
        ({"mlp_up_softcap": 30.0}, "situ_linear_beta"),
        ({"n_routed_experts": 64}, "expert_count"),
        ({"d_rel": 32}, "relative_length"),
        ({"rms_norm_eps": 1e-5}, "layer_norm_rms_epsilon"),
        ({"unpadded_vocab_size": 200000}, "vocab_size_unpadded"),
        ({"swa_num_attention_heads": 16}, "attention.head_count"),
        (
            {
                "loop_config": {
                    "core_start": 6,
                    "core_end": 18,
                    "num_passes": 2,
                    "gate_rank": 64,
                    "is_varloop": False,
                }
            },
            "num_loops",
        ),
        (
            {
                "loop_config": {
                    "core_start": 6,
                    "core_end": 18,
                    "num_passes": 3,
                    "gate_rank": 32,
                    "is_varloop": False,
                }
            },
            "loop.gate_rank",
        ),
    ],
)
def test_config_mismatch_names_the_key(override, key):
    with pytest.raises(Inkling2ConfigMismatch) as info:
        validate_inkling2_config(_metadata(), _config(**override).text_config)
    assert any(key in m for m in info.value.mismatches), info.value.mismatches


def test_missing_metadata_key_is_a_mismatch():
    kv = dict(_json("ud_iq4_xs.metadata.json"))
    del kv["inkling2.attention.relative_extent"]
    with pytest.raises(Inkling2ConfigMismatch, match="relative_extent: missing"):
        validate_inkling2_config(
            Inkling2GGUFMetadata.from_kv(kv), _config().text_config
        )


def test_vision_config_matches_the_mmproj():
    config = _config()
    validate_inkling2_vision_config(
        _json("mmproj_bf16.metadata.json"), config.vision_config
    )
    wrong = dict(config.vision_config) | {"num_hidden_layers": 24}
    with pytest.raises(Inkling2ConfigMismatch, match="block_count"):
        validate_inkling2_vision_config(_json("mmproj_bf16.metadata.json"), wrong)


def test_dspark_config_matches_the_draft():
    config = _config()
    kv = _json("dspark_q8_0.metadata.json")
    validate_inkling2_dspark_config(kv, config.mtp_config, config.text_config)
    wrong = dict(config.mtp_config) | {"target_layer_ids": [5, 23, 41, 47]}
    with pytest.raises(Inkling2ConfigMismatch, match="target_layers"):
        validate_inkling2_dspark_config(kv, wrong, config.text_config)


# ------------------------------------------------- value conversions
#
# The forward transforms below are transcribed from llama.cpp's
# conversion/inkling2.py; the adapter must invert each exactly.


def _llama_cpp_forward(conversion, hf):
    match conversion:
        case Conversion.UNSQUEEZE_CONV:
            return hf.squeeze(1)
        case Conversion.STATIC_CONV:
            return hf.flip(0).T.contiguous()
        case Conversion.TRANSPOSE:
            return hf.T.contiguous()
        case Conversion.PATCH_EMBED:
            p = 4
            return hf.reshape(hf.shape[0], p, p, 3).permute(0, 3, 1, 2)
    raise AssertionError(conversion)


_HF_SHAPES = {
    Conversion.UNSQUEEZE_CONV: (6, 1, 4),
    Conversion.STATIC_CONV: (4, 6),
    Conversion.TRANSPOSE: (3, 8),
    Conversion.PATCH_EMBED: (5, 4 * 4 * 3),
}

_SAMPLE_NAMES = {
    Conversion.UNSQUEEZE_CONV: ("blk.0.attn_post_conv.weight", (6, 4)),
    Conversion.STATIC_CONV: ("blk.5.attn_k_conv.weight", (6, 4)),
    Conversion.TRANSPOSE: ("blk.0.attn_rel_bias.weight", (8, 3)),
    Conversion.PATCH_EMBED: ("v.patch_embd.weight", (5, 3, 4, 4)),
}


def _tiny_plan(tensors):
    """A load plan over a few tensors of the real UD-IQ4_XS layout."""
    real = {**_tensors("ud_iq4_xs"), **_tensors("mmproj_bf16")}
    types = {n: real[n][0] for n in tensors}
    return build_load_plan(types, _metadata(), num_experts=4)


@pytest.mark.parametrize("conversion", list(_HF_SHAPES))
def test_conversions_invert_llama_cpp(conversion):
    gguf_name, _ = _SAMPLE_NAMES[conversion]
    plan = _tiny_plan(
        [gguf_name, "blk.5.attn_k_conv_proj.weight"]
        if conversion is Conversion.STATIC_CONV
        else [gguf_name]
    )
    target = plan.targets[gguf_name]
    assert target.conversion is conversion
    hf = torch.randn(_HF_SHAPES[conversion])
    stored = _llama_cpp_forward(conversion, hf)
    out = dict(inkling2_weights([(target.hf_name, stored)], plan))
    assert torch.equal(out[target.hf_name], hf)


# ---------------------------------------------------------- weight stream

E, INTER, H = 4, 6, 8  # experts, intermediate, hidden (tiny)
LAYER = "model.llm.layers.0."


def _expert_plan(gate_up_type=T.NVFP4, down_type=T.NVFP4, scaled=True):
    types = {
        "blk.0.ffn_gate_up_exps.weight": gate_up_type,
        "blk.0.ffn_down_exps.weight": down_type,
        "blk.1.ffn_gate_up_exps.weight": T.Q8_0,
        "blk.1.ffn_down_exps.weight": T.Q8_0,
    }
    if scaled:
        types |= {
            "blk.0.ffn_gate_up_exps.scale": T.F32,
            "blk.0.ffn_down_exps.scale": T.F32,
        }
    return build_load_plan(types, _metadata(), num_experts=E)


def _packed(rows, cols_bytes, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (E, rows, cols_bytes), dtype=torch.uint8, generator=g)


def _stream(plan, gate_up, down, scales=None):
    w13 = LAYER + "mlp.experts.w13_weight"
    w2 = LAYER + "mlp.experts.w2_weight"
    stream = [
        (w13 + "_type", torch.tensor(int(T.NVFP4))),
        (w13, gate_up),
        (w2 + "_type", torch.tensor(int(T.NVFP4))),
        (w2, down),
    ]
    if scales is not None:
        stream += [
            (LAYER + "mlp.experts.w13_expert_scale", scales[0]),
            (LAYER + "mlp.experts.w2_expert_scale", scales[1]),
        ]
    return dict(inkling2_weights(stream, plan))


def test_stacked_experts_are_served_per_expert_gate_then_up():
    plan = _expert_plan()
    gate_up, down = _packed(2 * INTER, 9, 0), _packed(H, 5, 1)
    out = _stream(plan, gate_up, down, (torch.rand(E), torch.rand(E)))
    base = LAYER + "mlp.experts."
    for e in range(E):
        assert torch.equal(out[f"{base}{e}.gate_proj.weight"], gate_up[e, :INTER])
        assert torch.equal(out[f"{base}{e}.up_proj.weight"], gate_up[e, INTER:])
        assert torch.equal(out[f"{base}{e}.down_proj.weight"], down[e])
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert int(out[f"{base}{e}.{proj}.weight_type"]) == int(T.NVFP4)
    assert not any(n.endswith(("w13_weight", "w2_weight", "_weight_type")) for n in out)


def test_expert_scales_are_served_per_expert():
    plan = _expert_plan()
    s13, s2 = torch.rand(E), torch.rand(E)
    out = _stream(plan, _packed(2 * INTER, 9, 0), _packed(H, 5, 1), (s13, s2))
    base = LAYER + "mlp.experts."
    for e in range(E):
        assert out[f"{base}{e}.gate_proj.expert_scale"].item() == s13[e].item()
        assert out[f"{base}{e}.down_proj.expert_scale"].item() == s2[e].item()
        assert f"{base}{e}.up_proj.expert_scale" not in out


def test_layers_without_nvfp4_scales_get_unit_scales():
    plan = _expert_plan()
    out = dict(inkling2_weights([], plan))
    base = "model.llm.layers.1.mlp.experts."
    for e in range(E):
        assert out[f"{base}{e}.gate_proj.expert_scale"].item() == 1.0
        assert out[f"{base}{e}.down_proj.expert_scale"].item() == 1.0
    assert not any(n.startswith(LAYER) for n in out)


def test_dense_experts_carry_their_ggml_type():
    plan = _expert_plan(gate_up_type=T.BF16, down_type=T.BF16, scaled=False)
    gate_up = torch.randn(E, 2 * INTER, H, dtype=torch.bfloat16)
    down = torch.randn(E, H, INTER, dtype=torch.bfloat16)
    stream = [
        (LAYER + "mlp.experts.w13_weight", gate_up),
        (LAYER + "mlp.experts.w2_weight", down),
    ]
    out = dict(inkling2_weights(stream, plan))
    base = LAYER + "mlp.experts."
    assert int(out[base + "0.gate_proj.weight_type"]) == int(T.BF16)
    assert int(out[base + "3.down_proj.weight_type"]) == int(T.BF16)
    assert torch.equal(out[base + "2.up_proj.weight"], gate_up[2, INTER:])


def test_dense_shard_of_a_quantized_q_proj_carries_its_type():
    """attn_sink_proj is BF16 beside quantized q / r / gate rows of the same
    merged q_proj: it is served by the GGUF linear with a BF16 type."""
    real = _tensors("ud_iq4_xs")
    shards = [
        f"blk.0.{n}.weight"
        for n in ("attn_q", "attn_rel", "attn_gate", "attn_sink_proj")
    ]
    plan = build_load_plan({n: real[n][0] for n in shards}, _metadata(), num_experts=E)
    assert plan.dense_in_quantized_modules == {"blk.0.attn_sink_proj.weight"}
    sink = torch.randn(24, 32, dtype=torch.bfloat16)
    name = LAYER + "attn.attn_sink_proj.weight"
    out = dict(inkling2_weights([(name, sink)], plan))
    assert int(out[LAYER + "attn.attn_sink_proj.weight_type"]) == int(T.BF16)
    assert torch.equal(out[name], sink)


def test_all_dense_q_proj_stays_unquantized():
    shards = [
        f"blk.0.{n}.weight"
        for n in ("attn_q", "attn_rel", "attn_gate", "attn_sink_proj")
    ]
    plan = build_load_plan(dict.fromkeys(shards, T.BF16), _metadata(), num_experts=E)
    assert plan.dense_in_quantized_modules == frozenset()


def test_quantized_tensors_pass_through_with_their_type():
    plan = _tiny_plan(["blk.0.attn_q.weight", "token_embd.weight"])
    q = torch.randint(0, 256, (3072, 2448), dtype=torch.uint8)
    name = LAYER + "attn.wq_du.weight"
    out = list(
        inkling2_weights([(name + "_type", torch.tensor(int(T.Q8_0))), (name, q)], plan)
    )
    assert [n for n, _ in out] == [name + "_type", name]


def test_value_conversion_of_a_quantized_tensor_is_refused():
    plan = _tiny_plan(["blk.0.attn_post_conv.weight"])
    name = LAYER + "attn_sconv.weight"
    stream = [
        (name + "_type", torch.tensor(int(T.Q8_0))),
        (name, torch.zeros(2304, 34, dtype=torch.uint8)),
    ]
    with pytest.raises(TypeError, match="attn_sconv"):
        list(inkling2_weights(stream, plan))


def test_dspark_stream_is_dense_bf16():
    from gguf.quants import quantize

    from vllm_gguf_plugin.weights_adapter.inkling2 import dspark_weights

    values = torch.randn(4, 64)
    packed = torch.from_numpy(quantize(values.numpy(), T.Q8_0))
    stream = [
        ("layers.0.self_attn.q_proj.weight_type", torch.tensor(int(T.Q8_0))),
        ("layers.0.self_attn.q_proj.weight", packed),
        ("norm.weight", torch.randn(8)),
        ("mask_embed", torch.randn(8)),
    ]
    names = ["layers.0.self_attn.q_proj.weight", "norm.weight", "mask_embed"]
    out = dict(dspark_weights(stream, names))
    assert set(out) == set(names)
    assert all(t.dtype is torch.bfloat16 for t in out.values())
    q = out["layers.0.self_attn.q_proj.weight"]
    assert q.shape == (4, 64)
    assert torch.allclose(q.float(), values, atol=0.05, rtol=0.02)


# ------------------------------------------------------------ loader glue


def test_pure_gguf_source_is_sent_to_the_repo(tmp_path):
    from gguf import GGUFWriter

    from vllm_gguf_plugin.config_parser import _reject_config_from_metadata_only

    path = tmp_path / "Inkling-2-Mini-UD-IQ4_XS.gguf"
    writer = GGUFWriter(str(path), "inkling2")
    writer.add_tensor("token_embd_norm.weight", torch.ones(4).numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with pytest.raises(RuntimeError, match="thinkingmachines/Inkling-2-Mini"):
        _reject_config_from_metadata_only(str(path))


def test_dense_shards_of_quantized_modules_are_not_unquantized(tmp_path):
    from gguf import GGUFWriter
    from gguf.quants import quantize

    from vllm_gguf_plugin.gguf_files import GGUFModelFiles
    from vllm_gguf_plugin.loader import _get_unquantized_modules

    path = tmp_path / "tiny.gguf"
    writer = GGUFWriter(str(path), "inkling2")
    writer.add_tensor(
        "blk.0.attn_q.weight",
        quantize(torch.randn(2, 32).numpy(), T.Q8_0),
        raw_dtype=T.Q8_0,
    )
    writer.add_tensor("blk.0.attn_sink_proj.weight", torch.randn(2, 32).numpy())
    writer.add_tensor("blk.0.attn_norm.weight", torch.ones(32).numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    names = {
        "blk.0.attn_q.weight": LAYER + "attn.wq_du.weight",
        "blk.0.attn_sink_proj.weight": LAYER + "attn.attn_sink_proj.weight",
        "blk.0.attn_norm.weight": LAYER + "attn_norm.weight",
    }
    files = GGUFModelFiles(backbone=(str(path),))
    assert _get_unquantized_modules(files, names) == (
        LAYER + "attn.attn_sink_proj",
        LAYER + "attn_norm",
    )
    assert _get_unquantized_modules(
        files, names, frozenset({"blk.0.attn_sink_proj.weight"})
    ) == (LAYER + "attn_norm",)


# ---------------------------------------------------------------- LM head
#
# Inkling2's config asks for an fp32 head (``head_dtype: float32``). A GGUF
# serves its packed ``output.weight`` as llama.cpp does, so the GGUF config
# parser resets the head to the model dtype. ``--hf-overrides
# '{"head_dtype": "float32"}'`` (applied after parsing) opts back in; vLLM's
# LogitsProcessor runs an fp32 head only over an unquantized lm_head, so the
# packed head is then dequantized to BF16 (the checkpoint's own unembedding
# dtype) on load.


def _parse_inkling2_config(monkeypatch, config):
    import vllm_gguf_plugin.config_parser as config_parser_module

    monkeypatch.setattr(
        config_parser_module.HFConfigParser,
        "parse",
        lambda self, model, trust_remote_code, **kwargs: (config.to_dict(), config),
    )
    return config_parser_module.GGUFConfigParser().parse(
        "thinkingmachines/Inkling-2-Mini", trust_remote_code=False
    )


def test_gguf_parser_serves_the_packed_head(monkeypatch):
    config = _config()
    assert config.head_dtype == "float32"
    config_dict, parsed = _parse_inkling2_config(monkeypatch, config)
    assert parsed.head_dtype == "model"
    assert config_dict["head_dtype"] == "model"
    assert not Inkling2GGUFAdapter.wants_dense_lm_head(parsed)
    assert Inkling2GGUFAdapter().unquantized_modules_for(parsed) == ()


def test_an_fp32_head_override_after_parsing_dequantizes_the_head(monkeypatch):
    _, parsed = _parse_inkling2_config(monkeypatch, _config())
    # vLLM applies ``--hf-overrides`` to the parsed config.
    parsed.update({"head_dtype": "float32"})
    assert Inkling2GGUFAdapter.wants_dense_lm_head(parsed)
    assert Inkling2GGUFAdapter().unquantized_modules_for(parsed) == ("lm_head",)


def test_gguf_parser_leaves_other_models_head_dtype(monkeypatch):
    from transformers import PretrainedConfig

    import vllm_gguf_plugin.config_parser as config_parser_module

    config = PretrainedConfig(model_type="qwen3_moe", head_dtype="float32")
    monkeypatch.setattr(
        config_parser_module.HFConfigParser,
        "parse",
        lambda self, model, trust_remote_code, **kwargs: ({}, config),
    )
    _, parsed = config_parser_module.GGUFConfigParser().parse(
        "Qwen/Qwen3-30B-A3B", trust_remote_code=False
    )
    assert parsed.head_dtype == "float32"


def test_lm_head_is_dequantized_to_bf16_for_an_fp32_head():
    from gguf.quants import dequantize, quantize

    real = _tensors("ud_iq4_xs")
    plan = build_load_plan(
        {"output.weight": real["output.weight"][0]},
        _metadata(),
        num_experts=E,
        dense_lm_head=True,
    )
    values = torch.randn(300, 256)
    packed = torch.from_numpy(quantize(values.numpy(), T.Q8_0))
    name = "model.llm.unembed.weight"
    out = dict(
        inkling2_weights(
            [(name + "_type", torch.tensor(int(T.Q8_0))), (name, packed)], plan
        )
    )
    assert set(out) == {name}
    assert out[name].dtype is torch.bfloat16
    want = torch.from_numpy(dequantize(packed.numpy(), T.Q8_0)).to(torch.bfloat16)
    assert torch.equal(out[name], want)


def test_lm_head_stays_packed_without_an_fp32_head():
    plan = build_load_plan(
        {"output.weight": T.Q6_K}, _metadata(), num_experts=E, dense_lm_head=False
    )
    name = "model.llm.unembed.weight"
    packed = torch.zeros(4, 210, dtype=torch.uint8)
    out = list(
        inkling2_weights(
            [(name + "_type", torch.tensor(int(T.Q6_K))), (name, packed)], plan
        )
    )
    assert [n for n, _ in out] == [name + "_type", name]


@pytest.mark.parametrize(("head_dtype", "dense"), [("float32", True), ("model", False)])
def test_adapter_follows_the_head_dtype(head_dtype, dense):
    adapter = Inkling2GGUFAdapter()
    config = _config()
    config.head_dtype = head_dtype
    assert adapter.wants_dense_lm_head(config) is dense
    assert ("lm_head" in adapter.unquantized_modules_for(config)) is dense
