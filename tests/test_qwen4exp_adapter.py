# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next (ggml ``qwen4exp``) GGUF -> vLLM Qwen4Exp.

The fixtures under ``fixtures/qwen4exp`` are header-only extracts of the
real unsloth files (UD-Q2_K_XL, UD-IQ3_XXS, mmproj-BF16): metadata, plus
every tensor's name, ggml type and logical shape in HF (row-major) order.
``hf_checkpoint_shapes.json`` lists the official Qwen/Qwen3.8-Flash-Next
checkpoint's tensor names with layer indices folded to ``N``.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType
from gguf.quants import quantize
from transformers import Qwen4ExpConfig

from vllm_gguf_plugin.weights_adapter import get_adapter_architecture
from vllm_gguf_plugin.weights_adapter.qwen4_exp import (
    Qwen4ExpGGUFAdapter,
    Qwen4ExpGGUFMetadata,
    qwen4exp_name_map,
    validate_qwen4exp_config,
)

pytestmark = [pytest.mark.cpu]

FIXTURES = Path(__file__).parent / "fixtures" / "qwen4exp"
QUANTS = ["ud_q2_k_xl", "ud_iq3_xxs"]


def _json(name):
    return json.loads((FIXTURES / name).read_text())


def _tensors(name):
    return {n: (t, tuple(s)) for n, t, s in _json(f"{name}.tensors.json")}


def _metadata():
    return Qwen4ExpGGUFMetadata.from_kv(_json("ud_q2_k_xl.metadata.json"))


def _hf_config():
    return Qwen4ExpConfig(**_json("config.json"))


def _fold(name: str) -> str:
    name = re.sub(r"\.(layers|blocks)\.\d+\.", r".\1.N.", name)
    return re.sub(r"\.experts\.\d+\.", ".experts.", name)


HF_SHAPES = {k: tuple(v[1]) for k, v in _json("hf_checkpoint_shapes.json").items()}


# --------------------------------------------------------------------- names


@pytest.mark.parametrize("quant", QUANTS)
def test_every_text_tensor_is_mapped(quant):
    names = list(_tensors(quant))
    name_map = qwen4exp_name_map(names, _metadata(), is_multimodal=True)
    assert set(name_map) == set(names)


@pytest.mark.parametrize("quant", QUANTS)
def test_mapped_names_exist_in_the_hf_checkpoint(quant):
    """Every target is an HF checkpoint name, apart from the three the
    adapter assembles itself (indexer halves, the PLE table)."""
    name_map = qwen4exp_name_map(list(_tensors(quant)), _metadata(), True)
    synthesized = {
        "model.language_model.layers.N.self_attn.indexer.index_q_proj.weight",
        "model.language_model.layers.N.self_attn.indexer.index_k_proj.weight",
        "model.language_model.layers.N.ple.ple_embedding.ngram_embedding.weight",
    }
    hf_names = set(HF_SHAPES)
    hf_names |= {
        # Stacked experts arrive per expert, as vLLM's expert mapping expects.
        "model.language_model.layers.N.mlp.experts.gate_proj.weight",
        "model.language_model.layers.N.mlp.experts.up_proj.weight",
        "model.language_model.layers.N.mlp.experts.down_proj.weight",
    }
    missing = {
        _fold(hf) for hf in name_map.values() if _fold(hf) not in hf_names | synthesized
    }
    assert not missing


@pytest.mark.parametrize("quant", QUANTS)
def test_mapped_shapes_match_the_hf_checkpoint(quant):
    tensors = _tensors(quant)
    name_map = qwen4exp_name_map(list(tensors), _metadata(), True)
    for gguf_name, hf_name in name_map.items():
        _, shape = tensors[gguf_name]
        folded = _fold(hf_name)
        if ".experts." in folded:
            num_experts, *expert = shape
            assert num_experts == 512
            fused = HF_SHAPES[
                folded.replace("gate_proj.weight", "gate_up_proj")
                .replace("up_proj.weight", "gate_up_proj")
                .replace("down_proj.weight", "down_proj")
            ]
            assert fused[0] == num_experts
            assert expert[1] == fused[2]
            continue
        if folded.endswith("conv1d.weight"):
            # ggml stores the depthwise kernel squeezed: [channels, width].
            assert shape == (HF_SHAPES[folded][0], HF_SHAPES[folded][2])
            continue
        if "index_q_proj" in folded or "index_k_proj" in folded:
            fused = HF_SHAPES[
                folded.replace("index_q_", "index_qk_").replace("index_k_", "index_qk_")
            ]
            assert shape[1] == fused[1]
            continue
        if folded.endswith("ngram_embedding.weight"):
            sharded = folded.replace("ngram_embedding.", "ngram_embedding.shard_S.")
            shard = HF_SHAPES[sharded]
            assert shape == (shard[0] * 128, shard[1])
            continue
        if folded.endswith("shared_expert_gate.weight"):
            assert (1, *shape) == HF_SHAPES[folded]
            continue
        assert shape == HF_SHAPES[folded], gguf_name


def test_hyper_connection_names():
    name_map = qwen4exp_name_map(list(_tensors("ud_q2_k_xl")), _metadata(), True)
    p = "model.language_model."
    assert name_map["blk.5.hc_attn_down.weight"] == (
        p + "layers.5.attn_hyper_connection.input_mix_weight_down.weight"
    )
    assert name_map["blk.5.hc_ffn_inject.weight"] == (
        p + "layers.5.mlp_hyper_connection.block_inject_weight.weight"
    )
    assert name_map["output_hc_up.weight"] == (
        p + "hyper_connection_mixer.input_mix_weight_up.weight"
    )
    assert name_map["blk.1.ple_key.weight"] == p + "layers.1.ple.key_proj.weight"
    assert name_map["per_layer_token_embd.weight"] == (
        p + "layers.1.ple.ple_embedding.ngram_embedding.weight"
    )


def test_text_only_prefix():
    name_map = qwen4exp_name_map(list(_tensors("ud_q2_k_xl")), _metadata(), False)
    assert name_map["token_embd.weight"] == "model.embed_tokens.weight"
    assert name_map["output.weight"] == "lm_head.weight"


def test_vision_tensors_are_mapped():
    tensors = _tensors("mmproj_bf16")
    name_map = qwen4exp_name_map(list(tensors), _metadata(), True)
    assert set(name_map) == set(tensors)
    hf_names = set(HF_SHAPES)
    for gguf_name, hf_name in name_map.items():
        base = hf_name.removesuffix(".1")
        assert _fold(base) in hf_names, gguf_name


def test_unknown_tensor_is_reported():
    with pytest.raises(ValueError, match="blk.0.mystery.weight"):
        qwen4exp_name_map(["blk.0.mystery.weight"], _metadata(), True)


# -------------------------------------------------------------------- config


def test_real_metadata_matches_the_official_config():
    validate_qwen4exp_config(_metadata(), _hf_config().get_text_config())


@pytest.mark.parametrize(
    "key, value",
    [
        ("qwen4exp.block_count", 49),
        ("qwen4exp.expert_count", 256),
        ("qwen4exp.hyper_connection.count", 2),
        ("qwen4exp.ple.layers", [2]),
        ("qwen4exp.ssm.group_count", 8),
        ("qwen4exp.attention.indexer.top_k", 1024),
    ],
)
def test_mismatched_metadata_is_rejected(key, value):
    kv = _json("ud_q2_k_xl.metadata.json")
    kv[key] = value
    with pytest.raises(ValueError, match=re.escape(key)):
        validate_qwen4exp_config(
            Qwen4ExpGGUFMetadata.from_kv(kv), _hf_config().get_text_config()
        )


def test_metadata_reads_ple_constants():
    meta = _metadata()
    assert meta.ple_layers == (1,)
    assert meta.ple_layer_multipliers == (
        23703573157769,
        20109073645365,
        8052911324071,
    )
    assert len(meta.ple_head_offsets) == 16
    assert meta.ple_head_vocab_sizes[0] == 20000003


def test_adapter_matches_and_declares_architecture():
    config = _hf_config()
    assert Qwen4ExpGGUFAdapter.matches(config)
    assert get_adapter_architecture(config) == "Qwen4ExpForConditionalGeneration"
    text = config.get_text_config()
    assert Qwen4ExpGGUFAdapter.matches(text)
    assert get_adapter_architecture(text) == "Qwen4ExpForCausalLM"


# ---------------------------------------------------------------- transforms


class _TextCfg(SimpleNamespace):
    def get_text_config(self):
        return self


def _small_model_config():
    text = _TextCfg(
        model_type="qwen4_exp_text",
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        split_ngram_parts=4,
    )
    root = SimpleNamespace(
        model_type="qwen4_exp",
        vision_config=SimpleNamespace(temporal_patch_size=2),
        get_text_config=lambda: text,
    )
    return SimpleNamespace(hf_config=root)


def _adapter_with(meta=None):
    adapter = Qwen4ExpGGUFAdapter()
    adapter.metadata = meta or _metadata()
    return adapter


P = "model.language_model.layers.0."


def _q8(rows, cols, seed=0):
    data = np.random.default_rng(seed).standard_normal((rows, cols)).astype(np.float32)
    return data, torch.from_numpy(quantize(data, GGMLQuantizationType.Q8_0))


_PLE_CONSTANTS = ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")


def _run(weights, meta=None, constants=False):
    """Transformed weights; the PLE hash constants (always appended) are
    dropped unless asked for."""
    adapter = _adapter_with(meta)
    out = adapter.transform_weights(iter(weights), _small_model_config())
    return [(n, w) for n, w in out if constants or not n.endswith(_PLE_CONSTANTS)]


def test_hyper_connection_weights_are_dequantized():
    dense, packed = _q8(4, 64)
    out = dict(
        _run(
            [
                (
                    P + "attn_hyper_connection.input_mix_weight_down.weight_type",
                    torch.tensor(GGMLQuantizationType.Q8_0),
                ),
                (P + "attn_hyper_connection.input_mix_weight_down.weight", packed),
            ]
        )
    )
    assert list(out) == [P + "attn_hyper_connection.input_mix_weight_down.weight"]
    got = out[P + "attn_hyper_connection.input_mix_weight_down.weight"]
    assert got.dtype == torch.bfloat16
    assert got.shape == (4, 64)
    torch.testing.assert_close(got.float(), torch.from_numpy(dense), atol=0.05, rtol=0)


def test_zero_centred_norms_are_restored():
    w = torch.full((8,), 1.25)
    names = [
        P + "attn_hyper_connection.hc_norm.weight",
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
        P + "self_attn.q_norm.weight",
        P + "self_attn.indexer.k_layernorm.weight",
        P + "ple.norm_key.weight",
        P + "ple.norm_query.weight",
        P + "ple.norm_conv.weight",
    ]
    out = dict(_run([(n, w) for n in names]))
    for n in names:
        torch.testing.assert_close(out[n], torch.full((8,), 0.25))


def test_linear_attention_norm_keeps_its_scale():
    w = torch.full((8,), 1.25)
    out = dict(_run([(P + "linear_attn.norm.weight", w)]))
    torch.testing.assert_close(out[P + "linear_attn.norm.weight"], w)


def test_indexer_halves_are_fused_q_first():
    q = torch.randn(6, 8).to(torch.bfloat16)
    k = torch.randn(2, 8).to(torch.bfloat16)
    out = dict(
        _run(
            [
                (P + "self_attn.indexer.index_k_proj.weight", k),
                (P + "self_attn.indexer.index_q_proj.weight", q),
            ]
        )
    )
    assert list(out) == [P + "self_attn.indexer.index_qk_proj.weight"]
    torch.testing.assert_close(
        out[P + "self_attn.indexer.index_qk_proj.weight"], torch.cat([q, k])
    )


def test_quantized_indexer_is_dequantized_before_fusing():
    dq, q = _q8(6, 64, seed=1)
    dk, k = _q8(2, 64, seed=2)
    qt = torch.tensor(GGMLQuantizationType.Q8_0)
    out = dict(
        _run(
            [
                (P + "self_attn.indexer.index_q_proj.weight_type", qt),
                (P + "self_attn.indexer.index_q_proj.weight", q),
                (P + "self_attn.indexer.index_k_proj.weight_type", qt),
                (P + "self_attn.indexer.index_k_proj.weight", k),
            ]
        )
    )
    fused = out.pop(P + "self_attn.indexer.index_qk_proj.weight")
    assert not out
    torch.testing.assert_close(
        fused.float(), torch.from_numpy(np.concatenate([dq, dk])), atol=0.05, rtol=0
    )


def test_ple_table_is_split_into_checkpoint_shards():
    rows, row_bytes = 10, 90
    table = torch.arange(rows * row_bytes, dtype=torch.int32).to(torch.uint8)
    table = table.reshape(rows, row_bytes)
    base = P.replace("layers.0.", "layers.1.") + "ple.ple_embedding."
    out = _run(
        [
            (
                base + "ngram_embedding.weight_type",
                torch.tensor(GGMLQuantizationType.IQ4_NL),
            ),
            (base + "ngram_embedding.weight", table),
        ]
    )
    shards = [(n, w) for n, w in out if "ngram_embedding" in n]
    # ceil(10 / 4) = 3 rows per shard, the last one short.
    assert [n for n, _ in shards] == [
        base + f"ngram_embedding.shard_{i}.weight" for i in range(4)
    ]
    assert [w.shape[0] for _, w in shards] == [3, 3, 3, 1]
    torch.testing.assert_close(torch.cat([w for _, w in shards]), table)


def test_ple_hash_constants_come_from_metadata():
    out = dict(_run([], constants=True))
    base = "model.language_model.layers.1.ple.ple_embedding."
    meta = _metadata()
    assert out[base + "layer_multipliers"].dtype == torch.int64
    assert out[base + "layer_multipliers"].tolist() == list(meta.ple_layer_multipliers)
    assert out[base + "ngram_heads_offsets"].tolist() == list(meta.ple_head_offsets)
    assert out[base + "ngram_heads_vocab_sizes"].tolist() == list(
        meta.ple_head_vocab_sizes
    )


def test_text_only_hash_constants_use_the_text_prefix():
    adapter = _adapter_with()
    model_config = _small_model_config()
    model_config.hf_config.vision_config = None
    out = dict(adapter.transform_weights(iter([]), model_config))
    assert "model.layers.1.ple.ple_embedding.layer_multipliers" in out


def test_gdn_and_moe_conventions_still_apply():
    a = -torch.exp(torch.arange(4, dtype=torch.float32))
    experts = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    gate = torch.ones(8)
    out = dict(
        _run(
            [
                (P + "linear_attn.A_log", a),
                (P + "mlp.experts.0.gate_proj.weight", experts),
                (P + "mlp.shared_expert_gate.weight", gate),
            ]
        )
    )
    # 2 key heads x ratio 2: ggml tiles value heads, HF groups them.
    torch.testing.assert_close(
        out[P + "linear_attn.A_log"], torch.tensor([0.0, 2.0, 1.0, 3.0])
    )
    assert out[P + "mlp.experts.2.gate_proj.weight"].shape == (2, 4)
    assert out[P + "mlp.shared_expert_gate.weight"].shape == (1, 8)


# -------------------------------------------------------- pure-GGUF sources


def test_pure_gguf_source_asks_for_the_hf_config(tmp_path):
    """transformers cannot build a qwen4exp config, and the file lacks fields
    the model needs; the parser must say where the config has to come from
    instead of surfacing transformers' 'architecture not supported'."""
    from gguf import GGUFWriter

    from vllm_gguf_plugin.config_parser import GGUFConfigParser

    path = tmp_path / "Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00004.gguf"
    writer = GGUFWriter(str(path), "qwen4exp")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    with pytest.raises(RuntimeError, match="--tokenizer Qwen/Qwen3.8-Flash-Next"):
        GGUFConfigParser().parse(str(path), trust_remote_code=False)
