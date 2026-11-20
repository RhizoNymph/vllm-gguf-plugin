# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next (ggml ``qwen4exp``) GGUF -> vLLM ``Qwen4Exp``.

Qwen4Exp keeps Qwen3.5's gated delta net and MoE, and adds hyper-connections
in place of every layer norm, QSA sparse attention with an indexer on the
full-attention layers, and a PLE n-gram hash table on one layer. The GGUF
names, the conversions llama.cpp applies, and the HF names they map back to
are documented in ``docs/features/qwen4exp.md``.

The config always comes from the HF repo (``--tokenizer`` or
``--hf-config-path``): transformers cannot build one from ``qwen4exp``
metadata, and several fields the model needs (n-gram vocabulary base, MTP
layout, output gate) are not in the file. The metadata is used to validate
that repo config against the checkpoint instead.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gguf
import numpy as np
import torch
from gguf import GGMLQuantizationType
from vllm.logger import init_logger

from ..gguf_files import GGUFModelFiles
from ..weight_utils import get_gguf_tensor_names
from .base import GGUFWeight
from .qwen3_5 import QWEN35_MOE_SUBSTR, Qwen35GGUFAdapter, build_qwen35_vision_mapper

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)

QWEN4EXP_ARCHITECTURES = {
    "qwen4_exp": "Qwen4ExpForConditionalGeneration",
    "qwen4_exp_text": "Qwen4ExpForCausalLM",
}

# Per-block GGUF tensor -> HF name below ``layers.N.``. Exact names, not
# substrings: several GGUF names are substrings of others (``attn_norm`` of
# ``hc_attn_norm``, ``ssm_a`` of ``ssm_alpha``).
_LAYER_TENSORS: dict[str, str] = {
    # hyper-connections (one per attention and MLP branch)
    **{
        f"hc_{branch}_{part}.weight": f"{hf_branch}_hyper_connection.{hf_part}.weight"
        for branch, hf_branch in (("attn", "attn"), ("ffn", "mlp"))
        for part, hf_part in (
            ("norm", "hc_norm"),
            ("down", "input_mix_weight_down"),
            ("up", "input_mix_weight_up"),
            ("inject", "block_inject_weight"),
        )
    },
    # QSA full attention; attn_q carries the sigmoid output gate beside Q.
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    # The QSA indexer's one HF projection is stored as two; the halves get
    # placeholder names and are fused again on load.
    "indexer.q_proj.weight": "self_attn.indexer.index_q_proj.weight",
    "indexer.k_proj.weight": "self_attn.indexer.index_k_proj.weight",
    "indexer.q_norm.weight": "self_attn.indexer.q_layernorm.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_layernorm.weight",
    # gated delta net, as in Qwen3.5
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    # PLE projections (the table itself is a top-level tensor)
    "ple_conv1d.weight": "ple.conv1d.weight",
    "ple_key.weight": "ple.key_proj.weight",
    "ple_value.weight": "ple.value_proj.weight",
    "ple_norm_conv.weight": "ple.norm_conv.weight",
    "ple_norm_key.weight": "ple.norm_key.weight",
    "ple_norm_query.weight": "ple.norm_query.weight",
    # MoE: routed experts arrive stacked and are split per expert later.
    **{
        f"{gguf_name}weight": f"{hf_name}weight"
        for gguf_name, hf_name in QWEN35_MOE_SUBSTR.items()
    },
}

_GLOBAL_TENSORS: dict[str, str] = {
    "token_embd.weight": "{backbone}embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_hc_norm.weight": "{backbone}hyper_connection_mixer.hc_norm.weight",
    "output_hc_down.weight": (
        "{backbone}hyper_connection_mixer.input_mix_weight_down.weight"
    ),
    "output_hc_up.weight": (
        "{backbone}hyper_connection_mixer.input_mix_weight_up.weight"
    ),
    "per_layer_token_embd.weight": (
        "{backbone}layers.{ple_layer}.ple.ple_embedding.ngram_embedding.weight"
    ),
}

# Modules built with ``quant_config=None`` or fused from two GGUF tensors:
# their GGUF payload is dequantized on load instead of served packed.
_DEQUANTIZE_ON_LOAD = (
    "_hyper_connection.",
    "hyper_connection_mixer.",
    "indexer.index_q_proj.",
    "indexer.index_k_proj.",
)

# Gemma-style norms the Qwen3.5 rule (``*norm.weight``) does not catch; ggml
# stores them with the ``+1`` folded in.
_EXTRA_ZERO_CENTRED_NORMS = (
    "ple.norm_conv.weight",
    "ple.norm_key.weight",
    "ple.norm_query.weight",
)

_PLE_TABLE_SUFFIX = "ple.ple_embedding.ngram_embedding.weight"


def _ints(value: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in value)


@dataclass(frozen=True)
class Qwen4ExpGGUFMetadata:
    """The ``qwen4exp.*`` metadata the adapter relies on."""

    kv: Mapping[str, Any]

    @classmethod
    def from_kv(cls, kv: Mapping[str, Any]) -> Qwen4ExpGGUFMetadata:
        if kv.get("general.architecture") != "qwen4exp":
            raise ValueError(
                "Not a qwen4exp GGUF: general.architecture="
                f"{kv.get('general.architecture')!r}"
            )
        return cls(dict(kv))

    @classmethod
    def from_file(cls, path: str) -> Qwen4ExpGGUFMetadata:
        reader = gguf.GGUFReader(path)
        kv: dict[str, Any] = {}
        for key, field in reader.fields.items():
            if key.startswith(("GGUF.", "tokenizer.ggml.")):
                continue
            kv[key] = field.contents()
        return cls.from_kv(kv)

    def __getitem__(self, key: str) -> Any:
        return self.kv[f"qwen4exp.{key}"]

    def get(self, key: str, default: Any = None) -> Any:
        return self.kv.get(f"qwen4exp.{key}", default)

    @property
    def ple_layers(self) -> tuple[int, ...]:
        return _ints(self.get("ple.layers", ()))

    @property
    def ple_layer_multipliers(self) -> tuple[int, ...]:
        return _ints(self["ple.layer_multipliers"])

    @property
    def ple_head_offsets(self) -> tuple[int, ...]:
        return _ints(self["ple.head_offsets"])

    @property
    def ple_head_vocab_sizes(self) -> tuple[int, ...]:
        return _ints(self["ple.head_vocab_sizes"])

    @property
    def ple_layer(self) -> int:
        """The single block carrying the PLE table."""
        layers = self.ple_layers
        if len(layers) != 1:
            raise NotImplementedError(
                f"qwen4exp GGUF with PLE layers {list(layers)}: only one "
                "per-layer table is supported"
            )
        return layers[0]


def validate_qwen4exp_config(
    meta: Qwen4ExpGGUFMetadata, text_config: PretrainedConfig
) -> None:
    """Raise if the HF config describes a different model than the GGUF."""
    c = text_config
    num_layers = c.num_hidden_layers
    attention_layers = [t != "linear_attention" for t in c.layer_types]
    rope = c.rope_parameters or {}
    ngram_heads = (c.ngram_size - 1) * c.heads_per_ngram
    expected: dict[str, Any] = {
        "block_count": num_layers + int(meta.get("nextn_predict_layers", 0)),
        "embedding_length": c.hidden_size,
        "attention.head_count": c.num_attention_heads,
        "attention.head_count_kv": c.num_key_value_heads,
        "attention.key_length": c.head_dim,
        "attention.value_length": c.head_dim,
        "expert_count": c.num_experts,
        "expert_used_count": c.num_experts_per_tok,
        "expert_feed_forward_length": c.moe_intermediate_size,
        "expert_shared_feed_forward_length": c.shared_expert_intermediate_size,
        "ssm.conv_kernel": c.linear_conv_kernel_dim,
        "ssm.state_size": c.linear_key_head_dim,
        "ssm.group_count": c.linear_num_key_heads,
        "ssm.time_step_rank": c.linear_num_value_heads,
        "ssm.inner_size": c.linear_num_value_heads * c.linear_value_head_dim,
        "hyper_connection.count": c.hc_count,
        "hyper_connection.low_rank": c.hc_lowrank,
        "attention.indexer.head_count": c.indexer_n_heads,
        "attention.indexer.key_length": c.indexer_head_dim,
        "attention.indexer.top_k": c.indexer_budget,
        "rope.dimension_count": int(
            c.head_dim * rope.get("partial_rotary_factor", 1.0)
        ),
        "rope.freq_base": float(rope.get("rope_theta", 0.0)),
    }
    if c.ple_layer_ids:
        expected |= {
            "ple.layers": [i - 1 for i in c.ple_layer_ids],
            "ple.ngram_size": c.ngram_size,
            "ple.heads_per_ngram": c.heads_per_ngram,
            "ple.conv_kernel": c.ple_conv_kernel_size,
            "embedding_length_per_layer_input": c.ple_embed_dim // ngram_heads,
        }
    mismatches: list[str] = []
    for key, want in expected.items():
        got = meta.get(key)
        if isinstance(want, float) and got is not None:
            ok = math.isclose(float(got), want, rel_tol=1e-6)
        elif isinstance(want, list):
            ok = got is not None and _ints(got) == tuple(want)
        else:
            ok = got == want
        if not ok:
            mismatches.append(f"qwen4exp.{key}: gguf={got!r} config={want!r}")
    ratios = meta.get("attention.compress_ratios")
    if ratios is not None:
        want_ratios = [
            c.indexer_compress_ratio if attn else 0 for attn in attention_layers
        ]
        if list(_ints(ratios))[:num_layers] != want_ratios:
            mismatches.append(
                "qwen4exp.attention.compress_ratios: full-attention layers "
                "disagree with config.layer_types"
            )
    sections = meta.get("rope.dimension_sections")
    if (
        sections is not None
        and "mrope_section" in rope
        and [s for s in _ints(sections) if s] != list(rope["mrope_section"])
    ):
        mismatches.append(
            f"qwen4exp.rope.dimension_sections: gguf={list(_ints(sections))} "
            f"config={rope['mrope_section']}"
        )
    if mismatches:
        raise ValueError(
            "The HF config does not describe this qwen4exp GGUF:\n  "
            + "\n  ".join(mismatches)
        )


def qwen4exp_name_map(
    tensor_names: Iterable[str],
    meta: Qwen4ExpGGUFMetadata,
    is_multimodal: bool,
) -> dict[str, str]:
    """Map every GGUF tensor name to its HF checkpoint name.

    Raises on any tensor without a mapping: an unmapped tensor is never
    loaded, and a parameter left at its initialisation corrupts the model
    without an error.
    """
    backbone = "model.language_model." if is_multimodal else "model."
    vision_mapper = build_qwen35_vision_mapper()
    name_map: dict[str, str] = {}
    unmapped: list[str] = []
    for name in tensor_names:
        hf_name: str | None = None
        if name.startswith(("v.", "mm.")):
            mapped = vision_mapper.apply_list([name])[0]
            hf_name = mapped if mapped != name else None
        elif name.startswith("blk."):
            _, index, rest = name.split(".", 2)
            suffix = _LAYER_TENSORS.get(rest)
            if suffix is not None:
                hf_name = f"{backbone}layers.{index}.{suffix}"
        elif name in _GLOBAL_TENSORS:
            template = _GLOBAL_TENSORS[name]
            ple_layer = meta.ple_layer if "{ple_layer}" in template else None
            hf_name = template.format(backbone=backbone, ple_layer=ple_layer)
        if hf_name is None:
            unmapped.append(name)
        else:
            name_map[name] = hf_name
    if unmapped:
        raise ValueError(
            f"No HF name for {len(unmapped)} qwen4exp GGUF tensor(s): {unmapped}"
        )
    return name_map


def _dequantize(weight: torch.Tensor, weight_type: int) -> torch.Tensor:
    """Dequantize packed GGUF rows on the CPU to bfloat16."""
    qtype = GGMLQuantizationType(weight_type)
    values = gguf.quants.dequantize(weight.cpu().numpy(), qtype)
    return torch.from_numpy(np.ascontiguousarray(values)).to(torch.bfloat16)


def _split_rows(name: str, table: torch.Tensor, parts: int) -> Iterable[GGUFWeight]:
    """Re-shard the PLE table the way the HF checkpoint stores it."""
    shard_rows = -(-table.shape[0] // parts)
    base = name.removesuffix("weight")
    for index in range(parts):
        rows = table[index * shard_rows : (index + 1) * shard_rows]
        if rows.shape[0]:
            yield f"{base}shard_{index}.weight", rows


class Qwen4ExpGGUFAdapter(Qwen35GGUFAdapter):
    """Qwen3.8-Flash-Next text, MoE and multimodal GGUF adapter."""

    # The QSA indexer's fused projection is assembled from two GGUF tensors
    # and always served as bfloat16.
    extra_unquantized_modules = ("indexer.index_qk_proj",)

    def __init__(self) -> None:
        self.metadata: Qwen4ExpGGUFMetadata | None = None

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in QWEN4EXP_ARCHITECTURES

    @classmethod
    def architecture(cls, config) -> str | None:
        return QWEN4EXP_ARCHITECTURES.get(config.model_type)

    def _load_metadata(self, files: GGUFModelFiles) -> Qwen4ExpGGUFMetadata:
        if self.metadata is None:
            self.metadata = Qwen4ExpGGUFMetadata.from_file(files.primary_backbone)
        return self.metadata

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        validate_qwen4exp_config(
            self._load_metadata(files), hf_config.get_text_config()
        )
        has_vision = getattr(hf_config, "vision_config", None) is not None
        if (
            has_vision
            and files.mm_proj is None
            and not getattr(hf_config, "language_model_only", False)
        ):
            raise RuntimeError(
                "Could not find mm_proj for multimodal Qwen3.8-Flash-Next GGUF. "
                "Place *mmproj*.gguf beside the backbone, pass "
                "model_loader_extra_config={'mm_proj': ...}, or serve "
                "language-only with --hf-overrides "
                "'{\"language_model_only\": true}'."
            )
        hf_config.architectures = [QWEN4EXP_ARCHITECTURES[hf_config.model_type]]
        return hf_config

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        is_multimodal = (
            getattr(model_config.hf_config, "vision_config", None) is not None
        )
        tensor_names = sorted(get_gguf_tensor_names(files.all_files))
        return qwen4exp_name_map(
            tensor_names, self._load_metadata(files), is_multimodal
        )

    def _ple_constants(self, backbone: str) -> Iterable[GGUFWeight]:
        meta = self.metadata
        if meta is None or not meta.ple_layers:
            return
        base = f"{backbone}layers.{meta.ple_layer}.ple.ple_embedding."
        for name, values in (
            ("layer_multipliers", meta.ple_layer_multipliers),
            ("ngram_heads_offsets", meta.ple_head_offsets),
            ("ngram_heads_vocab_sizes", meta.ple_head_vocab_sizes),
        ):
            yield base + name, torch.tensor(values, dtype=torch.int64)

    def _qwen4exp_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        """Conversions Qwen3.5's adapter does not know about."""
        text_config = model_config.hf_config.get_text_config()
        is_multimodal = (
            getattr(model_config.hf_config, "vision_config", None) is not None
        )
        backbone = "model.language_model." if is_multimodal else "model."
        pending_types: dict[str, int] = {}
        indexer_halves: dict[str, dict[str, torch.Tensor]] = {}

        for name, weight in weights:
            if name.endswith(".weight_type"):
                base = name.removesuffix("_type")
                if any(m in base for m in _DEQUANTIZE_ON_LOAD) or base.endswith(
                    _PLE_TABLE_SUFFIX
                ):
                    # Served unpacked (or by the PLE method): no weight_type.
                    pending_types[base] = int(weight)
                    continue
                yield name, weight
                continue

            weight_type = pending_types.pop(name, None)
            if name.endswith(_PLE_TABLE_SUFFIX):
                yield from _split_rows(name, weight, text_config.split_ngram_parts)
                continue
            if weight_type is not None and any(m in name for m in _DEQUANTIZE_ON_LOAD):
                weight = _dequantize(weight, weight_type)
            if name.endswith(_EXTRA_ZERO_CENTRED_NORMS):
                weight = weight - 1
            for half in ("q", "k"):
                marker = f"indexer.index_{half}_proj.weight"
                if name.endswith(marker):
                    fused = name.removesuffix(marker) + "indexer.index_qk_proj.weight"
                    halves = indexer_halves.setdefault(fused, {})
                    halves[half] = weight
                    break
            else:
                yield name, weight
                continue
            if len(halves) == 2:
                del indexer_halves[fused]
                yield fused, torch.cat([halves["q"], halves["k"]], dim=0)
        if indexer_halves:
            raise RuntimeError(
                "qwen4exp GGUF is missing an indexer projection half for "
                f"{sorted(indexer_halves)}"
            )
        yield from self._ple_constants(backbone)

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        yield from super().transform_weights(
            self._qwen4exp_weights(weights, model_config), model_config
        )

    def get_ple_weight_type(self, files: GGUFModelFiles) -> int | None:
        """ggml type of the PLE table, or None when the model has none."""
        for gguf_file in files.backbone:
            for tensor in gguf.GGUFReader(gguf_file).tensors:
                if tensor.name == "per_layer_token_embd.weight":
                    return int(tensor.tensor_type)
        return None
