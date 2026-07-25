# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf
import regex
import torch
from transformers import AutoModelForCausalLM
from vllm.logger import init_logger

from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..weight_utils import (
    get_gguf_extra_tensor_names,
    get_gguf_weight_type_map,
    gguf_quant_weights_iterator_multi,
)
from .base import BaseGGUFWeightsAdapter, GGUFLoadSpec

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)


# Qwen3.5 RMSNorms whose HF weights are zero-centered — the module applies
# (1 + w). ``linear_attn.norm`` is deliberately absent: the gated RMSNorm
# inside the delta-net stores a conventional scale and needs no adjustment.
_ZERO_CENTERED_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
)


def _is_zero_centered_norm(hf_name: str) -> bool:
    if hf_name.endswith(_ZERO_CENTERED_NORM_SUFFIXES):
        return True
    # Final trunk norm; must not catch ``...linear_attn.norm.weight``.
    return hf_name.endswith("model.norm.weight")


class GGUFWeightsAdapter(BaseGGUFWeightsAdapter):
    """Default adapter for GGUF models."""

    load_spec = None

    @classmethod
    def matches(cls, config) -> bool:
        del config
        return True

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        hf_config = maybe_patch_hf_config_from_gguf(model_path, hf_config)
        if hf_config.model_type in ("qwen3_5_text", "qwen3_5_moe_text"):
            self._patch_qwen35_layer_types_from_gguf(model_path, hf_config)
        return hf_config

    @staticmethod
    def _patch_qwen35_layer_types_from_gguf(
        model_path: str,
        hf_config: PretrainedConfig,
    ) -> None:
        layer_types = list(getattr(hf_config, "layer_types", []))
        if not layer_types:
            return
        tensor_names: set[str] = set()
        for gguf_file in GGUFWeightsAdapter._get_all_gguf_files(model_path):
            tensor_names.update(t.name for t in gguf.GGUFReader(gguf_file).tensors)
        for idx in range(min(len(layer_types), hf_config.num_hidden_layers)):
            if f"blk.{idx}.attn_qkv.weight" in tensor_names:
                layer_types[idx] = "linear_attention"
            elif f"blk.{idx}.attn_q.weight" in tensor_names:
                layer_types[idx] = "full_attention"
        hf_config.layer_types = layer_types

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        config = model_config.hf_config
        text_config = config.get_text_config()
        model_type = config.model_type
        is_multimodal = (
            hasattr(config, "vision_config") and config.vision_config is not None
        )

        gguf_to_hf_name_map: dict[str, str] = {}
        sideload_params: list[re.Pattern] = []

        if model_type == "cohere":
            model_type = "command-r"
        if model_type == "gemma3_text":
            model_type = "gemma3"
        if model_type == "gemma4_text":
            model_type = "gemma4"
        if model_type == "qwen3_5_text":
            model_type = "qwen35"
            for idx, layer_type in enumerate(config.layer_types):
                if layer_type == "linear_attention":
                    gguf_to_hf_name_map[f"blk.{idx}.ssm_dt.bias"] = (
                        f"model.layers.{idx}.linear_attn.dt_bias"
                    )
        if model_type == "qwen3_5_moe_text":
            model_type = "qwen35moe"
            for idx, layer_type in enumerate(config.layer_types):
                if layer_type == "linear_attention":
                    gguf_to_hf_name_map[f"blk.{idx}.ssm_dt.bias"] = (
                        f"model.layers.{idx}.linear_attn.dt_bias"
                    )
        if model_type in ("deepseek_v3", "deepseek_v2"):
            model_type = "deepseek2"
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.mlp.gate.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.append(
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )
        if model_type in ("qwen2_moe", "qwen3_moe"):
            model_type = model_type.replace("_", "")
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.append(
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )
        if model_type == "olmoe":
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.extend(
                    [
                        regex.compile(
                            f"model\\.layers\\.{idx}"
                            r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                        ),
                        regex.compile(
                            f"model\\.layers\\.{idx}"
                            r"\.mlp\.experts\.(gate_up_proj|down_proj)"
                        ),
                    ]
                )
        if model_type == "minimax_m2":
            model_type = "minimax-m2"
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.block_sparse_moe.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w2.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w1.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w3.weight"
                )
                sideload_params.append(
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.block_sparse_moe\.experts\.(gate_up_proj|down_proj)"
                    )
                )

        arch = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == model_type:
                arch = key
                break
        if arch is None:
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")

        text_name_map = gguf.get_tensor_name_map(arch, text_config.num_hidden_layers)

        if is_multimodal:
            mm_proj_arch = gguf.MODEL_ARCH.MMPROJ
            vision_name_map = gguf.get_tensor_name_map(
                mm_proj_arch, config.vision_config.num_hidden_layers
            )
        else:
            vision_name_map = None

        with torch.device("meta"):
            dummy_model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=model_config.trust_remote_code
            )

        state_dict = dummy_model.state_dict()
        if hf_checkpoint_map := getattr(
            dummy_model, "_checkpoint_conversion_mapping", None
        ):

            def revert_hf_rename(name: str) -> str:
                for original_name, hf_name in hf_checkpoint_map.items():
                    if hf_name in name:
                        name = name.replace(hf_name, original_name).lstrip("^")
                return name

            state_dict = {
                revert_hf_rename(name): tensor for name, tensor in state_dict.items()
            }

        if model_type == "minimax-m2" and not hf_checkpoint_map:
            state_dict = {
                name.replace(".mlp.", ".block_sparse_moe."): tensor
                for name, tensor in state_dict.items()
            }

        def find_hf_name_in_tensor_map(hf_name: str) -> str | None:
            if is_multimodal and hf_name.startswith("model."):
                hf_name = hf_name[6:]
            if hf_name.startswith("language_model."):
                hf_name = hf_name[15:]
                if is_multimodal:
                    hf_name = "model." + hf_name
            if hf_name.endswith((".weight", ".bias")):
                base_name, suffix = hf_name.rsplit(".", 1)
            else:
                # HF tensors whose name doesn't end in ".weight"/".bias"
                # (e.g. Gemma4's registered ``layer_scalar`` buffer) are still
                # stored in GGUF with a ".weight" suffix appended. For gemma4
                # default to "weight"; elsewhere the GGUF tensor is stored
                # bare (e.g. qwen35's ``blk.N.ssm_a`` backing ``A_log``) and
                # gets no suffix at all.
                base_name = hf_name
                suffix = "weight" if model_type == "gemma4" else ""
                if base_name.endswith("_weight"):
                    base_name = base_name[:-7]
                    suffix = "weight"
            gguf_name = None
            if vision_name_map is not None:
                gguf_name = vision_name_map.get_name(base_name)
            if gguf_name is None:
                gguf_name = text_name_map.get_name(base_name)
            if gguf_name is None:
                return None
            # An empty suffix means the GGUF tensor is stored bare. Joining
            # unconditionally would yield a trailing-dot key that matches no
            # tensor, and an unmatched key is silent: the parameter keeps its
            # initialisation and the model loads without a single warning.
            return f"{gguf_name}.{suffix}" if suffix else gguf_name

        unmapped_params = []
        for hf_name in state_dict:
            gguf_name_with_suffix = find_hf_name_in_tensor_map(hf_name)
            if gguf_name_with_suffix is not None:
                gguf_to_hf_name_map[gguf_name_with_suffix] = hf_name
                logger.debug("Mapped GGUF %s → HF %s", gguf_name_with_suffix, hf_name)
            elif hf_name not in gguf_to_hf_name_map.values():
                unmapped_params.append(hf_name)

        if unmapped_params:
            unmapped_params = [
                x
                for x in unmapped_params
                if not any(regex.fullmatch(p, x) for p in sideload_params)
            ]
        if unmapped_params:
            # Multimodal checkpoints (e.g. gemma-4) carry vision-tower / vision
            # tensors that the text-only vLLM model never loads; skip them
            # instead of failing the whole load (restores the pre-plugin
            # lenient behavior).
            _mm = [
                x
                for x in unmapped_params
                if any(
                    s in x
                    for s in (
                        "vision_tower",
                        "embed_vision",
                        "vision_model",
                        "multi_modal_projector",
                    )
                )
            ]
            if _mm:
                logger.warning(
                    "Skipping %d unmapped multimodal GGUF params for text-only "
                    "load (e.g. %s)",
                    len(_mm),
                    _mm[:3],
                )
                unmapped_params = [x for x in unmapped_params if x not in _mm]
        if unmapped_params:
            raise RuntimeError(
                f"Failed to map GGUF parameters "
                f"({len(unmapped_params)}): {unmapped_params}"
            )
        return gguf_to_hf_name_map

    def map_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for hf_name, weight in weights:
            weight = self.transform_weight(hf_name, weight)
            if weight.ndim == 3 and ".experts.0." in hf_name:
                for expert_id, expert_weight in enumerate(weight.unbind()):
                    expert_name = hf_name.replace(
                        ".experts.0.", f".experts.{expert_id}."
                    )
                    yield expert_name, expert_weight
            else:
                yield hf_name, weight

    def transform_weight(
        self,
        hf_name: str,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        is_qwen35 = self.config.model_type in ("qwen3_5_text", "qwen3_5_moe_text")
        if ".linear_attn.conv1d.weight" in hf_name and weight.ndim == 2 and is_qwen35:
            weight = weight.unsqueeze(1)
        if is_qwen35 and _is_zero_centered_norm(hf_name):
            # Qwen3.5's RMSNorms are zero-centered: the checkpoint stores w and
            # the kernel applies (1 + w). ggml has no such convention, so the
            # conversion folds the +1 into the stored value. Loading that
            # directly makes vLLM add 1 a second time, roughly doubling every
            # norm scale. Verified against the HF checkpoint: ggml's value is
            # exactly hf + 1.
            weight = weight - 1.0
        if hf_name.endswith(".linear_attn.A_log") and is_qwen35:
            # ggml stores the gated-delta-net decay already exponentiated, as
            # A = -exp(A_log); vLLM's kernels take A_log and exponentiate it
            # themselves. Verified against the HF checkpoint of the same
            # model: -exp(A_log) reproduces ggml's ssm_a exactly. Passing the
            # stored value through unchanged applies exp twice.
            if bool((weight < 0).all()):
                weight = weight.neg().log()
        if is_qwen35:
            weight = self._reorder_qwen35_value_heads(hf_name, weight)
        return weight

    def _qwen35_gdn_dims(self) -> tuple[int, int, int, int] | None:
        """(num_k_heads, num_v_heads, value_head_dim, key_dim) or None."""
        cfg = self.config.get_text_config()
        try:
            nk = int(cfg.linear_num_key_heads)
            nv = int(cfg.linear_num_value_heads)
            hv = int(cfg.linear_value_head_dim)
            hk = int(cfg.linear_key_head_dim)
        except (AttributeError, TypeError, ValueError):
            return None
        if nk <= 0 or nv <= 0 or nv % nk:
            return None
        return nk, nv, hv, nk * hk

    def _assert_head_block_aligned(self, hf_name: str, head_dim: int) -> None:
        """Fail loudly if a value head does not cover whole quant blocks.

        Only relevant when re-indexing along a packed row. Silently permuting
        a half-block would corrupt the scales and produce plausible-looking
        garbage, which is the failure mode this whole fix exists to remove.
        """
        wtype = (getattr(self, "_weight_type_map", None) or {}).get(hf_name)
        if wtype is None:
            return
        try:
            qtype = gguf.GGMLQuantizationType[wtype]
            block_size, _ = gguf.GGML_QUANT_SIZES[qtype]
        except (KeyError, AttributeError):
            return
        if block_size > 1 and head_dim % block_size:
            raise RuntimeError(
                f"Cannot re-index Qwen3.5 value heads for {hf_name}: a "
                f"{head_dim}-element head does not cover whole {wtype} blocks "
                f"of {block_size} elements, so permuting the packed row would "
                f"split a quantisation block. This model needs out_proj "
                f"dequantised at load time before re-indexing."
            )

    def _reorder_qwen35_value_heads(
        self,
        hf_name: str,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Re-index gated-delta-net value heads from ggml order to HF order.

        ggml lays the value heads out strided as ``r * num_k_heads + g``; HF
        groups them under their key head as ``g * ratio + r``. The two agree
        only when num_v_heads == num_k_heads, so this is invisible on models
        with a 1:1 ratio and scrambles which value head is paired with which
        key head on every other model.

        Rows are permuted whole, which is safe on packed quantised data. The
        out_proj case permutes along the *input* axis and so moves bytes
        inside each row; that is only valid when a head spans a whole number
        of quantisation blocks, which is checked before use.
        """
        dims = self._qwen35_gdn_dims()
        if dims is None:
            return weight
        nk, nv, hv, key_dim = dims
        ratio = nv // nk
        if ratio == 1:
            return weight

        # Quantised tensors reach us renamed: the weight iterator rewrites
        # ``<x>.weight`` to ``<x>.qweight`` (and emits a separate
        # ``<x>.qweight_type`` scalar). Re-indexing is a pure layout change,
        # valid on packed bytes as long as whole rows move, so match both
        # spellings. The scale/offset conversions above deliberately do *not*
        # do this: applying arithmetic to packed bytes would corrupt them,
        # and those tensors are only ever stored unquantised.
        if hf_name.endswith(".qweight_type"):
            return weight
        if hf_name.endswith(".qweight"):
            hf_name = hf_name[: -len(".qweight")] + ".weight"

        def rows(t: torch.Tensor, per_head: int) -> torch.Tensor:
            head_rows = per_head
            tail = t.shape[1:]
            return (
                t.reshape(ratio, nk, head_rows, *tail)
                .transpose(0, 1)
                .reshape(nv * head_rows, *tail)
            )

        if hf_name.endswith((".linear_attn.A_log", ".linear_attn.dt_bias")):
            return rows(weight, 1)
        if hf_name.endswith((".linear_attn.in_proj_b.weight",
                             ".linear_attn.in_proj_a.weight")):
            return rows(weight, 1)
        if hf_name.endswith(".linear_attn.in_proj_z.weight"):
            return rows(weight, hv)
        if hf_name.endswith(".linear_attn.in_proj_qkv.weight"):
            # Only the trailing value block is value-head indexed.
            qk, v = weight[:2 * key_dim], weight[2 * key_dim:]
            return torch.cat([qk, rows(v, hv)], dim=0)
        if hf_name.endswith(".linear_attn.conv1d.weight"):
            qk, v = weight[:2 * key_dim], weight[2 * key_dim:]
            return torch.cat([qk, rows(v, hv)], dim=0)
        if hf_name.endswith(".linear_attn.out_proj.weight"):
            # Value heads index the input axis, i.e. within each packed row,
            # so a head must cover a whole number of quantisation blocks.
            # Byte divisibility alone is not enough: Q5_K packs 256 elements
            # per block, so a 128-element head is half a block and reordering
            # would split it.
            self._assert_head_block_aligned(hf_name, hv)
            width = weight.shape[1]
            if width % nv:
                raise RuntimeError(
                    f"Cannot re-index Qwen3.5 value heads for {hf_name}: row "
                    f"width {width} is not divisible by {nv} value heads."
                )
            per_head = width // nv
            return (
                weight.reshape(-1, ratio, nk, per_head)
                .transpose(1, 2)
                .reshape(-1, width)
            )
        return weight

    @staticmethod
    def _get_all_gguf_files(model_path: str) -> list[str]:
        match = re.search(r"-(\d+)-of-(\d+)\.gguf$", model_path)
        if not match:
            return [model_path]
        total = int(match.group(2))
        num_digits = len(match.group(1))
        prefix = model_path[: match.start(1)]
        suffix = model_path[match.end(2) :]
        files = []
        for i in range(1, total + 1):
            shard_path = f"{prefix}{i:0{num_digits}d}-of-{total:0{num_digits}d}{suffix}"
            if os.path.isfile(shard_path):
                files.append(shard_path)
        if files:
            logger.info("Discovered %d GGUF shard files", len(files))
        return files if files else [model_path]

    def update_tie_word_embeddings(
        self,
        model_path: str,
        hf_config: PretrainedConfig,
        gguf_to_hf_name_map: dict[str, str],
    ) -> None:
        if "lm_head.weight" not in gguf_to_hf_name_map.values():
            return

        all_extra_names = []
        for gguf_file in self._get_all_gguf_files(model_path):
            all_extra_names.extend(
                get_gguf_extra_tensor_names(gguf_file, gguf_to_hf_name_map)
            )
        hf_config.update({"tie_word_embeddings": "lm_head.weight" in all_extra_names})

    def get_weight_type_map(
        self,
        model_path: str,
        gguf_to_hf_name_map: dict[str, str],
    ) -> dict[str, str]:
        weight_type_map = {}
        for gguf_file in self._get_all_gguf_files(model_path):
            weight_type_map.update(
                get_gguf_weight_type_map(gguf_file, gguf_to_hf_name_map)
            )
        return weight_type_map

    @staticmethod
    def get_unquantized_modules(weight_type_map: dict[str, str]) -> list[str]:
        modules = [
            name.removesuffix(".weight")
            for name, weight_type in weight_type_map.items()
            if weight_type in ("F32", "F16", "BF16") and name.endswith(".weight")
        ]
        module_set = set(modules)
        for module in tuple(module_set):
            if module.endswith(".linear_attn.in_proj_qkv"):
                sibling = module.removesuffix("in_proj_qkv")
                if f"{sibling}in_proj_z" in module_set:
                    modules.append(f"{sibling}in_proj_qkvz")
            elif module.endswith(".linear_attn.in_proj_b"):
                sibling = module.removesuffix("in_proj_b")
                if f"{sibling}in_proj_a" in module_set:
                    modules.append(f"{sibling}in_proj_ba")
        return modules

    def prepare_loading(
        self,
        model_path: str,
        model_config: ModelConfig,
    ) -> GGUFLoadSpec:
        model_config.hf_config = self.patch_hf_config(
            model_path, model_config.hf_config
        )
        gguf_to_hf_name_map = self.build_name_map(model_config)
        self.update_tie_word_embeddings(
            model_path, model_config.hf_config, gguf_to_hf_name_map
        )
        weight_type_map = self.get_weight_type_map(model_path, gguf_to_hf_name_map)
        # Kept for transform-time block-alignment checks (see
        # _reorder_qwen35_value_heads); keyed by HF name with a ".weight"
        # suffix, before the iterator rewrites it to ".qweight".
        self._weight_type_map = weight_type_map
        self.load_spec = GGUFLoadSpec(
            weights_source=self._get_all_gguf_files(model_path),
            gguf_to_hf_name_map=gguf_to_hf_name_map,
            unquantized_modules=self.get_unquantized_modules(weight_type_map),
        )
        return self.load_spec

    def prepare_weights(
        self,
        model_config: ModelConfig,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        del model_config
        weights = gguf_quant_weights_iterator_multi(
            self.load_spec.weights_source,
            self.load_spec.gguf_to_hf_name_map,
        )
        yield from self.map_weights(weights)
