# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thinking Machines' Inkling 2 Mini GGUF (ggml ``inkling2``) -> vLLM
``Inkling2LoopForConditionalGeneration``.

The config always comes from the HF repo (``--tokenizer`` or
``--hf-config-path thinkingmachines/Inkling-2-Mini``): the GGUF omits the
vision tower's and the DSpark draft's configs, the loop's ``is_varloop``
and the MTP feature layers. The backbone, mmproj and DSpark metadata are
checked against that config instead. See ``docs/features/inkling2.md``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf
from vllm.logger import init_logger

from ...gguf_files import GGUFModelFiles
from ..base import BaseGGUFWeightsAdapter, GGUFWeight
from .metadata import (
    Inkling2GGUFMetadata,
    read_gguf_kv,
    validate_inkling2_config,
    validate_inkling2_vision_config,
)
from .names import name_map
from .weights import Inkling2LoadPlan, build_load_plan, inkling2_weights

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

    from ...quantization import GGUFConfig

logger = init_logger(__name__)

INKLING2_ARCHITECTURES = {
    "inkling2_loop": "Inkling2LoopForConditionalGeneration",
    "inkling2": "Inkling2ForConditionalGeneration",
}


def _tensor_types(paths: Iterable[str]) -> dict[str, gguf.GGMLQuantizationType]:
    return {
        tensor.name: tensor.tensor_type
        for path in paths
        for tensor in gguf.GGUFReader(path).tensors
    }


def _builds_vision_tower(model_config: ModelConfig) -> bool:
    """Whether vLLM's Inkling2 model will build (and load) its vision tower."""
    if getattr(model_config.hf_config, "vision_config", None) is None:
        return False
    mm = getattr(model_config, "multimodal_config", None)
    return mm is not None and bool(mm.get_limit_per_prompt("image"))


class Inkling2GGUFAdapter(BaseGGUFWeightsAdapter):
    """Inkling 2 Mini backbone and vision GGUF adapter."""

    def __init__(self) -> None:
        self.metadata: Inkling2GGUFMetadata | None = None
        self.plan: Inkling2LoadPlan | None = None
        self.dense_lm_head = False

    @staticmethod
    def wants_dense_lm_head(hf_config: PretrainedConfig) -> bool:
        """Whether the config asks for an fp32 LM head.

        Only an explicit ``--hf-overrides '{"head_dtype": "float32"}'`` does
        (see :meth:`config_defaults`). vLLM's LogitsProcessor runs an fp32
        head only over an unquantized lm_head, so the packed GGUF head is
        then dequantized to BF16 (~0.5 GB more, ~5% slower decode).
        """
        return str(getattr(hf_config, "head_dtype", None)).lower() == "float32"

    def unquantized_modules_for(self, hf_config: PretrainedConfig) -> tuple[str, ...]:
        return ("lm_head",) if self.wants_dense_lm_head(hf_config) else ()

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in INKLING2_ARCHITECTURES

    @classmethod
    def architecture(cls, config) -> str | None:
        return INKLING2_ARCHITECTURES.get(config.model_type)

    @classmethod
    def config_defaults(cls, config) -> dict[str, str]:
        """Serve the GGUF's packed LM head, as llama.cpp does.

        Inkling2's config defaults to ``head_dtype: float32`` (BF16
        unembedding, fp32 logits); a GGUF computes the logits from its
        packed ``output.weight`` in the model dtype instead.
        """
        del config
        return {"head_dtype": "model"}

    def _load_metadata(self, files: GGUFModelFiles) -> Inkling2GGUFMetadata:
        if self.metadata is None:
            self.metadata = Inkling2GGUFMetadata.from_file(files.primary_backbone)
        return self.metadata

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        validate_inkling2_config(
            self._load_metadata(files), hf_config.get_text_config()
        )
        vision = getattr(hf_config, "vision_config", None)
        if vision is not None and files.mm_proj is not None:
            validate_inkling2_vision_config(read_gguf_kv(files.mm_proj), vision)
        hf_config.architectures = [INKLING2_ARCHITECTURES[hf_config.model_type]]
        self.dense_lm_head = self.wants_dense_lm_head(hf_config)
        # Read by the loader after this call.
        self.extra_unquantized_modules = self.unquantized_modules_for(hf_config)
        if self.dense_lm_head:
            logger.info(
                "Inkling2 GGUF: dequantizing the LM head to BF16 for the "
                "fp32 head override (head_dtype=float32)"
            )
        return hf_config

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        if _builds_vision_tower(model_config) and files.mm_proj is None:
            raise RuntimeError(
                "Inkling 2 Mini GGUF without its vision tower: place "
                "mmproj-BF16.gguf beside the backbone, pass "
                "model_loader_extra_config={'mm_proj': ...}, or serve text "
                "only with --limit-mm-per-prompt '{\"image\": 0}'."
            )
        text_config = model_config.hf_config.get_text_config()
        self.plan = build_load_plan(
            _tensor_types(files.all_files),
            self._load_metadata(files),
            num_experts=text_config.n_routed_experts,
            dense_lm_head=self.dense_lm_head,
        )
        return name_map(self.plan.targets)

    def dense_tensors_in_quantized_modules(self) -> frozenset[str]:
        return self.plan.dense_in_quantized_modules if self.plan else frozenset()

    def configure_quant_config(
        self,
        files: GGUFModelFiles,
        quant_config: GGUFConfig,
    ) -> None:
        """NVFP4 experts carry a per-expert output scale (``.scale``)."""
        del files
        quant_config.moe_expert_scales = True

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        del model_config
        if self.plan is None:
            raise RuntimeError("build_name_map must run before transform_weights")
        yield from inkling2_weights(weights, self.plan)
