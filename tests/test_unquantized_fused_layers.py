# SPDX-License-Identifier: Apache-2.0
"""Unquantized GGUF shards of a fused layer, after vLLM renames them.

``unquantized_modules`` holds checkpoint names that ``apply_vllm_mapper``
rewrites with the model's ``hf_to_vllm_mapper``. A stacked mapping renames a
shard to its fused module (``ple.key_proj`` -> ``ple.kv_proj``), after which
the shard names no longer occur in the list and the fused layer was treated
as quantized — loading bfloat16 tensors into a GGUF parameter that never
receives a weight type.
"""

import pytest
from vllm.model_executor.models.utils import WeightsMapper

from vllm_gguf_plugin.quantization import GGUFConfig
from vllm_gguf_plugin.quantization.utils import is_layer_skipped_gguf

pytestmark = [pytest.mark.cpu]

FUSED = {
    "kv_proj": ["key_proj", "value_proj"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}


def test_fused_layer_renamed_by_the_mapper_is_skipped():
    config = GGUFConfig(
        unquantized_modules=[
            "model.language_model.layers.1.ple.key_proj",
            "model.language_model.layers.1.ple.value_proj",
        ]
    )
    config.apply_vllm_mapper(
        WeightsMapper(
            orig_to_new_prefix={"model.language_model.": "language_model.model."},
            orig_to_new_stacked={
                "ple.key_proj": ("ple.kv_proj", 0),
                "ple.value_proj": ("ple.kv_proj", 1),
            },
        )
    )
    assert is_layer_skipped_gguf(
        "language_model.model.layers.1.ple.kv_proj",
        config.unquantized_modules,
        FUSED,
    )


def test_quantized_fused_layer_is_not_skipped():
    assert not is_layer_skipped_gguf(
        "language_model.model.layers.1.ple.kv_proj",
        ["language_model.model.layers.1.linear_attn.in_proj_ba"],
        FUSED,
    )


def test_partially_quantized_fused_layer_is_rejected():
    with pytest.raises(ValueError, match="same precision"):
        is_layer_skipped_gguf(
            "model.layers.0.linear_attn.in_proj_ba",
            ["model.layers.0.linear_attn.in_proj_b"],
            FUSED,
        )
