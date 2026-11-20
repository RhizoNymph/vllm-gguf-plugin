# SPDX-License-Identifier: Apache-2.0
"""Registration against the transformers/vLLM versions the plugin targets.

transformers 5.18 moved the GGUF tokenizer converters out of
``transformers.integrations.ggml`` and gained a native qwen35 GGUF config
path; vLLM now registers the Qwen3.5 text entry points itself. register()
must neither touch the removed converter tables nor shadow vLLM's own
model registrations.
"""

import pytest

pytestmark = [pytest.mark.cpu]


def test_register_does_not_need_removed_ggml_converters():
    from transformers.integrations import ggml

    import vllm_gguf_plugin

    assert not hasattr(ggml, "GGUF_TO_FAST_CONVERTERS")
    vllm_gguf_plugin.register()


def test_register_keeps_vllm_qwen35_text_registrations():
    from vllm.model_executor.models.registry import ModelRegistry

    before = {
        arch: ModelRegistry.models[arch]
        for arch in ("Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM")
    }

    import vllm_gguf_plugin

    vllm_gguf_plugin.register()

    after = {arch: ModelRegistry.models[arch] for arch in before}
    assert after == before
    for model in after.values():
        assert model.module_name == "vllm.models.qwen3_5"


def test_qwen35_gguf_config_is_native():
    from transformers.integrations.gguf import GGUF_CONFIG_ARCHS

    assert {"qwen35", "qwen35moe"} <= set(GGUF_CONFIG_ARCHS)


def test_gemma4_mapping_keeps_plugin_only_keys():
    """transformers maps gemma4 natively but omits keys the plugin relies on;
    registration must merge them in rather than lose them to setdefault."""
    from transformers.integrations import ggml

    import vllm_gguf_plugin

    vllm_gguf_plugin.register()
    mapping = ggml.GGUF_CONFIG_MAPPING["gemma4"]
    assert mapping["attention.shared_kv_layers"] == "num_kv_shared_layers"
    assert mapping["embedding_length_per_layer_input"] == "hidden_size_per_layer_input"
    # Deliberately unmapped: applying it saturates gemma4 logits.
    assert "final_logit_softcapping" not in mapping
