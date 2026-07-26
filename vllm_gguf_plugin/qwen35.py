# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5-family GGUF support."""

_PATCHED = False


def register_qwen35_gguf_support() -> None:
    """Teach transformers' GGUF loader about qwen35/qwen35moe."""
    global _PATCHED

    from transformers import configuration_utils as _cu
    from transformers import modeling_gguf_pytorch_utils as _mgu
    from transformers.integrations import ggml as _ggml
    from transformers.models.auto import tokenization_auto as _ta
    from vllm.model_executor.models.registry import ModelRegistry

    qwen35_mapping = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.key_length": "head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "full_attention_interval": "full_attention_interval",
        "nextn_predict_layers": "num_nextn_predict_layers",
        "ssm.conv_kernel": "linear_conv_kernel_dim",
        "ssm.state_size": "linear_key_head_dim",
        "ssm.group_count": "linear_num_key_heads",
        "ssm.inner_size": "linear_num_value_heads",
        "vocab_size": "vocab_size",
    }
    qwen35moe_mapping = {
        **qwen35_mapping,
        "feed_forward_length": "moe_intermediate_size",
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_tok",
    }

    _ggml.GGUF_CONFIG_MAPPING.setdefault("qwen35", qwen35_mapping)
    _ggml.GGUF_CONFIG_MAPPING.setdefault("qwen35moe", qwen35moe_mapping)
    for arch in ("qwen35", "qwen35moe"):
        if arch not in _mgu.GGUF_SUPPORTED_ARCHITECTURES:
            _mgu.GGUF_SUPPORTED_ARCHITECTURES.append(arch)
        _ggml.GGUF_TO_FAST_CONVERTERS.setdefault(
            arch, _ggml.GGUFQwen2Converter
        )
    ModelRegistry.register_model(
        "Qwen3_5ForCausalLM",
        "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM",
    )
    ModelRegistry.register_model(
        "Qwen3_5MoeForCausalLM",
        "vllm.model_executor.models.qwen3_5:Qwen3_5MoeForCausalLM",
    )

    if _PATCHED:
        return

    _orig_load = _mgu.load_gguf_checkpoint

    def _patched_load(*args, **kwargs):
        parsed = _orig_load(*args, **kwargs)
        cfg = parsed.get("config", {})
        if cfg.get("model_type") == "qwen35":
            cfg["model_type"] = "qwen3_5_text"
            cfg["architectures"] = ["Qwen3_5ForCausalLM"]
            _derive_linear_value_heads(cfg)
            _exclude_mtp_layers(cfg)
        elif cfg.get("model_type") == "qwen35moe":
            cfg["model_type"] = "qwen3_5_moe_text"
            cfg["architectures"] = ["Qwen3_5MoeForCausalLM"]
            cfg.setdefault("norm_topk_prob", True)
            _derive_linear_value_heads(cfg)
            _exclude_mtp_layers(cfg)
        return parsed

    _mgu.load_gguf_checkpoint = _patched_load
    _cu.load_gguf_checkpoint = _patched_load
    # AutoTokenizer imports load_gguf_checkpoint by name, so patch its module
    # reference as well. Otherwise GGUF tokenizer construction still sees the
    # raw qwen35/qwen35moe model_type and AutoConfig rejects it.
    _ta.load_gguf_checkpoint = _patched_load
    _PATCHED = True


def _exclude_mtp_layers(cfg: dict) -> None:
    """Drop the multi-token-prediction blocks from the decoder layer count.

    ggml's ``block_count`` counts every block in the file, including the
    trailing MTP/nextn blocks; HF's ``num_hidden_layers`` counts only the
    decoder stack. Leaving the ggml value in place makes vLLM build extra
    layers that no checkpoint tensor maps onto, so they keep their
    initialisation and quietly corrupt the residual stream — the model loads
    without a single warning and emits noise.

    MTP weights are skipped at load time, so the layers are not built here;
    the count is zeroed rather than carried.
    """
    total = cfg.get("num_hidden_layers")
    mtp = cfg.get("num_nextn_predict_layers")
    if not isinstance(total, int) or not isinstance(mtp, int):
        return
    if not 0 < mtp < total:
        return
    cfg["num_hidden_layers"] = total - mtp
    cfg["num_nextn_predict_layers"] = 0


def _derive_linear_value_heads(cfg: dict) -> None:
    inner_size = cfg.get("linear_num_value_heads")
    state_size = cfg.get("linear_key_head_dim")
    if not isinstance(inner_size, int) or not isinstance(state_size, int):
        return
    if state_size <= 0 or inner_size % state_size != 0:
        return
    cfg.setdefault("linear_value_head_dim", state_size)
    cfg["linear_num_value_heads"] = inner_size // state_size
