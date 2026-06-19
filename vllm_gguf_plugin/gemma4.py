# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 GGUF support.

Transformers ships ``GGUF_CONFIG_MAPPING`` with entries for gemma2/gemma3
but not gemma4, so ``load_gguf_checkpoint`` raises before vLLM ever sees a
gemma4 GGUF. The ``gguf`` library *does* know the ``gemma4`` architecture, so
once the config is parsed the default weights adapter maps tensors correctly
(see ``weights_adapter/default.py``). This module teaches transformers about
gemma4 at the config and tokenizer level via two idempotent monkeypatches that
are installed from ``plugin.register()``.
"""

from pathlib import Path

import gguf
from vllm.logger import init_logger

logger = init_logger(__name__)

_TOKENIZER_PATCHED = False


def _patch_gemma4_config() -> None:
    """Teach transformers' GGUF loader about the gemma4 architecture.

    Patches ``GGUF_CONFIG_MAPPING`` in-place and wraps ``load_gguf_checkpoint``
    to:
      * rename model_type "gemma4" -> "gemma4_text" (mirrors the gemma3 path)
      * reduce per-layer head_count_kv lists into the scalar fields that
        Gemma4TextConfig expects (num_key_value_heads for sliding layers,
        num_global_key_value_heads for full-attention layers)
      * derive layer_types from attention.sliding_window_pattern
      * build rope_parameters from rope.freq_base{,_swa}
      * split key_length{,_swa} into head_dim / global_head_dim
    """
    from transformers import modeling_gguf_pytorch_utils as _mgu
    from transformers.integrations import ggml as _ggml

    if "gemma4" in _ggml.GGUF_CONFIG_MAPPING:
        return

    _ggml.GGUF_CONFIG_MAPPING["gemma4"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "attention.head_count": "num_attention_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.sliding_window": "sliding_window",
        "attention.shared_kv_layers": "num_kv_shared_layers",
        # NOTE: final_logit_softcapping intentionally not mapped. llama.cpp's
        # gemma4 GGUFs carry a value of 30.0 inherited from gemma2/3, but
        # the HF Gemma4TextConfig defaults to None and applying it here
        # saturates all token logits to ±30 (hidden state RMS is large), so
        # softmax collapses to ~uniform. Leave it unset.
        "embedding_length_per_layer_input": "hidden_size_per_layer_input",
        "vocab_size": "vocab_size",
    }
    _mgu.GGUF_SUPPORTED_ARCHITECTURES.append("gemma4")
    # convert_gguf_tokenizer keys by the raw GGUF architecture string
    # ("gemma4"), while elsewhere we use the renamed model_type
    # ("gemma4_text"). Register both so either lookup path works.
    _ggml.GGUF_TO_FAST_CONVERTERS["gemma4"] = _ggml.GGUFGemmaConverter
    _ggml.GGUF_TO_FAST_CONVERTERS["gemma4_text"] = _ggml.GGUFGemmaConverter

    _orig_load = _mgu.load_gguf_checkpoint

    def _patched_load(*args, **kwargs):
        parsed = _orig_load(*args, **kwargs)
        cfg = parsed.get("config", {})
        if cfg.get("model_type") != "gemma4":
            return parsed
        gguf_path = args[0] if args else kwargs.get("gguf_checkpoint_path")
        cfg["model_type"] = "gemma4_text"
        if not cfg.get("architectures"):
            cfg["architectures"] = ["Gemma4ForCausalLM"]
        reader = gguf.GGUFReader(gguf_path)
        fields = reader.fields
        pat_field = fields.get("gemma4.attention.sliding_window_pattern")
        nkv_field = fields.get("gemma4.attention.head_count_kv")
        layer_pat = None
        if pat_field is not None:
            layer_pat = [
                bool(pat_field.parts[i].tolist()[0]) for i in pat_field.data
            ]
            cfg["layer_types"] = [
                "sliding_attention" if p else "full_attention" for p in layer_pat
            ]
        if nkv_field is not None:
            nkv_list = [
                int(nkv_field.parts[i].tolist()[0]) for i in nkv_field.data
            ]
            if layer_pat is not None and len(layer_pat) == len(nkv_list):
                sliding = {v for v, p in zip(nkv_list, layer_pat) if p}
                full = {v for v, p in zip(nkv_list, layer_pat) if not p}
                if len(sliding) == 1:
                    cfg["num_key_value_heads"] = sliding.pop()
                if len(full) == 1:
                    cfg["num_global_key_value_heads"] = full.pop()
            elif isinstance(cfg.get("num_key_value_heads"), list):
                cfg["num_key_value_heads"] = max(cfg["num_key_value_heads"])
        key_swa = fields.get("gemma4.attention.key_length_swa")
        key_full = fields.get("gemma4.attention.key_length")
        if key_swa is not None:
            cfg["head_dim"] = int(key_swa.parts[key_swa.data[0]].tolist()[0])
        if key_full is not None:
            cfg["global_head_dim"] = int(
                key_full.parts[key_full.data[0]].tolist()[0]
            )
        # Detect attention_k_eq_v: if any full-attention block is missing
        # an attn_v tensor, the checkpoint uses shared K/V weights for
        # full-attention layers.
        if layer_pat is not None:
            tensor_names = {t.name for t in reader.tensors}
            full_blocks_missing_v = any(
                not p and f"blk.{i}.attn_v.weight" not in tensor_names
                for i, p in enumerate(layer_pat)
            )
            if full_blocks_missing_v:
                cfg["attention_k_eq_v"] = True
        rope_swa = fields.get("gemma4.rope.freq_base_swa")
        rope_full = fields.get("gemma4.rope.freq_base")
        rope_parameters: dict = {}
        if rope_swa is not None:
            rope_parameters["sliding_attention"] = {
                "rope_type": "default",
                "rope_theta": float(
                    rope_swa.parts[rope_swa.data[0]].tolist()[0]
                ),
            }
        if rope_full is not None:
            # Verified against the GGUF's rope_freqs.weight: only the first
            # head_dim/4 frequency pairs are active (= partial_rotary_factor
            # 0.25 with head_dim=512 → 64 active angles, 192 zero-padded).
            rope_parameters["full_attention"] = {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": float(
                    rope_full.parts[rope_full.data[0]].tolist()[0]
                ),
            }
        if rope_parameters:
            cfg["rope_parameters"] = rope_parameters
        cfg.pop("rope_theta", None)
        return parsed

    _mgu.load_gguf_checkpoint = _patched_load
    # transformers.configuration_utils imports the function by name at module
    # load, so patching the source module alone misses the path that
    # PretrainedConfig.get_config_dict actually invokes.
    from transformers import configuration_utils as _cu

    _cu.load_gguf_checkpoint = _patched_load


def _patch_gemma4_tokenizer() -> None:
    """Fix BOS handling for gemma4 GGUF tokenizers.

    The transformers GGUF tokenizer converter doesn't propagate
    ``tokenizer.ggml.add_bos_token`` from the GGUF metadata, so the resulting
    fast tokenizer has ``add_bos_token=False`` even when the underlying model
    requires a BOS prefix. For Gemma4 (trained with a leading <bos>) that
    produces degenerate output — the model collapses to repeating the last two
    prompt tokens. Patch ``AutoTokenizer.from_pretrained`` to inject
    ``add_bos_token=True`` when loading a gemma4 GGUF, and post-fix
    ``bos_token_id`` to whatever the GGUF metadata reports (HF's converter
    sometimes leaves this at the wrong vocab id — e.g. 203 instead of 2).
    """
    global _TOKENIZER_PATCHED
    if _TOKENIZER_PATCHED:
        return

    from transformers import AutoTokenizer

    _orig_from_pretrained = AutoTokenizer.from_pretrained

    def _patched_tokenizer_from_pretrained(
        pretrained_model_name_or_path, *args, **kwargs
    ):
        # Locate the backing .gguf file. The model path may be the .gguf
        # itself, a dir containing it, or a path paired with a ``gguf_file``
        # kwarg.
        gguf_path: Path | None = None
        base = Path(pretrained_model_name_or_path)
        gguf_file = kwargs.get("gguf_file")
        if gguf_file:
            parent = base.parent if base.is_file() else base
            candidate = parent / Path(gguf_file).name
            if candidate.is_file():
                gguf_path = candidate
        elif base.suffix == ".gguf" and base.is_file():
            gguf_path = base
        elif base.is_dir() and not (
            (base / "tokenizer.json").is_file()
            or (base / "tokenizer_config.json").is_file()
        ):
            # GGUF-only directory (no HF tokenizer files): build the tokenizer
            # from the GGUF's embedded tokenizer. transformers needs to be
            # told which file via ``gguf_file``; without this it can't
            # instantiate any backend tokenizer. Applies to all GGUF archs,
            # not just gemma4.
            ggufs = sorted(
                g
                for g in base.glob("*.gguf")
                if not g.name.lower().startswith("mmproj")
                and "mtp" not in g.name.lower()
            )
            if ggufs:
                gguf_path = ggufs[0]
                kwargs["gguf_file"] = gguf_path.name

        gguf_bos: int | None = None
        gguf_eos: int | None = None
        is_gemma4 = False
        if gguf_path is not None:
            try:
                reader = gguf.GGUFReader(gguf_path)
                arch_field = reader.fields.get("general.architecture")
                if arch_field is not None:
                    arch_bytes = bytes(arch_field.parts[arch_field.data[0]])
                    is_gemma4 = arch_bytes == b"gemma4"
                if is_gemma4:
                    bos_field = reader.fields.get("tokenizer.ggml.bos_token_id")
                    if bos_field is not None:
                        gguf_bos = int(
                            bos_field.parts[bos_field.data[0]].tolist()[0]
                        )
                    eos_field = reader.fields.get("tokenizer.ggml.eos_token_id")
                    if eos_field is not None:
                        gguf_eos = int(
                            eos_field.parts[eos_field.data[0]].tolist()[0]
                        )
            except Exception as exc:
                logger.debug("gguf BOS/EOS probe failed for %s: %s", gguf_path, exc)

        if is_gemma4 and "add_bos_token" not in kwargs:
            kwargs["add_bos_token"] = True

        tokenizer = _orig_from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

        if is_gemma4 and gguf_bos is not None:
            try:
                bos_str = tokenizer.convert_ids_to_tokens(gguf_bos)
                if bos_str:
                    tokenizer.bos_token = bos_str
                tokenizer.bos_token_id = gguf_bos
            except Exception as exc:
                logger.debug("failed to set BOS attrs on tokenizer: %s", exc)

            # The fast tokenizer's post_processor template was baked at
            # conversion time with the wrong BOS id (often 203 for Gemma4
            # variants), so even after fixing the bos_token_id attribute,
            # encode() prepends the wrong id. Replace the post_processor
            # to use the GGUF-declared BOS.
            try:
                from tokenizers import processors

                tokenizer.backend_tokenizer.post_processor = (
                    processors.TemplateProcessing(
                        single=f"{bos_str} $A",
                        pair=f"{bos_str} $A {bos_str} $B",
                        special_tokens=[(bos_str, gguf_bos)],
                    )
                )
            except Exception as exc:
                logger.debug(
                    "failed to override post_processor for BOS: %s", exc
                )

        if is_gemma4 and gguf_eos is not None:
            # The GGUF tokenizer converter defaults eos to a generic token
            # (e.g. </s>, id 212) instead of Gemma4's turn terminator
            # ``<turn|>`` (the GGUF-declared eos_token_id). Without this,
            # generation never stops on turn end and the model rambles.
            try:
                eos_str = tokenizer.convert_ids_to_tokens(gguf_eos)
                if eos_str:
                    tokenizer.eos_token = eos_str
                tokenizer.eos_token_id = gguf_eos
            except Exception as exc:
                logger.debug("failed to set EOS attrs on tokenizer: %s", exc)
        return tokenizer

    AutoTokenizer.from_pretrained = _patched_tokenizer_from_pretrained
    _TOKENIZER_PATCHED = True


def register_gemma4_gguf_support() -> None:
    """Install gemma4 GGUF config + tokenizer patches (idempotent)."""
    _patch_gemma4_config()
    _patch_gemma4_tokenizer()
