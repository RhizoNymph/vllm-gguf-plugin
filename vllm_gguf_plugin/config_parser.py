# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from transformers import PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from vllm.transformers_utils.config import HFConfigParser
from vllm.transformers_utils.config_parser_base import ConfigParserBase

from .gguf_utils import (
    check_gguf_file,
    is_gguf,
    is_remote_gguf,
    maybe_patch_hf_config_from_gguf,
    split_remote_gguf,
)


def _find_local_gguf_file(model: str | Path) -> str | None:
    """Return a concrete local ``.gguf`` file for *model* (a file or dir).

    Pure-GGUF sources ship no ``config.json``; locating the actual file lets
    us hand transformers ``gguf_file`` so it reads the config from the GGUF
    metadata. Auxiliary files (mmproj projector, MTP) are skipped.
    """
    model_str = str(model)
    if check_gguf_file(model_str):
        return model_str
    path = Path(model_str)
    if path.is_dir():
        candidates = sorted(
            g
            for g in path.glob("*.gguf")
            if not g.name.lower().startswith("mmproj") and "mtp" not in g.name.lower()
        )
        if candidates:
            return str(candidates[0])
    return None


class GGUFConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        original_model = model
        gguf_file_path = _find_local_gguf_file(model)
        if gguf_file_path is not None:
            # Pure-GGUF source (no config.json): have transformers read the
            # config straight from the GGUF metadata via ``gguf_file``.
            resolved_model = str(Path(gguf_file_path).parent)
            kwargs = {**kwargs, "gguf_file": Path(gguf_file_path).name}
        else:
            resolved_model = self._resolve_config_source(model)
        config_dict, config = HFConfigParser().parse(
            resolved_model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            code_revision=code_revision,
            **kwargs,
        )

        if config.model_type == "qwen3_moe" and "norm_topk_prob" not in config_dict:
            config_dict["norm_topk_prob"] = True
            config.update({"norm_topk_prob": True})

        text_config = getattr(config, "text_config", None)
        if (
            text_config is not None
            and getattr(text_config, "model_type", None)
            in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        ):
            config = text_config
            config_dict = config.to_dict()

        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")

        # gemma4 'laptop variant' GGUFs share K/V on full-attention layers
        # (no attn_v tensor); signal the model via attention_k_eq_v so it
        # reuses K as V instead of building an empty V proj (garbage out).
        if (
            config.model_type in ("gemma4", "gemma4_text")
            and gguf_file_path is not None
        ):
            import gguf as _gguf
            _n = {t.name for t in _gguf.GGUFReader(gguf_file_path).tensors}
            _nl = getattr(config, "num_hidden_layers", 0)
            if any(
                f"blk.{i}.attn_q.weight" in _n
                and f"blk.{i}.attn_v.weight" not in _n
                for i in range(_nl)
            ):
                config.attention_k_eq_v = True
                config_dict["attention_k_eq_v"] = True
        # llama.cpp gemma4 GGUFs carry *_logit_softcapping (inherited from
        # gemma2/3); HF Gemma4 defaults to None and applying them saturates
        # logits -> garbage. Native transformers maps them from GGUF, so
        # unset (mirrors the plugin's in-tree gemma4 GGUF patch).
        for _scf in ("final_logit_softcapping", "attn_logit_softcapping"):
            if getattr(config, _scf, None) is not None:
                setattr(config, _scf, None)
                config_dict[_scf] = None
        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config_dict["architectures"] = [model_type]
        config.update({"architectures": [model_type]})

        patch_source = gguf_file_path if gguf_file_path is not None else original_model
        if is_gguf(patch_source):
            config = maybe_patch_hf_config_from_gguf(str(patch_source), config)

        return config_dict, config

    @staticmethod
    def _resolve_config_source(model: str | Path) -> str | Path:
        if check_gguf_file(model):
            return Path(model).parent
        if is_remote_gguf(model):
            repo_id, _ = split_remote_gguf(model)
            return repo_id
        return model
