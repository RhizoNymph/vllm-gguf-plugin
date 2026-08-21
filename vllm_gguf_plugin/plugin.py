# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from pathlib import Path

import gguf
import vllm.engine.arg_utils as arg_utils_module
import vllm.transformers_utils.config as config_module
from vllm.config.load import LoadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.model_loader import (
    _LOAD_FORMAT_TO_MODEL_LOADER,
    get_model_loader,
    register_model_loader,
)
from vllm.transformers_utils.config import get_config_parser, register_config_parser

from .config_parser import GGUFConfigParser
from .gemma4 import register_gemma4_gguf_support
from .gguf_utils import (
    check_gguf_file,
    is_gguf,
    is_local_gguf_quant,
    is_remote_gguf,
    split_remote_gguf,
)
from .loader import GGUFModelLoader
from .quantization import DiffusionGGUFConfig, GGUFConfig
from .qwen35 import register_qwen35_gguf_support
from .weight_utils import download_gguf, resolve_local_gguf
from .weights_adapter.diffusion.integration import _patch_diffusers_loader

OOTGGUFConfig = GGUFConfig
OOTGGUFModelLoader = GGUFModelLoader


def _is_gguf_reference(model: str | None) -> bool:
    if not model:
        return False
    model_path = Path(model)
    return (
        model.endswith(".gguf")
        or is_remote_gguf(model)
        or is_gguf(model)
        or _find_local_gguf_file(model_path) is not None
    )


def _find_local_gguf_file(model: str | Path) -> str | None:
    model_path = Path(model)
    if check_gguf_file(str(model_path)):
        return str(model_path)
    if not model_path.is_dir():
        return None
    candidates = sorted(
        gguf_file
        for gguf_file in model_path.glob("*.gguf")
        if not gguf_file.name.lower().startswith("mmproj")
        and "mtp" not in gguf_file.name.lower()
    )
    if candidates:
        return str(candidates[0])
    return None


def _resolve_gguf_weights(model: str) -> str:
    model_str = str(model)
    local_file = _find_local_gguf_file(model_str)
    if local_file is not None:
        return local_file
    if is_remote_gguf(model_str):
        repo_id, quant = split_remote_gguf(model_str)
        return download_gguf(repo_id, quant)
    if is_local_gguf_quant(model_str):
        local_dir, quant = model_str.rsplit(":", 1)
        return resolve_local_gguf(local_dir, quant)
    return model_str


def _get_gguf_architecture(model: str) -> str | None:
    if not check_gguf_file(model):
        return None
    try:
        reader = gguf.GGUFReader(model)
    except Exception:
        return None
    general_arch = reader.fields.get("general.architecture")
    if general_arch is None:
        return None
    value = general_arch.contents()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _resolve_local_gguf_dir(model: str) -> str:
    """Resolve a GGUF reference to the local directory containing the ``.gguf``.

    Pure-GGUF repos ship no ``config.json``, so pointing the loaders at the
    repo id alone fails. We resolve to the directory holding the actual file
    (downloading remote references here; ``snapshot_download`` is cached, so
    the loader's later fetch is a no-op) — a directory, not the file itself,
    because transformers' auxiliary loaders (image processor, etc.) treat
    ``model`` as a repo id / dir and reject a bare file path. The config
    parser then reads config from the ``.gguf`` in that dir via ``gguf_file``.
    """
    return str(Path(_resolve_gguf_weights(model)).parent)


def _get_gguf_config_source(
    model: str,
    tokenizer: str | None,
    hf_config_path: str | None,
) -> str:
    if hf_config_path is not None:
        return hf_config_path
    if tokenizer is not None and not _is_gguf_reference(tokenizer):
        return tokenizer
    return _resolve_local_gguf_dir(model)


def _patch_engine_args() -> None:
    if getattr(EngineArgs, "_gguf_create_model_config_patched", False):
        return

    original_create_model_config = EngineArgs.create_model_config

    @wraps(original_create_model_config)
    def create_model_config(self, *args, **kwargs):
        if _is_gguf_reference(self.model):
            gguf_model = self.model
            gguf_weights = _resolve_gguf_weights(str(gguf_model))
            if self.quantization is None:
                self.quantization = "gguf"
            if self.load_format == "auto":
                self.load_format = "gguf"
            if self.config_format == "auto":
                self.config_format = "gguf"
            if not self.model_weights:
                self.model_weights = gguf_weights
            if self.served_model_name is None:
                self.served_model_name = [gguf_model]
            hf_config_path = self.hf_config_path
            tokenizer = self.tokenizer if isinstance(self.tokenizer, str) else None
            # Reading the config out of the GGUF is a *fallback* for pure-GGUF
            # sources that ship no config.json. Only do it when nothing better
            # exists: an explicit --hf-config-path, or a tokenizer repo, is a
            # full HF config, including the vision section that a text-only
            # backbone GGUF cannot supply. Pinning the GGUF unconditionally
            # strips that section while the adapter still declares the
            # multimodal architecture, so vLLM builds a multimodal processor
            # against a text config and fails on the type mismatch.
            if (
                hf_config_path is None
                and (tokenizer is None or _is_gguf_reference(tokenizer))
                and check_gguf_file(str(gguf_weights))
            ):
                self.hf_config_path = gguf_weights
            self.model = _get_gguf_config_source(
                gguf_weights,
                tokenizer,
                hf_config_path,
            )
            # ``self.model`` is now the *directory* holding the weights, which
            # loses track of which .gguf was asked for. Pin the tokenizer to
            # the resolved file so it is never re-derived by globbing that
            # directory — a dir holding several .gguf files would otherwise
            # yield whichever sorts first, silently building the tokenizer
            # from an unrelated model.
            if tokenizer is None:
                self.tokenizer = gguf_weights
        return original_create_model_config(self, *args, **kwargs)

    EngineArgs.create_model_config = create_model_config
    EngineArgs._gguf_create_model_config_patched = True

    original_create_speculative_config = EngineArgs.create_speculative_config

    @wraps(original_create_speculative_config)
    def create_speculative_config(self, *args, **kwargs):
        configured_model = getattr(self, "spec_model", None)
        if self.speculative_config is not None:
            configured_model = configured_model or self.speculative_config.get("model")

        config = original_create_speculative_config(self, *args, **kwargs)
        gguf_model = self.model_weights
        if (
            config is not None
            and config.method == "mtp"
            and configured_model is None
            and _is_gguf_reference(gguf_model)
        ):
            config.draft_model_config.model_weights = gguf_model
        return config

    EngineArgs.create_speculative_config = create_speculative_config


def _patch_speculator_probe() -> None:
    if getattr(arg_utils_module, "_gguf_speculator_probe_patched", False):
        return

    original_maybe_override = arg_utils_module.maybe_override_with_speculators

    @wraps(original_maybe_override)
    def maybe_override_with_speculators(model, tokenizer, *args, **kwargs):
        if _is_gguf_reference(model):
            return model, tokenizer, kwargs.get("vllm_speculative_config")
        return original_maybe_override(model, tokenizer, *args, **kwargs)

    arg_utils_module.maybe_override_with_speculators = maybe_override_with_speculators
    config_module.maybe_override_with_speculators = maybe_override_with_speculators
    arg_utils_module._gguf_speculator_probe_patched = True
    config_module._gguf_speculator_probe_patched = True


def _register_omni_diffusion_quantization() -> None:
    try:
        from vllm_omni.quantization import register_quantization_override
    except ImportError:
        return

    register_quantization_override("gguf", lambda **kw: DiffusionGGUFConfig(**kw))


def register() -> None:
    """Register the out-of-tree GGUF integration."""
    register_quantization_config("gguf")(GGUFConfig)
    _register_omni_diffusion_quantization()

    if "gguf" not in _LOAD_FORMAT_TO_MODEL_LOADER or not isinstance(
        get_model_loader(LoadConfig(load_format="gguf")), GGUFModelLoader
    ):
        register_model_loader("gguf")(GGUFModelLoader)

    try:
        parser = get_config_parser("gguf")
    except ValueError:
        parser = None
    if not isinstance(parser, GGUFConfigParser):
        register_config_parser("gguf")(GGUFConfigParser)
    _patch_engine_args()
    _patch_speculator_probe()
    register_gemma4_gguf_support()
    register_qwen35_gguf_support()
    _patch_diffusers_loader()
