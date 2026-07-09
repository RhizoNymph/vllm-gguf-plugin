# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from pathlib import Path

import vllm.engine.arg_utils as arg_utils_module
import vllm.transformers_utils.config as config_module
from vllm.config.load import LoadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS,
    get_quantization_config,
    register_quantization_config,
)
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
from .qwen35 import register_qwen35_gguf_support
from .quantization import GGUFConfig
from .weight_utils import download_gguf, resolve_local_gguf

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
            if hf_config_path is None and check_gguf_file(str(gguf_weights)):
                self.hf_config_path = gguf_weights
            self.model = _get_gguf_config_source(
                gguf_weights,
                self.tokenizer if isinstance(self.tokenizer, str) else None,
                hf_config_path,
            )
        return original_create_model_config(self, *args, **kwargs)

    EngineArgs.create_model_config = create_model_config
    EngineArgs._gguf_create_model_config_patched = True


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


def register() -> None:
    """Register the out-of-tree GGUF integration."""
    if (
        "gguf" not in QUANTIZATION_METHODS
        or get_quantization_config("gguf") is not GGUFConfig
    ):
        register_quantization_config("gguf")(GGUFConfig)

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
