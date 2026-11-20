# SPDX-License-Identifier: Apache-2.0

# Registers ggml types newer than gguf-py before anything opens a GGUF.
from . import gguf_compat  # noqa: F401  # isort: skip
from .config_parser import GGUFConfigParser
from .loader import GGUFModelLoader
from .plugin import OOTGGUFConfig, OOTGGUFModelLoader, register
from .quantization import DiffusionGGUFConfig, GGUFConfig

__all__ = [
    "DiffusionGGUFConfig",
    "GGUFConfig",
    "GGUFConfigParser",
    "GGUFModelLoader",
    "OOTGGUFConfig",
    "OOTGGUFModelLoader",
    "register",
]
