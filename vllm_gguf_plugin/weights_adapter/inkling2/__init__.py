# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling 2 Mini GGUF support (``docs/features/inkling2.md``)."""

from .adapter import INKLING2_ARCHITECTURES, Inkling2GGUFAdapter
from .dspark import dspark_weights
from .metadata import (
    Inkling2ConfigMismatch,
    Inkling2GGUFMetadata,
    LoopLayout,
    read_gguf_kv,
    validate_inkling2_config,
    validate_inkling2_dspark_config,
    validate_inkling2_vision_config,
)
from .names import (
    Conversion,
    TensorTarget,
    UnmappedTensorsError,
    dspark_name_map,
    inkling2_targets,
)
from .weights import Inkling2LoadPlan, build_load_plan, inkling2_weights

__all__ = [
    "INKLING2_ARCHITECTURES",
    "Conversion",
    "Inkling2ConfigMismatch",
    "Inkling2GGUFAdapter",
    "Inkling2GGUFMetadata",
    "Inkling2LoadPlan",
    "LoopLayout",
    "TensorTarget",
    "UnmappedTensorsError",
    "build_load_plan",
    "dspark_name_map",
    "dspark_weights",
    "inkling2_targets",
    "inkling2_weights",
    "read_gguf_kv",
    "validate_inkling2_config",
    "validate_inkling2_dspark_config",
    "validate_inkling2_vision_config",
]
