# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Inkling 2 Mini DSpark draft from unsloth's ``dflash`` GGUF.

vLLM's ``Inkling2DSparkModel`` is an unquantized BF16 module that loads
``mtp.safetensors`` names (``layers.N.self_attn.q_proj.weight``, …) and
rejects any other dtype. ``dspark-Inkling-2-Mini-BF16.gguf`` holds those
tensors byte for byte (F32 ones are exact BF16 upcasts);
``dspark-Inkling-2-Mini-Q8_0.gguf`` stores the 27 projections as Q8_0,
which are dequantized here (0.40 GB file, 0.65 GB in BF16).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import gguf
import numpy as np
import torch
from gguf import GGMLQuantizationType

from ...weight_utils import weight_type_name
from ..base import GGUFWeight


def _dequantize(weight: torch.Tensor, weight_type: int) -> torch.Tensor:
    qtype = GGMLQuantizationType(weight_type)
    values = gguf.quants.dequantize(weight.cpu().numpy(), qtype)
    return torch.from_numpy(np.ascontiguousarray(values))


def dspark_weights(
    weights: Iterable[GGUFWeight],
    hf_names: Mapping[str, str] | Iterable[str],
    dtype: torch.dtype = torch.bfloat16,
) -> Iterable[GGUFWeight]:
    """Turn the DSpark GGUF stream (``mtp.safetensors`` names, ``weight_type``
    companions first) into dense *dtype* tensors."""
    names = set(hf_names.values() if isinstance(hf_names, Mapping) else hf_names)
    companions = {weight_type_name(n): n for n in names if "weight" in n}
    pending: dict[str, int] = {}
    for name, weight in weights:
        base = companions.get(name)
        if base is not None:
            pending[base] = int(weight)
            continue
        weight_type = pending.pop(name, None)
        if weight_type is not None:
            weight = _dequantize(weight, weight_type)
        yield name, weight.to(dtype)
    if pending:
        raise RuntimeError(
            f"ggml type companions without their tensor: {sorted(pending)}"
        )
