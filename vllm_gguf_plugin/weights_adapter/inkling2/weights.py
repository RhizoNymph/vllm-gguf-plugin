# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Inkling 2 weight stream: GGUF tensors (already under their HF names)
into what vLLM's ``Inkling2LoopForConditionalGeneration`` loads.

vLLM's side of the contract is ``notes/inkling2-expert-interface.md`` in
the vLLM fork: routed experts arrive per expert as
``…mlp.experts.<e>.{gate,up,down}_proj.<suffix>``; dense tensors keep their
HF names (vLLM casts F32 to the parameter dtype); quantized tensors keep
their ``weight_type`` companion.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cached_property

import gguf
import torch
from gguf import GGMLQuantizationType

from ...weight_utils import weight_type_name
from ..base import GGUFWeight
from .metadata import Inkling2GGUFMetadata
from .names import (
    LLM,
    W2_EXPERT_SCALE,
    W13_EXPERT_SCALE,
    Conversion,
    TensorTarget,
    inkling2_targets,
)

_DENSE_TYPES = {
    GGMLQuantizationType.F32: torch.float32,
    GGMLQuantizationType.F16: torch.float16,
    GGMLQuantizationType.BF16: torch.bfloat16,
}
_TORCH_TO_GGML = {v: k for k, v in _DENSE_TYPES.items()}

# The four GGUF tensors vLLM loads as the shards of one merged ``q_proj``
# (``Inkling2ForConditionalGeneration._split_shards``).
_Q_PROJ_SHARDS = (
    "attn_q.weight",
    "attn_rel.weight",
    "attn_gate.weight",
    "attn_sink_proj.weight",
)
_EXPERT_TENSORS = ("ffn_gate_up_exps.weight", "ffn_down_exps.weight")
_VALUE_CONVERSIONS = (
    Conversion.UNSQUEEZE_CONV,
    Conversion.STATIC_CONV,
    Conversion.TRANSPOSE,
    Conversion.PATCH_EMBED,
)


def _layer_of(gguf_name: str) -> int | None:
    head, _, rest = gguf_name.partition(".")
    index = rest.partition(".")[0]
    return int(index) if head == "blk" and index.isdigit() else None


@dataclass(frozen=True)
class Inkling2LoadPlan:
    """What the weight stream needs to know about one set of files."""

    targets: Mapping[str, TensorTarget]
    types: Mapping[str, GGMLQuantizationType]
    num_layers: int
    num_experts: int
    #: GGUF names of dense tensors loaded into a GGUF-quantized module: they
    #: get a ``weight_type`` companion and must not be listed as unquantized
    #: modules (a merged linear may not mix quant methods).
    dense_in_quantized_modules: frozenset[str] = field(default_factory=frozenset)
    #: GGUF names of packed tensors served dense: dequantized to BF16 on load
    #: (the LM head when the config asks for an fp32 head).
    dequantize_on_load: frozenset[str] = field(default_factory=frozenset)

    @cached_property
    def dequantized_hf_names(self) -> frozenset[str]:
        return frozenset(self.targets[g].hf_name for g in self.dequantize_on_load)

    @cached_property
    def by_hf_name(self) -> dict[str, tuple[str, TensorTarget]]:
        return {t.hf_name: (g, t) for g, t in self.targets.items()}

    @cached_property
    def type_companions(self) -> dict[str, str]:
        """``weight_type`` companion name -> HF tensor name."""
        return {weight_type_name(hf): hf for hf in self.by_hf_name if "weight" in hf}

    @cached_property
    def dense_typed_hf_names(self) -> frozenset[str]:
        return frozenset(
            self.targets[g].hf_name for g in self.dense_in_quantized_modules
        )

    @cached_property
    def expert_layers(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    layer
                    for g, t in self.targets.items()
                    if t.conversion is Conversion.EXPERTS_GATE_UP
                    and (layer := _layer_of(g)) is not None
                }
            )
        )

    @cached_property
    def scaled_layers(self) -> frozenset[int]:
        return frozenset(
            layer
            for g, t in self.targets.items()
            if t.conversion is Conversion.EXPERT_SCALE_GATE_UP
            and (layer := _layer_of(g)) is not None
        )


def build_load_plan(
    types: Mapping[str, GGMLQuantizationType],
    meta: Inkling2GGUFMetadata,
    num_experts: int,
    dense_lm_head: bool = False,
) -> Inkling2LoadPlan:
    """Plan the weight stream for GGUF tensors *types* (name -> ggml type).

    *dense_lm_head* serves a packed ``output.weight`` as BF16.
    """
    targets = inkling2_targets(types, meta)
    dense: set[str] = set()
    for layer in range(meta.num_layers):
        shards = [
            f"blk.{layer}.{s}" for s in _Q_PROJ_SHARDS if f"blk.{layer}.{s}" in types
        ]
        dense_shards = [s for s in shards if types[s] in _DENSE_TYPES]
        if dense_shards and len(dense_shards) < len(shards):
            dense.update(dense_shards)
        for name in _EXPERT_TENSORS:
            full = f"blk.{layer}.{name}"
            if full in types and types[full] in _DENSE_TYPES:
                dense.add(full)
    return Inkling2LoadPlan(
        targets=targets,
        types=dict(types),
        num_layers=meta.num_layers,
        num_experts=num_experts,
        dense_in_quantized_modules=frozenset(dense),
        dequantize_on_load=frozenset(
            {"output.weight"}
            if dense_lm_head
            and "output.weight" in types
            and types["output.weight"] not in _DENSE_TYPES
            else ()
        ),
    )


_DEQUANT_CHUNK_ROWS = 8192


def _dequantize_bf16(weight: torch.Tensor, weight_type: int) -> torch.Tensor:
    """Dequantize packed rows on the CPU to BF16, a chunk of rows at a time
    (the 201024-row LM head would need 1.9 GB of fp32 at once)."""
    qtype = GGMLQuantizationType(weight_type)
    block, type_size = gguf.GGML_QUANT_SIZES[qtype]
    rows = weight.reshape(-1, weight.shape[-1]).cpu()
    cols = rows.shape[-1] // type_size * block
    out = torch.empty(rows.shape[0], cols, dtype=torch.bfloat16)
    for start in range(0, rows.shape[0], _DEQUANT_CHUNK_ROWS):
        chunk = rows[start : start + _DEQUANT_CHUNK_ROWS].numpy()
        values = gguf.quants.dequantize(chunk, qtype)
        out[start : start + chunk.shape[0]] = torch.from_numpy(values)
    return out.reshape(*weight.shape[:-1], cols)


def _type_tensor(weight_type: int) -> torch.Tensor:
    return torch.tensor(int(weight_type))


def _convert_values(conversion: Conversion, weight: torch.Tensor) -> torch.Tensor:
    match conversion:
        case Conversion.UNSQUEEZE_CONV:
            return weight.unsqueeze(1)
        case Conversion.STATIC_CONV:
            return weight.T.flip(0).contiguous()
        case Conversion.TRANSPOSE:
            return weight.T.contiguous()
        case Conversion.PATCH_EMBED:
            return weight.permute(0, 2, 3, 1).reshape(weight.shape[0], -1).contiguous()
    raise AssertionError(conversion)


def _per_expert(
    hf_name: str,
    weight: torch.Tensor,
    weight_type: int,
    conversion: Conversion,
) -> Iterable[GGUFWeight]:
    base = hf_name.rsplit(".", 1)[0]  # …mlp.experts
    for expert, rows in enumerate(weight.unbind()):
        if conversion is Conversion.EXPERTS_GATE_UP:
            half = rows.shape[0] // 2
            parts = (("gate_proj", rows[:half]), ("up_proj", rows[half:]))
        else:
            parts = (("down_proj", rows),)
        for proj, part in parts:
            yield f"{base}.{expert}.{proj}.weight_type", _type_tensor(weight_type)
            yield f"{base}.{expert}.{proj}.weight", part


def _expert_scales(hf_name: str, scales: torch.Tensor) -> Iterable[GGUFWeight]:
    base = hf_name.rsplit(".", 1)[0]
    proj = "gate_proj" if hf_name.endswith(W13_EXPERT_SCALE) else "down_proj"
    for expert in range(scales.shape[0]):
        yield f"{base}.{expert}.{proj}.expert_scale", scales[expert : expert + 1]


def _unit_scales(plan: Inkling2LoadPlan) -> Iterable[GGUFWeight]:
    """Scales of 1 for expert layers whose GGUF has none (non-NVFP4)."""
    ones = torch.ones(plan.num_experts, dtype=torch.float32)
    for layer in plan.expert_layers:
        if layer in plan.scaled_layers:
            continue
        prefix = f"{LLM}layers.{layer}."
        yield from _expert_scales(prefix + W13_EXPERT_SCALE, ones)
        yield from _expert_scales(prefix + W2_EXPERT_SCALE, ones)


def inkling2_weights(
    weights: Iterable[GGUFWeight], plan: Inkling2LoadPlan
) -> Iterable[GGUFWeight]:
    """Convert the GGUF weight stream (HF names, ``weight_type`` companions
    before their tensors) into the tensors vLLM's Inkling2 model loads."""
    pending: dict[str, int] = {}
    for name, weight in weights:
        base = plan.type_companions.get(name)
        if base is not None:
            pending[base] = int(weight)
            continue
        entry = plan.by_hf_name.get(name)
        if entry is None:
            yield name, weight
            continue
        _, target = entry
        conversion = target.conversion
        weight_type = pending.pop(name, None)
        if name in plan.dequantized_hf_names and weight_type is not None:
            yield name, _dequantize_bf16(weight, weight_type)
            continue
        if weight_type is None and name in plan.dense_typed_hf_names:
            weight_type = int(_TORCH_TO_GGML[weight.dtype])

        if conversion in (Conversion.EXPERTS_GATE_UP, Conversion.EXPERTS_DOWN):
            if weight_type is None:
                weight_type = int(_TORCH_TO_GGML[weight.dtype])
            yield from _per_expert(name, weight, weight_type, conversion)
        elif conversion in (
            Conversion.EXPERT_SCALE_GATE_UP,
            Conversion.EXPERT_SCALE_DOWN,
        ):
            yield from _expert_scales(name, weight.to(torch.float32))
        elif conversion in _VALUE_CONVERSIONS:
            if weight_type is not None and weight_type not in map(int, _DENSE_TYPES):
                raise TypeError(
                    f"{name}: stored as ggml type "
                    f"{GGMLQuantizationType(weight_type).name}, but its "
                    f"{conversion.value} conversion needs dense values"
                )
            yield name, _convert_values(conversion, weight)
        else:
            if weight_type is not None:
                yield weight_type_name(name), _type_tensor(weight_type)
            yield name, weight
    if pending:
        raise RuntimeError(
            f"ggml type companions without their tensor: {sorted(pending)}"
        )
    yield from _unit_scales(plan)
