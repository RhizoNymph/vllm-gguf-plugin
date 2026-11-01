# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import os

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from .layout import GGUFLinearLayout
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    GGUFWeightParameter,
    _gguf_ordered_shard_ids,
    _materialize_gguf_weight_parameter,
    _materialize_gguf_weight_type_parameter,
    _resolve_gguf_weight_loader,
    _resolve_gguf_weight_type_loader,
)
from .utils import (
    DEQUANT_TYPES,
    IMATRIX_QUANT_TYPES,
    MMQ_QUANT_TYPES,
    MMVQ_QUANT_TYPES,
    UNQUANTIZED_TYPES,
)


# Batch size at or above which the fused MMQ kernel loses to
# dequantize-then-cuBLAS, so prefill-shaped GEMMs take the dequant path instead.
#
# `mul_mat_q4_K` computes a 32x4 output tile -- one output per thread over the
# full K loop -- so operand-fetch instructions vastly outnumber math. Nsight
# Compute on an RTX 3090 (M=684, 5120x17408 Q4_K) shows Mem Pipes Busy at 95.44%
# while DRAM sits at 46.63% and the tensor pipes are completely idle, giving
# ~5.7 TFLOP/s *flat* from M=32 to M=8192 -- 8% of the card's fp16 peak, and
# batch-independent, so larger batches cannot help.
#
# Materializing bf16 weights and calling cuBLAS reaches ~55 TFLOP/s (77% of peak,
# via a 256x128 tensor-core tile): 9.5x for Q4_K, 12.4x Q5_K, 12.7x Q6_K at
# M=2048, and still ~8.9x ahead even paying a full dequantization on every call.
# Crossover is near M=32. Accuracy also improves 6-8x, because MMQ additionally
# quantizes activations to int8 (`quantize_q8_1`) whereas this path keeps bf16.
#
# Measured end to end on Qwen3.6-27B-UD-Q4_K_XL (RTX 3090, torch.compile +
# CUDA graphs): prefill 105.8 -> 692.3 tok/s (6.54x), mean TTFT 43.9 -> 6.7 s,
# for 7.6% less KV cache (peak activation grows by the transient bf16 weight,
# ~0.29 GiB here). Decode is untouched: decode batches are <= max_num_seqs, far
# below this threshold, so they still take the MMVQ/MMQ path.
#
# Opt-in for now: the numbers above come from a single model on a single GPU
# (RTX 3090, sm_86), so the default stays 0 (always MMQ, unchanged behaviour)
# until validated more widely. Set VLLM_GGUF_MMQ_SAFE=32 to enable.
_MMQ_SAFE = int(os.environ.get("VLLM_GGUF_MMQ_SAFE", "0"))


# Whether ggml_mul_mat_a8 runs the vendored llama.cpp tensor-core MMQ kernel. Read with
# the same rule as the extension (gguf_kernel.cu, mma_mmq_enabled): set and not "0".
_MMA_MMQ_ENV = os.environ.get("VLLM_GGUF_MMA_MMQ", "")
_MMA_MMQ = _MMA_MMQ_ENV != "" and not _MMA_MMQ_ENV.startswith("0")
# Whether ggml_mul_mat_vec_a8 runs the vendored llama.cpp multi-column MMVQ
# (gguf_kernel.cu, mmvq_multi_enabled), which reads each weight block once for up to
# 8 activation rows.
_MMVQ_MULTI_ENV = os.environ.get("VLLM_GGUF_MMVQ_MULTI", "")
_MMVQ_MULTI = _MMVQ_MULTI_ENV != "" and not _MMVQ_MULTI_ENV.startswith("0")
# Rows up to which multi-column MMVQ beats tensor-core MMQ. RTX 3090, Qwen3.6-27B
# weights: Q4_K 17408x5120 64/69/83/106 us at M=1/2/3/4 vs MMQ 95-97 us; Q6_K and
# Q8_0 stay ahead to M=4, Q4_K/Q5_K cross over between 3 and 4.
_MMVQ_MULTI_MAX_ROWS = 3


class GGUFW8A8ConfigError(ValueError):
    """VLLM_GGUF_W8A8 / VLLM_GGUF_W8A8_TYPES hold an unsupported value."""


_W8A8_GROUPS = (128, 256)
_W8A8_DEFAULT_GROUP = 128


def _parse_w8a8_group(raw: str) -> int:
    """VLLM_GGUF_W8A8: unset / "0" -> 0 (off), "1" -> the default group, or a group
    size (K values per int8 scale) from _W8A8_GROUPS."""
    raw = raw.strip()
    if raw in ("", "0"):
        return 0
    if raw == "1":
        return _W8A8_DEFAULT_GROUP
    if raw.isdigit() and int(raw) in _W8A8_GROUPS:
        return int(raw)
    raise GGUFW8A8ConfigError(
        f"VLLM_GGUF_W8A8={raw!r}: expected 0, 1 or one of {_W8A8_GROUPS}"
    )


# Weight types the W8A8 kernel can decode, and the default subset it is used for.
# Re-quantising Q6_K / Q8_0 to int8 adds a second rounding that MMQ (which keeps
# their weights exact) does not: on real Qwen3.6-27B prefill activations W8A8 at
# G=128 is ~1.6e-2 relative error there vs ~0.9e-2 for MMQ, while on Q4_K / Q5_K
# it is below MMQ (1.4e-2 vs 1.7e-2 median). End to end, all four types moved
# greedy replies at non-tie margins (up to 0.875 nats), so by default only Q4_K
# and Q5_K -- 79% of the model's GEMM FLOPs -- take W8A8.
_W8A8_TYPE_IDS = {
    "Q8_0": int(WeightType.Q8_0),
    "Q4_K": int(WeightType.Q4_K),
    "Q5_K": int(WeightType.Q5_K),
    "Q6_K": int(WeightType.Q6_K),
}
_W8A8_DEFAULT_TYPES = "Q4_K,Q5_K"


def _parse_w8a8_types(raw: str) -> frozenset[int]:
    """VLLM_GGUF_W8A8_TYPES: comma-separated GGUF type names (default Q4_K,Q5_K)."""
    names = [n.strip().upper() for n in (raw or _W8A8_DEFAULT_TYPES).split(",") if n.strip()]
    unknown = [n for n in names if n not in _W8A8_TYPE_IDS]
    if unknown or not names:
        raise GGUFW8A8ConfigError(
            f"VLLM_GGUF_W8A8_TYPES={raw!r}: expected names from {sorted(_W8A8_TYPE_IDS)}"
        )
    return frozenset(_W8A8_TYPE_IDS[n] for n in names)


# Group-scaled int8 tensor-core GEMM for prefill (csrc/gguf/w8a8). With it on, GEMMs
# that would take the dequantize+cuBLAS path (M >= VLLM_GGUF_MMQ_SAFE) instead
# re-quantise the GGUF weight and the activations to int8 with one fp32 scale per
# (row, group of K values) and run on the int8 MMA, which has twice the bf16
# tensor-core rate on sm_86. Off by default: it trades accuracy (int8 activations,
# the MMQ numerics class decode already uses) for prefill speed.
_W8A8_GROUP = _parse_w8a8_group(os.environ.get("VLLM_GGUF_W8A8", ""))
_W8A8_TYPES = _parse_w8a8_types(os.environ.get("VLLM_GGUF_W8A8_TYPES", ""))


@functools.cache
def _mma_capable(device_index: int) -> bool:
    """Turing or newer: the MMA kernels' own condition (turing_mma_available)."""
    return torch.cuda.get_device_capability(device_index) >= (7, 5)


def _mmvq_max_rows(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int) -> int:
    """Largest batch the vector kernel (MMVQ) takes.

    The plugin's MMVQ handles one activation row per pass, so its cost grows linearly
    with the batch. The tensor-core MMQ kernel reads the weight once for up to 128 rows:
    on an RTX 3090 with Qwen3.6-27B weights it is already faster at two rows (Q4_K
    17408x5120: 98 us vs 135 us at M=2, 99 us vs 512 us at M=8), so with it enabled only
    single-row GEMMs keep MMVQ -- or up to three rows with the multi-column MMVQ.
    """
    if (
        _MMA_MMQ
        and qweight_type in MMQ_QUANT_TYPES
        and x.is_cuda
        and _mma_capable(x.device.index if x.device.index is not None else 0)
    ):
        return _MMVQ_MULTI_MAX_ROWS if _MMVQ_MULTI else 1
    if qweight_type in IMATRIX_QUANT_TYPES:
        return 8 if qweight.shape[0] > 5120 else 16
    return 2 if qweight.shape[0] > 5120 else 6


def _dequant_gemm(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    """Dequantize `qweight` to `x.dtype`, then a plain (tensor-core) GEMM."""
    block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
    shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
    weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
    return x @ weight.T


def _prefer_dequant_gemm(x: torch.Tensor, qweight_type: int) -> bool:
    """True when this GEMM is large enough that MMQ is the slower choice."""
    return (
        _MMQ_SAFE > 0
        and x.shape[0] >= _MMQ_SAFE
        and qweight_type in DEQUANT_TYPES
    )


def _use_w8a8(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int) -> bool:
    """True when the W8A8 GEMM is enabled and computes this (prefill-sized) GEMM."""
    if not _W8A8_GROUP or qweight_type not in _W8A8_TYPES:
        return False
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16):
        return False
    if not x.is_contiguous() or not ops.w8a8_available():
        return False
    block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
    cols = qweight.shape[1] // type_size * block_size
    return cols == x.shape[1] and ops.w8a8_supported(
        qweight_type, qweight.shape[0], cols, _W8A8_GROUP
    )


def _fused_mul_mat_gguf(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    mmvq_safe = _mmvq_max_rows(x, qweight, qweight_type)
    if x.shape[0] == 0:
        return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T
    if x.shape[0] <= mmvq_safe and qweight_type in MMVQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    elif _prefer_dequant_gemm(x, qweight_type):
        if _use_w8a8(x, qweight, qweight_type):
            y = ops.ggml_mul_mat_w8a8(
                qweight, x, qweight_type, qweight.shape[0], _W8A8_GROUP
            )
        else:
            y = _dequant_gemm(x, qweight, qweight_type)
    elif qweight_type in MMQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    elif qweight_type in DEQUANT_TYPES:
        y = _dequant_gemm(x, qweight, qweight_type)
    else:
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")
    return y


def _fused_mul_mat_gguf_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
) -> torch.Tensor:
    return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)


try:
    direct_register_custom_op(
        op_name="_fused_mul_mat_gguf",
        op_func=_fused_mul_mat_gguf,
        fake_impl=_fused_mul_mat_gguf_fake,
    )
    fused_mul_mat_gguf = torch.ops.vllm._fused_mul_mat_gguf
except AttributeError as error:
    raise error


@register_weight_loader_v2_supported_method
class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF."""

    # Store merged layers whose shards have different quant types (Qwen3.5's
    # in_proj_qkvz: Q6_K/Q4_K qkv + Q5_K z; attention qkv: Q4_K q, k + Q6_K v) as one
    # contiguous weight per run of same-typed shards, instead of slicing and copying
    # the padded merged weight on every call. Subclasses with their own apply() that
    # reads the padded weight turn it off.
    split_mixed_shards = True

    def __init__(
        self,
        quant_config,
        layout: GGUFLinearLayout | None = None,
    ) -> None:
        self.quant_config = quant_config
        self.layout = layout

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del output_size
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        fallback_weight_loader = extra_weight_attrs.pop("weight_loader", None)
        weight_loader = _resolve_gguf_weight_loader(layer, fallback_weight_loader)
        assert weight_loader is not None

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "weight_loader": weight_loader,
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        weight_loader_type = _resolve_gguf_weight_type_loader(
            layer, fallback_weight_loader
        )
        assert weight_loader_type is not None
        qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            qweight_type,
            {
                "weight_loader": weight_loader_type,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": len(output_partition_sizes),
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

        if self.layout is not None:
            set_weight_attrs(
                qweight,
                {
                    "gguf_layout": self.layout,
                    "gguf_logical_input_size": input_size,
                    "gguf_weight_type_parameter": qweight_type,
                },
            )

    def process_weights_after_loading(self, layer: torch.nn.Module):
        self._materialize_gguf_parameters(layer)
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            qweight_type = WeightType(qweight_type)
            raise ValueError(
                f"Unsupported GGUF quantization type {qweight_type} in layer {layer}."
            )
        self._create_padded_weight_param(layer)
        if self.split_mixed_shards:
            self._split_mixed_shard_groups(layer)

    def _split_mixed_shard_groups(self, layer: torch.nn.Module) -> None:
        """Mixed-type merged weight -> `qweight_group<i>` parameters, one per run of
        consecutive same-typed shards, in output order; `layer.gguf_shard_groups` names
        them with their types. The padded merged data is released."""
        qweight = layer.qweight
        shard_ids = list(qweight.shard_id)
        if not shard_ids:
            return
        order = ["q", "k", "v"] if "q" in shard_ids else shard_ids
        fallback = layer.qweight_type.weight_type
        types = [layer.qweight_type.shard_weight_type.get(i, fallback) for i in order]
        if len(set(types)) == 1:
            return
        runs: list[list[int]] = []  # [start, end, width, type]
        for idx, wtype in zip(order, types):
            start, end, width = qweight.shard_offset_map[idx]
            last = runs[-1] if runs else None
            if last and last[3] == wtype and last[1] == start and last[2] == width:
                last[1] = end
            else:
                runs.append([start, end, width, wtype])
        groups = []
        for i, (start, end, width, wtype) in enumerate(runs):
            name = f"qweight_group{i}"
            # clone, not contiguous(): a slice that is already contiguous would be a view
            # keeping the whole padded merged weight alive.
            data = qweight.data[start:end, :width].clone()
            layer.register_parameter(name, torch.nn.Parameter(data, requires_grad=False))
            groups.append((name, int(wtype)))
        layer.gguf_shard_groups = tuple(groups)
        qweight.data = torch.empty(0, dtype=qweight.dtype, device=qweight.device)

    def _materialize_gguf_parameters(self, layer: torch.nn.Module) -> None:
        self._materialize_qweight(layer)
        self._materialize_qweight_type(layer)

    def _materialize_qweight(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_parameter(layer, "qweight")

    def _materialize_qweight_type(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_type_parameter(layer, "qweight_type")

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1, ValueError(
                f"Data container has mixed dtypes: {dtype}"
            )
            dtype = next(iter(dtype))
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            shard_offset_map = dict[str, tuple[int, int, int]]()
            ordered_shard_ids = _gguf_ordered_shard_ids(shard_id)
            current_offset = 0
            for idx in ordered_shard_ids:
                id_in_container = shard_id_map[idx]
                start = current_offset
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
                current_offset = end
            padded_param = GGUFWeightParameter(
                data=padded_data,
                weight_loader=qweight.weight_loader,
                input_dim=qweight.input_dim,
                output_dim=qweight.output_dim,
                tensor_shape=qweight.tensor_shape,
            )
            padded_param.data_container = []
            padded_param.shard_id = ordered_shard_ids
            padded_param.shard_id_map = dict(qweight.shard_id_map)
            if hasattr(qweight, "ignore_warning"):
                padded_param.ignore_warning = qweight.ignore_warning
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            qweight.data_container.clear()
            qweight.shard_id.clear()
            qweight.shard_id_map.clear()
            if qweight.data.numel() > 0:
                qweight.data = torch.empty(
                    0, dtype=qweight.dtype, device=qweight.device
                )
            layer.register_parameter("qweight", padded_param)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from . import fused_mul_mat_gguf as fused_mul_mat_gguf_op

        if self.layout is not None:
            x = self.layout.input_to_gguf(x)

        groups = getattr(layer, "gguf_shard_groups", None)
        if groups is not None:
            out = torch.cat(
                [fused_mul_mat_gguf_op(x, getattr(layer, name), wtype) for name, wtype in groups],
                dim=1,
            )
            if bias is not None:
                out.add_(bias)
            return out

        shard_id = layer.qweight.shard_id
        if shard_id:
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            qweight = layer.qweight
            fallback_wtype = layer.qweight_type.weight_type
            shard_weight_types = [
                layer.qweight_type.shard_weight_type.get(idx, fallback_wtype)
                for idx in shard_id
            ]
            if len(set(shard_weight_types)) == 1:
                out = fused_mul_mat_gguf_op(x, qweight, shard_weight_types[0])
                if bias is not None:
                    out.add_(bias)
                return out
            result = []
            for idx in shard_id:
                start, end, offset = layer.qweight.shard_offset_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type.get(
                    idx, fallback_wtype
                )
                result.append(
                    fused_mul_mat_gguf_op(
                        x, qweight[start:end, :offset].contiguous(), qweight_type
                    )
                )
            out = torch.cat(result, axis=1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            out = fused_mul_mat_gguf_op(x, qweight, qweight_type)
        if bias is not None:
            out.add_(bias)
        return out
