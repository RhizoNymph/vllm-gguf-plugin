# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF-packed storage for Qwen4Exp's PLE n-gram table.

vLLM picks the table's storage format in
``Qwen4ExpPLEEmbeddingMethod.from_quant_config``, which knows FP8, NVFP4 and
unquantized tables and falls back to unquantized for any other quant config.
For Qwen3.8-Flash-Next that fallback is 320M x 160 bfloat16 = 102 GB; the
unsloth GGUFs ship the table as IQ4_NL (28.8 GB) or Q8_0/Q5_1.

``GGUFPLEEmbeddingMethod`` keeps the rows packed exactly as the GGUF stores
them, wherever vLLM allocates the table (device, or pinned host with engram
CPU offload), and decodes only the looked-up rows, mirroring vLLM's NVFP4
method. ``install_qwen4exp_ple_support`` routes ``GGUFConfig`` to it.
"""

from __future__ import annotations

import torch
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import dequantize as _dequantize_numpy
from torch import nn
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)

from .. import ops
from .config import GGUFConfig
from .utils import UNQUANTIZED_TYPES

_INSTALLED = False


def _decode_rows(
    rows: torch.Tensor, weight_type: int, dim: int, dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize packed GGUF rows ``[N, row_bytes]`` to ``[N, dim]``."""
    if rows.is_cuda:
        return ops.ggml_dequantize(rows, weight_type, rows.shape[0], dim, dtype)
    values = _dequantize_numpy(rows.numpy(), GGMLQuantizationType(weight_type))
    return torch.from_numpy(values.reshape(rows.shape[0], dim)).to(dtype)


class GGUFPLEEmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """PLE rows stored as packed GGUF blocks, decoded per lookup."""

    def __init__(self, weight_type: int) -> None:
        super().__init__()
        self.weight_type = GGMLQuantizationType(weight_type)
        self.block_size, self.type_size = GGML_QUANT_SIZES[self.weight_type]

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
            Qwen4ExpPLEDiskOffloadEmbedding,
        )

        if isinstance(layer, Qwen4ExpPLEDiskOffloadEmbedding):
            raise NotImplementedError(
                "Engram disk offload reads PLE rows from safetensors shards; a "
                "GGUF table needs cpu_offload (the default) or device storage"
            )
        if input_size_per_partition % self.block_size:
            raise ValueError(
                f"PLE row of {input_size_per_partition} values is not a multiple "
                f"of {self.block_size} ({self.weight_type.name} block size)"
            )
        row_bytes = input_size_per_partition // self.block_size * self.type_size
        weight = ModelWeightParameter(
            data=layer.allocate_embedding_weight(
                sum(output_partition_sizes), row_bytes, torch.uint8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)

    def lookup_dtype(self, layer: nn.Module) -> torch.dtype:
        # Rows are decoded during the lookup, as NVFP4 does.
        return layer.params_dtype

    def _decode_owned(
        self,
        layer: nn.Module,
        weight: torch.Tensor,
        ids: torch.Tensor,
        vocab_start: int,
        vocab_end: int,
    ) -> torch.Tensor:
        ids = ids.reshape(-1)
        owned = (ids >= vocab_start) & (ids < vocab_end)
        local = torch.where(owned, ids - vocab_start, torch.zeros_like(ids))
        rows = weight.index_select(0, local)
        values = _decode_rows(
            rows, self.weight_type, layer.embedding_dim, layer.params_dtype
        )
        return values * owned.unsqueeze(-1).to(values.dtype)

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        # vLLM has already masked the ids to this rank's local rows.
        values = self._decode_owned(
            layer, layer.weight, input_, 0, layer.weight.shape[0]
        )
        return values.reshape(*input_.shape, layer.embedding_dim)

    def lookup_from_pinned(
        self,
        layer: nn.Module,
        ids: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        values = self._decode_owned(
            layer,
            layer._uva_weight,
            ids,
            layer.shard_indices.org_vocab_start_index,
            layer.shard_indices.org_vocab_end_index,
        )
        output.view(-1, layer.embedding_dim).copy_(values)

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        del layer
        # The lookup already decoded the rows to the activation dtype.
        return embeddings.to(output_dtype)


def install_qwen4exp_ple_support() -> None:
    """Route ``GGUFConfig`` PLE tables to :class:`GGUFPLEEmbeddingMethod`."""
    global _INSTALLED
    if _INSTALLED:
        return
    original = Qwen4ExpPLEEmbeddingMethod.from_quant_config

    def from_quant_config(quant_config, prefix, embedding_dtype=None):
        if isinstance(quant_config, GGUFConfig) and embedding_dtype is None:
            weight_type = quant_config.ple_weight_type
            if weight_type is None or weight_type in UNQUANTIZED_TYPES:
                return Qwen4ExpPLEUnquantizedEmbeddingMethod()
            return GGUFPLEEmbeddingMethod(weight_type)
        return original(quant_config, prefix, embedding_dtype)

    Qwen4ExpPLEEmbeddingMethod.from_quant_config = staticmethod(from_quant_config)
    _INSTALLED = True
