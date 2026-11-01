# SPDX-License-Identifier: Apache-2.0

import functools
import os

import torch

from .triton.dequantize.interface import ggml_dequantize_triton
from .triton.fused_moe.interface import ggml_moe_a8_triton
from .triton.fused_moe.utils import get_triton_moe_block_m
from .triton.gemm.interface import ggml_mul_mat_a8_triton
from .triton.gemm.utils import (
    GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ4_XS,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
)

try:
    from torch.library import register_fake
except ImportError:
    from torch.library import impl_abstract as register_fake

# Backend selection: use CUDA kernels by default, unless explicitly disabled.
_USE_CUDA = os.environ.get("VLLM_GGUF_USE_CUDA", "1") == "1"

# Try importing CUDA extension
try:
    from . import _C_gguf  # noqa: F401

    _CUDA_AVAILABLE = True
except ImportError:
    _C_gguf = None
    _CUDA_AVAILABLE = False


# Effective CUDA usage: only when enabled AND available.
_CUDA_ENABLED = _USE_CUDA and _CUDA_AVAILABLE

_CUDA_GEMV_QUANT_TYPES = frozenset(
    {
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
        GGML_TYPE_IQ2_XXS,
        GGML_TYPE_IQ2_XS,
        GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ1_S,
        GGML_TYPE_IQ4_NL,
        GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ4_XS,
        GGML_TYPE_IQ1_M,
    }
)
_CUDA_GEMM_QUANT_TYPES = frozenset(
    {
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
    }
)


def _cuda_kernel_available(op_name: str, quant_type: int | None = None) -> bool:
    if not _CUDA_ENABLED:
        return False
    namespace = getattr(torch.ops, "_C_gguf", None)
    if namespace is None or not hasattr(namespace, op_name):
        return False
    if quant_type is None:
        return True
    return int(quant_type) in _CUDA_GEMV_QUANT_TYPES


def _cuda_gemm_kernel_available(op_name: str, quant_type: int) -> bool:
    return _cuda_kernel_available(op_name) and int(quant_type) in _CUDA_GEMM_QUANT_TYPES


# --- Fake implementations for CUDA custom ops (needed for torch.compile) ---

if (
    _CUDA_AVAILABLE
    and hasattr(torch.ops, "_C_gguf")
    and hasattr(torch.ops._C_gguf, "ggml_dequantize")
):

    @register_fake("_C_gguf::ggml_dequantize")
    def _ggml_dequantize_fake(
        W: torch.Tensor,
        quant_type: int,
        m: torch.SymInt,
        n: torch.SymInt,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return torch.empty((m, n), dtype=dtype or torch.float16, device=W.device)

    @register_fake("_C_gguf::ggml_mul_mat_vec_a8")
    def _ggml_mul_mat_vec_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.shape[0], row), dtype=X.dtype, device=W.device)

    @register_fake("_C_gguf::ggml_mul_mat_a8")
    def _ggml_mul_mat_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.size(0), row), dtype=X.dtype, device=W.device)

    if hasattr(torch.ops._C_gguf, "ggml_mul_mat_w8a8"):

        @register_fake("_C_gguf::ggml_mul_mat_w8a8")
        def _ggml_mul_mat_w8a8_fake(
            W: torch.Tensor,
            X: torch.Tensor,
            quant_type: int,
            row: torch.SymInt,
            group: int,
        ) -> torch.Tensor:
            return torch.empty((X.size(0), row), dtype=X.dtype, device=W.device)

    @register_fake("_C_gguf::ggml_moe_a8")
    def _ggml_moe_a8_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
        top_k: torch.SymInt,
        tokens: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.size(0) * top_k, row), dtype=X.dtype, device=W.device)


if (
    _CUDA_AVAILABLE
    and hasattr(torch.ops, "_C_gguf")
    and hasattr(torch.ops._C_gguf, "ggml_moe_a8_vec")
):

    @register_fake("_C_gguf::ggml_moe_a8_vec")
    def _ggml_moe_a8_vec_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        quant_type: int,
        row: torch.SymInt,
        tokens: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.size(0) * top_k, row), dtype=X.dtype, device=W.device)


# --- Public API ---


def ggml_dequantize(
    W: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None
) -> torch.Tensor:
    if _cuda_kernel_available("ggml_dequantize", quant_type):
        return torch.ops._C_gguf.ggml_dequantize(W, quant_type, m, n, dtype)
    return ggml_dequantize_triton(W, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    if _cuda_kernel_available("ggml_mul_mat_vec_a8", quant_type):
        return torch.ops._C_gguf.ggml_mul_mat_vec_a8(W, X, quant_type, row)
    return ggml_mul_mat_a8_triton(W, X, quant_type, row)


def ggml_mul_mat_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    if _cuda_gemm_kernel_available("ggml_mul_mat_a8", quant_type):
        return torch.ops._C_gguf.ggml_mul_mat_a8(W, X, quant_type, row)
    return ggml_mul_mat_a8_triton(W, X, quant_type, row)


def w8a8_available() -> bool:
    """True when the extension has the W8A8 prefill GEMM (csrc/gguf/w8a8)."""
    return _cuda_kernel_available("ggml_mul_mat_w8a8")


@functools.cache
def w8a8_supported(quant_type: int, row: int, col: int, group: int) -> bool:
    """True when the W8A8 GEMM computes this weight type, shape and group (cached:
    asked on every prefill GEMM)."""
    return torch.ops._C_gguf.w8a8_supported(int(quant_type), int(row), int(col), int(group))


def ggml_mul_mat_w8a8(
    W: torch.Tensor, X: torch.Tensor, quant_type: int, row: int, group: int
) -> torch.Tensor:
    """X @ dequant(W).T through the group-scaled int8 tensor-core GEMM: W re-quantised
    and X quantised to int8 with one scale per (row, `group` K values)."""
    return torch.ops._C_gguf.ggml_mul_mat_w8a8(W, X, int(quant_type), row, group)


def w8a8_quantize_act(X: torch.Tensor, aq: torch.Tensor, sa: torch.Tensor, group: int) -> None:
    """Stage 1 of ggml_mul_mat_w8a8, into caller-provided outputs (tests, timing)."""
    torch.ops._C_gguf.w8a8_quantize_act(X, aq, sa, group)


def w8a8_requant_weight(
    W: torch.Tensor, bq: torch.Tensor, sb: torch.Tensor, quant_type: int, group: int
) -> None:
    """Stage 2 of ggml_mul_mat_w8a8: GGUF blocks -> int8 [N, K] + scales [K / G, N]."""
    torch.ops._C_gguf.w8a8_requant_weight(W, bq, sb, int(quant_type), group)


def w8a8_gemm(
    aq: torch.Tensor, sa: torch.Tensor, bq: torch.Tensor, sb: torch.Tensor,
    Y: torch.Tensor, group: int,
) -> None:
    """Stage 3 of ggml_mul_mat_w8a8: the group-scaled int8 GEMM into Y."""
    torch.ops._C_gguf.w8a8_gemm(aq, sa, bq, sb, Y, group)


def ggml_moe_a8(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    if _cuda_gemm_kernel_available("ggml_moe_a8", quant_type):
        return torch.ops._C_gguf.ggml_moe_a8(
            X,
            W,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            quant_type,
            row,
            top_k,
            tokens,
        )
    return ggml_moe_a8_triton(
        X,
        W,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        quant_type,
        row,
        top_k,
        tokens,
    )


def ggml_moe_a8_vec(
    X: torch.Tensor,
    W: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
) -> torch.Tensor:
    if _cuda_kernel_available("ggml_moe_a8_vec", quant_type):
        return torch.ops._C_gguf.ggml_moe_a8_vec(
            X, W, topk_ids, top_k, quant_type, row, tokens
        )
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    E = W.shape[0]
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, get_triton_moe_block_m(quant_type), E
    )
    return ggml_moe_a8_triton(
        X,
        W,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        quant_type,
        row,
        top_k,
        tokens,
    )


def ggml_moe_get_block_size(quant_type: int) -> int:
    if _cuda_gemm_kernel_available("ggml_moe_get_block_size", quant_type):
        return torch.ops._C_gguf.ggml_moe_get_block_size(quant_type)
    return get_triton_moe_block_m(quant_type)


def moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    torch.ops._moe_C.moe_sum(input, output)
