# SPDX-License-Identifier: Apache-2.0
"""Fused MoE GEMM over ggml NVFP4 experts.

Each 64-value block is decoded in registers as ``(sub-block, byte, nibble)``
(the order ``tl.join(low, high)`` produces); the activations are gathered in
that same order, so no shuffle of the decoded weights is needed. The
per-expert ``.scale`` is applied by the caller on the output rows.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...fp4_tables import get_e2m1_table, ue4m3_to_f32
from ...gemm.utils import GGML_TYPE_NVFP4
from ..utils import load_moe_token_info, run_triton_fused_moe_kernel


@triton.jit
def nvfp4_moe_kernel(
    x_ptr,
    w_u8_ptr,
    y_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    num_valid_tokens,
    top_k,
    n,
    num_k_blocks,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    values_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_output, offs_token, token_mask = load_moe_token_info(
        sorted_token_ids_ptr, pid_m, top_k, num_valid_tokens, BLOCK_M=BLOCK_M
    )
    expert = tl.load(expert_ids_ptr + pid_m)
    if expert < 0:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < n
    offs_kb = tl.arange(0, BLOCK_K_BLOCKS)
    offs_sub = tl.arange(0, 4)
    offs_byte = tl.arange(0, 8)
    offs_half = tl.arange(0, 2)
    # Element of a block at decoded position (sub, byte, half).
    elem = tl.reshape(
        offs_sub[:, None, None] * 16
        + offs_half[None, None, :] * 8
        + offs_byte[None, :, None],
        (64,),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n[:, None] * stride_wn
    x_row_ptrs = x_ptr + offs_token[:, None] * stride_xm

    for kb_start in range(0, num_k_blocks, BLOCK_K_BLOCKS):
        cur_kb = kb_start + offs_kb
        kb_mask = cur_kb < num_k_blocks
        x_k = tl.reshape(cur_kb[:, None] * 64 + elem[None, :], (BLOCK_K_BLOCKS * 64,))
        x_k_mask = tl.reshape(
            kb_mask[:, None] & (elem[None, :] >= 0), (BLOCK_K_BLOCKS * 64,)
        )
        x_tile = tl.load(
            x_row_ptrs + x_k[None, :] * stride_xk,
            mask=token_mask[:, None] & x_k_mask[None, :],
            other=0.0,
        )
        x_dtype = x_tile.dtype

        block_ptrs = w_row_ptrs + cur_kb[None, :] * 36
        wmask = n_mask[:, None] & kb_mask[None, :]
        scale = ue4m3_to_f32(
            tl.load(
                block_ptrs[:, :, None] + offs_sub[None, None, :],
                mask=wmask[:, :, None],
                other=0,
            )
        ).to(x_dtype)
        packed = tl.load(
            block_ptrs[:, :, None, None]
            + 4
            + offs_sub[None, None, :, None] * 8
            + offs_byte[None, None, None, :],
            mask=wmask[:, :, None, None],
            other=0,
        )
        low = tl.load(values_ptr + (packed & 0x0F).to(tl.int32)).to(x_dtype)
        high = tl.load(values_ptr + ((packed >> 4) & 0x0F).to(tl.int32)).to(x_dtype)
        low = low * scale[:, :, :, None]
        high = high * scale[:, :, :, None]
        q_tile = tl.reshape(tl.join(low, high), (BLOCK_N, BLOCK_K_BLOCKS * 64))
        acc = tl.dot(x_tile, tl.trans(q_tile), acc=acc)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=token_mask[:, None] & n_mask[None, :])


def ggml_moe_nvfp4_triton(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    return run_triton_fused_moe_kernel(
        nvfp4_moe_kernel,
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_NVFP4,
        extra_args=(get_e2m1_table(W.device),),
        block_k_blocks=2,
    )
