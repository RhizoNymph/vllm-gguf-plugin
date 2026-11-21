# SPDX-License-Identifier: Apache-2.0
"""ggml NVFP4 -> float (see ``quantization/nvfp4.py`` for the layout)."""

import torch
import triton
import triton.language as tl

from ...fp4_tables import get_e2m1_table, ue4m3_to_f32
from ...gemm.utils import GGML_TYPE_NVFP4
from ..utils import dequant_offsets, run_dequantize_kernel


@triton.jit
def nvfp4_dequantize_kernel(
    w_ptr,
    y_ptr,
    total,
    values_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // 64) * 36
    pos = offs % 64
    sub = pos // 16
    j = pos % 16
    packed = tl.load(block_ptrs + 4 + sub * 8 + (j % 8), mask=mask, other=0)
    code = tl.where(j < 8, packed & 0x0F, (packed >> 4) & 0x0F)
    scale = ue4m3_to_f32(tl.load(block_ptrs + sub, mask=mask, other=0))
    value = tl.load(values_ptr + code.to(tl.int32), mask=mask, other=0.0)
    tl.store(y_ptr + offs, value * scale, mask=mask)


def ggml_dequantize_nvfp4_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return run_dequantize_kernel(
        nvfp4_dequantize_kernel,
        W,
        m,
        n,
        dtype,
        GGML_TYPE_NVFP4,
        extra_args=(get_e2m1_table(W.device),),
    )
