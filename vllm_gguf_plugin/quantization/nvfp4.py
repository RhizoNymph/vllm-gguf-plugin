# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ggml NVFP4, decoded in torch (the reference for the Triton kernels).

A block is 64 values in 36 bytes: four UE4M3 scales (one per 16 values),
then 32 bytes of E2M1 codes. Sub-block ``s`` uses bytes ``4 + 8s .. 11 + 8s``:
value ``16s + j`` is the low nibble of byte ``j`` (j < 8) or the high nibble
of byte ``j - 8``. A value is ``e2m1(code) * ue4m3(scale)``; ggml decodes
the scale codes 0x00 and 0x7F as 0. A tensor may carry a separate F32
``.scale`` (per expert for routed experts) that multiplies the matmul output;
it is not part of the blocks.
"""

from __future__ import annotations

import torch

NVFP4_BLOCK_VALUES = 64
NVFP4_BLOCK_BYTES = 36
NVFP4_SUB_BLOCK = 16

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0) + tuple(
    -v for v in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
)


def ue4m3_to_float(codes: torch.Tensor) -> torch.Tensor:
    """Decode unsigned E4M3 scale codes (bias 7) to float32."""
    codes = codes.to(torch.int32)
    exponent = (codes >> 3) & 0xF
    mantissa = (codes & 0x7).to(torch.float32)
    normal = (1.0 + mantissa / 8.0) * torch.exp2((exponent - 7).to(torch.float32))
    value = torch.where(exponent == 0, mantissa * 2.0**-9, normal)
    return torch.where((codes == 0) | (codes == 0x7F), torch.zeros_like(value), value)


def dequantize_nvfp4(data: torch.Tensor) -> torch.Tensor:
    """Decode packed NVFP4 rows ``[..., blocks * 36]`` (uint8) to float32
    ``[..., blocks * 64]``."""
    if data.dtype is not torch.uint8:
        raise TypeError(f"NVFP4 data must be torch.uint8, got {data.dtype}")
    row_bytes = data.shape[-1]
    if row_bytes % NVFP4_BLOCK_BYTES:
        raise ValueError(
            f"NVFP4 row of {row_bytes} bytes is not a multiple of "
            f"{NVFP4_BLOCK_BYTES}-byte blocks"
        )
    lead = data.shape[:-1]
    blocks = data.reshape(-1, NVFP4_BLOCK_BYTES)
    scale = ue4m3_to_float(blocks[:, :4]).reshape(-1, 4, 1)
    qs = blocks[:, 4:].reshape(-1, 4, 8)
    codes = torch.cat([qs & 0x0F, qs >> 4], dim=-1).to(torch.int64)
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=data.device)
    values = table[codes] * scale
    return values.reshape(*lead, row_bytes // NVFP4_BLOCK_BYTES * NVFP4_BLOCK_VALUES)
