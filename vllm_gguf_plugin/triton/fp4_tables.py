# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the ggml NVFP4 Triton kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_E2M1_TABLES: dict[tuple[str, int | None], torch.Tensor] = {}


def get_e2m1_table(device: torch.device) -> torch.Tensor:
    """The 16 E2M1 values as float32 on *device*."""
    key = (device.type, device.index)
    table = _E2M1_TABLES.get(key)
    if table is None:
        # Imported here: quantization/ imports the kernels at package import.
        from ..quantization.nvfp4 import E2M1_VALUES

        table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=device)
        _E2M1_TABLES[key] = table
    return table


@triton.jit
def ue4m3_to_f32(code):
    """Unsigned E4M3 (bias 7) scale codes to float32, exactly; ggml reads
    0x00 and 0x7F as 0."""
    code = code.to(tl.int32)
    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7
    # 2**(e - 7) * (1 + m / 8) as an fp32 bit pattern (exponent field e + 120).
    normal = tl.cast(
        ((exponent + 120) << 23) | (mantissa << 20), tl.float32, bitcast=True
    )
    subnormal = mantissa.to(tl.float32) * 0.001953125  # m * 2**-9
    value = tl.where(exponent == 0, subnormal, normal)
    return tl.where((code == 0) | (code == 0x7F), 0.0, value)
