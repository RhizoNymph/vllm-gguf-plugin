# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ggml tensor types newer than the installed gguf-py.

ggml added Q1_0 (41), Q2_0 (42) and F8_E4M3 (43) after the last gguf-py
release (0.19.0). ``GGUFReader`` resolves every tensor's type through
``GGMLQuantizationType(raw)`` and raises on an unknown value, so a single
Q2_0 tensor makes the whole file unreadable — including its metadata.

``ensure_new_ggml_types`` registers the missing members (and their block
sizes) on gguf-py's own enum, so the reader, ``GGML_QUANT_SIZES`` lookups
and name lookups all see them. It is a no-op once gguf-py ships them.

The plugin has no Q2_0 kernels. ``transcode_q2_0_to_q4_0`` rewrites Q2_0
blocks as Q4_0, which every GGUF linear and MoE path supports: Q2_0 stores
``(q - 1) * d`` with ``q`` in 0..3 and one fp16 ``d`` per 64 values, and
Q4_0 stores ``(q - 8) * d`` with ``q`` in 0..15 and one fp16 ``d`` per 32,
so ``q4 = q2 + 7`` with the same ``d`` reproduces every value bit for bit,
at twice the storage.

ggml's MXFP4 has no plugin kernels either. ``transcode_mxfp4_to_q5_0``
rewrites it as Q5_0: an MXFP4 block holds ``k * 2**(e - 128)`` (``k`` the
doubled E2M1 code, -12..12; ``e`` the E8M0 byte), and Q5_0 holds
``(q - 16) * d``, so ``q = k + 16`` with ``d = 2**(e - 128)`` is exact
whenever that power of two is an fp16 value (``e`` in 104..143), at 22
bytes per 32 values instead of 17.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType


class _NewType(NamedTuple):
    name: str
    value: int
    block_size: int
    type_size: int


# Values and block geometry from ggml/include/ggml.h and gguf-py/constants.py
# on llama.cpp master.
_NEW_TYPES = (
    _NewType("Q1_0", 41, 128, 2 + 16),
    _NewType("Q2_0", 42, 64, 2 + 16),
    _NewType("F8_E4M3", 43, 1, 1),
)

_Q2_0_BLOCK_BYTES = 18
_Q4_0_BLOCK_BYTES = 18
_MXFP4_BLOCK_BYTES = 17
_Q5_0_BLOCK_BYTES = 22
# E8M0 exponents whose scale 2**(e - 128) is an fp16 value (subnormals
# included: 2**-24 is the smallest).
_MXFP4_FP16_EXPONENTS = range(104, 144)
_E2M1_DOUBLED = np.array(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int8
)


def _register(new: _NewType) -> GGMLQuantizationType:
    existing = GGMLQuantizationType._value2member_map_.get(new.value)
    if existing is not None:
        if existing.name != new.name:
            raise RuntimeError(
                f"gguf-py maps ggml type {new.value} to {existing.name}, "
                f"expected {new.name}"
            )
        member = existing
    else:
        member = int.__new__(GGMLQuantizationType, new.value)
        member._name_ = new.name
        member._value_ = new.value
        # Attribute access first: EnumType refuses to set a name that is
        # already in the member map.
        setattr(GGMLQuantizationType, new.name, member)
        GGMLQuantizationType._value2member_map_[new.value] = member
        GGMLQuantizationType._member_map_[new.name] = member
        GGMLQuantizationType._member_names_.append(new.name)
    GGML_QUANT_SIZES.setdefault(member, (new.block_size, new.type_size))
    return member


def ensure_new_ggml_types() -> None:
    """Register ggml tensor types the installed gguf-py does not know."""
    for new in _NEW_TYPES:
        _register(new)


ensure_new_ggml_types()

Q1_0 = GGMLQuantizationType(41)
Q2_0 = GGMLQuantizationType(42)
F8_E4M3 = GGMLQuantizationType(43)


def transcode_q2_0_to_q4_0(data: np.ndarray) -> np.ndarray:
    """Rewrite packed Q2_0 rows as Q4_0 rows holding identical values.

    ``data`` is a uint8 array whose last axis is a whole number of Q2_0
    blocks; leading axes (rows, experts) are preserved.
    """
    data = np.ascontiguousarray(data, dtype=np.uint8)
    row_bytes = data.shape[-1]
    if row_bytes % _Q2_0_BLOCK_BYTES:
        raise ValueError(
            f"Q2_0 row of {row_bytes} bytes is not a multiple of "
            f"{_Q2_0_BLOCK_BYTES}-byte blocks"
        )
    lead = data.shape[:-1]
    blocks = data.reshape(-1, _Q2_0_BLOCK_BYTES)
    scale = blocks[:, :2]
    # Element j of a Q2_0 block sits in byte j // 4 at bit (j % 4) * 2.
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    q2 = (blocks[:, 2:, None] >> shifts) & 0x03
    q4 = q2.reshape(-1, 2, 32) + np.uint8(7)
    # Q4_0 element j sits in the low nibble of byte j (j < 16) or the high
    # nibble of byte j - 16.
    packed = q4[..., :16] | (q4[..., 16:] << 4)
    out = np.concatenate(
        [np.broadcast_to(scale[:, None, :], (scale.shape[0], 2, 2)), packed],
        axis=-1,
    )
    assert out.shape[-1] == _Q4_0_BLOCK_BYTES
    return out.reshape(*lead, row_bytes * 2)


class MXFP4ScaleOutOfRange(ValueError):
    """An MXFP4 block scale has no exact fp16 (Q5_0) counterpart."""

    def __init__(self, exponents: tuple[int, ...]) -> None:
        self.exponents = exponents
        super().__init__(
            "MXFP4 block scale(s) 2**(e - 128) outside the fp16 range for "
            f"e = {list(exponents)}; Q5_0 cannot hold these blocks exactly"
        )


def transcode_mxfp4_to_q5_0(data: np.ndarray) -> np.ndarray:
    """Rewrite packed MXFP4 rows as Q5_0 rows holding identical values.

    ``data`` is a uint8 array whose last axis is a whole number of MXFP4
    blocks; leading axes (rows, experts) are preserved. Raises
    :class:`MXFP4ScaleOutOfRange` if a block scale is not an fp16 value.
    """
    data = np.ascontiguousarray(data, dtype=np.uint8)
    row_bytes = data.shape[-1]
    if row_bytes % _MXFP4_BLOCK_BYTES:
        raise ValueError(
            f"MXFP4 row of {row_bytes} bytes is not a multiple of "
            f"{_MXFP4_BLOCK_BYTES}-byte blocks"
        )
    lead = data.shape[:-1]
    blocks = data.reshape(-1, _MXFP4_BLOCK_BYTES)
    e = blocks[:, 0].astype(np.int32)
    bad = ~((e >= _MXFP4_FP16_EXPONENTS.start) & (e < _MXFP4_FP16_EXPONENTS.stop))
    if bad.any():
        raise MXFP4ScaleOutOfRange(tuple(sorted({int(x) for x in e[bad]})))
    d = np.ldexp(np.float32(1.0), e - 128).astype(np.float16).view(np.uint8)
    qs = blocks[:, 1:]
    # Element j of both formats sits in the low nibble of byte j (j < 16) or
    # the high nibble of byte j - 16; Q5_0 keeps the fifth bit in qh.
    codes = np.concatenate([qs & 0x0F, qs >> 4], axis=-1)
    q5 = (_E2M1_DOUBLED[codes].astype(np.int16) + 16).astype(np.uint8)
    qh = np.packbits(q5 >> 4, axis=-1, bitorder="little")
    packed = (q5[:, :16] & 0x0F) | (q5[:, 16:] << 4)
    out = np.concatenate([d.reshape(-1, 2), qh, packed], axis=-1)
    assert out.shape[-1] == _Q5_0_BLOCK_BYTES
    return out.reshape(*lead, row_bytes // _MXFP4_BLOCK_BYTES * _Q5_0_BLOCK_BYTES)
