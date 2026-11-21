# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 pieces of the plugin's CUDA headers agree with the vendored
llama.cpp copy and with the torch decoder.

``csrc/gguf/ggml-common.h`` cannot include ``llamacpp/ggml-common.h`` (the two
redefine the same block structs), so the NVFP4 block and the doubled E2M1
table are copied; these checks keep the copies honest without a GPU.
"""

import re
from pathlib import Path

import pytest

from vllm_gguf_plugin.quantization.nvfp4 import (
    E2M1_VALUES,
    NVFP4_BLOCK_BYTES,
    NVFP4_BLOCK_VALUES,
    NVFP4_SUB_BLOCK,
)

pytestmark = [pytest.mark.cpu]

GGUF_CSRC = Path(__file__).resolve().parents[1] / "vllm_gguf_plugin" / "csrc" / "gguf"
PLUGIN_COMMON = GGUF_CSRC / "ggml-common.h"
VENDORED_COMMON = GGUF_CSRC / "llamacpp" / "ggml-common.h"


def _int_table(text: str, name: str) -> list[int]:
    match = re.search(
        rf"int8_t\W[^;{{]*?\b{name}\s*(?:\[16\]\s*=\s*\{{|,\s*16\))(.*?)(?:\}};|GGML_TABLE_END)",
        text,
        re.S,
    )
    assert match, name
    return [int(v) for v in re.findall(r"-?\d+", match.group(1))]


def _define(text: str, name: str) -> str:
    match = re.search(rf"#define\s+{name}\s+(.+)", text)
    assert match, name
    return match.group(1).split("//")[0].strip()


def test_e2m1_table_matches_vendored_and_torch():
    plugin = _int_table(PLUGIN_COMMON.read_text(), "kvalues_mxfp4")
    vendored = _int_table(VENDORED_COMMON.read_text(), "kvalues_mxfp4")
    assert plugin == vendored
    # The table holds 2 * E2M1; the UE4M3 decode halves the scale to match.
    assert [v / 2 for v in plugin] == list(E2M1_VALUES)


def test_nvfp4_block_geometry_matches():
    plugin = PLUGIN_COMMON.read_text()
    vendored = VENDORED_COMMON.read_text()
    for name in ("QK_NVFP4", "QK_NVFP4_SUB", "QR_NVFP4", "QI_NVFP4"):
        assert _define(plugin, name) == _define(vendored, name), name
    assert int(_define(plugin, "QK_NVFP4")) == NVFP4_BLOCK_VALUES
    assert int(_define(plugin, "QK_NVFP4_SUB")) == NVFP4_SUB_BLOCK
    assert re.search(rf"sizeof\(block_nvfp4\)\s*==\s*{NVFP4_BLOCK_BYTES}\b", plugin)
