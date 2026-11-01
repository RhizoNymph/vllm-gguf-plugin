# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The vendored llama.cpp multi-column MMVQ (VLLM_GGUF_MMVQ_MULTI=1).

The extension reads the variable once per process, so these tests only run when it is
set: `VLLM_GGUF_MMVQ_MULTI=1 pytest tests/test_mmvq_multi.py`. Each batch size it
serves must agree with the dequantized reference as closely as the single-row MMVQ
does, and with the single-row kernel's result row by row.
"""

import os

import gguf
import pytest
import torch
from gguf import GGMLQuantizationType

from vllm_gguf_plugin import ops

from .utils import get_gguf_sample_tensors, seed_everything

ENABLED = os.environ.get("VLLM_GGUF_MMVQ_MULTI", "") not in ("", "0")
HIDDEN_SIZE = 1024
TYPES = [
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q8_0,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not ENABLED, reason="set VLLM_GGUF_MMVQ_MULTI=1 to run")
@pytest.mark.parametrize("quant_type", TYPES)
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 5, 8])
@torch.inference_mode()
def test_multi_column_mmvq_matches_reference(quant_type, num_tokens):
    seed_everything(0)
    tensor = get_gguf_sample_tensors(HIDDEN_SIZE, quant_type)[0]
    qweight = torch.tensor(tensor.data, device="cuda")
    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    rows, cols = qweight.shape[0], qweight.shape[1] // type_size * block_size
    x = torch.rand((num_tokens, cols), dtype=torch.bfloat16, device="cuda") - 0.5

    weight = ops.ggml_dequantize(qweight, quant_type, rows, cols, torch.float32)
    ref = x.float() @ weight.T
    out = ops.ggml_mul_mat_vec_a8(qweight, x, quant_type, rows).float()
    rel = ((out - ref).norm() / ref.norm()).item()
    assert rel < 2e-2, rel
    # Row by row, a batch of one must give the same as row i of the batch.
    for i in range(num_tokens):
        single = ops.ggml_mul_mat_vec_a8(qweight, x[i : i + 1].contiguous(), quant_type, rows)
        torch.testing.assert_close(single.float()[0], out[i], rtol=1e-2, atol=1e-2)
