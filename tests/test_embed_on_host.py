# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The host-resident (UVA) input embedding gives the same lookups as the device one."""

import pytest
import torch
from gguf import GGMLQuantizationType

from vllm_gguf_plugin.quantization.vocal_embeds import _apply_gguf_embedding, _move_qweight_to_host

from .utils import get_gguf_sample_tensors, seed_everything

HIDDEN_SIZE = 256


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("quant_type", [GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q6_K])
@torch.inference_mode()
def test_host_embedding_matches_device(quant_type):
    seed_everything(0)
    data = torch.tensor(get_gguf_sample_tensors(HIDDEN_SIZE, quant_type)[0].data, device="cuda")
    layer = torch.nn.Module()
    layer.register_parameter("qweight", torch.nn.Parameter(data, requires_grad=False))
    ids = torch.randint(0, data.shape[0], (3, 17), device="cuda")

    expected = _apply_gguf_embedding(ids, layer.qweight, int(quant_type), HIDDEN_SIZE, torch.bfloat16)
    _move_qweight_to_host(layer)
    assert layer._gguf_host_qweight.is_pinned()
    assert layer.qweight.is_cuda
    assert layer.qweight.data_ptr() != data.data_ptr()
    got = _apply_gguf_embedding(ids, layer.qweight, int(quant_type), HIDDEN_SIZE, torch.bfloat16)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
