# SPDX-License-Identifier: Apache-2.0
"""Merged GGUF linears whose shards have different ggml types.

Inkling 2's ``q_proj`` merges four checkpoint tensors (``wq_du``, ``wr_du``,
``gate_attn``, ``attn_sink_proj``). In unsloth's UD quants the first three
are Q8_0/Q6_K/Q5_K and the sink is BF16. Shards of different types are kept
back to back in one byte buffer, each read as its own type, instead of
zero-padding every row to the widest shard (BF16 rows are 4608 bytes beside
2448-byte Q8_0 rows: 0.34 GB of padding over the model).
"""

import pytest
import torch
from gguf import GGMLQuantizationType
from gguf.quants import dequantize, quantize

from vllm_gguf_plugin import quantization as gguf_quantization
from vllm_gguf_plugin.quantization import linear as gguf_linear

pytestmark = [pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    import vllm.model_executor.parameter as parameter_module

    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )


T = GGMLQuantizationType
K = 256


def _reference_ops(monkeypatch):
    """Quantized GEMMs through gguf-py dequantization (no CUDA)."""

    def matmul(weight, x, qtype, row):
        dense = torch.from_numpy(dequantize(weight.cpu().numpy(), T(qtype)))
        return (x.float() @ dense.T).to(x.dtype)

    for name in ("ggml_mul_mat_vec_a8", "ggml_mul_mat_a8"):
        monkeypatch.setattr(gguf_linear.ops, name, matmul)

    def dequant(weight, qtype, m, n, dtype):
        dense = dequantize(weight.cpu().numpy(), T(qtype))
        return torch.from_numpy(dense).reshape(m, n).to(dtype)

    monkeypatch.setattr(gguf_linear.ops, "ggml_dequantize", dequant)
    monkeypatch.setattr(
        gguf_quantization, "fused_mul_mat_gguf", gguf_linear._fused_mul_mat_gguf
    )


def _layer(shards):
    """A merged GGUF linear over *shards*: (rows, ggml type, dense values)."""
    method = gguf_linear.GGUFLinearMethod(gguf_quantization.GGUFConfig())
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=K,
        output_partition_sizes=[rows for rows, _, _ in shards],
        input_size=K,
        output_size=sum(rows for rows, _, _ in shards),
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    for shard_id, (_, qtype, values) in enumerate(shards):
        if qtype is T.BF16:
            data = values.to(torch.bfloat16)
        else:
            data = torch.from_numpy(quantize(values.numpy(), qtype))
        layer.weight._store(data, shard_id=shard_id)
        layer.weight_type._store(torch.tensor(int(qtype)), shard_id=shard_id)
    method.process_weights_after_loading(layer)
    return method, layer


def _shards(types, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = (32, 16, 32, 24)
    return [(r, t, torch.randn(r, K, generator=g) / 4) for r, t in zip(rows, types)]


def _expected(x, shards):
    outs = []
    for _, qtype, values in shards:
        if qtype is T.BF16:
            dense = values.to(torch.bfloat16).float()
        else:
            dense = torch.from_numpy(dequantize(quantize(values.numpy(), qtype), qtype))
        outs.append(x.float() @ dense.T)
    return torch.cat(outs, dim=1)


@pytest.mark.parametrize("tokens", [1, 7])
def test_quantized_shards_beside_a_bf16_shard(monkeypatch, tokens):
    _reference_ops(monkeypatch)
    shards = _shards([T.Q8_0, T.Q5_0, T.Q4_0, T.BF16])
    method, layer = _layer(shards)
    x = torch.randn(tokens, K).to(torch.bfloat16)
    out = method.apply(layer, x)
    assert out.shape == (tokens, sum(r for r, _, _ in shards))
    torch.testing.assert_close(out.float(), _expected(x, shards), rtol=2e-2, atol=2e-2)


def test_mixed_shards_are_stored_without_row_padding(monkeypatch):
    _reference_ops(monkeypatch)
    shards = _shards([T.Q8_0, T.Q5_0, T.Q4_0, T.BF16])
    _, layer = _layer(shards)
    row_bytes = {
        T.Q8_0: K // 32 * 34,
        T.Q5_0: K // 32 * 22,
        T.Q4_0: K // 32 * 18,
        T.BF16: 2 * K,
    }
    payload = sum(r * row_bytes[t] for r, t, _ in shards)
    assert layer.weight.dtype == torch.uint8
    # Each shard starts 16-byte aligned; nothing else is added.
    assert payload <= layer.weight.numel() < payload + 16 * len(shards)


def test_uniform_shards_keep_the_fused_layout(monkeypatch):
    _reference_ops(monkeypatch)
    shards = _shards([T.Q8_0] * 4)
    method, layer = _layer(shards)
    assert layer.weight.shape == (sum(r for r, _, _ in shards), K // 32 * 34)
    x = torch.randn(3, K).to(torch.bfloat16)
    torch.testing.assert_close(
        method.apply(layer, x).float(), _expected(x, shards), rtol=2e-2, atol=2e-2
    )


def test_dense_f32_shard_is_cast_to_the_activation_dtype(monkeypatch):
    _reference_ops(monkeypatch)
    x = torch.randn(2, K).to(torch.bfloat16)
    w = torch.randn(5, K)
    out = gguf_linear._fused_mul_mat_gguf(x, w, int(T.F32))
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, x @ w.bfloat16().T)
