# SPDX-License-Identifier: Apache-2.0
"""GGUF-quantized Qwen4Exp PLE n-gram table.

The table is 320M rows x 160 values: 102 GB as bfloat16, 28.8 GB as the
IQ4_NL unsloth ships. vLLM would otherwise fall back to an unquantized
table, so the rows stay packed in storage (device or pinned host) and a
lookup decodes only the selected rows.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType
from gguf.quants import dequantize
from torch import nn
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)

from vllm_gguf_plugin.quantization import GGUFConfig
from vllm_gguf_plugin.quantization.ple_embedding import (
    GGUFPLEEmbeddingMethod,
    install_qwen4exp_ple_support,
)

pytestmark = [pytest.mark.cpu]

DIM = 160
IQ4_NL = GGMLQuantizationType.IQ4_NL


@pytest.fixture(autouse=True)
def _installed(monkeypatch):
    import vllm.model_executor.parameter as parameter_module

    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    install_qwen4exp_ple_support()


def _table(rows: int, seed: int = 0):
    """Random IQ4_NL rows (gguf-py cannot quantize IQ4_NL, only decode it):
    an fp16 scale and 16 bytes of 4-bit codebook indices per 32 values."""
    rng = np.random.default_rng(seed)
    blocks = DIM // 32
    scale = rng.uniform(0.01, 0.1, (rows, blocks, 1)).astype(np.float16)
    codes = rng.integers(0, 256, (rows, blocks, 16), dtype=np.uint8)
    packed = np.concatenate([scale.view(np.uint8), codes], axis=-1)
    packed = packed.reshape(rows, blocks * 18)
    return dequantize(packed, IQ4_NL).reshape(rows, DIM), torch.from_numpy(packed)


class _Layer(nn.Module):
    """The slice of Qwen4ExpPLEEmbedding the method touches."""

    def __init__(self, rows: int, start: int = 0):
        super().__init__()
        self.embedding_dim = DIM
        self.params_dtype = torch.bfloat16
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=start, org_vocab_end_index=start + rows
        )

    def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
        return torch.zeros(num_embeddings, embedding_dim, dtype=dtype)


def _method_with_table(rows: int, start: int = 0):
    reference, packed = _table(rows)
    layer = _Layer(rows, start)
    method = GGUFPLEEmbeddingMethod(IQ4_NL)
    method.create_weights(layer, DIM, [rows], DIM, rows, torch.bfloat16)
    layer.weight.data.copy_(packed)
    layer._uva_weight = layer.weight.data
    return method, layer, reference


def test_gguf_config_selects_the_packed_method():
    config = GGUFConfig()
    config.ple_weight_type = int(IQ4_NL)
    method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(config, "x.ngram_embedding")
    assert isinstance(method, GGUFPLEEmbeddingMethod)
    assert method.weight_type == IQ4_NL


def test_unquantized_gguf_table_stays_unquantized():
    config = GGUFConfig()
    config.ple_weight_type = int(GGMLQuantizationType.BF16)
    method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(config, "x.ngram_embedding")
    assert isinstance(method, Qwen4ExpPLEUnquantizedEmbeddingMethod)


def test_other_quant_configs_keep_vllm_dispatch():
    method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(None, "x.ngram_embedding")
    assert isinstance(method, Qwen4ExpPLEUnquantizedEmbeddingMethod)


def test_install_is_idempotent():
    before = Qwen4ExpPLEEmbeddingMethod.from_quant_config
    install_qwen4exp_ple_support()
    assert Qwen4ExpPLEEmbeddingMethod.from_quant_config is before


def test_weight_is_packed_rows():
    method, layer, _ = _method_with_table(7)
    # IQ4_NL: 32 values in 18 bytes -> 5 blocks per 160-value row.
    assert layer.weight.dtype == torch.uint8
    assert layer.weight.shape == (7, 90)
    assert method.lookup_dtype(layer) == torch.bfloat16


def test_rows_must_be_whole_blocks():
    layer = _Layer(4)
    with pytest.raises(ValueError, match="multiple of 32"):
        GGUFPLEEmbeddingMethod(IQ4_NL).create_weights(
            layer, 100, [4], 100, 4, torch.bfloat16
        )


def test_embedding_decodes_selected_rows():
    method, layer, reference = _method_with_table(9)
    ids = torch.tensor([[3, 0], [8, 3]])
    out = method.embedding(layer, ids)
    assert out.shape == (2, 2, DIM)
    assert out.dtype == torch.bfloat16
    expected = torch.from_numpy(reference[ids.reshape(-1).numpy()]).reshape(2, 2, DIM)
    torch.testing.assert_close(out.float(), expected.to(torch.bfloat16).float())


def test_pinned_lookup_zeroes_rows_owned_elsewhere():
    """A rank decodes only its vocabulary range; the ETP reduction sums the
    zero rows of every other rank away."""
    method, layer, reference = _method_with_table(6, start=10)
    ids = torch.tensor([10, 3, 15, 16])
    out = torch.full((4, DIM), 7.0, dtype=torch.bfloat16)
    method.lookup_from_pinned(layer, ids, out)
    torch.testing.assert_close(
        out[0].float(), torch.from_numpy(reference[0]).to(torch.bfloat16).float()
    )
    torch.testing.assert_close(
        out[2].float(), torch.from_numpy(reference[5]).to(torch.bfloat16).float()
    )
    assert not out[1].any() and not out[3].any()


def test_dequantize_only_casts():
    method, layer, _ = _method_with_table(2)
    rows = torch.randn(3, DIM, dtype=torch.bfloat16)
    torch.testing.assert_close(
        method.dequantize(layer, rows, torch.float32), rows.float()
    )


def _write_gguf(path, name, data, qtype):
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), "qwen4exp")
    writer.add_tensor(name, data, raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_adapter_configures_the_ple_table_type(tmp_path):
    from vllm_gguf_plugin.gguf_files import GGUFModelFiles
    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    _, packed = _table(4)
    path = tmp_path / "ple.gguf"
    _write_gguf(path, "per_layer_token_embd.weight", packed.numpy(), IQ4_NL)
    config = GGUFConfig()
    Qwen4ExpGGUFAdapter().configure_quant_config(
        GGUFModelFiles(backbone=(str(path),)), config
    )
    assert config.ple_weight_type == IQ4_NL
    method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(config, "x")
    assert isinstance(method, GGUFPLEEmbeddingMethod)


def test_large_tensors_are_served_without_a_copy(tmp_path, monkeypatch):
    """The 28.8 GB table must not be duplicated into RAM before it is split
    into shards; tensors past the threshold are views of the file mapping."""
    from vllm_gguf_plugin import weight_utils

    monkeypatch.setattr(weight_utils, "_ZERO_COPY_MIN_BYTES", 0)
    _, packed = _table(4)
    path = tmp_path / "ple.gguf"
    _write_gguf(path, "per_layer_token_embd.weight", packed.numpy(), IQ4_NL)
    weights = dict(
        weight_utils.gguf_quant_weights_iterator_multi(
            [str(path)], {"per_layer_token_embd.weight": "t.weight"}
        )
    )
    torch.testing.assert_close(weights["t.weight"], packed)
