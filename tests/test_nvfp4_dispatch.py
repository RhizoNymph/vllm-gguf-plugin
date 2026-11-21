# SPDX-License-Identifier: Apache-2.0
"""Which kernel serves ggml NVFP4, decided on the CPU.

NVFP4 has CUDA vector kernels (dense MMVQ and the per-token MoE kernel) but
no CUDA dequantize or MMQ kernel: decode-sized batches go to CUDA, prefill
stays on Triton (grouped fused MoE, dequantize + GEMM for dense).
"""

from types import SimpleNamespace

import pytest
import torch
from gguf import GGMLQuantizationType as T

from vllm_gguf_plugin import ops
from vllm_gguf_plugin.quantization import fused_moe as gguf_fused_moe
from vllm_gguf_plugin.quantization import linear as gguf_linear
from vllm_gguf_plugin.quantization.fused_moe import (
    GROUPED_MOE_MIN_TOKENS,
    MoEKernel,
    select_moe_kernel,
)
from vllm_gguf_plugin.quantization.utils import (
    FP4_QUANT_TYPES,
    MMQ_QUANT_TYPES,
    MMVQ_QUANT_TYPES,
    MOE_VEC_QUANT_TYPES,
)

pytestmark = [pytest.mark.cpu]

NVFP4 = int(T.NVFP4)


# ------------------------------------------------------------- type sets


def test_nvfp4_has_cuda_vector_kernels_only():
    assert NVFP4 in ops.CUDA_OP_QUANT_TYPES["ggml_mul_mat_vec_a8"]
    assert NVFP4 in ops.CUDA_OP_QUANT_TYPES["ggml_moe_a8_vec"]
    for op in ("ggml_dequantize", "ggml_mul_mat_a8", "ggml_moe_a8"):
        assert NVFP4 not in ops.CUDA_OP_QUANT_TYPES[op], op
    assert NVFP4 not in ops.CUDA_OP_QUANT_TYPES["ggml_moe_get_block_size"]


def test_every_cuda_op_has_a_type_set():
    assert set(ops.CUDA_OP_QUANT_TYPES) == {
        "ggml_dequantize",
        "ggml_mul_mat_vec_a8",
        "ggml_mul_mat_a8",
        "ggml_moe_a8",
        "ggml_moe_a8_vec",
        "ggml_moe_get_block_size",
    }


def test_cuda_type_sets_for_existing_types_are_unchanged():
    vec = ops.CUDA_OP_QUANT_TYPES["ggml_mul_mat_vec_a8"]
    dequant = ops.CUDA_OP_QUANT_TYPES["ggml_dequantize"]
    assert vec - {NVFP4} == dequant
    assert ops.CUDA_OP_QUANT_TYPES["ggml_moe_a8_vec"] == vec
    gemm = ops.CUDA_OP_QUANT_TYPES["ggml_mul_mat_a8"]
    assert gemm == ops.CUDA_OP_QUANT_TYPES["ggml_moe_a8"]
    assert gemm == ops.CUDA_OP_QUANT_TYPES["ggml_moe_get_block_size"]
    assert gemm <= {int(t) for t in MMQ_QUANT_TYPES}


def test_nvfp4_is_an_mmvq_type():
    assert T.NVFP4 in FP4_QUANT_TYPES
    assert T.NVFP4 in MMVQ_QUANT_TYPES
    assert T.NVFP4 in MOE_VEC_QUANT_TYPES
    assert T.NVFP4 not in MMQ_QUANT_TYPES


def test_cuda_disabled_means_no_cuda_kernel(monkeypatch):
    monkeypatch.setattr(ops, "_CUDA_ENABLED", False)
    assert not ops._cuda_kernel_available("ggml_moe_a8_vec", NVFP4)
    assert not ops._cuda_kernel_available("ggml_mul_mat_vec_a8", NVFP4)


def test_cuda_kernel_needs_the_type_in_that_ops_set(monkeypatch):
    monkeypatch.setattr(ops, "_CUDA_ENABLED", True)
    namespace = SimpleNamespace(**{name: object() for name in ops.CUDA_OP_QUANT_TYPES})
    monkeypatch.setattr(torch.ops, "_C_gguf", namespace, raising=False)
    assert ops._cuda_kernel_available("ggml_moe_a8_vec", NVFP4)
    assert ops._cuda_kernel_available("ggml_mul_mat_vec_a8", NVFP4)
    assert not ops._cuda_kernel_available("ggml_dequantize", NVFP4)
    assert not ops._cuda_kernel_available("ggml_moe_a8", NVFP4)
    assert ops._cuda_kernel_available("ggml_moe_a8", int(T.Q4_K))


# --------------------------------------------------------- MoE selection


@pytest.mark.parametrize("tokens", [1, 2, 8, 32, GROUPED_MOE_MIN_TOKENS])
def test_nvfp4_decode_takes_the_vector_kernel(tokens):
    assert select_moe_kernel(NVFP4, NVFP4, tokens) is MoEKernel.VECTOR


@pytest.mark.parametrize("tokens", [GROUPED_MOE_MIN_TOKENS + 1, 512, 8192])
def test_nvfp4_prefill_takes_the_grouped_kernel(tokens):
    assert select_moe_kernel(NVFP4, NVFP4, tokens) is MoEKernel.GROUPED


@pytest.mark.parametrize("other", [T.Q5_0, T.Q4_K, T.IQ4_XS])
@pytest.mark.parametrize("tokens", [1, 512])
def test_nvfp4_beside_another_type_takes_the_vector_kernel(other, tokens):
    # A grouped launch shares one block size across both GEMMs; Triton's
    # (NVFP4) and the CUDA MMQ kernel's differ, so mixed pairs stay per token.
    assert select_moe_kernel(NVFP4, int(other), tokens) is MoEKernel.VECTOR
    assert select_moe_kernel(int(other), NVFP4, tokens) is MoEKernel.VECTOR


@pytest.mark.parametrize(
    "qtype,tokens,kernel",
    [
        (T.Q4_K, 1, MoEKernel.VECTOR),
        (T.Q4_K, GROUPED_MOE_MIN_TOKENS, MoEKernel.VECTOR),
        (T.Q4_K, GROUPED_MOE_MIN_TOKENS + 1, MoEKernel.GROUPED),
        (T.IQ4_XS, 4096, MoEKernel.VECTOR),
        (T.BF16, 1, MoEKernel.PER_TOKEN),
        (T.BF16, 4096, MoEKernel.PER_TOKEN),
    ],
)
def test_other_types_keep_their_kernels(qtype, tokens, kernel):
    assert select_moe_kernel(int(qtype), int(qtype), tokens) is kernel


# ------------------------------------------- _fused_moe_gguf call routing


E, TOP_K, H, INTER = 4, 2, 128, 64


@pytest.fixture
def recorded(monkeypatch):
    """Replace the MoE entry points with shape-correct stand-ins that log
    which one ran."""
    calls: list[tuple[str, int]] = []

    def moe_a8(x, w, sorted_ids, expert_ids, num_post, qtype, row, top_k, tokens):
        calls.append(("grouped", qtype))
        return torch.zeros(tokens * top_k, row, dtype=x.dtype)

    def moe_a8_vec(x, w, topk_ids, top_k, qtype, row, tokens):
        calls.append(("vector", qtype))
        return torch.zeros(tokens * top_k, row, dtype=x.dtype)

    def align(topk_ids, block_size, num_experts):
        calls.append(("align", block_size))
        n = topk_ids.numel()
        return (
            torch.arange(n, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([n], dtype=torch.int32),
        )

    def moe_sum(inp, out):
        out.copy_(inp.sum(dim=1))

    def activation(act, out, inp, *, activation_config=None, **_):
        out.copy_(inp[..., : out.shape[-1]])

    import vllm.model_executor.layers.fused_moe.fused_moe as vllm_fused_moe

    monkeypatch.setattr(gguf_fused_moe.ops, "ggml_moe_a8", moe_a8)
    monkeypatch.setattr(gguf_fused_moe.ops, "ggml_moe_a8_vec", moe_a8_vec)
    monkeypatch.setattr(gguf_fused_moe.ops, "moe_sum", moe_sum)
    monkeypatch.setattr(gguf_fused_moe, "apply_moe_activation", activation)
    monkeypatch.setattr(vllm_fused_moe, "moe_align_block_size", align)
    return calls


def _run(tokens: int, t1: int = NVFP4, t2: int = NVFP4) -> torch.Tensor:
    x = torch.zeros(tokens, H)
    ids = torch.zeros(tokens, TOP_K, dtype=torch.int32)
    weights = torch.ones(tokens, TOP_K)
    w13 = torch.zeros(E, 2 * INTER, H // 64 * 36, dtype=torch.uint8)
    w2 = torch.zeros(E, H, INTER // 64 * 36, dtype=torch.uint8)
    return gguf_fused_moe._fused_moe_gguf(
        x, w13, w2, weights, ids, t1, t2, "silu", torch.ones(E), torch.ones(E)
    )


def test_nvfp4_decode_calls_the_vector_op_twice(recorded):
    out = _run(tokens=3)
    assert out.shape == (3, H)
    assert recorded == [("vector", NVFP4), ("vector", NVFP4)]


def test_nvfp4_prefill_calls_the_grouped_op_with_triton_block_size(recorded):
    from vllm_gguf_plugin.triton.fused_moe.utils import get_triton_moe_block_m

    out = _run(tokens=GROUPED_MOE_MIN_TOKENS + 1)
    assert out.shape == (GROUPED_MOE_MIN_TOKENS + 1, H)
    assert recorded == [
        ("align", get_triton_moe_block_m(NVFP4)),
        ("grouped", NVFP4),
        ("grouped", NVFP4),
    ]


# ---------------------------------------------------------- dense linear


@pytest.fixture
def dense_ops(monkeypatch):
    calls: list[str] = []

    def mmvq(w, x, qtype, row):
        calls.append("mmvq")
        return torch.zeros(x.shape[0], row, dtype=x.dtype)

    def mmq(w, x, qtype, row):
        calls.append("mmq")
        return torch.zeros(x.shape[0], row, dtype=x.dtype)

    def dequantize(w, qtype, m, n, dtype):
        calls.append("dequantize")
        return torch.zeros(m, n, dtype=dtype)

    monkeypatch.setattr(gguf_linear.ops, "ggml_mul_mat_vec_a8", mmvq)
    monkeypatch.setattr(gguf_linear.ops, "ggml_mul_mat_a8", mmq)
    monkeypatch.setattr(gguf_linear.ops, "ggml_dequantize", dequantize)
    return calls


@pytest.mark.parametrize("tokens,path", [(1, "mmvq"), (6, "mmvq"), (7, "dequantize")])
def test_dense_nvfp4_decode_takes_mmvq(dense_ops, tokens, path):
    weight = torch.zeros(128, 4 * 36, dtype=torch.uint8)
    x = torch.zeros(tokens, 256)
    out = gguf_linear._fused_mul_mat_gguf(x, weight, NVFP4)
    assert out.shape == (tokens, 128)
    assert dense_ops == [path]
