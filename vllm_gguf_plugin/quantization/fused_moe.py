# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from functools import partial
from typing import Optional

import torch
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    _gguf_moe_weight_loader,
    _gguf_moe_weight_type_loader,
)
from .utils import FP4_QUANT_TYPES, MMQ_QUANT_TYPES, MOE_VEC_QUANT_TYPES, logger


class MoEKernel(enum.Enum):
    """How ``_fused_moe_gguf`` runs the expert GEMMs."""

    # ``ggml_moe_a8`` over token slots grouped by expert
    # (``moe_align_block_size``): CUDA MMQ, or Triton for NVFP4.
    GROUPED = enum.auto()
    # ``ggml_moe_a8_vec``: one matrix-vector product per (token, expert) pair
    # (CUDA MMVQ; Triton where CUDA has no kernel for the type).
    VECTOR = enum.auto()
    # ``fused_mul_mat_gguf`` per token and expert (dense or unknown types).
    PER_TOKEN = enum.auto()


# Above this many tokens the grouped kernels read each expert's weights once
# per block of tokens instead of once per (token, expert) pair.
GROUPED_MOE_MIN_TOKENS = 64


def select_moe_kernel(
    weight_type: int, weight_type2: int, num_tokens: int
) -> MoEKernel:
    """Pick the MoE kernel for gate/up type ``weight_type`` and down type
    ``weight_type2``.

    A grouped launch aligns tokens once with one block size for both GEMMs,
    so both types must use the same grouped backend: CUDA MMQ, or the
    Triton FP4 kernel (NVFP4 has no CUDA MMQ kernel). Mixed pairs and
    smaller batches run per (token, expert) pair.
    """
    types = {weight_type, weight_type2}
    same_grouped_backend = types <= MMQ_QUANT_TYPES or types <= FP4_QUANT_TYPES
    if num_tokens > GROUPED_MOE_MIN_TOKENS and same_grouped_backend:
        return MoEKernel.GROUPED
    if types <= MOE_VEC_QUANT_TYPES:
        return MoEKernel.VECTOR
    return MoEKernel.PER_TOKEN


def _mask_padded_slots(
    topk_weights: torch.Tensor, topk_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Point padded routing slots at expert 0 with no weight.

    vLLM's routers mark padded tokens (CUDA graph capture, DP padding) with
    expert id -1 and weight 0; the GGUF kernels index experts by id, so a
    negative id reads before the weights. Returns new tensors.
    """
    padded = topk_ids < 0
    return topk_weights.masked_fill(padded, 0), topk_ids.clamp_min(0)


def _fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weight_type: int,
    weight_type2: int,
    activation: str,
    w13_scale: Optional[torch.Tensor] = None,  # noqa: UP045 (op schema)
    w2_scale: Optional[torch.Tensor] = None,  # noqa: UP045
    situ_beta: Optional[float] = None,  # noqa: UP045
    situ_linear_beta: Optional[float] = None,  # noqa: UP045
) -> torch.Tensor:
    """Routed experts ``w2 @ act(w13 @ x)``, weighted by ``topk_weights``.

    ``w13_scale`` / ``w2_scale`` (``[E]``, optional) multiply each expert's
    GEMM output, as ggml applies an NVFP4 tensor's ``.scale``: the gate/up
    rows before the activation, the down rows before the routing weight.
    ``situ_beta`` / ``situ_linear_beta`` configure the SITU activation.
    """
    activation_enum = MoEActivation.from_str(activation)
    topk_weights, topk_ids = _mask_padded_slots(topk_weights, topk_ids)
    activation_config = (
        None
        if situ_beta is None
        else ApplyMoEActivationConfig(
            activation_situ_beta=situ_beta,
            activation_situ_linear_beta=situ_linear_beta,
        )
    )

    def act(inp: torch.Tensor):
        d = inp.shape[-1] // 2
        output_shape = inp.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
        apply_moe_activation(
            activation_enum, out, inp, activation_config=activation_config
        )
        return out

    def scale_rows(out: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
        # Rows are (token, k) pairs in topk_ids order.
        if scale is None:
            return out
        return out.mul_(scale[topk_ids.reshape(-1).long()].to(out.dtype)[:, None])

    route_weights = topk_weights
    if w2_scale is not None:
        route_weights = topk_weights * w2_scale[topk_ids.long()].to(topk_weights.dtype)

    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    out_hidden_states = torch.empty_like(x)
    kernel = select_moe_kernel(weight_type, weight_type2, x.shape[0])
    match kernel:
        case MoEKernel.GROUPED:
            num_tokens, _ = x.shape
            E, N, _ = w1.shape
            top_k = topk_ids.shape[1]
            block_size = ops.ggml_moe_get_block_size(weight_type)

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                topk_ids, block_size, E
            )
            out = ops.ggml_moe_a8(
                x,
                w1,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                weight_type,
                N,
                top_k,
                num_tokens,
            )
            out = act(scale_rows(out, w13_scale))
            out = ops.ggml_moe_a8(
                out,
                w2,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                weight_type2,
                w2.shape[1],
                1,
                num_tokens * top_k,
            )
            out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
                route_weights.view(num_tokens, top_k, 1)
            )
            ops.moe_sum(out, out_hidden_states)
        case MoEKernel.VECTOR:
            num_tokens, _ = x.shape
            E, N, _ = w1.shape
            top_k = topk_ids.shape[1]

            out = ops.ggml_moe_a8_vec(
                x, w1, topk_ids, top_k, weight_type, N, num_tokens
            )
            out = act(scale_rows(out, w13_scale))

            out = ops.ggml_moe_a8_vec(
                out, w2, topk_ids, 1, weight_type2, w2.shape[1], num_tokens * top_k
            )
            out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
                route_weights.view(num_tokens, top_k, 1)
            )
            ops.moe_sum(out, out_hidden_states)
        case MoEKernel.PER_TOKEN:
            from . import fused_mul_mat_gguf as fused_mul_mat_gguf_op

            logger.warning_once(
                "There is no support for fast MoE kernel "
                "for current quantization method. "
                "Falling back to slow implementation. "
            )
            for tok, (w, idx) in enumerate(zip(route_weights, topk_ids)):
                inp = x[tok].reshape((1,) + x.shape[1:])
                current_hidden_state = None
                for ww, ii in zip(w, idx):
                    out = fused_mul_mat_gguf_op(inp, w1[ii], weight_type)
                    if w13_scale is not None:
                        out = out * w13_scale[ii].to(out.dtype)
                    out = act(out)
                    current_state = fused_mul_mat_gguf_op(
                        out, w2[ii], weight_type2
                    ).mul_(ww)
                    if current_hidden_state is None:
                        current_hidden_state = current_state
                    else:
                        current_hidden_state.add_(current_state)
                out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


def _gguf_moe_expert_scale_loader(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    return_success: bool = False,
) -> bool | None:
    """Load one expert's output scale (``…experts.<e>.<proj>.expert_scale``)."""
    del weight_name, shard_id
    param.data[expert_id] = loaded_weight.reshape(()).to(param.dtype)
    return True if return_success else None


def _fused_moe_gguf_fake(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weight_type: int,
    weight_type2: int,
    activation: str,
    w13_scale: Optional[torch.Tensor] = None,  # noqa: UP045
    w2_scale: Optional[torch.Tensor] = None,  # noqa: UP045
    situ_beta: Optional[float] = None,  # noqa: UP045
    situ_linear_beta: Optional[float] = None,  # noqa: UP045
) -> torch.Tensor:
    del w1, w2, topk_weights, topk_ids, weight_type, weight_type2, activation
    del w13_scale, w2_scale, situ_beta, situ_linear_beta
    return torch.empty_like(x)


try:
    direct_register_custom_op(
        op_name="_fused_moe_gguf",
        op_func=_fused_moe_gguf,
        fake_impl=_fused_moe_gguf_fake,
    )
    fused_moe_gguf = torch.ops.vllm._fused_moe_gguf
except AttributeError as error:
    raise error


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF."""

    def __init__(
        self,
        quant_config,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del params_dtype
        base_weight_loader = extra_weight_attrs.pop("weight_loader")
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        w13_weight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w13_weight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w13_weight", w13_weight)

        w13_weight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w13_weight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w13_weight_type, extra_weight_attrs)
        layer.register_parameter("w13_weight_type", w13_weight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        w2_weight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w2_weight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)

        w2_weight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w2_weight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w2_weight_type, extra_weight_attrs)
        layer.register_parameter("w2_weight_type", w2_weight_type)

        if getattr(self.quant_config, "moe_expert_scales", False):
            for name in ("w13_expert_scale", "w2_expert_scale"):
                scale = torch.nn.Parameter(
                    torch.ones(num_experts, dtype=torch.float32), requires_grad=False
                )
                set_weight_attrs(
                    scale, {"weight_loader": _gguf_moe_expert_scale_loader}
                )
                layer.register_parameter(name, scale)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for"
                "fused GGUF MoE method."
            )

        from . import fused_moe_gguf as fused_moe_gguf_op

        moe_config = getattr(layer, "moe_config", None)
        return fused_moe_gguf_op(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            layer.w13_weight_type.weight_type,
            layer.w2_weight_type.weight_type,
            layer.activation.value,
            getattr(layer, "w13_expert_scale", None),
            getattr(layer, "w2_expert_scale", None),
            getattr(moe_config, "activation_situ_beta", None),
            getattr(moe_config, "activation_situ_linear_beta", None),
        )
