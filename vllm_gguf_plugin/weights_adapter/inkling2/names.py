# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF -> HF checkpoint names of Inkling 2 Mini, and the value conversion
each tensor needs.

The HF names are those of ``thinkingmachines/Inkling-2-Mini`` @ ed40e7da
(``model.safetensors.index.json``, and ``mtp.safetensors`` for the DSpark
draft). Every table is exact: several GGUF names are prefixes of others
(``attn_k_conv`` of ``attn_k_conv_proj``), so nothing is matched by
substring. A GGUF tensor without an entry is an error.

The conversions undo llama.cpp's ``convert_hf_to_gguf.py`` (``Inkling2Model``
and ``Inkling2VisionModel``); each was checked against the bytes of the
published files (see ``docs/features/inkling2.md``).
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .metadata import Inkling2GGUFMetadata

LLM = "model.llm."
VISUAL = "model.visual."


class Conversion(enum.Enum):
    """How a GGUF tensor's values become the HF tensor's."""

    #: Same values and shape (F32 tensors are exact upcasts of BF16 ones).
    NONE = "none"
    #: Depthwise conv kernel squeezed by ggml: ``[C, K]`` -> ``[C, 1, K]``.
    UNSQUEEZE_CONV = "unsqueeze_conv"
    #: Static short-conv taps: ggml stores ``static_w[K, C].flip(0).T``.
    STATIC_CONV = "static_conv"
    #: ``rel_logits_proj.proj[d_rel, extent]`` stored transposed.
    TRANSPOSE = "transpose"
    #: Linear patch embedding over HWC patches stored as a conv2d weight
    #: ``[O, 3, P, P]``; HF has ``[O, P * P * 3]``.
    PATCH_EMBED = "patch_embed"
    #: Routed experts ``[E, 2I, ·]`` with gate rows then up rows (HF's
    #: ``w13_weight`` interleaves them); served per expert.
    EXPERTS_GATE_UP = "experts_gate_up"
    #: Routed experts ``[E, H, ·]``; served per expert.
    EXPERTS_DOWN = "experts_down"
    #: Per-expert output scale of an NVFP4 ``w13`` (``amax / denominator``).
    EXPERT_SCALE_GATE_UP = "expert_scale_gate_up"
    #: Per-expert output scale of an NVFP4 ``w2``.
    EXPERT_SCALE_DOWN = "expert_scale_down"


@dataclass(frozen=True)
class TensorTarget:
    """The HF tensor a GGUF tensor becomes."""

    hf_name: str
    conversion: Conversion = Conversion.NONE


# Names the plugin gives the per-expert NVFP4 scales. The HF checkpoint
# stores them as ``w13_weight_amax / w13_weight_outer_scale_denominator``
# (and ``w2_…``); the GGUF holds the quotient, bit for bit.
W13_EXPERT_SCALE = "mlp.experts.w13_expert_scale"
W2_EXPERT_SCALE = "mlp.experts.w2_expert_scale"

# Per-layer tensors -> name below ``model.llm.layers.N.``.
_LAYER: dict[str, TensorTarget] = {
    "attn_q.weight": TensorTarget("attn.wq_du.weight"),
    "attn_k.weight": TensorTarget("attn.wk_dv.weight"),
    "attn_v.weight": TensorTarget("attn.wv_dv.weight"),
    "attn_output.weight": TensorTarget("attn.wo_ud.weight"),
    "attn_gate.weight": TensorTarget("attn.gate_attn.weight"),
    "attn_rel.weight": TensorTarget("attn.wr_du.weight"),
    "attn_sink_proj.weight": TensorTarget("attn.attn_sink_proj.weight"),
    "attn_rel_bias.weight": TensorTarget(
        "attn.rel_logits_proj.proj", Conversion.TRANSPOSE
    ),
    "attn_q_clip.weight": TensorTarget("attn.q_l2_clip.scale"),
    "attn_rel_clip.weight": TensorTarget("attn.rel_l2_clip.scale"),
    "attn_k_conv_proj.weight": TensorTarget("attn.k_sconv.weight_proj"),
    "attn_v_conv_proj.weight": TensorTarget("attn.v_sconv.weight_proj"),
    "attn_norm.weight": TensorTarget("attn_norm.weight"),
    "attn_q_inp_norm.weight": TensorTarget("q_input_norm.weight"),
    "attn_k_inp_norm.weight": TensorTarget("k_input_norm.weight"),
    "attn_v_inp_norm.weight": TensorTarget("v_input_norm.weight"),
    "attn_post_conv.weight": TensorTarget(
        "attn_sconv.weight", Conversion.UNSQUEEZE_CONV
    ),
    "ffn_norm.weight": TensorTarget("mlp_norm.weight"),
    "ffn_post_conv.weight": TensorTarget("mlp_sconv.weight", Conversion.UNSQUEEZE_CONV),
    "ffn_gate_inp.weight": TensorTarget("mlp.gate.weight"),
    "exp_probs_b.bias": TensorTarget("mlp.gate.bias"),
    "ffn_gate_up_exps.weight": TensorTarget(
        "mlp.experts.w13_weight", Conversion.EXPERTS_GATE_UP
    ),
    "ffn_down_exps.weight": TensorTarget(
        "mlp.experts.w2_weight", Conversion.EXPERTS_DOWN
    ),
    "ffn_gate_up_exps.scale": TensorTarget(
        W13_EXPERT_SCALE, Conversion.EXPERT_SCALE_GATE_UP
    ),
    "ffn_down_exps.scale": TensorTarget(W2_EXPERT_SCALE, Conversion.EXPERT_SCALE_DOWN),
    # loop controllers (core layers only)
    **{
        f"loop_{part}_{kind}.weight": TensorTarget(f"controllers.{part}.{kind}_weight")
        for part in ("attn", "k", "v", "mlp")
        for kind in ("down", "gain", "forget")
    },
}

# The two short convs of the attention: a static kernel on the local
# layers (``k_sconv.weight [C, 1, K]``) and, on the global layers, static
# taps plus a dynamic projection (``k_sconv.static_w [K, C]`` beside
# ``k_sconv.weight_proj``). The GGUF name is the same for both; the
# presence of ``attn_*_conv_proj`` tells them apart.
_ATTN_CONVS = {"attn_k_conv.weight": "k", "attn_v_conv.weight": "v"}

_GLOBAL: dict[str, TensorTarget] = {
    "token_embd.weight": TensorTarget(LLM + "embed.weight"),
    "token_embd_norm.weight": TensorTarget(LLM + "embed_norm.weight"),
    "output.weight": TensorTarget(LLM + "unembed.weight"),
    "output_norm.weight": TensorTarget(LLM + "norm.weight"),
    "output_res_query.weight": TensorTarget(LLM + "attn_res.output_query"),
    "loop_res_query.weight": TensorTarget(LLM + "middle_cycle_cross_pass_queries"),
    "loop_kv_res_query.weight": TensorTarget(
        LLM + "middle_cycle_cross_pass_kv_queries"
    ),
}

_VISION_BLOCK: dict[str, str] = {
    "ln1.weight": "ln_1.weight",
    "ln1.bias": "ln_1.bias",
    "ln2.weight": "ln_2.weight",
    "ln2.bias": "ln_2.bias",
    "ls1.weight": "ls_1.gamma",
    "ls2.weight": "ls_2.gamma",
    "ffn_up.weight": "mlp.c_fc.weight",
    "ffn_up.bias": "mlp.c_fc.bias",
    "ffn_down.weight": "mlp.c_proj.weight",
    "ffn_down.bias": "mlp.c_proj.bias",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_out.weight": "self_attn.out_proj.weight",
    "attn_out.bias": "self_attn.out_proj.bias",
}

_VISION_GLOBAL: dict[str, TensorTarget] = {
    "v.class_embd": TensorTarget(VISUAL + "backbone.embeddings.class_embedding"),
    "v.class_position_embd.weight": TensorTarget(
        VISUAL + "backbone.embeddings.cls_position_embedding"
    ),
    "v.position_embd.weight": TensorTarget(
        VISUAL + "backbone.embeddings.position_embedding"
    ),
    "v.patch_embd.weight": TensorTarget(
        VISUAL + "backbone.embeddings.patch_embedding.weight", Conversion.PATCH_EMBED
    ),
    "v.pre_ln.weight": TensorTarget(VISUAL + "backbone.ln_pre.weight"),
    "v.pre_ln.bias": TensorTarget(VISUAL + "backbone.ln_pre.bias"),
    # llama.cpp numbers the projector by its layer: proj_in = mm.1.
    "mm.1.weight": TensorTarget(VISUAL + "proj_in.weight"),
    "mm.2.weight": TensorTarget(VISUAL + "proj_out.weight"),
}

# DSpark draft (``dflash``) -> ``mtp.safetensors`` names.
_DSPARK_LAYER: dict[str, str] = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_k_ctx.weight": "self_attn.k_proj_ctx.weight",
    "attn_v_ctx.weight": "self_attn.v_proj_ctx.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_k_ctx_norm.weight": "self_attn.k_norm_ctx.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "dflare_norm.weight": "dflare_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

_DSPARK_GLOBAL: dict[str, str] = {
    "dspark_fuse.weight": "fuse_weight",
    "dspark_hidden_out.weight": "markov_head.hidden_out.weight",
    "dspark_hidden_out.bias": "markov_head.hidden_out.bias",
    "dspark_norm.weight": "markov_head.dflare_norm.weight",
    "joint_proj.weight": "markov_head.rnn_proj.weight",
    "joint_proj.bias": "markov_head.rnn_proj.bias",
    "markov_w1.weight": "markov_head.markov_w1.weight",
    "mask_embd.weight": "mask_embed",
    "output_norm.weight": "norm.weight",
}


class UnmappedTensorsError(ValueError):
    """GGUF tensors with no HF counterpart."""

    def __init__(self, source: str, names: Iterable[str]) -> None:
        self.names = tuple(sorted(names))
        super().__init__(
            f"No HF name for {len(self.names)} {source} tensor(s): {list(self.names)}"
        )


def _split_block(name: str) -> tuple[int, str] | None:
    head, _, rest = name.partition(".")
    if head != "blk":
        return None
    index, _, suffix = rest.partition(".")
    if not index.isdigit():
        return None
    return int(index), suffix


def _layer_target(
    layer: int, suffix: str, dynamic_conv_layers: frozenset[int]
) -> TensorTarget | None:
    prefix = f"{LLM}layers.{layer}."
    if suffix == "attn_res_query.weight":
        # The input query of layer i is attn_res.attn_queries.{i - 1}
        # (layer 0 has none; its input is the embedding).
        if layer == 0:
            return None
        return TensorTarget(f"{LLM}attn_res.attn_queries.{layer - 1}")
    if suffix == "ffn_res_query.weight":
        return TensorTarget(f"{LLM}attn_res.mlp_queries.{layer}")
    if suffix in _ATTN_CONVS:
        kv = _ATTN_CONVS[suffix]
        if layer in dynamic_conv_layers:
            return TensorTarget(
                f"{prefix}attn.{kv}_sconv.static_w", Conversion.STATIC_CONV
            )
        return TensorTarget(
            f"{prefix}attn.{kv}_sconv.weight", Conversion.UNSQUEEZE_CONV
        )
    target = _LAYER.get(suffix)
    if target is None:
        return None
    return TensorTarget(prefix + target.hf_name, target.conversion)


def _vision_target(name: str) -> TensorTarget | None:
    if name in _VISION_GLOBAL:
        return _VISION_GLOBAL[name]
    if not name.startswith("v.blk."):
        return None
    block = _split_block(name.removeprefix("v."))
    if block is None:
        return None
    index, suffix = block
    hf = _VISION_BLOCK.get(suffix)
    if hf is None:
        return None
    return TensorTarget(f"{VISUAL}backbone.layers.{index}.{hf}")


def inkling2_targets(
    tensor_names: Iterable[str], meta: Inkling2GGUFMetadata
) -> dict[str, TensorTarget]:
    """Map every backbone and mmproj GGUF tensor to its HF target.

    Raises :class:`UnmappedTensorsError` on any tensor without one (or on a
    block index past the file's layer count): a tensor that is not loaded
    leaves its parameter at initialisation without an error.
    """
    names = list(tensor_names)
    dynamic_conv_layers = frozenset(
        block[0]
        for name in names
        if (block := _split_block(name)) is not None
        and block[1] == "attn_k_conv_proj.weight"
    )
    loop = meta.loop
    core = loop.core if loop is not None else range(0)
    targets: dict[str, TensorTarget] = {}
    unmapped: list[str] = []
    for name in names:
        target: TensorTarget | None = None
        if name.startswith(("v.", "mm.")):
            target = _vision_target(name)
        elif (block := _split_block(name)) is not None:
            layer, suffix = block
            if layer < meta.num_layers and (
                not suffix.startswith("loop_") or layer in core
            ):
                target = _layer_target(layer, suffix, dynamic_conv_layers)
        elif name.startswith("loop_") and loop is None:
            target = None
        else:
            target = _GLOBAL.get(name)
        if target is None:
            unmapped.append(name)
        else:
            targets[name] = target
    if unmapped:
        raise UnmappedTensorsError("inkling2 GGUF", unmapped)
    return targets


def dspark_name_map(tensor_names: Iterable[str], num_layers: int) -> dict[str, str]:
    """Map every DSpark (``dflash``) GGUF tensor to its ``mtp.safetensors``
    name. Raises :class:`UnmappedTensorsError` on any other tensor."""
    mapped: dict[str, str] = {}
    unmapped: list[str] = []
    for name in tensor_names:
        hf: str | None = _DSPARK_GLOBAL.get(name)
        if (block := _split_block(name)) is not None:
            layer, suffix = block
            local = _DSPARK_LAYER.get(suffix)
            if local is not None and layer < num_layers:
                hf = f"layers.{layer}.{local}"
        if hf is None:
            unmapped.append(name)
        else:
            mapped[name] = hf
    if unmapped:
        raise UnmappedTensorsError("inkling2 DSpark GGUF", unmapped)
    return mapped


def name_map(targets: Mapping[str, TensorTarget]) -> dict[str, str]:
    return {name: target.hf_name for name, target in targets.items()}
