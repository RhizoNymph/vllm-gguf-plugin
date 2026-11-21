# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF metadata of Inkling 2 files, and checks of the HF config against it.

Three files carry metadata: the backbone (``general.architecture`` =
``inkling2``), the vision projector (``clip``, projector ``inkling2v``) and
the DSpark draft (``dflash``). The HF config always comes from the
``thinkingmachines`` repo; these checks make sure it describes the model in
the files before anything is built.

The backbone describes its loop by *layer visits*: ``block_count`` is the
number of decoder layers run per token (48 for Mini: 6 prelude + 3 passes of
the 12-layer core + 6 coda), and the per-layer arrays
(``attention.head_count``, ``attention.sliding_window_pattern``, …) have one
entry per visit. The tensors exist once per physical layer (24).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gguf

if TYPE_CHECKING:
    from transformers import PretrainedConfig

BACKBONE_ARCH = "inkling2"
# Mainline llama.cpp's layout of the same tensors (unsloth's Mainline_Support
# files): different names, a stacked loop-query layout and per-layer arrays.
MAINLINE_LOOP_ARCH = "inkling2_loop"
VISION_PROJECTOR = "inkling2v"
DSPARK_ARCH = "dflash"


class Inkling2ConfigMismatch(ValueError):
    """The HF config and a GGUF file describe different models."""

    def __init__(self, source: str, mismatches: Sequence[str]) -> None:
        self.source = source
        self.mismatches = tuple(mismatches)
        super().__init__(
            f"The HF config does not describe this {source}:\n  "
            + "\n  ".join(self.mismatches)
        )


def read_gguf_kv(path: str) -> dict[str, Any]:
    """Every metadata key of *path* except the tokenizer tables."""
    reader = gguf.GGUFReader(path)
    return {
        key: field.contents()
        for key, field in reader.fields.items()
        if not key.startswith(("GGUF.", "tokenizer.ggml."))
    }


def _ints(value: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in value)


def _bools(value: Any) -> tuple[bool, ...]:
    return tuple(bool(v) for v in value)


def _compare(
    prefix: str, kv: Mapping[str, Any], expected: Mapping[str, Any]
) -> list[str]:
    mismatches: list[str] = []
    for key, want in expected.items():
        full = f"{prefix}{key}"
        got = kv.get(full)
        if got is None:
            mismatches.append(f"{full}: missing from the GGUF (config={want!r})")
            continue
        if isinstance(want, float):
            ok = math.isclose(float(got), want, rel_tol=1e-6, abs_tol=0.0)
        elif isinstance(want, tuple) and want and isinstance(want[0], bool):
            ok = _bools(got) == want
        elif isinstance(want, tuple):
            ok = _ints(got) == want
        else:
            ok = got == want
        if not ok:
            mismatches.append(f"{full}: gguf={got!r} config={want!r}")
    return mismatches


@dataclass(frozen=True)
class LoopLayout:
    """The loop core of a looped checkpoint (``loop_config``)."""

    core_start: int
    core_end: int
    num_passes: int
    gate_rank: int

    @property
    def core(self) -> range:
        return range(self.core_start, self.core_end)


@dataclass(frozen=True)
class Inkling2GGUFMetadata:
    """The ``inkling2.*`` metadata of a backbone file."""

    kv: Mapping[str, Any]

    @classmethod
    def from_kv(cls, kv: Mapping[str, Any]) -> Inkling2GGUFMetadata:
        arch = kv.get("general.architecture")
        if arch == MAINLINE_LOOP_ARCH:
            raise NotImplementedError(
                "This is a mainline-llama.cpp inkling2_loop GGUF (unsloth's "
                "Mainline_Support layout). The plugin reads the root "
                "inkling2 files of unsloth/Inkling-2-Mini-GGUF, which hold the "
                "same tensors byte for byte."
            )
        if arch != BACKBONE_ARCH:
            raise ValueError(f"Not an inkling2 GGUF: general.architecture={arch!r}")
        return cls(dict(kv))

    @classmethod
    def from_file(cls, path: str) -> Inkling2GGUFMetadata:
        return cls.from_kv(read_gguf_kv(path))

    def __getitem__(self, key: str) -> Any:
        return self.kv[f"{BACKBONE_ARCH}.{key}"]

    def get(self, key: str, default: Any = None) -> Any:
        return self.kv.get(f"{BACKBONE_ARCH}.{key}", default)

    @property
    def num_layer_visits(self) -> int:
        return int(self["block_count"])

    @property
    def loop(self) -> LoopLayout | None:
        if self.get("loop.core_start") is None:
            return None
        return LoopLayout(
            core_start=int(self["loop.core_start"]),
            core_end=int(self["loop.core_end"]),
            num_passes=int(self["num_loops"]),
            gate_rank=int(self["loop.gate_rank"]),
        )

    @property
    def num_layers(self) -> int:
        """Physical decoder layers (one set of tensors each)."""
        loop = self.loop
        if loop is None:
            return self.num_layer_visits
        return self.num_layer_visits - (loop.num_passes - 1) * len(loop.core)


def _visit_layers(text_config: PretrainedConfig) -> list[int]:
    """The physical layer of every visit, in execution order."""
    layers = [0] * text_config.num_layer_visits
    for layer in range(text_config.num_hidden_layers):
        for visit in text_config.layer_visit_ids(layer):
            layers[visit] = layer
    return layers


def validate_inkling2_config(
    meta: Inkling2GGUFMetadata, text_config: PretrainedConfig
) -> None:
    """Raise :class:`Inkling2ConfigMismatch` if *text_config* is not the
    model whose backbone metadata is *meta*."""
    c = text_config
    local = set(c.local_layer_ids)
    visits = _visit_layers(c)
    expected: dict[str, Any] = {
        "block_count": c.num_layer_visits,
        "context_length": c.model_max_length,
        "embedding_length": c.hidden_size,
        "feed_forward_length": c.intermediate_size,
        "expert_feed_forward_length": c.intermediate_size,
        "expert_count": c.n_routed_experts,
        "expert_used_count": c.num_experts_per_tok,
        "expert_weights_scale": float(c.route_scale),
        "vocab_size": c.padded_vocab_size,
        "vocab_size_unpadded": c.vocab_size,
        "attention.head_count": tuple(
            c.swa_num_attention_heads if layer in local else c.num_attention_heads
            for layer in visits
        ),
        "attention.head_count_kv": tuple(
            c.swa_num_key_value_heads if layer in local else c.num_key_value_heads
            for layer in visits
        ),
        "attention.key_length": c.head_dim,
        "attention.value_length": c.head_dim,
        "attention.key_length_swa": c.swa_head_dim,
        "attention.value_length_swa": c.swa_head_dim,
        "attention.sliding_window": c.sliding_window_size,
        "attention.sliding_window_pattern": tuple(layer in local for layer in visits),
        "attention.layer_norm_rms_epsilon": float(c.rms_norm_eps),
        "attention.relative_length": c.d_rel,
        "attention.relative_extent": c.rel_extent,
        "attention.qk_clip": float(c.qk_l2_clip_threshold),
        "attention.relative_clip": float(c.rel_l2_clip_threshold),
        "attention.log_scale_floor": c.log_scaling_n_floor,
        "attention.log_scale_alpha": float(c.log_scaling_alpha),
        "rope.dimension_count": 0,
        "shortconv.l_cache": c.sconv_kernel_size,
        "attn_res.block_size": c.n_layers_per_attn_res_block,
        "activation.situ_beta": float(c.mlp_gate_softcap),
        "activation.situ_linear_beta": float(c.mlp_up_softcap),
        "final_logit_softcapping": float(c.final_logit_softcapping),
    }
    loop = c.loop_config
    if loop is not None:
        expected |= {
            "num_loops": c.num_loop_passes,
            "loop.core_start": loop["core_start"],
            "loop.core_end": loop["core_end"],
            "loop.gate_rank": loop["gate_rank"],
        }
    mismatches = _compare(f"{BACKBONE_ARCH}.", meta.kv, expected)
    if loop is None and meta.loop is not None:
        mismatches.append(
            f"{BACKBONE_ARCH}.loop.*: the GGUF is looped, the config has no loop_config"
        )
    if c.n_shared_experts:
        mismatches.append(
            f"n_shared_experts={c.n_shared_experts}: the GGUF has no shared experts"
        )
    if mismatches:
        raise Inkling2ConfigMismatch("inkling2 GGUF", mismatches)


def validate_inkling2_vision_config(
    kv: Mapping[str, Any], vision_config: Mapping[str, Any] | PretrainedConfig
) -> None:
    """Raise if the mmproj metadata *kv* is not the config's vision tower."""
    v = vision_config if isinstance(vision_config, Mapping) else vars(vision_config)
    mismatches: list[str] = []
    if kv.get("clip.vision.projector_type") != VISION_PROJECTOR:
        mismatches.append(
            "clip.vision.projector_type: "
            f"gguf={kv.get('clip.vision.projector_type')!r} "
            f"config={VISION_PROJECTOR!r}"
        )
    patch = v["native_patch_size"]
    hidden = v["hidden_size"]
    mismatches += _compare(
        "clip.vision.",
        kv,
        {
            "block_count": v["num_hidden_layers"],
            "embedding_length": hidden,
            "feed_forward_length": int(hidden * v.get("mlp_ratio", 4.0)),
            "attention.head_count": v["num_attention_heads"],
            "attention.layer_norm_epsilon": float(v["layer_norm_eps"]),
            "patch_size": patch,
            "image_size": v["posemb_grid_size"] * patch,
            "projection_dim": v["text_hidden_size"],
            "spatial_merge_size": v["adapter_merge_size"],
        },
    )
    if mismatches:
        raise Inkling2ConfigMismatch("inkling2 mmproj GGUF", mismatches)


def validate_inkling2_dspark_config(
    kv: Mapping[str, Any],
    mtp_config: Mapping[str, Any],
    text_config: PretrainedConfig,
) -> None:
    """Raise if the DSpark GGUF metadata *kv* is not the config's
    ``mtp_config`` draft of *text_config*."""
    m = mtp_config
    if kv.get("general.architecture") != DSPARK_ARCH:
        raise ValueError(
            "Not an Inkling2 DSpark GGUF: general.architecture="
            f"{kv.get('general.architecture')!r}"
        )
    local = set(m.get("local_layer_ids", ()))
    mismatches = _compare(
        f"{DSPARK_ARCH}.",
        kv,
        {
            "block_count": m["num_hidden_layers"],
            "embedding_length": m["hidden_size"],
            "feed_forward_length": m["intermediate_size"],
            "attention.head_count": m["num_attention_heads"],
            "attention.head_count_kv": m["num_key_value_heads"],
            "attention.key_length": m["head_dim"],
            "attention.value_length": m["head_dim"],
            "attention.layer_norm_rms_epsilon": float(m["rms_norm_eps"]),
            "attention.sliding_window": m["sliding_window_size"],
            "attention.sliding_window_pattern": tuple(
                i in local for i in range(m["num_hidden_layers"])
            ),
            "rope.freq_base": float(m["rope_theta"]),
            "rope.dimension_count": m["head_dim"],
            "block_size": m["block_size"],
            "target_layers": tuple(m["target_layer_ids"]),
            "dspark.markov_rank": m["markov_rank"],
            "vocab_size": text_config.padded_vocab_size,
            "vocab_size_unpadded": text_config.vocab_size,
            "final_logit_softcapping": float(text_config.final_logit_softcapping),
        },
    )
    if mismatches:
        raise Inkling2ConfigMismatch("inkling2 DSpark GGUF", mismatches)
