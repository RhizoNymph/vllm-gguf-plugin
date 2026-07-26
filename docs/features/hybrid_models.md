# Hybrid Models

## Scope

Serving GGUF checkpoints whose layer stack mixes full attention with a
recurrent/linear-attention mechanism — currently the Qwen3.5 family
(`qwen35`, `qwen35moe`), which interleaves gated-delta-net layers with full
attention.

Not in scope: the GDN kernels themselves (in vLLM), tensor-name mapping, or
quantization.

## Why hybrid models need special care

The two layer kinds size their caches by unrelated rules:

- **Attention** page size scales with `block_size` — tokens per block × heads ×
  head dim × 2 (K and V) × dtype width.
- **Recurrent** page size is a fixed per-layer constant — conv state plus SSM
  state — independent of `block_size`.

vLLM allocates one uniform paged pool, so these must be reconciled. That is
`Platform._align_hybrid_block_size()`: it raises `block_size` until the
attention page is at least as large as the recurrent page, then pads the
recurrent page so the two are exactly equal. On this model it settles at
`block_size=400` with 0.25% mamba padding.

Everything downstream assumes that pass ran. Skip it and
`unify_kv_cache_spec_page_size()` later finds page sizes that do not divide and
raises `NotImplementedError: The page size of the layer is not divisible by the
maximum page size.`

## The gate

Both the alignment pass and `MambaModelConfig` (which defaults
`mamba_block_size`) are gated on `ModelConfig.is_hybrid`, which is
`_model_info.is_hybrid` — derived by inspecting the resolved model class for
the `IsHybrid` marker.

Crucially, **vLLM inspects the model class in a subprocess**
(`ModelRegistry`'s `_run_in_subprocess`). The marker must therefore be a real
property of the class on disk; monkey-patching `is_hybrid` in the parent
process has no effect.

`qwen35.py` registers the text-only entry points:

```python
ModelRegistry.register_model(
    "Qwen3_5ForCausalLM", "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM"
)
```

so those classes must themselves carry `IsHybrid` and expose the mamba state
calculators. In this fork they inherit `Qwen3_5HybridStateMixin`, which bundles
the marker with `get_mamba_state_shape_from_config`,
`get_mamba_state_dtype_from_config`, and `get_mamba_state_copy_func` — a class
cannot advertise hybridness without supplying the shapes needed to size its
cache.

## Value conventions ggml does not share with HF

Three conversions are required on load. All were verified element-wise against
the HF release of the same checkpoint; the F32 ones reproduce it exactly.

| Tensor | Conversion | Why |
|---|---|---|
| RMSNorm weights | subtract 1 | Qwen3.5 norms are zero-centered (the module applies `1 + w`); ggml folds the `+1` in, so vLLM would apply it twice. Excludes `linear_attn.norm`, which stores a conventional scale. |
| `A_log` | `log(-x)` | ggml stores the decay pre-exponentiated as `A = -exp(A_log)`; the kernels exponentiate it themselves. |
| every value-head-indexed tensor | re-index | ggml strides value heads `r * num_k_heads + g`; HF groups them `g * ratio + r`. |

The re-indexing covers `A_log`, `dt_bias`, `in_proj_a`, `in_proj_b`, the value
block of `conv1d` and `in_proj_qkv`, all of `in_proj_z` (rows), and `out_proj`
(columns — value heads index its *input* axis). `norm` is indexed by head
*dimension*, not head, and must not move.

Two properties make this easy to get wrong:

- **A 1:1 model cannot detect a mistake.** When `num_v_heads == num_k_heads`
  the two orderings are identical, so a 0.8B model loads and generates
  fluently while every larger sibling is silently scrambled.
- **Quantised tensors arrive renamed.** The weight iterator rewrites
  `<x>.weight` to `<x>.qweight`. Matching only `.weight` skips exactly the
  projections — and leaving the projections un-reordered while the recurrent
  state *is* reordered is worse than doing nothing, because the two halves of
  each layer then disagree.

Re-indexing is a layout change and is safe on packed data when whole rows
move. `out_proj` is the exception: value heads index its *input* axis, so
re-indexing moves bytes within each packed row, which is only valid when a
head spans whole quantisation blocks.

| `out_proj` type | block | 128-element head | handling |
|---|---|---|---|
| Q8_0 | 32 | 4 whole blocks | re-indexed in place, zero copy |
| Q5_K / Q4_K / Q6_K | 256 | half a block | dequantised at load |

`_qwen35_dequant_on_load` detects the misaligned case during
`prepare_loading` and adds those tensors to `unquantized_modules`, so vLLM
builds a plain `Linear`; `map_weights` then unpacks the payload, drops the
now-unused `qweight_type` scalar, and re-indexes in element space. The cost
is roughly 380 MB on a 4B model, paid only by models with grouped value
heads and a 256-element-block `out_proj`. `_assert_head_block_aligned`
remains as a backstop: a *packed* misaligned tensor reaching the transform
raises rather than silently splitting a block.

## What the plugin must not do

Earlier revisions carried `_patch_qwen35_mamba_cache_args`, which set
`engine_args.mamba_block_size = block_size` to paper over the missing default.
It has been removed, because it made things strictly worse:

- assigning `mamba_block_size` sets
  `CacheConfig.user_specified_mamba_block_size` (`vllm/config/cache.py`);
- `_align_hybrid_block_size` reads that flag and treats the value as a
  deliberate user choice, honouring `16` instead of computing an aligned size;
- so the very band-aid that hid the missing default also disabled the fix.

The two symptoms swapped places depending on `--enable-prefix-caching`, which
made the shared root cause hard to see:

| Prefix caching | Symptom |
|---|---|
| on | `NotImplementedError` — page sizes not divisible |
| off | `AssertionError` at `mamba/abstract.py` — `mamba_block_size is None` |

**Invariant: the plugin never sets cache-geometry knobs.** Its job is to
describe the model accurately; sizing the cache is vLLM's.

## Files

| File | Role |
|---|---|
| `vllm_gguf_plugin/qwen35.py` | GGUF→HF config mapping for `qwen35`/`qwen35moe`; registers the text-only model classes; derives `linear_*_heads` fields ggml stores implicitly |
| `vllm_gguf_plugin/plugin.py` | `create_model_config` patch — must leave `mamba_block_size` unset |
| `tests/test_plugin.py` | `test_register_leaves_qwen35_mamba_block_size_unset` |

Requires the companion vLLM change adding `Qwen3_5HybridStateMixin` to
`vllm/model_executor/models/qwen3_5.py` and the `Qwen3_5*ForCausalLM` entries
in `MODELS_CONFIG_MAP`.
