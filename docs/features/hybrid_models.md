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
