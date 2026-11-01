# Model Resolution

## Scope

Turning whatever the user passed as `--model` into one concrete `.gguf` path,
and keeping every downstream consumer — config, tokenizer, weights — pointed at
that same file.

Not in scope: parsing GGUF metadata into an HF config (see the config-synthesis
subsystem), tensor-name mapping, or quantization kernels.

## Why it is not trivial

vLLM assumes `--model` names an HF repo or a directory laid out like one. A
GGUF checkpoint is a single file, often sitting in a directory beside unrelated
models. Three consumers each need a path, and they do not all accept the same
form:

| Consumer | Accepts | Supplied as |
|---|---|---|
| Config parser | the `.gguf` file | `hf_config_path` |
| `AutoTokenizer` | `(directory, gguf_file=name)` — never a bare file path | `tokenizer` |
| Aux HF loaders (image processor, …) | a repo id or directory | `model` |

Because the aux loaders reject a bare file path, `create_model_config` rewrites
`model` to the *containing directory*. That rewrite is lossy: the directory
alone no longer records which `.gguf` was asked for.

## Control flow

`_patch_engine_args()` wraps `EngineArgs.create_model_config`. For a GGUF
reference it:

1. `_resolve_gguf_weights(model)` → absolute `.gguf` path. Handles a direct
   file, a directory (`_find_local_gguf_file` picks the first `.gguf`, skipping
   `mmproj*` and `*mtp*` sidecars), a local `dir:QUANT` spec, and a remote
   `repo:QUANT`, which is downloaded.
2. Forces `quantization`, `load_format`, and `config_format` to `gguf`, and
   records `model_weights` / `served_model_name`.
3. Pins `hf_config_path` to the resolved file.
4. Rewrites `model` to the containing directory via `_get_gguf_config_source`.
5. **Pins `tokenizer` to the resolved file** (only when the caller did not
   supply one), restoring the information step 4 discarded.

`gemma4._patch_gemma4_tokenizer` wraps `AutoTokenizer.from_pretrained` and
normalises whatever it receives into transformers' `(dir, gguf_file)` form:

- explicit `gguf_file` kwarg → honoured as-is;
- a `.gguf` file path → restated as `(parent_dir, gguf_file=name)`;
- a directory with no `tokenizer.json`/`tokenizer_config.json` → scanned, first
  `.gguf` by name wins, **warning** when the choice is ambiguous;
- a directory with real HF tokenizer files → left alone.

It also probes the GGUF for gemma4 and, for that architecture only, fixes
`bos_token_id` and the fast tokenizer's post-processor template.

## The failure this prevents

Step 5 exists because of a silent-corruption bug. With `~/Models/` holding
several checkpoints and `--model ~/Models/Qwen3.6-27B-UD-Q4_K_XL.gguf`:

- step 4 rewrote `model` to `~/Models`;
- `tokenizer` defaulted to `model`, i.e. the directory;
- the directory scan sorted `*.gguf` and took the first. ASCII order puts
  uppercase before lowercase, so `Laguna-XS-2.1-Q4_K_M.gguf` preceded
  `Qwen3.6-…`, and the tokenizer was built from an unrelated model.

That surfaced as `ValueError: GGUF model with architecture laguna is not
supported yet` — an error naming a model the user never asked for. Had the
sibling been an architecture transformers *does* know, it would not have raised
at all: vLLM would have served Qwen weights through a foreign tokenizer and
produced garbage.

## Same-checkpoint drafts (MTP)

A speculative draft that reuses the target checkpoint (`method: mtp` without a
`model` of its own; the GGUF's `nextn` block) is built by vLLM's
`SpeculativeConfig.__post_init__` as `ModelConfig(model=target.model,
runner="draft", ...)`. For a GGUF target `target.model` is the directory from
step 4 and the draft gets no `hf_config_path`, so its config was parsed from
the first `.gguf` in that directory — the same Laguna failure as below, for
the draft only. `_patch_engine_args` therefore wraps `create_speculative_config`
in `_draft_inherits_target_hf_config`, which, while the speculative config is
built, gives a draft `ModelConfig` for the target's directory the target's
`hf_config_path`. The draft's `model_weights` is then pinned to the target
file as before. A draft with its own `model` is left alone.

## Invariants

- Exactly one `.gguf` backs a served model; config, tokenizer, and weights all
  resolve to it.
- Once resolved, the file is never re-derived by globbing a directory.
- An explicitly supplied `--tokenizer` is never overridden.
- A same-checkpoint draft reads its config from the target's file, never from
  the directory.
- Directory scanning is a last resort, and never silent when ambiguous.
- `mmproj*` and `*mtp*` files are sidecars, not model candidates.

## Files

| File | Role |
|---|---|
| `vllm_gguf_plugin/plugin.py` | `_resolve_gguf_weights`, `_find_local_gguf_file`, `_get_gguf_config_source`, `_patch_engine_args`, `_draft_inherits_target_hf_config` |
| `vllm_gguf_plugin/gemma4.py` | `_patch_gemma4_tokenizer` — tokenizer path normalisation and gemma4 BOS fixups |
| `vllm_gguf_plugin/gguf_utils.py` | remote/quant reference parsing, `download_gguf`, `resolve_local_gguf` |
| `tests/test_plugin.py` | engine-args rewriting, tokenizer pinning, same-checkpoint draft config |
| `tests/test_gguf_tokenizer_resolution.py` | tokenizer path normalisation and ambiguity warning |
