# Tokenizer Added Tokens

## Scope

Restoring the added-token vocabulary when a tokenizer is built from GGUF
metadata: which vocab entries must be matched verbatim by `encode()` rather
than passed through BPE.

Not in scope: which `.gguf` backs the tokenizer (see
[model_resolution](model_resolution.md)), BOS/EOS attribute repair, or the
chat template itself.

## The problem

A GGUF tags every vocab entry in `tokenizer.ggml.token_type`:

| Type | Name | HF equivalent |
|---|---|---|
| 1 | `NORMAL` | ordinary BPE vocab |
| 2 | `UNKNOWN` | — |
| 3 | `CONTROL` | added token, `special=True` |
| 4 | `USER_DEFINED` | added token, `special=False` |
| 5 | `UNUSED` | padding to a round vocab size |
| 6 | `BYTE` | byte fallback |

Types 3 and 4 are what HF keeps in `added_tokens_decoder`. An added token is
matched as a literal string ahead of BPE, so it always encodes to exactly one
id.

transformers' GGUF converters largely ignore that array:

| Converter | Used for | Added tokens registered |
|---|---|---|
| `GGUFLlamaConverter` | llama | CONTROL only (`token_type == 3`) |
| `GGUFQwen2Converter` | qwen2, **qwen35, qwen35moe** | 3 hardcoded strings |
| `GGUFGemmaConverter` | gemma, **gemma4** | none |
| `GGUFGPTConverter` | gpt2 | none |

The two converters this plugin routes to — `GGUFQwen2Converter` via
`qwen35.py`, `GGUFGemmaConverter` via `gemma4.py` — never read `token_type`
at all. Qwen2 hardcodes `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`; Gemma
adds nothing.

## Why it fails silently

The dropped tokens are still present in the BPE vocab at their correct ids.
So `convert_ids_to_tokens`, `decode`, and `convert_tokens_to_ids` all behave
normally, and **only `encode()` is wrong**:

```
<think>  ->  [13314, 741, 29]      ("<th", "ink", ">")     instead of [248068]
```

On Qwen3.5/3.6 this breaks chat and nothing else. The chat template's final
branch prefills an opening `<think>`:

```jinja
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- else %}
    {{- '<think>\n' }}
{%- endif %}
```

The model receives three nonsense tokens where one belongs and emits an
immediate `<|im_end|>`. The response is an empty string with `finish_reason:
"stop"` — no error, no warning. `/v1/completions` is unaffected, because
nothing prefills `<think>` there. Both template branches emit `<think>`, so
`enable_thinking: false` is not a workaround.

Measured on `Qwen3.6-27B-UD-Q4_K_XL`, same prompt, only the prefill encoding
differing:

| prefill tokens | output |
|---|---|
| `13314, 741, 29` (BPE-split) | `''` — immediate stop |
| `248068` (atomic) | `"Here's a thinking process: 1. **Analyze User Input:**..."` |

The blast radius is wider than `<think>`: `<tool_call>`, `<|vision_start|>`,
and the `<|fim_*|>` tokens are dropped the same way.

## The fix

`vllm_gguf_plugin/tokenizer.py` re-registers types 3 and 4 after the
converter runs, mapping `special = (token_type == CONTROL)`. This is the rule
transformers already applies on its *non*-GGUF SentencePiece path
(`convert_slow_tokenizer.py`: `p.type in [3, 4]`, special `p.type == 3`); the
GGUF converters simply never got it.

Validated against the HF release of the same checkpoint:

- GGUF CONTROL (27) + USER_DEFINED (6) = **33**
- HF `added_tokens_decoder` = **33**, zero difference in either direction
- flags agree — `<|im_start|>` `special=True`, `<think>` `special=False`

## Invariants

**Registration must never change the vocab.** Only tokens the built tokenizer
already resolves to their GGUF index are registered:

```python
if tokenizer.convert_tokens_to_ids(added.content) == vocab_id:
```

`add_tokens` reuses an existing vocab entry rather than allocating, so no id
can land past `vocab_size` — which would index off the end of the embedding
matrix. Anything absent or sitting at a different id is skipped with a
warning. On the 27B checkpoint `len(tokenizer)` is unchanged by the restore
and all 33 tokens encode to their GGUF id.

**Never raises.** This runs inside `AutoTokenizer.from_pretrained`. A
tokenizer that splits `<think>` is worse than one that doesn't, but far
better than a model that won't load. `GGUFReader` has no typed failure
contract — a truncated file raises `IndexError` from its header parse, not
the `ValueError` its own magic check would suggest — so the read boundary
catches broadly and returns 0.

**UNUSED (5) is excluded.** This vocab carries 243 of them; they are padding
to a round size and registering them would bloat the added-token trie that
every `encode()` consults.

## Files

| File | Role |
|---|---|
| `vllm_gguf_plugin/tokenizer.py` | `added_tokens_from_gguf_vocab` (pure selection), `restore_gguf_added_tokens` (id-checked registration) |
| `vllm_gguf_plugin/gemma4.py` | hosts the `AutoTokenizer.from_pretrained` wrapper; calls the restore for every arch, not just gemma4 |
| `tests/test_gguf_added_tokens.py` | type selection, special-flag mapping, id-drift guard, failure tolerance |

The wrapper lives in `gemma4.py` for historical reasons — it is the only
point where the plugin sees a tokenizer between construction and use. The
BOS/EOS repair around it remains gemma4-only; the added-token restore is not.

## Known gap

`GGUFQwen2Converter` also registers `<s>` and `</s>` as added tokens even
when they are absent from the vocab, so they receive fresh ids at
`vocab_size` and `vocab_size + 1` (248320/248321 on the 27B). Those ids index
past the embedding matrix. Nothing emits them in practice — the GGUF's
declared EOS is `<|im_end|>` — so this is recorded, not fixed. The restore
above cannot introduce such a token, by the id-check invariant.
