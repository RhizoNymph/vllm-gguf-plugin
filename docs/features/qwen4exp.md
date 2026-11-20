# Qwen3.8-Flash-Next (qwen4exp) GGUF

## Scope

Loading unsloth's `Qwen3.8-Flash-Next-GGUF` (ggml architecture `qwen4exp`,
vLLM `Qwen4ExpForConditionalGeneration` / `Qwen4ExpForCausalLM`, ~125B-A5B)
into vLLM's native Qwen4Exp model: name mapping, value conventions, the
vision tower from `mmproj-*.gguf`, the PLE n-gram table, and the ggml types
newer than gguf-py that these files use.

Not in scope: building an HF config from `qwen4exp` metadata (the config
comes from `Qwen/Qwen3.8-Flash-Next`), the MTP draft heads under `MTP/`
(separate GGUFs; see "MTP" below), Qwen4Exp's kernels (vLLM's), and engram
disk offload.

## Serving

```
vllm serve unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ3_XXS \
  --tokenizer Qwen/Qwen3.8-Flash-Next \
  --pipeline-parallel-size 3
```

`--tokenizer` (or `--hf-config-path`) is required: with a pure GGUF source
`GGUFConfigParser` raises and names the repo. The plugin's
`create_model_config` patch then takes the config from that repo and the
weights from the GGUF. `mmproj-*.gguf` beside the backbone (or
`model_loader_extra_config={"mm_proj": ...}`) supplies the vision tower;
without one, serve language-only with
`--hf-overrides '{"language_model_only": true}'`.

## Data flow

1. `EngineArgs.create_model_config` (patched in `plugin.py`) resolves the
   GGUF shards, sets `quantization/load_format/config_format=gguf`, and —
   because `--tokenizer` is a repo — points `model` at the repo.
2. `GGUFConfigParser.parse` loads the repo config; `get_adapter_architecture`
   finds `Qwen4ExpGGUFAdapter` (model_type `qwen4_exp`/`qwen4_exp_text`).
   A pure GGUF path is rejected by `_reject_config_from_metadata_only`.
3. `GGUFModelLoader._prepare_adapter`:
   * `patch_hf_config` reads `qwen4exp.*` metadata from shard 1
     (`Qwen4ExpGGUFMetadata`) and `validate_qwen4exp_config` compares it with
     the repo config (layer count, heads, experts, SSM, hyper-connections,
     indexer, PLE layout, rope, per-layer compress ratios). Any mismatch
     raises with every differing key.
   * `build_name_map` → `qwen4exp_name_map`: exact per-block names
     (`_LAYER_TENSORS`), globals (`_GLOBAL_TENSORS`), vision via Qwen3.5's
     mapper. Unmapped tensors raise.
   * `get_linear_layouts` (inherited): GDN `out_proj` activation reorder.
4. `load_model` → `configure_quant_config` records the PLE table's ggml type
   on `GGUFConfig.ple_weight_type` and installs the PLE storage hook
   (`quantization/ple_embedding.py`) before the model is built.
5. `gguf_quant_weights_iterator_multi` streams tensors (Q2_0 transcoded to
   Q4_0; tensors >= 4 GiB as zero-copy views of the mapping), and
   `transform_weights` runs `_qwen4exp_weights` and then Qwen3.5's transforms.
6. vLLM's `Qwen4ExpForConditionalGeneration.load_weights` maps HF names to
   its fused modules (`in_proj_qkvz`, `in_proj_ba`, `kv_proj`,
   `input_mix_weight_down_block_inject`, …).

## Tensor mapping

`B` = `model.language_model.` (multimodal) or `model.` (text-only);
`L` = `B + layers.N.`. GGUF shapes are logical, in HF (row-major) order.

| GGUF | HF | Conversion |
|---|---|---|
| `token_embd.weight` [248320, 2560] | `B embed_tokens.weight` | packed GGUF embedding |
| `output.weight` | `lm_head.weight` | packed GGUF lm_head |
| `output_hc_norm.weight` [10240] | `B hyper_connection_mixer.hc_norm.weight` | −1 |
| `output_hc_down.weight` [320, 10240] | `B hyper_connection_mixer.input_mix_weight_down.weight` | dequantize → bf16 |
| `output_hc_up.weight` [10240, 320] | `B hyper_connection_mixer.input_mix_weight_up.weight` | dequantize → bf16 |
| `blk.N.hc_{attn,ffn}_norm.weight` [10240] | `L {attn,mlp}_hyper_connection.hc_norm.weight` | −1 |
| `blk.N.hc_{attn,ffn}_down.weight` [320, 10240] | `L …_hyper_connection.input_mix_weight_down.weight` | dequantize → bf16 |
| `blk.N.hc_{attn,ffn}_up.weight` [10240, 320] | `L …_hyper_connection.input_mix_weight_up.weight` | dequantize → bf16 |
| `blk.N.hc_{attn,ffn}_inject.weight` [4, 10240] | `L …_hyper_connection.block_inject_weight.weight` | dequantize → bf16 |
| `blk.N.attn_q.weight` [12288, 2560] | `L self_attn.q_proj.weight` | Q and sigmoid gate interleaved, as HF |
| `blk.N.attn_{k,v}.weight` [512, 2560] | `L self_attn.{k,v}_proj.weight` | |
| `blk.N.attn_output.weight` | `L self_attn.o_proj.weight` | |
| `blk.N.attn_{q,k}_norm.weight` [256] | `L self_attn.{q,k}_norm.weight` | −1 |
| `blk.N.indexer.q_proj.weight` [512, 2560] + `indexer.k_proj.weight` [128, 2560] | `L self_attn.indexer.index_qk_proj.weight` [640, 2560] | concat q;k, bf16 |
| `blk.N.indexer.{q,k}_norm.weight` [128] | `L self_attn.indexer.{q,k}_layernorm.weight` | −1 |
| `blk.N.attn_qkv.weight` [10240, 2560] | `L linear_attn.in_proj_qkv.weight` | V rows re-grouped |
| `blk.N.attn_gate.weight` [6144, 2560] | `L linear_attn.in_proj_z.weight` | rows re-grouped |
| `blk.N.ssm_{alpha,beta}.weight` [48, 2560] | `L linear_attn.in_proj_{a,b}.weight` | rows re-grouped |
| `blk.N.ssm_conv1d.weight` [10240, 4] | `L linear_attn.conv1d.weight` [10240, 1, 4] | V channels re-grouped, unsqueeze |
| `blk.N.ssm_a` [48] | `L linear_attn.A_log` | `log(-x)`, re-grouped |
| `blk.N.ssm_dt.bias` [48] | `L linear_attn.dt_bias` | re-grouped |
| `blk.N.ssm_norm.weight` [128] | `L linear_attn.norm.weight` | none (conventional scale) |
| `blk.N.ssm_out.weight` [2560, 6144] | `L linear_attn.out_proj.weight` | packed: activation reorder; dense: column re-group |
| `blk.N.ffn_gate_inp.weight` [512, 2560] | `L mlp.gate.weight` | F32 |
| `blk.N.ffn_gate_inp_shexp.weight` [2560] | `L mlp.shared_expert_gate.weight` [1, 2560] | unsqueeze |
| `blk.N.ffn_{gate,up,down}_shexp.weight` | `L mlp.shared_expert.{gate,up,down}_proj.weight` | |
| `blk.N.ffn_{gate,up}_exps.weight` [512, 640, 2560] | `L mlp.experts.E.{gate,up}_proj.weight` | split per expert |
| `blk.N.ffn_down_exps.weight` [512, 2560, 640] | `L mlp.experts.E.down_proj.weight` | split per expert |
| `blk.1.ple_key.weight` [10240, 2560] | `L ple.key_proj.weight` | |
| `blk.1.ple_value.weight` [2560, 2560] | `L ple.value_proj.weight` | |
| `blk.1.ple_conv1d.weight` [10240, 4] | `L ple.conv1d.weight` [10240, 1, 4] | unsqueeze |
| `blk.1.ple_norm_{conv,key,query}.weight` [10240] | `L ple.norm_{conv,key,query}.weight` | −1 |
| `per_layer_token_embd.weight` [320001536, 160] | `L ple.ple_embedding.ngram_embedding.shard_{0..127}.weight` [2500012, 160] | row-split, kept packed |
| metadata `ple.layer_multipliers`, `ple.head_offsets`, `ple.head_vocab_sizes` | `L ple.ple_embedding.{layer_multipliers,ngram_heads_offsets,ngram_heads_vocab_sizes}` | int64 tensors |
| mmproj `v.blk.N.{attn_qkv,attn_out,ffn_up,ffn_down,ln1,ln2}.*` | `model.visual.blocks.N.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2,norm1,norm2}.*` | |
| mmproj `v.patch_embd.weight` + `.weight.1` | `model.visual.patch_embed.proj.weight` [1152, 3, 2, 16, 16] | stack on the temporal axis |
| mmproj `v.position_embd.weight`, `v.post_ln.*`, `mm.0.*`, `mm.2.*` | `model.visual.pos_embed.weight`, `merger.norm.*`, `merger.linear_fc1.*`, `merger.linear_fc2.*` | |

GGUF `ple.layers` is 0-based (`[1]`); HF `ple_layer_ids` is 1-based (`[2]`).
vLLM builds the PLE layer where `layer_idx + 1 in ple_layer_ids`.

### MTP

The main quant files carry no MTP block (`block_count` 48). The `MTP/`
heads are separate `qwen4exp` GGUFs with `blk.48.*` (attention, MoE,
hyper-connections as above) plus `nextn.eh_proj` [2560, 5120] =
`[mtp.fc_embedding | mtp.fc_hidden]` (split on the input axis),
`nextn.enorm` → `mtp.pre_fc_norm_embedding`, `nextn.hnorm` [10240] →
`mtp.pre_fc_norm_hidden`, `nextn.hc_head_{norm,down,up}` →
`mtp.hyper_connection_mixer.*`. The `shared-` variants omit embeddings and
lm_head (`nextn_shared_target_tensors`). No adapter maps these yet.

## Storage decisions

* **Hyper-connections** are built with `quant_config=None` (bf16
  `ReplicatedLinear`/`MergedColumnParallelLinear`), so their Q8_0 payload is
  dequantized on the CPU at load and its `weight_type` dropped
  (`_DEQUANTIZE_ON_LOAD`). +0.6 GB over Q8_0.
* **QSA indexer**: vLLM's projection is one `[640, 2560]` linear; it is
  forced unquantized (`extra_unquantized_modules`), so vLLM keeps it out of
  the fused q/k/v GEMM (different quant methods).
* **PLE table**: vLLM falls back to an unquantized table for unknown quant
  configs (102 GB bf16). `GGUFPLEEmbeddingMethod` keeps rows in their GGUF
  type (IQ4_NL: 90 bytes per 160-value row, 28.8 GB) in whatever storage
  vLLM chose — pinned host under the default `engram_config.cpu_offload`,
  else device — and decodes only looked-up rows (`ggml_dequantize` on CUDA,
  gguf-py on CPU). Rows outside a rank's vocabulary range decode to zero so
  the ETP reduction is unchanged. Disk offload is rejected.
* **Q2_0** (`ffn_down_exps` of 16 layers in UD-Q2_K_XL, 3 in UD-IQ3_XXS) is
  transcoded to Q4_0 on load (`gguf_compat.py`): `q4 = q2 + 7`, same fp16
  scale, bit-exact, twice the bytes (+3.8 GB / +0.7 GB).

## Files

| File | Role |
|---|---|
| `vllm_gguf_plugin/weights_adapter/qwen4_exp.py` | `Qwen4ExpGGUFAdapter`, `Qwen4ExpGGUFMetadata`, `validate_qwen4exp_config`, `qwen4exp_name_map` |
| `vllm_gguf_plugin/quantization/ple_embedding.py` | `GGUFPLEEmbeddingMethod`, `install_qwen4exp_ple_support` (patches `Qwen4ExpPLEEmbeddingMethod.from_quant_config`) |
| `vllm_gguf_plugin/gguf_compat.py` | registers ggml Q1_0/Q2_0/F8_E4M3 on gguf-py's enum; `transcode_q2_0_to_q4_0` |
| `vllm_gguf_plugin/weight_utils.py` | Q2_0 transcode, suffix-only `weight_type_name`, zero-copy views for >= 4 GiB tensors |
| `vllm_gguf_plugin/config_parser.py` | `_reject_config_from_metadata_only` |
| `vllm_gguf_plugin/weights_adapter/base.py` | `configure_quant_config` hook |
| `vllm_gguf_plugin/quantization/utils.py` | fused-layer skip check after `apply_vllm_mapper` renames shards |
| `tests/test_qwen4exp_adapter.py`, `tests/test_qwen4exp_ple.py`, `tests/test_gguf_compat.py`, `tests/test_weight_type_naming.py`, `tests/test_unquantized_fused_layers.py` | CPU tests |
| `tests/fixtures/qwen4exp/` | header-only extracts of the real UD-Q2_K_XL / UD-IQ3_XXS / mmproj files, the repo `config.json`, and the HF checkpoint's names and shapes |

## Invariants

* Every GGUF tensor maps to an HF checkpoint name, or loading fails.
* The repo config and the GGUF metadata describe the same model, or
  loading fails before the model is built.
* All Gemma-style norms (hyper-connection, QSA q/k, indexer q/k, PLE) are
  stored by ggml with `+1` folded in and are restored with `−1`;
  `linear_attn.norm` and vision LayerNorms are not touched.
* The PLE table is never materialised as bf16 and never copied into RAM
  before it is split; its shards are views of the file mapping, which must
  not be written in place.
* A packed tensor's `weight_type` companion renames only the trailing
  `weight` of its name.
