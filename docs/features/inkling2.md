# Inkling 2 Mini (inkling2) GGUF

## Scope

Loading unsloth's `Inkling-2-Mini-GGUF` (revision ea0c79b2; ggml architecture
`inkling2`, root-layout files) into vLLM's
`Inkling2LoopForConditionalGeneration` (HF `model_type: inkling2_loop`) on
the fork's `QUANT_METHOD` backend, so the GGUF kernels serve the attention
linears, embeddings, LM head and routed experts on Ampere: name and value
mapping onto the official `thinkingmachines/Inkling-2-Mini` @ ed40e7da
checkpoint, config validation, the `mmproj-*.gguf` vision tower, ggml NVFP4
experts with their per-expert scales, ggml MXFP4 (as exact Q5_0), and the
DSpark draft's name map.

Not in scope: the model's kernels (relative attention, short convs, loop
executor, router: vLLM's), unsloth's `Mainline_Support/*.gguf_file` layout
(`inkling2_loop`; rejected with a pointer to the root files, which hold the
same tensors byte for byte), wiring the DSpark draft through vLLM's
speculative loader (mapping and conversion exist, the loader hook does not),
and TP > 1 (the fork's `QUANT_METHOD` backend is TP = 1).

## Serving

```
vllm serve unsloth/Inkling-2-Mini-GGUF:UD-IQ4_XS \
  --tokenizer thinkingmachines/Inkling-2-Mini \
  --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice
```

`--tokenizer` (or `--hf-config-path`) is required: a pure `inkling2` GGUF is
rejected by `_reject_config_from_metadata_only`, naming the repo. The repo
config is validated against the GGUF. `mmproj-BF16.gguf` beside the backbone
(or `model_loader_extra_config={"mm_proj": ...}`) supplies the vision tower;
without one, serve text-only with `--limit-mm-per-prompt '{"image": 0}'`.

## Data flow

1. `EngineArgs.create_model_config` (patched) resolves the GGUF; the config
   comes from the repo (`Inkling2LoopMMConfig`, registered by the fork).
2. `get_adapter_architecture` finds `Inkling2GGUFAdapter` (model types
   `inkling2_loop`, `inkling2`).
3. `GGUFModelLoader._prepare_adapter`:
   * `patch_hf_config` validates the text config against `inkling2.*`
     metadata (`validate_inkling2_config`) and, with an mmproj, the vision
     config against `clip.vision.*` (`validate_inkling2_vision_config`).
     Every differing key is reported (`Inkling2ConfigMismatch`).
   * `build_name_map` reads every tensor's ggml type, builds an
     `Inkling2LoadPlan` (`build_load_plan`) and returns its exact
     GGUF -> HF name map (`inkling2_targets`). Unmapped tensors raise
     `UnmappedTensorsError`.
   * `dense_tensors_in_quantized_modules`: dense tensors served by a GGUF
     method (the BF16 `attn_sink_proj` shard beside quantized `q_proj`
     shards; dense experts) are kept out of the unquantized-module list.
   * `patch_hf_config` also decides the LM head (see Storage decisions) and
     sets `extra_unquantized_modules` accordingly.
4. `load_model` -> `configure_quant_config` sets
   `GGUFConfig.moe_expert_scales`, so `GGUFMoEMethod.create_weights` adds
   `w13_expert_scale` / `w2_expert_scale` (`[E]`, F32, default 1).
5. `gguf_quant_weights_iterator_multi` streams tensors (Q2_0 -> Q4_0 and
   MXFP4 -> Q5_0 transcoded, `weight_type` companions first) and
   `inkling2_weights` converts them (below).
6. vLLM's `load_weights` renames HF names to its modules, merges
   `wq_du/wr_du/gate_attn/attn_sink_proj` into `attn.q_proj` shards 0-3, and
   loads per-expert tensors through `load_quant_expert_weight`
   (`…experts.<e>.{gate,up,down}_proj.<suffix>` -> `w13_<suffix>` /
   `w2_<suffix>`).
7. At run time Inkling routes and calls `GGUFMoEMethod.apply`, which passes
   the expert scales and `moe_config.activation_situ_{,linear_}beta` to
   `_fused_moe_gguf`.

## Layer visits and the loop

`inkling2.block_count` is 48: layer visits, not layers (6 prelude + 3 x 12
core + 6 coda). Per-visit arrays: `attention.head_count`,
`attention.head_count_kv`, `attention.sliding_window_pattern` (global
visits 5, 11, 17, 23, 29, 35, 41, 47 = layers 5, 11, 17, 23). The loop is
`num_loops` = 3, `loop.core_start` = 6, `loop.core_end` = 18,
`loop.gate_rank` = 64. Tensors exist once per physical layer (24);
`Inkling2GGUFMetadata.num_layers` derives it. `is_varloop` is not in the
file (taken from the repo).

## Tensor mapping

`L` = `model.llm.layers.N.`; shapes are GGUF logical shapes (row-major).
Checked against the bytes of both repos (134 range-request comparisons).

| GGUF | HF | Conversion |
|---|---|---|
| `token_embd.weight` [201024, 2304] | `model.llm.embed.weight` | packed GGUF embedding |
| `token_embd_norm.weight` | `model.llm.embed_norm.weight` | F32 upcast |
| `output.weight` | `model.llm.unembed.weight` | dequantized to BF16 for the fp32 head (default), else packed |
| `output_norm.weight` | `model.llm.norm.weight` | |
| `output_res_query.weight` | `model.llm.attn_res.output_query` | |
| `loop_res_query.weight` [24, 2304] | `model.llm.middle_cycle_cross_pass_queries` | |
| `loop_kv_res_query.weight` [24, 2304] | `model.llm.middle_cycle_cross_pass_kv_queries` | |
| `blk.N.attn_res_query.weight` (N >= 1; [3, 2304] on 4, 10, 16, 22 → layers 5, 11, 17, 23) | `model.llm.attn_res.attn_queries.{N-1}` | index shift |
| `blk.N.ffn_res_query.weight` | `model.llm.attn_res.mlp_queries.N` | |
| `blk.N.attn_q.weight` [3072, 2304] | `L attn.wq_du.weight` | |
| `blk.N.attn_rel.weight` [384, 2304] | `L attn.wr_du.weight` | |
| `blk.N.attn_gate.weight` [3072, 2304] | `L attn.gate_attn.weight` | |
| `blk.N.attn_sink_proj.weight` [24, 2304] BF16 | `L attn.attn_sink_proj.weight` | typed BF16 shard |
| `blk.N.attn_k.weight` / `attn_v.weight` [1024, 2304] | `L attn.wk_dv.weight` / `attn.wv_dv.weight` | |
| `blk.N.attn_output.weight` [2304, 3072] | `L attn.wo_ud.weight` | |
| `blk.N.attn_rel_bias.weight` [1024, 16] | `L attn.rel_logits_proj.proj` [16, 1024] | transpose |
| `blk.N.attn_q_clip.weight` / `attn_rel_clip.weight` | `L attn.q_l2_clip.scale` / `attn.rel_l2_clip.scale` | |
| `blk.N.attn_{k,v}_conv.weight` [1024, 4], local layers | `L attn.{k,v}_sconv.weight` [1024, 1, 4] | unsqueeze |
| `blk.N.attn_{k,v}_conv.weight` [1024, 4], global layers | `L attn.{k,v}_sconv.static_w` [4, 1024] | `.T.flip(0)` |
| `blk.N.attn_{k,v}_conv_proj.weight` [32, 2304] BF16 | `L attn.{k,v}_sconv.weight_proj` | |
| `blk.N.attn_post_conv.weight` / `ffn_post_conv.weight` [2304, 4] | `L attn_sconv.weight` / `mlp_sconv.weight` [2304, 1, 4] | unsqueeze |
| `blk.N.attn_norm.weight` (local) / `ffn_norm.weight` | `L attn_norm.weight` / `mlp_norm.weight` | |
| `blk.N.attn_{q,k,v}_inp_norm.weight` (global) | `L {q,k,v}_input_norm.weight` | |
| `blk.N.ffn_gate_inp.weight` [128, 2304] F32 | `L mlp.gate.weight` | |
| `blk.N.exp_probs_b.bias` [128] | `L mlp.gate.bias` | |
| `blk.N.ffn_gate_up_exps.weight` [128, 2304, 2304] | `L mlp.experts.w13_weight` (interleaved rows) | gate rows then up rows; per expert `…experts.e.{gate,up}_proj` |
| `blk.N.ffn_down_exps.weight` [128, 2304, 1152] | `L mlp.experts.w2_weight` | per expert `…experts.e.down_proj` |
| `blk.N.ffn_gate_up_exps.scale` [128] (NVFP4 layers 0-19) | `w13_weight_amax / w13_weight_outer_scale_denominator` | per expert `…experts.e.gate_proj.expert_scale` |
| `blk.N.ffn_down_exps.scale` [128] | `w2_weight_amax / w2_weight_outer_scale_denominator` | per expert `…experts.e.down_proj.expert_scale` |
| `blk.N.loop_{attn,k,v,mlp}_{down,gain,forget}.weight` (layers 6-17) | `L controllers.{attn,k,v,mlp}.{down,gain,forget}_weight` | |
| mmproj `v.patch_embd.weight` [1024, 3, 20, 20] | `model.visual.backbone.embeddings.patch_embedding.weight` [1024, 1200] | `permute(0, 2, 3, 1)` (HWC) |
| mmproj `v.class_embd`, `v.class_position_embd.weight`, `v.position_embd.weight`, `v.pre_ln.*` | `…embeddings.class_embedding`, `…cls_position_embedding`, `…position_embedding`, `…ln_pre.*` | |
| mmproj `v.blk.N.{ln1,ln2}.*`, `ls1/ls2.weight` | `…layers.N.ln_{1,2}.*`, `ls_{1,2}.gamma` | |
| mmproj `v.blk.N.attn_{q,k,v,out}.*`, `ffn_up/ffn_down.*` | `…self_attn.{q,k,v,out}_proj.*`, `mlp.c_fc/c_proj.*` | |
| mmproj `mm.1.weight` / `mm.2.weight` | `model.visual.proj_in.weight` / `proj_out.weight` | |

Layers 20-23 have no `.scale`: their experts were MXFP8 in the checkpoint and
are requantized (Q8_0 / Q5_K / IQ4_XS …); the adapter feeds them unit scales.

DSpark (`dflash`, 54 tensors) -> `mtp.safetensors`: `blk.N.attn_{q,k,v}`,
`attn_{k,v}_ctx`, `attn_output`, `attn_{q,k}_norm`, `attn_k_ctx_norm`,
`attn_norm` (input_layernorm), `ffn_norm` (post_attention_layernorm),
`dflare_norm`, `ffn_{gate,up,down}` -> `layers.N.…`; `dspark_fuse` ->
`fuse_weight`, `dspark_hidden_out` -> `markov_head.hidden_out`,
`dspark_norm` -> `markov_head.dflare_norm`, `joint_proj` ->
`markov_head.rnn_proj`, `markov_w1` -> `markov_head.markov_w1`,
`mask_embd` -> `mask_embed`, `output_norm` -> `norm`. No conversions (no rope
permutation). `dspark_weights` dequantizes Q8_0 to BF16 (the draft module is
unquantized).

## Conventions reversed

From llama.cpp `conversion/inkling2.py` (`Inkling2Model`,
`Inkling2LoopModel`, `Inkling2VisionModel`) and unsloth's root layout:

* `w13_weight` rows are interleaved (2c gate, 2c+1 up); ggml stores gate rows
  then up rows. vLLM's `RoutedExperts` wants `[gate; up]` too, so the GGUF
  rows are split in halves, never re-interleaved.
* NVFP4 experts are the checkpoint's own NVFP4 codes and E4M3 block scales,
  repacked into ggml blocks (bit-exact); `amax / outer_scale_denominator`
  becomes the per-expert `.scale`, which ggml multiplies into each expert's
  matmul output (`build_lora_mm_id`): before the activation for gate/up.
* Static short-conv taps: `static_w[K, C].flip(0).T`.
* `rel_logits_proj.proj` transposed; depthwise conv kernels squeezed.
* `attn_res.attn_queries.i` feeds layer `i + 1`.
* No norm offsets (no `+1` folded), no Q/K permutation (no rope in the
  backbone; the DSpark rope projections are stored unpermuted).
* F32 tensors (norms, queries, controllers, router, conv) are exact upcasts
  of the BF16 originals.
* Root layout vs mainline: root keeps the raw cross-pass query matrices
  (`loop_*res_query`) and names controllers `loop_{attn,k,v,mlp}_*`;
  mainline stacks four query rows per core layer (`loop_msg_query`) and uses
  `loop_{q,k,v,ffn}_ctrl_*`.

## ggml types

| Type | Where | Served by |
|---|---|---|
| NVFP4 (40) | experts of layers 0-19, every quant | new Triton dequantize + fused MoE kernels (`triton/*/fp4_quant/nvfp4.py`); MoE takes the per-token path (`MOE_VEC_QUANT_TYPES`) |
| MXFP4 (39) | one `ffn_down_exps` in UD-Q2_K_XL | `transcode_mxfp4_to_q5_0` on load, bit-exact (+29% for that tensor) |
| Q2_0 (42) | `ffn_down_exps` in UD-IQ1_*, IQ2_XXS, Q2_K_XL | existing Q2_0 -> Q4_0 transcode |
| BF16 | `attn_sink_proj`, `attn_*_conv_proj`, Q8_K_XL linears | dense; the sink is a typed shard of a quantized `q_proj` |
| Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q5_0, IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS, IQ2_*, IQ1_S | linears and layers 20-23 experts | existing kernels |

## Storage decisions

* Merged linears whose shards differ in ggml type (Inkling's `q_proj`:
  Q8_0/Q6_K/Q5_K beside a BF16 sink) are stored back to back in one byte
  buffer (16-byte aligned shards) and read per shard as its type, instead of
  padding every row to the widest shard (4608-byte BF16 rows beside
  2448-byte Q8_0 rows would add ~0.34 GB). Uniform-type merges keep the
  fused 2-D layout.
* Dense experts (UD-Q8_K_XL layers 20-23, BF16) carry a `weight_type`
  companion so `GGUFMoEMethod` serves them (slow fallback path).
* The DSpark Q8_0 projections are dequantized to BF16 (0.40 -> 0.65 GB).
* LM head: Inkling2's config sets `head_dtype: float32` (BF16 unembedding,
  fp32 logits), and vLLM's `LogitsProcessor` only runs a different head
  dtype over an unquantized `lm_head`. The adapter then lists `lm_head` as
  unquantized and dequantizes `output.weight` to BF16 on load, in 8192-row
  chunks (UD-IQ4_XS Q6_K 0.38 GB -> 0.93 GB; UD-Q4_K_XL Q8_0 0.49 GB ->
  0.93 GB). `--hf-overrides '{"head_dtype": "model"}'` keeps it packed on
  the GGUF kernels, with bf16 logits.

## Files

| File | Role |
|---|---|
| `vllm_gguf_plugin/weights_adapter/inkling2/metadata.py` | `Inkling2GGUFMetadata`, `LoopLayout`, `validate_inkling2_{,vision_,dspark_}config`, `Inkling2ConfigMismatch` |
| `vllm_gguf_plugin/weights_adapter/inkling2/names.py` | exact name tables, `Conversion`, `TensorTarget`, `inkling2_targets`, `dspark_name_map`, `UnmappedTensorsError` |
| `vllm_gguf_plugin/weights_adapter/inkling2/weights.py` | `Inkling2LoadPlan`, `build_load_plan`, `inkling2_weights` |
| `vllm_gguf_plugin/weights_adapter/inkling2/adapter.py` | `Inkling2GGUFAdapter` |
| `vllm_gguf_plugin/weights_adapter/inkling2/dspark.py` | `dspark_weights` |
| `vllm_gguf_plugin/quantization/nvfp4.py` | torch NVFP4 decoder (reference) |
| `vllm_gguf_plugin/triton/fp4_tables.py`, `triton/dequantize/fp4_quant/`, `triton/fused_moe/fp4_quant/` | NVFP4 Triton kernels |
| `vllm_gguf_plugin/quantization/fused_moe.py` | expert scales, SITU betas, NVFP4 routing |
| `vllm_gguf_plugin/quantization/linear.py` | unpadded mixed-type merged shards |
| `vllm_gguf_plugin/gguf_compat.py`, `weight_utils.py` | MXFP4 -> Q5_0 |
| `vllm_gguf_plugin/loader.py`, `weights_adapter/base.py` | `dense_tensors_in_quantized_modules` hook |
| `vllm_gguf_plugin/config_parser.py` | `inkling2` -> repo pointer |
| `tests/test_inkling2_adapter.py`, `tests/test_inkling2_vllm_names.py`, `tests/test_nvfp4.py`, `tests/test_gguf_moe_expert_scales.py`, `tests/test_mixed_shard_linear.py`, `tests/test_mxfp4_transcode.py` | CPU tests (Triton ones under `TRITON_INTERPRET=1`) |
| `tests/fixtures/inkling2/` | header extracts of UD-IQ4_XS, UD-Q4_K_XL, mmproj-BF16, DSpark Q8_0; ggml types of every file; the repo config; the checkpoint's names/dtypes/shapes; real NVFP4 rows |

## Invariants

* Every GGUF tensor maps to an HF checkpoint name (or a documented plugin
  name for the expert scales), and every HF checkpoint tensor is covered
  apart from the block scales / amax / denominator folded into ggml blocks.
* The repo config and the GGUF metadata (backbone, and mmproj when given)
  describe the same model, or loading fails before the model is built.
* Every expert layer receives a scale for every expert (1 when the GGUF has
  none), so vLLM's missing-tensor check holds.
* A value conversion is never applied to packed bytes (`TypeError`).
* NVFP4 decoding matches gguf-py and the official checkpoint's own NVFP4
  encoding bit for bit.
