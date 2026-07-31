# Overview

`vllm-gguf-plugin` teaches vLLM to serve GGUF checkpoints out-of-tree. It ships
as a `vllm.general_plugins` entry point (`gguf = "vllm_gguf_plugin:register"`),
so importing vLLM is enough to activate it — there is no explicit setup call in
user code.

A GGUF file is not the shape vLLM expects. There is no `config.json`, no
`tokenizer.json`, no HF repo layout, and weights are quantised in ggml block
formats rather than safetensors. The plugin's job is to synthesise the missing
HF-shaped surfaces from GGUF metadata and to supply kernels that read ggml
blocks directly.

```yaml
Overview:
  description: >
    Out-of-tree GGUF support for vLLM: resolves a GGUF reference to a concrete
    file, synthesises an HF config and tokenizer from its metadata, registers a
    GGUF model loader and quantization backend, and dequantises ggml blocks via
    Triton/CUDA kernels.
  subsystems:
    registration: >
      register() in plugin.py wires everything into vLLM's registries:
      quantization config, model loader (load_format="gguf"), config parser
      (config_format="gguf"), plus monkey-patches over EngineArgs and the
      transformers GGUF loader. Every step is idempotent — register() may run
      more than once per process.
    reference_resolution: >
      Turns whatever the user passed (--model) into a concrete .gguf path:
      a local file, a directory containing one, a "dir:QUANT" spec, or a remote
      "repo:QUANT" that is downloaded. See docs/features/model_resolution.md.
    config_synthesis: >
      GGUFConfigParser + the per-architecture shims (gemma4.py, qwen35.py)
      translate GGUF metadata keys into an HF config dict, mapping the ggml
      architecture name onto a vLLM architecture and filling fields that ggml
      stores implicitly.
    tokenizer_synthesis: >
      A patched AutoTokenizer.from_pretrained (in gemma4.py) restates the
      GGUF reference in the (dir, gguf_file) form transformers needs, then
      repairs what its converters get wrong: BOS/EOS ids for gemma4, and —
      for every architecture — the CONTROL/USER_DEFINED tokens they drop
      from the added-token vocabulary. See docs/features/
      tokenizer_added_tokens.md.
    weight_loading: >
      GGUFModelLoader plus weights_adapter/* map ggml tensor names onto vLLM
      parameter names, handling per-architecture fusions (qkv, gate_up) and
      multimodal/mmproj sidecars.
    quantization: >
      quantization/* implements the GGUF quantization backend (linear, MoE,
      embeddings) over ggml block layouts; triton/ and csrc/ hold the
      dequantisation and quantized-GEMM kernels. csrc/gguf/llamacpp/ vendors
      llama.cpp's tensor-core MMQ for prefill — see
      docs/features/mma_mmq_kernel.md.
  data_flow: >
    EngineArgs.create_model_config (patched) resolves the GGUF reference,
    forces quantization/load_format/config_format to "gguf", rewrites `model`
    to the containing directory, and pins `hf_config_path` and `tokenizer` to
    the resolved file. vLLM then asks the registered config parser for a
    config, AutoTokenizer (patched) for a tokenizer, and GGUFModelLoader for
    weights — each reading the same .gguf.

Features Index:
  model_resolution:
    description: >
      Resolving a GGUF reference to one concrete file, and keeping every
      downstream consumer (config, tokenizer, weights) pointed at that same
      file.
    entry_points: [vllm_gguf_plugin.plugin.register, EngineArgs.create_model_config]
    depends_on: []
    doc: docs/features/model_resolution.md
  hybrid_models:
    description: >
      Serving hybrid attention/recurrent GGUF models (Qwen3.5 gated-delta-net),
      where cache geometry depends on vLLM recognising the model as hybrid.
    entry_points: [vllm_gguf_plugin.qwen35.register_qwen35_gguf_support]
    depends_on: [model_resolution]
    doc: docs/features/hybrid_models.md
  tokenizer_added_tokens:
    description: >
      The added-token vocabulary of a GGUF-derived tokenizer: re-registering
      the CONTROL/USER_DEFINED tokens the transformers converters drop, so
      tokens like <think> encode to one id instead of being BPE-split, and
      naming bos/eos from GGUF metadata so the backend's <s>/</s> defaults
      never mint an id past the embedding matrix.
    entry_points:
      - vllm_gguf_plugin.tokenizer.restore_gguf_added_tokens
      - vllm_gguf_plugin.tokenizer.gguf_special_token_kwargs
    depends_on: [model_resolution]
    doc: docs/features/tokenizer_added_tokens.md
  mma_mmq_kernel:
    description: >
      Routing quantized prefill GEMMs at llama.cpp's tensor-core (MMA) MMQ
      kernels instead of the plugin's DP4A-era copy, whose 4-token-deep output
      tile pins prefill at ~8% of peak. Opt-in via VLLM_GGUF_MMA_MMQ; decode,
      MoE and dequantization are untouched.
    entry_points: [gguf_mma::supported, gguf_mma::launch, ggml_mul_mat_a8]
    depends_on: []
    doc: docs/features/mma_mmq_kernel.md
```

## Invariants

- `register()` is idempotent; each patch guards with a module-level flag or a
  registry lookup.
- Exactly one `.gguf` backs a served model. Config, tokenizer, and weights must
  all resolve to that same file — never re-derive it independently.
- The plugin does not set cache-geometry knobs (`block_size`,
  `mamba_block_size`). Those belong to vLLM, which computes them from the model
  once the architecture is described correctly.
- Repairing a synthesised tokenizer never changes its vocabulary. Added tokens
  are only re-registered at ids the tokenizer already resolves them to, and
  bos/eos are named from GGUF metadata before construction, so no token can be
  minted past `vocab_size` and index off the embedding matrix.
