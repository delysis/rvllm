# Generated Tiny Gemma-Shaped Reference Export

The generated tiny HF/Gemma-shaped fixture is synthetic evidence. It is not a real checkpoint. The export hook gives external reference code stable artifacts for comparing the Rust CPU reference and the generated fixture outside the Metal backend.

Run:

```bash
RVLLM_GENERATED_TINY_HF_REFERENCE_DIR=/tmp/rvllm-generated-tiny-reference \
  cargo test -p rvllm-runtime --features apple generated_tiny_hf_reference_bundle_can_be_exported -- --nocapture
```

The output directory contains:

| File | Purpose |
| --- | --- |
| `config.json` | Generated HF/Gemma-shaped model config used by the Rust fixture |
| `model.safetensors` | Generated tiny fixture weights |
| `expected_reference.json` | Prompt tokens, expected generated tokens, and full CPU logits per decode step |

External comparison code should load `config.json` and `model.safetensors`, run prompt `[2, 4]` for two decode steps, and compare every logit in each step against `expected_reference.json`.
