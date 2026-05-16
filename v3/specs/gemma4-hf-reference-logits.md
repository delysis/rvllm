# Gemma4 HF Reference Logits Hook

This hook records Hugging Face/Transformers logits for a local Gemma checkpoint so rvLLM outputs can be compared later. It is not an rvLLM execution path and does not imply Metal, ANE, or production readiness.

Example with explicit token IDs:

```bash
python3 scripts/dump_gemma4_hf_reference_logits.py "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt-token-ids 2,4 \
  --decode-steps 2 \
  --selected-token-ids 3,5 \
  --top-k 16 \
  --output /tmp/gemma4-hf-reference-logits.json
```

Use `--full-logits` only when the output size is acceptable. For real Gemma vocabularies, selected logits plus top-k are usually the safer first artifact. The script requires PyTorch and Transformers in the active Python environment.
