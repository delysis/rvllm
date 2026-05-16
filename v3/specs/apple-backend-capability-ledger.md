# Apple Backend Capability Evidence Ledger

This ledger records current Apple backend evidence. It is not a production readiness claim, and the current project is not a production candidate for Gemma inference.

Evidence classes used here:

- `SYNTHETIC-SMOKE`: generated toy weights with hardware token/argmax smoke evidence only.
- `SYNTHETIC-NUMERIC`: generated toy weights with CPU/reference numerical comparison.
- `GENERATED-HF-NUMERIC`: generated HF/Gemma-shaped fixtures with CPU/reference numerical comparison.
- `REAL-CHECKPOINT-DRYRUN`: real-checkpoint metadata/shape validation only; no decode claim.
- `REAL-CHECKPOINT-NUMERIC`: reserved for real-checkpoint numerical comparison; no current row has this class.
- `SCAFFOLD`: planning, API, acceptance, or fallback structure without direct execution evidence.

| Capability | Evidence class | Tests | Hardware? | Limitations | Next proof |
| --- | --- | --- | --- | --- | --- |
| zero-layer real-weight decode | SYNTHETIC-SMOKE | `tiny_zero_layer_model_backend_decodes_token_2_to_3`, Engine zero-layer smokes | Yes, ignored Metal tests | Argmax/token smoke only; no real checkpoint decode | CPU/reference logit parity and real checkpoint path |
| one-layer prefused no-op | SYNTHETIC-SMOKE | `tiny_one_layer_noop_model_backend_decodes_token_2_to_3`, Engine no-op smoke | Yes, ignored Metal tests | Synthetic no-op weights only | Numerical parity for hidden/logits |
| one-layer HF-style no-op | SYNTHETIC-SMOKE | `tiny_one_layer_hf_style_noop_model_backend_decodes_token_2_to_3`, Engine HF-style no-op smoke | Yes, ignored Metal tests | Synthetic HF-shaped names; no nonzero attention/MLP evidence | Add nonzero HF-shaped parity |
| FFN nonzero | SYNTHETIC-SMOKE | `cpu_reference_one_layer_ffn_nonzero_fixture_argmax_is_3`, Metal/Engine FFN smokes | Yes, ignored Metal tests | Synthetic argmax/token evidence | Full logits and residual parity |
| attention nonzero | SYNTHETIC-SMOKE | `cpu_reference_one_layer_attention_nonzero_fixture_argmax_is_3`, Metal/Engine attention smokes | Yes, ignored Metal tests | Synthetic argmax/token evidence | Multi-head/GQA numerical parity |
| multi-head grouped-KV attention | SYNTHETIC-SMOKE | `cpu_reference_multihead_gqa_attention_fixture_argmax_is_3`, `tiny_multihead_gqa_attention_model_backend_decodes_token_2_to_3`, `engine_multihead_gqa_attention_prefill_then_decode_token_2_to_3` | Yes, ignored Metal tests | Synthetic argmax/token evidence | Full logits/residual parity for GQA |
| q_dim distinct from hidden | SYNTHETIC-SMOKE | `cpu_reference_qdim_not_hidden_fixture_argmax_is_3`, `tiny_qdim_not_hidden_model_backend_decodes_token_2_to_3`, `engine_qdim_not_hidden_prefill_then_decode_token_2_to_3` | Yes, ignored Metal tests | Synthetic argmax/token evidence; no real checkpoint decode | Full logits/residual parity for q_dim != hidden |
| full one-layer nonzero argmax | SYNTHETIC-SMOKE | `cpu_reference_one_layer_full_nonzero_fixture_argmax_is_3`, `tiny_one_layer_full_nonzero_model_backend_decodes_token_2_to_3` | Yes, ignored Metal test | Argmax/token evidence plus selected numerical checks | Full residual and full-logit parity |
| selected logits parity | SYNTHETIC-NUMERIC | `tiny_one_layer_full_nonzero_model_backend_selected_logits_match_cpu` | Yes, ignored Metal test | Selected logits only | Full vector comparison when vocab is small |
| full residual parity | SYNTHETIC-NUMERIC | `cpu_reference_one_layer_full_nonzero_residual_vector_is_stable`, `tiny_one_layer_full_nonzero_model_backend_full_residual_matches_cpu` | Yes, ignored Metal test | Synthetic one-layer fixture only | Broaden to real-shape synthetic fixtures |
| short prefill selected logits | SYNTHETIC-NUMERIC | `tiny_prompt_len_two_prefill_selected_logits_match_cpu`, `engine_prompt_len_two_prefill_selected_logits_match_cpu` | Yes, ignored Metal tests | Selected logits only | Full prefill logits for small vocab |
| multi-step decode full logits | GENERATED-HF-NUMERIC | `cpu_reference_generated_tiny_hf_full_logits_are_stable`, `engine_generated_gemma4_hf_end_to_end_full_logits_match_cpu` | Yes, ignored Metal test | Synthetic generated HF/Gemma-shaped fixture | Broaden beyond tiny vocab fixture |
| generated tiny Gemma-shaped full logits | GENERATED-HF-NUMERIC | `tiny_generated_gemma4_hf_end_to_end_model_backend_full_logits_match_cpu` | Yes, ignored Metal test | Synthetic generated HF/Gemma-shaped fixture | Real-shape synthetic dimensions |
| generated tiny reference export bundle | GENERATED-HF-NUMERIC | `generated_tiny_hf_reference_bundle_can_be_exported` | No | Env-gated export hook; exports generated fixture and CPU logits only | Compare exported bundle against Python/PyTorch or another external implementation |
| Gemma dry-run validation | REAL-CHECKPOINT-DRYRUN | `generated_tiny_gemma4_hf_fixture_uses_real_names_and_dry_run_validates`, `gemma4_dry_run_*`, `dry_run_*` tests | No Metal execution required | Metadata/shape validation only; generated fixtures pass; real model dir still optional | Run against real model dir when available |
| optional real Gemma dry-run | REAL-CHECKPOINT-DRYRUN | `gemma4_dry_run_real_model_dir_validates_when_env_is_set`, `real_gemma4_model_dir_dry_run_validates_when_env_is_set` | No decode; optional model dir gate | Skips when `RVLLM_GEMMA4_MODEL_DIR` is unset | Run against real model dir and fix valid shape mismatches |
| ANE planning | SCAFFOLD | `ane_partition_selection_models_dense_blocks`, strict unavailable tests | No real ANE execution | Planning and gating only | Compile dry-run only after Metal correctness gates |
| disaggregated fallback scaffold | SCAFFOLD | `synthetic_one_layer_ane_ffn_*_matches_metal_only_fallback` | No real ANE execution | Fallback scaffold; no production disaggregated inference | Hardware-backed partition execution evidence |
| production acceptance model | SCAFFOLD | `current_incomplete_evidence_fails_with_clear_reasons`, `complete_evidence_can_pass_evaluator_in_isolation` | No | Acceptance criteria model only | Fill with real correctness, real checkpoint, and perf evidence |
