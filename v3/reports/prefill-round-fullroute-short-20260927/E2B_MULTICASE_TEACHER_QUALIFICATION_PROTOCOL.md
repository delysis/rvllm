# E2B multi-case teacher-session route qualification (prospective, source-only)

This is a correctness check for the feature-gated multi-case teacher session, not a model-quality or timing experiment. No device job has been submitted. The source was selected before inspecting any output from these cases and is distinct from the failed balanced16 v1 cohort.

## Fixed source and model

- Text: `e2b-multicase-teacher-qual-source-v1.json`, SHA-256 `9a8e792014de7096b0ea89c9a2440448acc00797e26676ce67dd35a4b13cb17c`. Both cases are Codex-authored synthetic text; there is no independent corpus or quality oracle.
- Model: cached `google/gemma-4-E2B-it-qat-q4_0-unquantized` snapshot `6befbaca7398925921802abd1f277b495b78b738`, not the 12B W4 package. `model.safetensors` SHA-256 `33fe0cece08fb527ffefbd1a3a9ce73bd71073727993a283506293e5c6bf0137`; `config.json` `bbeff1e2fd3fe282536e7ace02309d43e0dbd9b6ac4b6a149b97e3ab6942a878`; `generation_config.json` `b69207f9be617e982d13cc273cce6fd88c98dda99a4bdc5e2d52ffe0a0d9f0a9`; `tokenizer.json` `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.
- The existing safe-Rust heldout tokenizer validated exact prompt-prefix continuation for both cases. Observatory: 184 prompt tokens, forced targets `[108355,236761]`. Kitchen: 184 prompt tokens, forced targets `[6819,236761]`. No model output was consulted to choose them.

## Planned immutable arms

After exact executable, source, input and manifest hashes are sealed, submit once through the existing serial queue, in this order: observatory standalone teacher, kitchen standalone teacher, then observatory+kitchen in one direct Metal teacher session. Use fresh IDs containing `e2b-multicase-teacher-qual-v1-20261001`; do not use any completed job ID. All arms use the same frozen executable, E2B model, selector `off`, no HF reference or EOS list, zero thermal dwell, a 32 GiB free-disk floor, no competing GPU work, and bounded run/wait times. The two-case input order is fixed. Every prompt and target must be pinned by hash before admission. Teacher readback makes every time and throughput value invalid.

## Frozen acceptance logic to implement before submission

Require all three outer queue jobs terminal succeeded with exit 0, unchanged pinned files, eligible complete condition journals (including sampled activity), and no violations. Require full five-file receipt hashes and the exact pinned command, environment, model, executable and input identities. The standalone reports must each have two teacher-forced steps. The batched report must use `rvllm.metal_teacher_forced_session.v1`, have exactly two cases in order, `checkpoint_complete: true`, `timing_valid: false`, and no timing/throughput fields. For each corresponding case require exact prompt, prompt-token, forced-token, output-token, finish-reason, per-step target/sampled ID, logit, rank and NLL, and aggregate teacher-score equality between standalone and batched arms; require complete finite values and selector-off dispatch without overflow. Any mismatch or missing evidence rejects the cohort. Preserve failure and do not retry, cherry-pick, widen a tolerance, or call a partial report complete.

Passing would qualify only case independence for these two fresh E2B synthetic inputs under this exact runtime and environment. It would not validate the 12B W4 package, natural-question quality, generic prefill-kernel attribution, speed, full-route parity, or production promotion. A separate prospective natural-quality cohort would still be required.
