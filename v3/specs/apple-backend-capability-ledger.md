# Apple Backend Capability Evidence Ledger

This ledger records current Apple backend evidence. It is not a broad
production-readiness claim. The current Metal pass has real local execution
evidence for cached Gemma 4 E2B, E4B-it, 26B-A4B-it, and 31B-it snapshots,
including the 26B-A4B MoE block. It does not establish ANE execution, XLA
parity, CUDA parity on this macOS host, broad BF16 numerical parity, or coverage
for missing local safetensors snapshots.

Evidence classes used here:

- `SYNTHETIC-SMOKE`: generated toy weights with hardware token/argmax smoke evidence only.
- `SYNTHETIC-NUMERIC`: generated toy weights with CPU/reference numerical comparison.
- `GENERATED-HF-NUMERIC`: generated HF/Gemma-shaped fixtures with CPU/reference numerical comparison.
- `REAL-CHECKPOINT-DRYRUN`: real-checkpoint metadata/shape validation only; no decode claim.
- `REAL-CHECKPOINT-NUMERIC`: real-checkpoint numerical comparison against an external reference, scoped by prompt, logits, layers, and tolerance.
- `SCAFFOLD`: planning, API, acceptance, or fallback structure without direct execution evidence.

## Cooperative Microbatch Projection Evidence (2026-07-15)

Scope: Metal projection scheduling for decode and prompt microbatches only.
Projection+RMSNorm sites with `M <= 19` use the cooperative GEMV path followed
by explicit RMSNorm; larger shapes keep the one-encoder fused fallback. The
cutoff comes from matched M4 Max E2B measurements: cooperative was faster at
M=17 (365.97 vs 395.78 ms) and M=18 (382.60 vs 398.30 ms), effectively tied at
M=19 (400.52 vs 405.37 ms), and slower at M=20 (444.40 vs 405.76 ms). It was
also clearly slower at M=24 (543.77 vs 419.01 ms), supporting the fused path
for larger prefills. These crossover probes preserved the same greedy token
between policies but are not external long-context numerical references.

The policy is covered by a structural boundary test, its additional encoders
are included in runtime counters, and checked-in E2B HF token artifacts replace
the previous test dependency on `/tmp` files. The model suite now supports
fail-closed per-case `min_tok_per_s`, `max_prefill_ms`, and `max_decode_ms`
gates so local hardware baselines can prevent silent performance regressions.

The final enforced E2B four-token run cleared `min_tok_per_s=20.0`,
`max_prefill_ms=300.0`, and `max_decode_ms=250.0` with 26.92 tok/s, 166.62 ms
prefill, 148.60 ms decode, exact HF-token match, and no gate failures. Prepared
HTTP serving returned the same token sequence at 25.49 tok/s and 156.92 ms
decode, with `/healthz` confirming one-time model preparation.

Real-checkpoint results preserved the established greedy tokens for E2B
four-step (`[236764,108,236777,735]`), E4B-it one-step (`[236888]`), and
26B-A4B-it MoE one-step (`[993]`). Three E2B measurements were 25.1--26.2
tok/s with 152.7--159.6 ms total decode, compared with the prior 4.40 tok/s and
908.1 ms. E4B-it measured 66.0 ms decode versus 459.4 ms previously; 26B-A4B
measured 71.8 ms versus 234.4 ms previously. This is strong local
real-checkpoint token-correctness and performance-regression evidence, but not
CUDA, ANE, long-context numerical parity, energy, or broad production-readiness
evidence.

## Metal Layer-Scale Fusion Pass (2026-05-30)

Scope: Metal-only optimization and instrumentation. ANE code and shared-KV
optimization were not touched. No Python tooling was added. This pass folds the
final per-layer residual add and following layer-scale multiply into
`residual_add_then_scale_f16` when the layer shape has a scalar layer scale,
preserving the old half-rounded sequence (`f16(residual + addition)`, then
`f16(updated * scale)`). Unsupported shapes continue to use the existing
fallback kernels. Reports now expose `layer_scale_encoder_fusions` and the
`encoder_counts_by_kernel_family.layer_scale_fused` counter on the diagnostic
CLI, session reports, model suite payloads, bench/eval/PPL paths, and prepared
server Metal reports.

Required checks run after the change:

```bash
cargo test -p rvllm-apple-metal -- --nocapture
cargo test -p rvllm-runtime --features apple --bin rvllm_metal_infer -- --nocapture
cargo test -p rvllm-runtime --features apple --bin rvllm_metal_model_suite -- --nocapture
cargo test -p rvllm-bench --features apple apple_metal_text -- --nocapture
cargo test -p rvllm-serve --features apple -- --nocapture
```

All listed checks passed. The `rvllm-apple-metal` suite includes the new
kernel-level CPU/Metal reference test
`residual_add_then_scale_matches_sequential_half_rounded_reference`.

Reference-backed E2B single-prompt evidence after the fusion:

```bash
target/debug/rvllm_metal_infer \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
  --prompt Hello \
  --max-new-tokens 1 \
  --max-total-tokens 128 \
  --hf-reference /tmp/rvllm-gemma4-metal-family-f16-refs/gemma4-e2b-hello-1tok.json \
  --json > /tmp/rvllm-e2b-hello-one-token-layer-scale-fused-bf16.json

target/debug/rvllm_metal_infer \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
  --prompt Hello \
  --max-new-tokens 4 \
  --max-total-tokens 128 \
  --hf-reference /tmp/rvllm-e2b-text-reference-suite-layer-scale-fused/gemma4-e2b-hf-text-reference-hello-steps4.json \
  --json > /tmp/rvllm-e2b-hello-four-token-layer-scale-fused-bf16.json
```

Observed one-token pass: generated `[236764]`, `hf_reference.matched=true`,
`metal_weight_dtype=bfloat16`, `decode_ms=225.284125`, `tok_per_s=4.438839`,
`encoders=993`, `layer_encoders=980`, `layer_scale_encoder_fusions=70`, and
`forced_waits=2`. This reduces the previous one-token BF16 encoder count from
`1063` to `993`. Observed four-token pass: generated
`[236764,108,236777,735]`, `hf_reference.matched=true`,
`decode_ms=908.139792`, `tok_per_s=4.404608`, `encoders=2487`,
`layer_encoders=2450`, `layer_scale_encoder_fusions=175`, and
`forced_waits=5`, reducing the previous four-token encoder count from `2662`
to `2487`.

Reference-backed E2B session evidence:

```bash
target/debug/rvllm_metal_infer \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
  --reference-manifest /tmp/rvllm-e2b-text-reference-suite-layer-scale-fused/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --session-backend direct \
  --max-total-tokens 128 \
  --case-timeout-seconds 120 \
  --report /tmp/rvllm-e2b-direct-session-layer-scale-fused-bf16.json \
  --json > /tmp/rvllm-e2b-direct-session-layer-scale-fused-bf16.stdout.json

target/debug/rvllm_metal_infer \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
  --reference-manifest /tmp/rvllm-e2b-text-reference-suite-layer-scale-fused/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --session-backend engine \
  --max-total-tokens 128 \
  --case-timeout-seconds 120 \
  --report /tmp/rvllm-e2b-engine-session-layer-scale-fused-bf16.json \
  --json > /tmp/rvllm-e2b-engine-session-layer-scale-fused-bf16.stdout.json
```

Direct session passed all three reference cases: `Hello -> [236764]`,
`Once upon a time -> [236764]`, `The capital of France is -> [9079]`,
`prepare_ms=1308.634792`, `prefill_ms=860.986708`,
`decode_ms=680.701083`, `tok_per_s=4.407221`, `encoders=2979`,
`layer_scale_encoder_fusions=210`, and `forced_waits=6`. Engine session passed
the same cases with `prepare_ms=1309.398833`, `prefill_ms=435.702084`,
`decode_ms=251.777542`, `tok_per_s=11.915280`, `encoders=993`,
`layer_scale_encoder_fusions=70`, and `forced_waits=2`.

All-local Gemma 4 safetensors suite after the fusion:

```bash
target/debug/rvllm_metal_model_suite \
  --manifest /tmp/rvllm-gemma4-metal-all-local-safetensors-suite-f16refs-manifest.json \
  --report /tmp/rvllm-gemma4-metal-family-layer-scale-fused-bf16-report.json \
  --json > /tmp/rvllm-gemma4-metal-family-layer-scale-fused-bf16.stdout.json
```

Observed status `pass`, `passed=4`, `skipped=0`. E2B generated `[236764]`,
`decode_ms=223.047125`, `encoders=993`,
`layer_scale_encoder_fusions=70`; E4B-it generated `[236888]`,
`decode_ms=459.360709`, `encoders=1189`,
`layer_scale_encoder_fusions=84`; 26B-A4B-it generated `[993]`,
`decode_ms=234.360792`, `encoders=965`,
`layer_scale_encoder_fusions=60`; 31B-it generated `[9259]`,
`decode_ms=3323.586083`, `encoders=1205`,
`layer_scale_encoder_fusions=120`. All four cases used
`metal_weight_dtype=bfloat16` and matched their configured token references.

Session profile report:

```bash
target/debug/rvllm_metal_infer \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
  --reference-manifest /tmp/rvllm-e2b-text-reference-suite-layer-scale-fused/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --session-backend engine \
  --max-total-tokens 128 \
  --case-timeout-seconds 120 \
  --profile-samples 3 \
  --profile-report /tmp/rvllm-e2b-engine-session-layer-scale-fused-profile-bf16.json \
  --json > /tmp/rvllm-e2b-engine-session-layer-scale-fused-profile-bf16.stdout.json
```

Median profile values: `prepare_ms=1314.781542`, `prefill_ms=422.955542`,
`decode_ms=255.039875`, `tok_per_s=11.762866`,
`latency_ms_per_token=85.013292`,
`command_buffers_per_token=0.6666666666666666`,
`encoders_per_token=331.0`, `forced_waits_per_token=0.6666666666666666`,
`cpu_encode_ms_per_token=2.302570`,
`command_buffer_wait_ms_per_token=223.704931`, and
`layer_scale_fusions_per_token=23.333333333333332`.

Public E2B surfaces after the same fusion:

- `rvllm-eval` with `RVLLM_BACKEND_PROFILE=apple`, `RVLLM_PROMPT=Hello`,
  `RVLLM_MAX_TOKENS=1`, and `RVLLM_METAL_MAX_TOTAL_TOKENS=128` generated
  `","`; stderr reported `tok_per_sec=4.424`, `prepare_ms=1352.1`,
  `prefill_ms=320.7`, and `decode_ms=226.0`.
- `rvllm-bench` wrote
  `/tmp/rvllm-e2b-bench-layer-scale-fused-bf16.json`: schema
  `rvllm.apple_metal_bench.v1`, `tok_per_sec=1.7621903566553982`,
  `prepare_ms=1837.891375`, `encoders=993`,
  `layer_scale_encoder_fusions=70`, and `forced_waits=2`.
- `rvllm-ppl` wrote
  `/tmp/rvllm-e2b-ppl-layer-scale-fused-bf16.json`:
  `perplexity=329.24177482909147`, `tokens=1`,
  `prepare_ms=1914.683375`, `encoders=996`,
  `layer_scale_encoder_fusions=70`, and `forced_waits=3`.
  Bench and PPL JSON now include
  `encoder_counts_by_kernel_family.layer_scale_fused`.

External profiler status: `xcrun xctrace record --template 'Power Profiler'`
failed because the Power Profiler instrument is not supported on macOS. A
`Metal System Trace` launch did not terminate cleanly under `xctrace`; the
recording had to be terminated and `xcrun xctrace export --toc` reported
`Document Missing Template Error`. Therefore this pass has internal timing and
counter evidence only, not external energy or Instruments evidence.

Acceptance status for this pass: correctness stayed green for the current E2B
one-token, four-token, direct-session, and Engine-session token gates; cached
E2B/E4B-it/26B-A4B-it/31B-it one-token suite cases passed; and the structural
performance gate improved by reducing E2B one-token encoder count below the
previous `1063` counter. The path is still not recorded as production-ready
because external energy/profiler evidence is missing, CUDA parity is not
runnable on this host, and broad long-context/performance coverage remains
open.

## Metal Production-Candidate Pass (2026-05-26)

Scope: Metal-only Gemma 4 execution. XLA was intentionally excluded for this
pass; ANE and shared-KV optimization were not touched. CUDA parity was not
runnable on this macOS host (`nvidia-smi` was not present), so no CUDA result is
claimed. The current supported local slice is cached E2B, E4B-it, 26B-A4B-it,
and 31B-it execution through the default checkpoint-native Metal path. BF16
Google safetensors now stay BF16-native by default; `RVLLM_METAL_DTYPE=f16` is
an explicit fallback/conversion mode. The 26B-A4B-it Google safetensors snapshot
is a MoE model; it is no longer recorded as an expected error, and the Rust
suite treats it as a required case. Native BF16 Metal passes the current
family token-reference gates listed below, but the HF BF16 CPU/MPS reference
surfaces disagree for 26B-A4B, so broad production readiness and CUDA/BF16
parity are not claimed.

Implementation changes:

- Replaced the old short probe-token cap on `rvllm_metal_infer`,
  `rvllm_metal_model_suite`, `rvllm-bench`, `rvllm-eval`, `rvllm-ppl`, and
  prepared `rvllm-server --backend metal-direct` with configurable Metal arena
  sizing: `RVLLM_METAL_MAX_TOTAL_TOKENS`, `RVLLM_METAL_MAX_BATCH_TOKENS`, and
  `RVLLM_METAL_MAX_BATCH_SEQUENCES`. The legacy
  `RVLLM_METAL_MAX_PROBE_TOKENS` remains a compatibility fallback.
- `--large-model-opt-in` remains accepted for compatibility but is no longer
  required by the public Metal text paths for E2B/31B.
- The Rust model-suite manifest now supports optional cases with
  `required:false`; missing optional model dirs are reported as
  `skip_missing_model` rather than silently omitted or treated as a successful
  decode.
- The Rust model-suite manifest rejects `expected_error_contains`; present
  local model snapshots must pass the configured case or fail loudly. Each case
  records command, stdout, stderr, return code, duration, timeout, and parsed
  report evidence.
- The Metal loader now validates and maps Gemma 4 MoE tensors
  (`router.*`, `experts.*`, and the additional MoE feed-forward norms), and the
  Metal layer path executes the dense MLP branch plus the routed expert branch.

Additional Google safetensors snapshots included on 2026-05-26:

- `google/gemma-4-E4B-it` at `/Users/george/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/d6436b3d62967e1af08bbb046c6300b2a9ae8e85`
- `google/gemma-4-26B-A4B-it` at `/Users/george/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/b2a81a03d25f927590a91d84ba43f96e8ef7349f`

Validation:

- E4B-it dry-run passed: 42 layers, hidden size 2560, 35 sliding + 7 full
  layers, tied embeddings, Gemma 4 tensor shapes validated.
- 26B-A4B-it dry-run passed: 30 layers, hidden size 2816, 25 sliding + 5 full
  layers, 5 global layers using `attention_k_eq_v`, tied embeddings, MoE tensor
  shapes validated.
- 31B-it dry-run passed: 60 layers, hidden size 5376, 50 sliding + 10 full
  layers, 10 global layers using `attention_k_eq_v`, tied embeddings, Gemma 4
  tensor shapes validated.

Reference-backed E2B single prompt gates:

```bash
target/debug/rvllm_metal_infer \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
  --prompt Hello \
  --max-new-tokens 1 \
  --max-total-tokens 128 \
  --hf-reference /tmp/gemma4-e2b-hf-text-infer-hello-step1.json \
  --json > /tmp/rvllm-e2b-hello-step1-configurable-context.json

target/debug/rvllm_metal_infer \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
  --prompt Hello \
  --max-new-tokens 4 \
  --max-total-tokens 128 \
  --hf-reference /tmp/gemma4-e2b-hf-text-infer-hello-steps4.json \
  --json > /tmp/rvllm-e2b-hello-step4-configurable-context.json
```

Observed passes: one-token `[236764]`, `hf_reference.matched=true`,
`prepare_ms=6318.612541`, `decode_ms=220.994541`,
`tok_per_s=4.5249986514372775`, `encoders=1063`, `forced_waits=2`. Four-token
`[236764,108,236777,735]`, `hf_reference.matched=true`,
`decode_ms=893.389417`, `tok_per_s=4.477330852465203`.

Reference-backed E2B session gates:

```bash
target/debug/rvllm_metal_infer \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
  --reference-manifest /tmp/rvllm-e2b-text-reference-suite-current/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --session-backend direct \
  --max-total-tokens 128 \
  --case-timeout-seconds 900 \
  --report /tmp/rvllm-e2b-direct-session-configurable-context.json \
  --json > /tmp/rvllm-e2b-direct-session-configurable-context.stdout.json

target/debug/rvllm_metal_infer \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
  --reference-manifest /tmp/rvllm-e2b-text-reference-suite-current/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --session-backend engine \
  --max-total-tokens 128 \
  --case-timeout-seconds 900 \
  --report /tmp/rvllm-e2b-engine-session-configurable-context.json \
  --json > /tmp/rvllm-e2b-engine-session-configurable-context.stdout.json
```

Observed direct session: status `pass`, all three references matched, generated
tokens `Hello -> [236764]`, `Once upon a time -> [236764]`,
`The capital of France is -> [9079]`, `prepare_ms=5377.449209`, and
`tok_per_s=4.449827912175971`. Observed Engine session: status `pass`, all
three references matched, `prepare_ms=5331.200708`, `prefill_ms=433.373875`,
`decode_ms=255.023375`, `tok_per_s=11.763627549827541`, `encoders=1063`, and
`forced_waits=2`.

Session profile:

```bash
target/debug/rvllm_metal_infer \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
  --reference-manifest /tmp/rvllm-e2b-text-reference-suite-current/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --session-backend engine \
  --max-total-tokens 128 \
  --case-timeout-seconds 900 \
  --profile-samples 2 \
  --profile-report /tmp/rvllm-e2b-engine-session-profile-configurable-context.json \
  --json > /tmp/rvllm-e2b-engine-session-profile-configurable-context.stdout.json
```

Median profile values: `prepare_ms=5232.5353335`,
`prefill_ms=428.78175`, `decode_ms=251.57218749999998`,
`tok_per_s=11.925064075849907`,
`command_buffers_per_token=0.6666666666666666`,
`encoders_per_token=354.3333333333333`,
`forced_waits_per_token=0.6666666666666666`,
`cpu_encode_ms_per_token=2.2561531666666665`, and
`command_buffer_wait_ms_per_token=224.52197916666665`.

Rust model suite:

```bash
target/debug/rvllm_metal_model_suite \
  --manifest /tmp/rvllm-gemma4-metal-production-candidate-suite-manifest.json \
  --report /tmp/rvllm-gemma4-metal-production-candidate-suite-report.json \
  --json > /tmp/rvllm-gemma4-metal-production-candidate-suite.stdout.json
```

Earlier pre-download observed status was `pass`, `passed=3`, `skipped=2`. E2B one-token HF matched
`[236764]` with `decode_ms=221.136`; E2B four-token HF matched
`[236764,108,236777,735]` with `decode_ms=891.5538750000001`; 31B-it one-token
execution produced `[9259]` with `decode_ms=3307.8813330000003` and no local HF
reference. At that time, optional missing Google safetensors cases
`gemma4-e4b-it-missing-safetensors-snapshot` and
`gemma4-26b-a4b-it-missing-safetensors-snapshot` were explicit
`skip_missing_model` entries.

Updated all-local safetensors suite after making `auto`/unset
`RVLLM_METAL_DTYPE` checkpoint-native:

```bash
target/debug/rvllm_metal_model_suite \
  --manifest /tmp/rvllm-gemma4-metal-all-local-safetensors-suite-f16refs-manifest.json \
  --report /tmp/rvllm-gemma4-metal-family-native-bf16-default-report.json \
  --json > /tmp/rvllm-gemma4-metal-family-native-bf16-default.stdout.json
```

Observed status `pass`, `passed=4`, `skipped=0`, with no expected-error
contract. All four cases used the existing token reference artifacts and ran
with default Metal BF16 execution. The final report records
`metal_compute_dtype=bfloat16`, `metal_weight_dtype=bfloat16`, and
`metal_moe_router_weight_dtype=bfloat16`:

- E2B generated `[236764]`, output `"Hello,"`, `hf_reference.matched=true`,
  `prepare_ms=2400.1242500000003`, `prefill_ms=323.321542`,
  `decode_ms=224.055625`, `tok_per_s=4.463177391774922`, `encoders=1063`,
  `forced_waits=2`, `arena_bytes=9306476470`.
- E4B-it generated `[236888]`, output `"Hello!"`,
  `hf_reference.matched=true`, `prepare_ms=3966.834417`,
  `prefill_ms=633.378458`, `decode_ms=470.596`,
  `tok_per_s=2.124964938078522`, `encoders=1273`, `forced_waits=2`,
  `arena_bytes=15055987300`.
- 26B-A4B-it generated `[993]`, output `"Hello there"`,
  `hf_reference.matched=true`, `prepare_ms=10960.55775`,
  `prefill_ms=805.81625`, `decode_ms=239.548375`,
  `tok_per_s=4.174522160711797`, `encoders=1025`, `forced_waits=2`,
  `arena_bytes=50551220556`.
- 31B-it generated `[9259]`, output `"HelloHello"`,
  `hf_reference.matched=true`, `prepare_ms=18911.961834`,
  `prefill_ms=5445.652125`, `decode_ms=3346.544625`,
  `tok_per_s=0.2988156776782859`, `encoders=1325`, `forced_waits=2`,
  `arena_bytes=61739678024`.

The explicit F16 fallback 26B-A4B MoE safetensors gate was also rerun directly
before switching the default to checkpoint-native BF16:

```bash
RVLLM_METAL_DTYPE=f16 \
target/debug/rvllm_metal_infer \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/b2a81a03d25f927590a91d84ba43f96e8ef7349f \
  --prompt Hello \
  --max-new-tokens 1 \
  --max-total-tokens 128 \
  --hf-reference /tmp/rvllm-gemma4-metal-family-f16-refs/gemma4-26b-a4b-it-hello-1tok.json \
  --json > /tmp/rvllm-gemma4-26b-a4b-it-hello-1tok-default-f16-after-dtype-fix.json
```

Observed generated token `[993]`, `hf_reference.matched=true`,
`metal_compute_dtype=float16`, `metal_weight_dtype=float16`,
`metal_moe_router_weight_dtype=float32`, `prepare_ms=26389.015042`,
`prefill_ms=871.1594580000001`, `decode_ms=235.684541`, and
`tok_per_s=4.24295966021802`.

The separate 26B-A4B HF BF16 reference comparison remains a reference-policy
blocker, not an expected pass:

```bash
RVLLM_METAL_DTYPE=bf16 target/debug/rvllm_metal_infer \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/b2a81a03d25f927590a91d84ba43f96e8ef7349f \
  --prompt Hello \
  --max-new-tokens 1 \
  --max-total-tokens 128 \
  --hf-reference /tmp/rvllm-gemma4-metal-family-f16-refs/gemma4-26b-a4b-it-hello-1tok-bf16.json \
  --top-logits 8 \
  --json > /tmp/rvllm-gemma4-26b-a4b-it-hello-1tok-explicit-native-bf16-after-bf16-guard.json
```

Observed native BF16 Metal generated `[993]` while BF16 HF generated `[107]`;
`hf_reference.matched=false`, with `metal_compute_dtype=bfloat16`,
`metal_weight_dtype=bfloat16`, and `metal_moe_router_weight_dtype=bfloat16`.
The diagnostic top logits were token `993` at `7.25`, `1018` at `6.75`,
`236761` at `6.65625`, and `107` at `6.5`. This is the current numeric blocker
for claiming CUDA/BF16 parity or broad production readiness for 26B-A4B, not a
reason to convert BF16 checkpoints to F16 by default.

An additional HF CPU `torch_dtype=auto` BF16 reference using the same prompt
token IDs `[2,9259]` was generated at
`/tmp/rvllm-gemma4-26b-a4b-it-hello-bos-1tok-hf-cpu-auto-bf16.json`; it
selected token `[532]` with top logits `532=5.0`, `107=4.71875`, and
`993=4.5625`. This shows the current BF16 reference surface is not stable
across HF CPU/MPS devices either; it does not make native Metal BF16 a passing
gate.

E2B three-prompt session gates were rerun with unset `RVLLM_METAL_DTYPE`, so
the default BF16 path was used:

- Direct session report
  `/tmp/rvllm-e2b-direct-session-native-bf16-default-report.json`: status
  `pass`, generated tokens `[236764]`, `[236764]`, `[9079]`, all
  `hf_reference.matched=true`, `metal_weight_dtype=bfloat16`,
  `prepare_ms=2375.521`, `prefill_ms=849.0745830000001`,
  `decode_ms=677.807666`, `tok_per_s=4.426034331691963`, `encoders=3189`, and
  `forced_waits=6`.
- Engine session report
  `/tmp/rvllm-e2b-engine-session-native-bf16-default-report.json`: status
  `pass`, generated tokens `[236764]`, `[236764]`, `[9079]`, all
  `hf_reference.matched=true`, `metal_weight_dtype=bfloat16`,
  `prepare_ms=1498.920708`, `prefill_ms=428.586709`,
  `decode_ms=253.98162499999998`, `tok_per_s=11.81187812307288`,
  `encoders=1063`, and `forced_waits=2`.

E2B four-token decode was rerun under the default BF16 path at
`/tmp/rvllm-e2b-hello-four-token-native-bf16-default-report.json`: generated
`[236764,108,236777,735]`, `hf_reference.matched=true`,
`metal_weight_dtype=bfloat16`, `prepare_ms=1452.681167`,
`prefill_ms=334.675958`, `decode_ms=886.9545420000001`,
`tok_per_s=4.509813987738731`, `encoders=2662`, and `forced_waits=5`.

26B-A4B four-token F16 gate:

```bash
RVLLM_METAL_MAX_TOTAL_TOKENS=128 target/debug/rvllm_metal_infer \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/b2a81a03d25f927590a91d84ba43f96e8ef7349f \
  --prompt Hello \
  --max-new-tokens 4 \
  --max-total-tokens 128 \
  --hf-reference /tmp/rvllm-gemma4-metal-family-f16-refs/gemma4-26b-a4b-it-hello-4tok.json \
  --json > /tmp/rvllm-gemma4-26b-a4b-it-hello-4tok-f16-report.json
```

Observed generated tokens `[993,236888,1030,236789]`, output
`"Hello there! It'"`, `hf_reference.matched=true`, `decode_ms=963.987125`,
`tok_per_s=4.149433012396302`, `command_buffers=5`, `encoders=2567`, and
`forced_waits=5`.

Public E2B surfaces:

- `rvllm-eval`: `RVLLM_BACKEND_PROFILE=apple RVLLM_MODEL_DIR=<E2B>
  RVLLM_PROMPT=Hello RVLLM_MAX_TOKENS=1 RVLLM_METAL_MAX_TOTAL_TOKENS=128
  target/debug/rvllm-eval` generated `","`, `prepare_ms=6216.8`,
  `decode_ms=225.0`, `tok_per_sec=4.445`.
- `rvllm-bench`: same model/prompt with `RVLLM_BATCH=1 RVLLM_ITERS=1
  RVLLM_WARMUP=0` emitted schema `rvllm.apple_metal_bench.v1`,
  `prepare_ms=5342.3167920000005`, `tok_per_sec=1.823201423191031`,
  `command_buffers=2`, `encoders=1063`, `forced_waits=2`.
- `rvllm-ppl`: `RVLLM_PROMPT='Hello world' RVLLM_PPL_CHUNK=3
  RVLLM_PPL_CHUNKS=1` emitted `perplexity=504.1798937686529`, `tokens=2`,
  `command_buffers=6`, `encoders=2132`, `final_logits_encoders=6`.
- Prepared server: `target/debug/rvllm-server --model-dir <E2B> --backend
  metal-direct --addr 127.0.0.1:18082 --max-new-tokens 1 --max-total-tokens
  128`; health reported `prepared_once=true`, and `/v1/completions` generated
  `","` with token `[236764]`, `decode_ms=223.831042`, `tok_per_s=4.467655563163576`,
  `encoders=1063`, `forced_waits=2`.

External profiler:

```bash
xcrun xctrace record \
  --template 'Metal System Trace' \
  --output /tmp/rvllm-e2b-metal-configurable-context.trace \
  --launch -- target/debug/rvllm_metal_infer \
    --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
    --prompt Hello \
    --max-new-tokens 1 \
    --max-total-tokens 128 \
    --hf-reference /tmp/gemma4-e2b-hf-text-infer-hello-step1.json \
    --json
```

Exported `metal-application-command-buffer-submissions` showed two command
buffers, with `495` and `498` encoders. Total exported command-buffer duration
was `11.619543 ms`; total exported encoder time was `9.736795 ms`. The previous
fused trace had the same two-buffer `495/498` encoder shape, so this pass does
not claim a new trace-level encoder reduction.

Acceptance status for this pass: default checkpoint-native BF16 Metal execution
now covers the cached local E2B, E4B-it, 26B-A4B-it MoE, and 31B-it snapshots.
Correctness did not regress for E2B, the one-token E2B direct decode stayed
around the recorded `~226 ms` baseline, 31B still executes, public surfaces
pass, and all JSON claims still avoid broad production readiness. The 26B-A4B HF
BF16 reference disagreement above keeps the broader production/parity gate open.

## Metal Completion Pass Evidence (2026-05-26)

This pass keeps the claim boundary unchanged: bounded Apple Metal evidence only,
not production readiness, CUDA/XLA parity, ANE execution, long-context coverage,
or broad prompt/size coverage.

The real decode path now has a conservative fused QKV+RoPE+KV-cache-write
kernel for supported Gemma 4 layer shapes, with fallback to the previous
separate QKV/RoPE/KV-write path for unsupported/debug/trace shapes. Reports now
include CPU encode time, command-buffer wait time, per-step timing slots, and
encoder counts by runtime family (`embedding`, `ple_input`, `layer_body`,
`final_sample`, `final_logits_diagnostic`).

Cached local Gemma 4 snapshots found for this pass:

- `google/gemma-4-E2B` at `/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca`
- `google/gemma-4-31B-it` at `/Users/george/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/419b2efe421994fdfd3394e621983d4cc511cd4f`

No other `models--google--gemma-4*` snapshots were present under the local HF
cache during this run; they are not claimed or silently skipped.

E2B one-token HF gate:

```bash
RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 target/debug/rvllm_metal_infer \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
  --prompt Hello \
  --max-new-tokens 1 \
  --max-total-tokens 16 \
  --large-model-opt-in \
  --hf-reference /tmp/rvllm-e2b-text-reference-suite-current/gemma4-e2b-hf-text-reference-hello-step1.json \
  --json > /tmp/rvllm-e2b-hello-step1-qkv-rope-cache-family.json
```

Observed pass: generated token `[236764]`, text `","`,
`hf_reference.matched=true`, `prepare_ms=5389.090125`,
`prefill_ms=329.112125`, `decode_ms=221.244292`,
`tok_per_s=4.5198906193701935`, `command_buffers=2`, `encoders=1063`
(`embedding=2`, `ple_input=8`, `layer_body=1050`, `final_sample=3`),
`forced_waits=2`, `cpu_encode_ns=8647624`,
`command_buffer_wait_ns=541699208`. This improves the previous recorded
one-token direct decoder count from `1203` to `1063` encoders and moves decode
below the previous `~226 ms` baseline without changing the HF-matched token.

E2B four-token HF gate:

```bash
RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 target/debug/rvllm_metal_infer \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
  --prompt Hello \
  --max-new-tokens 4 \
  --max-total-tokens 16 \
  --large-model-opt-in \
  --hf-reference /private/tmp/gemma4-e2b-hf-text-infer-hello-steps4.json \
  --json > /tmp/rvllm-e2b-hello-step4-qkv-rope-cache-family.json
```

Observed pass: generated tokens `[236764,108,236777,735]`, generated text
`",\n\nI have"`, `hf_reference.matched=true`, `decode_ms=891.2090000000001`,
`tok_per_s=4.488285015075027`, `command_buffers=5`, `encoders=2662`,
`forced_waits=5`, `cpu_encode_ns=18947292`, and
`command_buffer_wait_ns=1196148833`.

The E2B three-prompt direct session report
`/tmp/rvllm-e2b-direct-session-qkv-rope-cache-family.json` passed all HF
references with generated tokens `Hello -> [236764]`,
`Once upon a time -> [236764]`, and `The capital of France is -> [9079]`.
Observed totals: `prepare_ms=6302.7246669999995`,
`prefill_ms=896.1980410000001`, `decode_ms=705.6414179999999`,
`tok_per_s=4.2514511244293205`, `generated_tokens=3`,
`command_buffers=6`, `encoders=3189`, `forced_waits=6`,
`cpu_encode_ns=36875210`, and `command_buffer_wait_ns=1564934791`.

The E2B three-prompt Engine session report
`/tmp/rvllm-e2b-engine-session-qkv-rope-cache-family.json` also passed all HF
references. Observed totals: `prepare_ms=5631.603333`,
`prefill_ms=434.200125`, `decode_ms=255.08958399999997`,
`tok_per_s=11.760574277309576`, `generated_tokens=3`,
`command_buffers=2`, `encoders=1063`, `forced_waits=2`,
`cpu_encode_ns=7940500`, and `command_buffer_wait_ns=681331042`.

The Engine session profile report
`/tmp/rvllm-e2b-engine-session-profile-qkv-rope-cache-family.json` used
`--profile-samples 2`; both samples passed the HF gates. Median profile values:
`prepare_ms=5646.743937499999`, `prefill_ms=433.625271`,
`decode_ms=250.7965625`, `tok_per_s=11.963415507189936`,
`command_buffers_per_token=0.6666666666666666`,
`encoders_per_token=354.3333333333333`,
`forced_waits_per_token=0.6666666666666666`,
`latency_ms_per_token=83.59885416666665`,
`cpu_encode_ms_per_token=2.2533191666666665`, and
`command_buffer_wait_ms_per_token=225.88188216666668`.

A Rust model-suite runner report was recorded at
`/tmp/rvllm-gemma4-metal-model-suite-qkv-rope-cache-report.json` using manifest
`/tmp/rvllm-gemma4-metal-model-suite-qkv-rope-cache-manifest.json`. Status was
`pass` for two cases: E2B one-token HF matched `[236764]` with `encoders=1063`
and `decode_ms=224.762166`; 31B-it bounded one-token execution produced
`[9259]` with `encoders=1325` and `decode_ms=3310.648375`. The 31B case has no
local HF reference artifact, so it is bounded execution evidence only.

Direct 31B-it gate:

```bash
RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 target/debug/rvllm_metal_infer \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/419b2efe421994fdfd3394e621983d4cc511cd4f \
  --prompt Hello \
  --max-new-tokens 1 \
  --max-total-tokens 16 \
  --large-model-opt-in \
  --json > /tmp/rvllm-gemma4-31b-hello-step1-qkv-rope-cache-family.json
```

Observed bounded pass: generated token `[9259]`, text `"Hello"`,
`prepare_ms=44112.850833000004`, `prefill_ms=5031.555125`,
`decode_ms=3287.7048750000004`, `tok_per_s=0.304163554218047`,
`arena_bytes=61863697960`, `command_buffers=2`, `encoders=1325`
(`embedding=2`, `layer_body=1320`, `final_sample=3`), `forced_waits=2`,
`cpu_encode_ns=9944208`, and `command_buffer_wait_ns=8309307292`.

Public E2B surfaces after the same fusion:

- `rvllm-eval`: generated `","`, `prepare_ms=5337.1`,
  `prefill_ms=331.6`, `decode_ms=226.8`, `tok_per_sec=4.410`.
- `rvllm-bench`: schema `rvllm.apple_metal_bench.v1`,
  `prepare_ms=5311.950583`, `ms_per_step=549.7180000000001`,
  `tok_per_sec=1.8191145278124419`, `command_buffers=2`,
  `encoders=1063`, `forced_waits=2`, `cpu_encode_ns=7477292`,
  `command_buffer_wait_ns=542179041`.
- `rvllm-ppl`: `perplexity=504.1798937686529`, `tokens=2`,
  `chunk_len=3`, `elapsed_s=1.0013175`, `prepare_ms=5363.461459`,
  `command_buffers=6`, `encoders=2132`, `final_logits_encoders=6`,
  `forced_waits=6`.
- Prepared `rvllm-server --backend metal-direct` completion:
  `/tmp/rvllm-e2b-server-completion-qkv-rope-cache-family.json`, generated
  token `[236764]`, text `","`, `decode_ms=228.244917`,
  `tok_per_s=4.381258575848154`, `command_buffers=2`, `encoders=1063`,
  `forced_waits=2`.

Fresh external Metal profiling was captured at
`/tmp/rvllm-e2b-metal-qkv-rope-cache-fused.trace` (`96M`) with:

```bash
xcrun xctrace record \
  --template 'Metal System Trace' \
  --output /tmp/rvllm-e2b-metal-qkv-rope-cache-fused.trace \
  --launch -- target/debug/rvllm_metal_infer \
    --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
    --prompt Hello \
    --max-new-tokens 1 \
    --max-total-tokens 16 \
    --large-model-opt-in \
    --hf-reference /tmp/rvllm-e2b-text-reference-suite-current/gemma4-e2b-hf-text-reference-hello-step1.json \
    --json
```

Exporting `metal-application-command-buffer-submissions` showed two
application command-buffer submissions: `495` encoders with `6.67 ms` encoder
time and `7.50 ms` duration, then `498` encoders with `2.92 ms` encoder time
and `3.76 ms` duration. This improves the previous trace's submission encoder
counts (`565` and `568`) and remains bottleneck-localization evidence only.

## Current Metal E2B Session Evidence (2026-05-26)

Baseline was verified on `main@144cdd0`: `rvllm_metal_infer` ran real E2B
Metal for `Hello`, emitted schema `rvllm.apple_metal_text_infer.v1`, generated
token `[236764]` / `Hello,`, and kept the non-production claim boundary.

HF/Transformers references for the three-prompt text suite were generated
outside the repo with the existing reference script:

```bash
uv run --no-project --python 3.12 --with torch --with transformers python scripts/dump_gemma4_e2b_hf_text_reference_suite.py \
  /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
  --prompt "Hello" \
  --prompt "Once upon a time" \
  --prompt "The capital of France is" \
  --decode-steps 1 \
  --top-k 16 \
  --output-dir /tmp/rvllm-e2b-text-reference-suite-current \
  --skip-existing
```

The manifest is
`/tmp/rvllm-e2b-text-reference-suite-current/gemma4-e2b-hf-text-reference-suite-manifest.json`.
Reference generated tokens were `Hello -> [236764]`, `Once upon a time ->
[236764]`, and `The capital of France is -> [9079]`.

Direct session evidence used one prepared `ModelMetalBackend` and the Rust
manifest/session path:

```bash
cargo run -p rvllm-runtime --features apple --bin rvllm_metal_infer -- \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
  --reference-manifest /tmp/rvllm-e2b-text-reference-suite-current/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --session-backend direct \
  --large-model-opt-in \
  --case-timeout-seconds 120 \
  --report /tmp/rvllm-e2b-text-session-direct-shared-scratch-token-only-report.json \
  --json
```

Observed current result after the range-read loader fix, shared per-layer
scratch arena, native Metal matmul/fusion work, and token-only final sampling:
schema `rvllm.apple_metal_text_session.v1`, backend `direct`, status `pass`,
report `/tmp/rvllm-e2b-text-session-direct-shared-scratch-token-only-report.json`,
`prepare_ms=5165.747291`, `prefill_ms=858.6368320000001`,
`decode_ms=679.4907089999999`, `tok_per_s=4.415071406075694`,
`arena_bytes=9321447442`, `command_buffers=6`, `encoders=3609`, and
`forced_waits=6`. The loader fix reads only tensor payload ranges instead of
rereading whole shards per tensor. The active matmul/fusion path uses a native
Metal microbatch GEMM kernel plus one-pass projection/RMSNorm kernels, a fused
Q/K/V headwise projection/RMSNorm encoder, and a final LM-head argmax-tile path
that avoids full-logit materialization during normal sampling. Full logits are
still materialized on demand for diagnostic readback/comparison paths.
All three cases reported `hf_reference.matched=true`:
`Hello` prompt IDs `[2,9259]` generated `[236764]` and decoded `Hello,`;
`Once upon a time` prompt IDs `[2,14946,3324,496,990]` generated `[236764]`
and decoded `Once upon a time,`; `The capital of France is` prompt IDs
`[2,818,5279,529,7001,563]` generated `[9079]` and decoded
`The capital of France is Paris`.

The previous `prepare_ms` values around `416000..496000 ms` were caused by the
same loader defect and are superseded by the range-read evidence above. Direct
one-token E2B decode is now about `222..228 ms/token` in this three-prompt
suite and emits `1203` encoders per case. This is a meaningful Metal execution
improvement, but it is still not a production-readiness or CUDA/XLA parity
claim.

Engine session evidence used one prepared `Engine`/`ModelMetalBackend` and the
scheduler path:

```bash
cargo run -p rvllm-runtime --features apple --bin rvllm_metal_infer -- \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
  --reference-manifest /tmp/rvllm-e2b-text-reference-suite-current/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --session-backend engine \
  --large-model-opt-in \
  --case-timeout-seconds 120 \
  --report /tmp/rvllm-e2b-text-session-engine-shared-scratch-token-only-report.json \
  --json
```

Observed current result: schema `rvllm.apple_metal_text_session.v1`, backend
`engine`, status `pass`, report
`/tmp/rvllm-e2b-text-session-engine-shared-scratch-token-only-report.json`,
`prepare_ms=5242.516375`, `prefill_ms=436.179209`,
`decode_ms=249.59758300000001`, `tok_per_s=12.019347158501931`,
`arena_bytes=9321447442`, `command_buffers=2`, `encoders=1203`, and
`forced_waits=2`. All generated tokens and HF comparisons matched the direct
session evidence.

Session profiling was run after the direct and Engine gates:

```bash
cargo run -p rvllm-runtime --features apple --bin rvllm_metal_infer -- \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
  --reference-manifest /tmp/rvllm-e2b-text-reference-suite-current/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --session-backend direct \
  --large-model-opt-in \
  --case-timeout-seconds 120 \
  --profile-samples 1 \
  --profile-report /tmp/rvllm-e2b-text-session-profile-direct-shared-scratch-token-only-summary.json \
  --json
```

Observed profile result: schema
`rvllm.apple_metal_text_session_profile.v1`, backend `direct`,
`sample_count=1`, median/min/max `prepare_ms=5132.582208`,
`prefill_ms=866.86025`, `decode_ms=682.452416`,
`tok_per_s=4.39591087915498`, `command_buffers_per_token=2.0`,
`encoders_per_token=1203.0`, and `forced_waits_per_token=2.0`. All three
profile sample cases reported `hf_reference.matched=true`.

The Engine profile report
`/tmp/rvllm-e2b-text-session-profile-engine-shared-scratch-token-only-summary.json` recorded
schema `rvllm.apple_metal_text_session_profile.v1`, backend `engine`,
`sample_count=1`, median/min/max `prepare_ms=5156.567083999999`,
`prefill_ms=430.395917`, `decode_ms=249.685042`,
`tok_per_s=12.015137054145196`,
`command_buffers_per_token=0.6666666666666666`,
`encoders_per_token=401.0`, and
`forced_waits_per_token=0.6666666666666666`. The profile claim is
`bounded Apple Metal text session profile only; not production performance,
ANE, external profiler, or production readiness evidence`.

Prepared serving evidence was then captured with the in-process
`rvllm-server --backend metal-direct` path, which prepares one
`ModelMetalBackend` before listening and reuses it for completion requests:

```bash
cargo run -p rvllm-serve --features apple --bin rvllm-server -- \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
  --backend metal-direct \
  --addr 127.0.0.1:18082 \
  --max-new-tokens 1 \
  --max-total-tokens 16 \
  --large-model-opt-in
```

`GET /healthz` reported `backend=metal-direct`, `prepared_once=true`,
`prepare_ms=5310.338875`, `arena_bytes=9321447442`,
`max_supported_total_tokens=16`, and `debug_sync=false`. A real request:

```bash
curl -sS -X POST http://127.0.0.1:18082/v1/completions \
  -H 'content-type: application/json' \
  --data '{"prompt":"Hello","max_tokens":1}'
```

returned schema `rvllm.openai_completions.v1` with backend report schema
`rvllm.apple_metal_server_completion.v1`, token `[236764]`, text `","`,
`output_text="Hello,"`, `prefill_ms=330.087916`,
`decode_ms=223.971834`, `tok_per_s=4.464847128947473`,
`command_buffers=2`, `encoders=1203`, and `forced_waits=2`. The backend report
keeps the non-production claim boundary.

Public benchmark/eval/PPL surfaces were then wired to the same bounded Metal
text path under `RVLLM_BACKEND_PROFILE=apple` and checked against the cached
Gemma 4 E2B snapshot.

```bash
RVLLM_BACKEND_PROFILE=apple \
RVLLM_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
RVLLM_PROMPT='Hello' \
RVLLM_MAX_TOKENS=1 \
RVLLM_APPLE_LARGE_MODEL_OPT_IN=1 \
cargo run -p rvllm-bench --features apple --bin rvllm-eval
```

Observed current result: generated text `,`, `generated=1`,
`tok_per_sec=4.403`, `prepare_ms=5539.1`, `prefill_ms=333.1`, and
`decode_ms=227.1`. The equivalent direct session run above produced token
`[236764]` / `Hello,` and kept every HF reference match true.

```bash
RVLLM_BACKEND_PROFILE=apple \
RVLLM_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
RVLLM_PROMPT='Hello' \
RVLLM_MAX_TOKENS=1 \
RVLLM_BATCH=1 \
RVLLM_ITERS=1 \
RVLLM_WARMUP=0 \
RVLLM_APPLE_LARGE_MODEL_OPT_IN=1 \
cargo run -p rvllm-bench --features apple --bin rvllm-bench
```

Observed current result: schema `rvllm.apple_metal_bench.v1`, backend `apple`,
`prepare_ms=5318.452708`, `ms_per_step=559.057125`,
`tok_per_sec=1.788725973217853`, `generated_tokens=1`,
`command_buffers=2`, `encoders=1203`, `forced_waits=2`, and
`arena_bytes=9321447442`.

```bash
RVLLM_BACKEND_PROFILE=apple \
RVLLM_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
RVLLM_PROMPT='Hello world' \
RVLLM_PPL_CHUNK=3 \
RVLLM_PPL_CHUNKS=1 \
RVLLM_APPLE_LARGE_MODEL_OPT_IN=1 \
cargo run -p rvllm-bench --features apple --bin rvllm-ppl
```

Observed current result: `perplexity=504.3815019862137`, `tokens=2`,
`elapsed_s=1.019207042`, and `prepare_ms=5203.621333`.

The public surface claim for these commands is
`bounded Apple Metal bench/eval/ppl path; not production-ready or
parity-complete`.

These runs improve bounded Metal E2B session and prepared-serving operability,
and they make the public bench/eval/PPL command surfaces operational for
bounded Metal E2B diagnostics. They do not establish broad prompt coverage,
long-context decode, private ANE execution, an accepted shared-KV optimization,
external profiler coverage, CUDA/XLA parity, or a production performance claim.

Longer-token E2B text evidence was captured for the cached E2B `Hello` prompt
against `/private/tmp/gemma4-e2b-hf-text-infer-hello-steps4.json`:

```bash
RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 \
/Users/george/Downloads/rvllm/v3/target/debug/rvllm_metal_infer \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
  --prompt Hello \
  --max-new-tokens 4 \
  --max-total-tokens 16 \
  --large-model-opt-in \
  --hf-reference /private/tmp/gemma4-e2b-hf-text-infer-hello-steps4.json \
  --json > /tmp/rvllm-e2b-hello-steps4-shared-scratch-token-only.json
```

Observed result: schema `rvllm.apple_metal_text_infer.v1`,
`hf_reference.matched=true`, generated tokens `[236764,108,236777,735]`,
generated text `",\n\nI have"`, `output_text="Hello,\n\nI have"`,
`prepare_ms=5327.267`, `prefill_ms=333.035833`, `decode_ms=909.065`,
`tok_per_s=4.400125403574002`, `arena_bytes=9321447442`,
`command_buffers=5`, `encoders=3012`, and `forced_waits=5`. This is a
four-token quality/performance diagnostic only, not long-context coverage.

External Metal profiling was captured with Instruments `xctrace`; `xcrun
--find metal-capture` was unavailable on this host, so the selected external
tool was the `Metal System Trace` template:

```bash
rm -rf /tmp/rvllm-e2b-metal-token-only.trace
xcrun xctrace record \
  --template 'Metal System Trace' \
  --output /tmp/rvllm-e2b-metal-token-only.trace \
  --time-limit 20s \
  --no-prompt \
  --target-stdout /tmp/rvllm-e2b-metal-token-only-xctrace-stdout.json \
  --env RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 \
  --launch -- /Users/george/Downloads/rvllm/v3/target/debug/rvllm_metal_infer \
    --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca \
    --prompt Hello \
    --max-new-tokens 1 \
    --max-total-tokens 16 \
    --large-model-opt-in \
    --json
```

The trace artifact is `/tmp/rvllm-e2b-metal-token-only.trace` (`20M`), recorded
on macOS 15.6 with target process `rvllm_metal_infer` exit status `0` and Metal
device `M4 Max`. The target stdout reported token `[236764]`, `prepare_ms`
`6101.478292000001`, `prefill_ms=332.536`, `decode_ms=226.575083`,
`tok_per_s=4.413547980472328`, `arena_bytes=9321447442`,
`command_buffers=2`, `encoders=1203`, and `forced_waits=2`. Exporting
`metal-application-command-buffer-submissions` showed two application command
buffer submissions: `565` encoders with `7.03 ms` encoder time and `7.96 ms`
duration, then `568` encoders with `3.72 ms` encoder time and `4.87 ms`
duration. These profiler counters show very small GPU encoder durations
relative to host-visible decode wall time, so this is bottleneck-localization
evidence only; it is not a production performance claim.

Large Gemma 4 coverage was checked against the cached
`google/gemma-4-31B-it` snapshot at
`/Users/george/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/419b2efe421994fdfd3394e621983d4cc511cd4f`:

```bash
RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/419b2efe421994fdfd3394e621983d4cc511cd4f \
RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 \
cargo test -p rvllm-apple-metal real_gemma4_model_dir -- --ignored --nocapture
```

The shared-scratch and fused-weight arena accounting fix reduced the 31B dry-run
estimate from the earlier failing `100323760168` byte allocation shape to
`61863697960` bytes (`57.62 GiB`) for a 60-layer text model with `hidden=5376`,
`vocab=262144`, `sliding:50`, `full:10`, tied embeddings, and no FP8 scaled
weights.

An actual one-token 31B decode then ran:

```bash
RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 \
cargo run -p rvllm-runtime --features apple --bin rvllm_metal_infer -- \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/419b2efe421994fdfd3394e621983d4cc511cd4f \
  --prompt Hello \
  --max-new-tokens 1 \
  --max-total-tokens 8 \
  --large-model-opt-in \
  --json > /tmp/rvllm-gemma4-31b-hello-step1-token-only.json
```

Observed result: schema `rvllm.apple_metal_text_infer.v1`, generated token
`[9259]`, generated text `"Hello"`, `output_text="HelloHello"`,
`finish_reason="length"`, `prepare_ms=44562.049332999995`,
`prefill_ms=6318.7690410000005`, `decode_ms=3261.7925840000003`,
`tok_per_s=0.30657988644197615`, `arena_bytes=61681657960`,
`command_buffers=2`, `encoders=1565`, `forced_waits=2`, and
`max_supported_total_tokens=8`. This proves that the cached 31B snapshot can
execute a bounded one-token Metal decode on this host, but it has no HF
reference match and is far below any production performance target.

## Current Backend Parity Status (2026-05-25)

The Rust production acceptance evaluator now includes structured backend parity
evidence slots for CUDA, Metal, and XLA. Current repository status is not a
parity candidate: CUDA has feature-gated bringup/eval/ppl/bench paths, Metal
has bounded direct/session/server/bench/eval/PPL paths, and XLA has no
executable backend in this checkout. The only XLA references found are
`rvllm-bench` profile metadata and side-by-side JSON labels. The parity gate
therefore fails until there is a runnable XLA backend and the Metal public
surfaces move beyond the current bounded probe arena into equivalent
correctness, context/batch, and performance-gate coverage.

Focused gate tests:

```bash
cargo test -p rvllm-apple profiling::tests -- --nocapture
cargo test -p rvllm-apple production -- --nocapture
```

| Capability | Evidence class | Tests | Hardware? | Limitations | Next proof |
| --- | --- | --- | --- | --- | --- |
| zero-layer synthetic-weight decode | SYNTHETIC-SMOKE | `tiny_zero_layer_model_backend_decodes_token_2_to_3`, Engine zero-layer smokes | Yes, ignored Metal tests | Argmax/token smoke only; no real checkpoint decode | CPU/reference logit parity and real checkpoint path |
| one-layer prefused no-op | SYNTHETIC-SMOKE | `tiny_one_layer_noop_model_backend_decodes_token_2_to_3`, Engine no-op smoke | Yes, ignored Metal tests | Synthetic no-op weights only | Numerical parity for hidden/logits |
| one-layer HF-style no-op | SYNTHETIC-SMOKE | `tiny_one_layer_hf_style_noop_model_backend_decodes_token_2_to_3`, Engine HF-style no-op smoke | Yes, ignored Metal tests | Synthetic HF-shaped names; no nonzero attention/MLP evidence | Add nonzero HF-shaped parity |
| FFN nonzero | SYNTHETIC-SMOKE | `cpu_reference_one_layer_ffn_nonzero_fixture_argmax_is_3`, Metal/Engine FFN smokes | Yes, ignored Metal tests | Synthetic argmax/token evidence | Full logits and residual parity |
| attention nonzero | SYNTHETIC-SMOKE | `cpu_reference_one_layer_attention_nonzero_fixture_argmax_is_3`, Metal/Engine attention smokes | Yes, ignored Metal tests | Synthetic argmax/token evidence | Multi-head/GQA numerical parity |
| multi-head grouped-KV attention | SYNTHETIC-NUMERIC | `cpu_reference_multihead_gqa_attention_full_logits_are_stable`, `cpu_reference_multihead_gqa_attention_residual_vector_is_stable`, direct/Engine full-logit and full-residual GQA tests | Yes, ignored Metal tests | Synthetic one-token evidence; no real checkpoint decode; softmax is degenerate | Add multi-token/both-KV-group GQA fixture, then real checkpoint dry-run |
| q_dim distinct from hidden | SYNTHETIC-NUMERIC | `cpu_reference_qdim_not_hidden_full_logits_are_stable`, `cpu_reference_qdim_not_hidden_residual_vector_is_stable`, direct/Engine full-logit and full-residual q_dim tests | Yes, ignored Metal tests | Synthetic one-token evidence; no real checkpoint decode | Broaden to generated HF-shaped q_dim != hidden and real checkpoint dry-run |
| layered batch-two prefill/decode metadata | SYNTHETIC-SMOKE | `tiny_one_layer_noop_prefill_batch_two_then_decode_batch_two_returns_token_3` | Yes, ignored Metal test | Synthetic one-layer no-op fixture only; both batch entries use the same token; not E2B batching, mixed prompt lengths, scheduler batching, or throughput evidence | Add nonzero attention batch fixture, mixed prompt lengths, and then bounded real-checkpoint batch diagnostics |
| full one-layer nonzero argmax | SYNTHETIC-SMOKE | `cpu_reference_one_layer_full_nonzero_fixture_argmax_is_3`, `tiny_one_layer_full_nonzero_model_backend_decodes_token_2_to_3` | Yes, ignored Metal test | Argmax/token evidence plus selected numerical checks | Full residual and full-logit parity |
| selected logits parity | SYNTHETIC-NUMERIC | `tiny_one_layer_full_nonzero_model_backend_selected_logits_match_cpu` | Yes, ignored Metal test | Selected logits only | Full vector comparison when vocab is small |
| full residual parity | SYNTHETIC-NUMERIC | `cpu_reference_one_layer_full_nonzero_residual_vector_is_stable`, `tiny_one_layer_full_nonzero_model_backend_full_residual_matches_cpu` | Yes, ignored Metal test | Synthetic one-layer fixture only | Broaden to real-shape synthetic fixtures |
| short prefill selected logits | SYNTHETIC-NUMERIC | `tiny_prompt_len_two_prefill_selected_logits_match_cpu`, `engine_prompt_len_two_prefill_selected_logits_match_cpu` | Yes, ignored Metal tests | Selected logits only | Full prefill logits for small vocab |
| multi-step decode full logits | GENERATED-HF-NUMERIC | `cpu_reference_generated_tiny_hf_full_logits_are_stable`, `engine_generated_gemma4_hf_end_to_end_full_logits_match_cpu` | Yes, ignored Metal test | Synthetic generated HF/Gemma-shaped fixture | Broaden beyond tiny vocab fixture |
| generated tiny Gemma-shaped full logits | GENERATED-HF-NUMERIC | `tiny_generated_gemma4_hf_end_to_end_model_backend_full_logits_match_cpu` | Yes, ignored Metal test | Synthetic generated HF/Gemma-shaped fixture | Real-shape synthetic dimensions |
| generated tiny reference export bundle | GENERATED-HF-NUMERIC | `generated_tiny_hf_reference_bundle_can_be_exported`, `scripts/verify_generated_tiny_reference.py`, `scripts/compare_generated_tiny_reference_torch.py`, `scripts/check_generated_tiny_reference_bundle.sh` | No | Env-gated export hook plus one-command standalone/PyTorch verifier chain; synthetic generated fixture only; no Transformers or real checkpoint claim | Compare a real checkpoint against HF/Transformers once dry-run passes |
| Gemma dry-run validation | REAL-CHECKPOINT-DRYRUN | `generated_tiny_gemma4_hf_fixture_uses_real_names_and_dry_run_validates`, `gemma4_dry_run_*`, `dry_run_*` tests | No Metal execution required | Metadata/shape validation only; generated fixtures and cached Gemma 4 E2B snapshot validate; no decode claim | Add real checkpoint prepare/load gate with explicit unsupported-boundary reporting |
| Gemma dry-run/load arch contract | REAL-CHECKPOINT-DRYRUN | `gemma4_arch::*`, `gemma4_dry_run_validates_text_config_and_model_language_model_prefix`, `gemma4_dry_run_validates_language_model_model_prefix`, `gemma4_load_accepts_language_model_model_prefix`, `gemma4_dry_run_rejects_bad_gemma4_model_type`, `config::model::tests::parses_gemma4_causallm_identity`, `config::model::tests::rejects_gemma4_bad_text_model_type`, generated tiny dry-run validation | No Metal execution required | Dry-run, direct Gemma4 arch parsing, core probe config parsing, and loader prefix resolution share accepted Gemma4 architecture/model_type names; still metadata-only | Run against real model dirs, including sharded and single-shard checkpoints |
| Gemma FP8 scale dry-run validation | REAL-CHECKPOINT-DRYRUN | `gemma4_dry_run_validates_fp8_linears_with_bf16_scales`, `gemma4_dry_run_rejects_fp8_linear_missing_scale`, `gemma4_dry_run_rejects_fp8_linear_f16_scale`, `gemma4_dry_run_rejects_fp8_lm_head_missing_scale` | No Metal execution required | Safetensor metadata/header validation only; synthetic FP8 fixtures; no real FP8 checkpoint validated here | Run against a real FP8 Gemma model dir and compare loader behavior |
| Gemma packed quantization dry-run guard | REAL-CHECKPOINT-DRYRUN | `safetensors::tests::parses_packed_u32_and_u8_metadata`, `gemma4_dry_run_accepts_unused_packed_u32_u8_metadata`, `gemma4_dry_run_rejects_packed_required_linear_weights` | No Metal execution required | U8/U32 headers can be scanned, but packed required Gemma tensors are explicitly unsupported; no MXFP8 execution claim | Add real MLX/MXFP8 dry-run classification and implement packed-weight validation only when the loader supports it |
| Gemma 4 MoE dry-run/load support | REAL-CHECKPOINT-DRYRUN | `gemma4_arch::tests::from_dir_allows_dense_moe_placeholders`, `gemma4_arch::tests::from_dir_parses_explicit_moe_config`, `gemma4_dry_run_allows_dense_moe_placeholders`, `gemma4_dry_run_rejects_incomplete_moe_metadata`, `gemma4_dry_run_allows_extra_moe_tensor_metadata_when_dense`; `target/debug/probe_gemma4_load --dry-run /Users/george/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/b2a81a03d25f927590a91d84ba43f96e8ef7349f` | No Metal execution required for dry-run; real execution recorded in the all-local suite above | Explicit MoE config markers are parsed, incomplete MoE metadata is rejected, and the cached 26B-A4B-it snapshot validates required router/expert/norm tensor shapes. This replaces the old unsupported-MoE guard. | Add broader MoE prompt/decode references and resolve the 26B BF16 parity blocker |
| real Gemma 4 E2B dry-run harness (env-gated) | REAL-CHECKPOINT-DRYRUN | `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca cargo test -p rvllm-loader gemma4_dry_run_real_model_dir_validates_when_env_is_set -- --nocapture`; `cargo test -p rvllm-apple-metal --lib real_gemma4_model_dir_dry_run_validates_when_env_is_set -- --ignored --nocapture`; `cargo run -p rvllm-runtime --features apple --bin probe_gemma4_load -- --dry-run "$RVLLM_GEMMA4_MODEL_DIR"` | No decode; optional model dir gate | Observed pass for cached Google Gemma 4 E2B snapshot only. Metadata/shape validation found 35 layers, hidden 1536, vocab 262144, tied embeddings, sliding:28/full:7. No Metal prepare/load or inference claim. | Generate HF reference logits, then add env-gated Metal prepare/load boundary test |
| real Gemma 4 E2B Metal prepare boundary | SCAFFOLD | `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca cargo test -p rvllm-runtime --features apple real_gemma4_e2b_model_backend_prepare_reports_current_large_model_gate -- --ignored --nocapture` | Yes, ignored Metal test | Dry-run validates first, then default Metal prepare stops before arena allocation with `unsupported_probe_num_layers_without_large_model_opt_in` for E2B's 35 layers. This is an intentional default safety boundary, not a decode or numeric claim. | Keep the default boundary; use explicit opt-in only for env-gated real-checkpoint tests |
| real Gemma 4 E2B Metal prepare/load (large-model opt-in) | REAL-CHECKPOINT-DRYRUN | `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca cargo test -p rvllm-apple-metal --lib real_gemma4_model_dir_large_probe_arena_bytes_when_opted_in -- --ignored --nocapture`; `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca cargo test -p rvllm-runtime --features apple real_gemma4_e2b_model_backend_prepare_with_large_model_opt_in -- --ignored --nocapture` | Yes, ignored Metal tests | Observed pass for cached Gemma 4 E2B under `RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1`; arena requirement is about 6.47 GiB and runtime prepare/load completed in 358.71s on the audited host. This does not run prefill, decode, logits, ANE, or production inference. | Keep explicit opt-in and broaden numerical evidence beyond selected logits |
| diagnostic raw-token/text E2B Metal decode CLI | SCAFFOLD | `cargo run -p rvllm-runtime --features apple --bin probe_apple_metal_decode -- --model-dir "$RVLLM_GEMMA4_MODEL_DIR" --prompt-token-ids 2,4 --decode-steps 1 --top-k 16 --large-model-opt-in`; `cargo run -p rvllm-runtime --features apple --bin probe_apple_metal_decode -- --model-dir "$RVLLM_GEMMA4_MODEL_DIR" --prompt-text "Hello" --decode-steps 1 --top-k 16 --large-model-opt-in --decode-text`; `cargo test -p rvllm-runtime --features apple probe_apple_metal_decode_cli_args -- --nocapture`; `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca cargo test -p rvllm-runtime --features apple probe_apple_metal_decode_e2b_raw_tokens_smoke -- --ignored --nocapture`; `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca cargo test -p rvllm-runtime --features apple probe_apple_metal_decode_e2b_prompt_text_smoke -- --ignored --nocapture` | Yes, ignored Metal smoke when env/model are present | Adds a user-runnable diagnostic command that uses the existing `ModelMetalBackend` path, requires explicit large-model opt-in for E2B, and prints sampled token IDs, per-step top-k logits, timing, arena size, command buffers, encoders, forced waits, debug sync, and a non-production claim. It defaults to raw token IDs; optional `--prompt-text` loads `tokenizer.json`, prepends BOS token `2` unless `--no-bos` is supplied, and `--decode-text` reports decoded sampled/output text for inspection. Post layer-scalar fix, observed pass for cached E2B raw-token prompt `[2,4]` and one decode step on 2026-05-22 against `/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-step1.json`: `hf_reference.matched=true`, sampled token `[954]`, prepare `448050 ms`, prefill `584 ms`, decode `447 ms`, `2` command buffers, `1343` encoders, and `2` forced waits. This is a diagnostic probe only; text mode is not a production text workflow, ANE execution, throughput claim, or production inference promotion. | Broaden reference-backed text/token equality evidence and keep production-facing serving evidence on the prepared `rvllm-server --backend metal-direct` path |
| bounded E2B Metal text inference CLI | SCAFFOLD | `cargo run -p rvllm-runtime --features apple --bin rvllm_metal_infer -- --model-dir "$RVLLM_GEMMA4_MODEL_DIR" --prompt "Hello" --max-new-tokens 1 --large-model-opt-in --json`; `cargo test -p rvllm-runtime --features apple --bin rvllm_metal_infer -- --nocapture`; ignored real E2B text smokes for one-step, raised-cap, HF-backed, two-step, four-step, and Engine one-step paths | Yes, ignored Metal smoke when env/model/artifact are present | Adds a text-in/text-out command shape on the existing `ModelMetalBackend`: it loads `tokenizer.json`, prepends BOS token `2` unless `--no-bos` is supplied, runs prefill plus greedy decode, stops on configurable EOS token IDs or `--max-new-tokens`, decodes generated/output text, emits text or schema `rvllm.apple_metal_text_infer.v1` JSON, and reports timing/counter fields. Unlike the diagnostic top-k probe, this path does not read debug logits by default. If `--hf-reference` is supplied, it compares tokenizer-derived prompt IDs, decode step count, and generated token IDs with an existing HF artifact and reports `hf_reference.matched`. It still requires explicit large-model opt-in for E2B and fails clearly when prompt tokens plus max new tokens exceed the configurable Metal E2B probe arena cap. Current bounded direct, Engine session, and prepared server evidence is recorded above. This is workflow scaffolding and bounded diagnostic evidence, not broad tokenizer/text correctness, throughput, ANE, long-context, or production performance evidence. | Add broader text prompt coverage, longer decode within explicit caps, real batching throughput, and external profiler-backed benchmark samples |
| bounded E2B Metal text session manifest runner | SCAFFOLD | `cargo run -p rvllm-runtime --features apple --bin rvllm_metal_infer -- --model-dir "$RVLLM_GEMMA4_MODEL_DIR" --reference-manifest /tmp/rvllm-e2b-text-reference-suite-current/gemma4-e2b-hf-text-reference-suite-manifest.json --session-backend direct --large-model-opt-in --report /tmp/rvllm-e2b-text-session-direct-report.json --json`; `cargo run -p rvllm-runtime --features apple --bin rvllm_metal_infer -- --model-dir "$RVLLM_GEMMA4_MODEL_DIR" --reference-manifest /tmp/rvllm-e2b-text-reference-suite-current/gemma4-e2b-hf-text-reference-suite-manifest.json --session-backend engine --large-model-opt-in --report /tmp/rvllm-e2b-text-session-engine-report.json --json`; `cargo run -p rvllm-runtime --features apple --bin rvllm_metal_infer -- --model-dir "$RVLLM_GEMMA4_MODEL_DIR" --reference-manifest /tmp/rvllm-e2b-text-reference-suite-current/gemma4-e2b-hf-text-reference-suite-manifest.json --session-backend direct --large-model-opt-in --case-timeout-seconds 900 --profile-samples 1 --profile-report /tmp/rvllm-e2b-text-session-profile-direct-summary.json --json` | Yes, real Metal session runs observed | Adds Rust-native manifest and JSONL session modes to `rvllm_metal_infer`. The direct path prepares one `ModelMetalBackend` and runs all cases sequentially. The Engine path prepares one `Engine`/`ModelMetalBackend`, enqueues cases through the scheduler, and stops requests when the CLI EOS policy fires. The profile path records bounded local per-token timing/counter summaries only. Manifest mode now rejects missing or duplicate case names, rejects non-positive decode steps, fails clearly on missing reference artifacts, and supports per-case timeout. The current three-prompt direct and Engine suites passed with every `hf_reference.matched=true`; details and exact counters are recorded in the current session evidence section. This is bounded operational evidence, not ANE, external profiler, broad prompt/decode coverage, or production performance evidence. | Broaden prompt/decode coverage and capture external profiler-backed benchmark samples before any production claim |
| prepared E2B Metal server completion | SCAFFOLD | `cargo test -p rvllm-serve -- --nocapture`; `cargo test -p rvllm-serve --features apple -- --nocapture`; `cargo run -p rvllm-serve --features apple --bin rvllm-server -- --model-dir "$RVLLM_GEMMA4_MODEL_DIR" --backend metal-direct --addr 127.0.0.1:18081 --max-new-tokens 1 --max-total-tokens 16 --large-model-opt-in`; `curl -sS http://127.0.0.1:18081/healthz`; `curl -sS -X POST http://127.0.0.1:18081/v1/completions -H 'content-type: application/json' --data '{"prompt":"Hello","max_tokens":1}'` | Yes, real prepared Metal server run observed | Adds an OpenAI-compatible `/v1/completions` surface and an explicit `--backend metal-direct` serving mode that prepares one `ModelMetalBackend` in-process before listening and reuses it across requests. The observed E2B request returned token `[236764]`, text `","`, `output_text="Hello,"`, `prefill_ms=577.8385000000001`, `decode_ms=440.632542`, `command_buffers=2`, `encoders=1343`, and `forced_waits=2`. The default `subprocess` backend remains available for compatibility but prepares per request and is not the production-facing evidence path. This is prepared serving workflow evidence inside the current bounded probe arena; it is not broad API coverage, concurrency, streaming, long-context, external profiler, ANE, or production performance evidence. | Add concurrent request scheduling, streaming/chat surfaces if required, broader prompt/decode references, and external profiler captures |
| reference-backed raw-token E2B Metal decode CLI comparison | SCAFFOLD | `cargo test -p rvllm-runtime --features apple probe_apple_metal_decode -- --nocapture`; `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca cargo test -p rvllm-runtime --features apple probe_apple_metal_decode_e2b_reference_backed_smoke -- --ignored --nocapture`; optional runtime form: `cargo run -p rvllm-runtime --features apple --bin probe_apple_metal_decode -- --model-dir "$RVLLM_GEMMA4_MODEL_DIR" --prompt-token-ids 2,4 --decode-steps 1 --top-k 16 --large-model-opt-in --hf-reference /tmp/gemma4-e2b-hf-full-logits-prompt-2-4-step1.json --json` | Optional real Metal run when model and artifact are present | The CLI can read an existing HF/Transformers reference JSON artifact and report selected-logit, top-k, and sampled-token comparison status as `hf_reference_match` plus structured mismatches in text mode, or as `hf_reference.matched` and `hf_reference.mismatches` in schema `rvllm.apple_metal_decode_probe.v1` JSON mode. Top-k comparison checks token membership and logit tolerance so exact rank order is not overclaimed for near-ties. The post-fix optional runtime command above observed `hf_reference.matched=true` and no mismatches for prompt `[2,4]`. This is a comparison/reporting hook, not production acceptance or a Gemma correctness claim. | Broaden the reference-backed command across the reference-suite manifest before promoting broad numeric evidence |
| diagnostic E2B CLI reference-suite runner | SCAFFOLD | `python3 tools/run_apple_metal_e2b_cli_suite.py --help`; `python3 tools/run_apple_metal_e2b_cli_suite.py --manifest /tmp/rvllm-e2b-reference-suite/gemma4-e2b-hf-reference-suite-manifest.json --model-dir "$RVLLM_GEMMA4_MODEL_DIR" --cargo-bin probe_apple_metal_decode --dry-run` | No Metal execution in dry-run; real Metal only when run without `--dry-run` | Adds a manifest-driven diagnostic runner that invokes `probe_apple_metal_decode --json` once per HF reference-suite case, checks `hf_reference.matched`, prints a pass/fail table, and writes `/tmp/rvllm-e2b-cli-suite-report.json` by default. This consumes existing `/tmp` artifacts and does not generate references, run ANE, or make a production claim. | Run the full suite only for a concrete CLI/reporting boundary after refreshing HF artifacts |
| real Gemma 4 E2B selected logits vs HF | REAL-CHECKPOINT-NUMERIC | `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 cargo test -p rvllm-runtime --features apple real_gemma4_e2b_model_backend_prefill_decode_selected_logits_match_hf_reference -- --ignored --nocapture` using `/tmp/gemma4-e2b-hf-reference-logits.json` | Yes, ignored Metal test | Observed pass for prompt token IDs `[2,4]`, one decode step, selected token IDs `0..5`, and HF greedy token `954`; selected-logit deltas were within tolerance. This is not broad prompt coverage, text generation quality, ANE execution, performance, or production readiness. | Add more prompts, decode-loop checks, and cross-layer traces past selected logits |
| real Gemma 4 E2B full-vocab logits vs HF | REAL-CHECKPOINT-NUMERIC | `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 cargo test -p rvllm-runtime --features apple real_gemma4_e2b_model_backend_prefill_decode_full_vocab_logits_match_hf_reference -- --ignored --nocapture` using `/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-step1.json` | Yes, ignored Metal test | Observed pass for prompt token IDs `[2,4]`, one decode step, all `262144` logits, and HF greedy token `954`; max delta was `0.390625`, mean delta was about `0.0727`, p99 delta was `0.21875`, and p999 delta was `0.2734375` under tolerance `1.0`. This is one prompt and one decode step only; no broad prompt, longer decode, batching, ANE, performance, or production-readiness claim. | Add broader prompt and multi-step full-vocab HF comparisons |
| real Gemma 4 E2B two-step full-vocab decode loop vs HF | REAL-CHECKPOINT-NUMERIC | `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 cargo test -p rvllm-runtime --features apple real_gemma4_e2b_model_backend_prefill_decode_two_steps_full_vocab_logits_match_hf_reference -- --ignored --nocapture` using `/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-steps2.json` | Yes, ignored Metal test | Observed pass for prompt token IDs `[2,4]`, two decode steps, all `262144` logits per step, and generated tokens `[954,1289]`; step 2 max delta was `0.328125`, mean delta was about `0.0784`, p99 delta was `0.203125`, and p999 delta was `0.2421875` under tolerance `1.0`. This is still one prompt and bounded two-step decode only; no broad prompt, long-context, batching, ANE, performance, or production-readiness claim. | Add broader prompts and longer bounded decode loops |
| real Gemma 4 E2B broader-prompt full-vocab logits vs HF | REAL-CHECKPOINT-NUMERIC | `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 cargo test -p rvllm-runtime --features apple real_gemma4_e2b_model_backend_broader_prompt_full_vocab_logits_match_hf_reference -- --ignored --nocapture` using `/tmp/gemma4-e2b-hf-full-logits-prompt-2-17-42-4-step1.json` | Yes, ignored Metal test | Observed pass for prompt token IDs `[2,17,42,4]`, one decode step, all `262144` logits, and HF greedy token `236743`; max delta was `0.2890625`, mean delta was about `0.0516`, p99 delta was `0.16796875`, and p999 delta was `0.2109375` under tolerance `1.0`. This is bounded prompt coverage only; no long-context, batching, ANE, performance, or production-readiness claim. | Add more prompts and longer bounded decode loops |
| real Gemma 4 E2B four-step full-vocab decode loop vs HF | REAL-CHECKPOINT-NUMERIC | `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 cargo test -p rvllm-runtime --features apple real_gemma4_e2b_model_backend_prefill_decode_four_steps_full_vocab_logits_match_hf_reference -- --ignored --nocapture` using `/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-steps4.json` | Yes, ignored Metal test | Observed pass for prompt token IDs `[2,4]`, four decode steps, all `262144` logits per step, and generated tokens `[954,1289,236813,655]`; step 4 max delta was `0.4609375`, mean delta was about `0.2509`, p99 delta was `0.359375`, and p999 delta was `0.3984375` under tolerance `1.0`. This remains bounded by the current probe arena and is not long-context, batching, ANE, performance, or production-readiness evidence. | Extend arena/KV limits deliberately before longer context claims |
| real Gemma 4 E2B eight-step forced-HF-token full-vocab logits vs HF | REAL-CHECKPOINT-NUMERIC | `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 cargo test -p rvllm-runtime --features apple real_gemma4_e2b_model_backend_prefill_decode_eight_steps_forced_hf_tokens_full_vocab_logits_match_hf_reference -- --ignored --nocapture` using `/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-steps8.json` | Yes, ignored Metal test | Probe token cap raised from `8` to `16`; observed pass for prompt token IDs `[2,4]`, eight HF-forced decode contexts, and all `262144` logits per step. Step 8 max delta was `0.30859375`, mean delta about `0.0621`, p99 `0.19140625`, p999 `0.234375` under tolerance `1.0`. Step 7 has an HF exact tie: HF token `464` and Metal-sampled token `21841` both have HF logit `18.875`; therefore this is forced-context numeric evidence, not exact greedy-token evidence beyond the tie. | Resolve tie/precision sampling policy before claiming longer greedy generation |
| real Gemma 4 E2B batch-two bounded prefill finite diagnostic | SCAFFOLD | `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 cargo test -p rvllm-runtime --features apple real_gemma4_e2b_batch_two_prefill_layers_0_to_4_residuals_are_finite -- --ignored --nocapture` | Yes, ignored Metal test | Observed pass for batch-two prompts `[2,4]` and `[2,17]` through bounded prefill stop layers `0..4`; all `6144/6144` residual values were finite at each stop. This is a diagnostic smoke only: no HF comparison, final logits, decode, scheduling, throughput, or production batching claim. | Add batched HF logits parity or trace comparison before promoting batching evidence |
| real Gemma 4 E2B batch-two full-vocab logits vs HF | REAL-CHECKPOINT-NUMERIC | `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 cargo test -p rvllm-runtime --features apple real_gemma4_e2b_batch_two_prefill_decode_full_vocab_logits_match_hf_reference -- --ignored --nocapture`; `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca cargo test -p rvllm-runtime --features apple real_gemma4_e2b_engine_batch_two_prefill_decode_full_vocab_logits_match_hf_reference -- --ignored --nocapture` using `/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-step1.json` and `/tmp/gemma4-e2b-hf-full-logits-prompt-2-17-step1.json` | Yes, ignored Metal tests | Observed pass for one batched prefill/decode step with prompts `[2,4]` and `[2,17]`; both direct backend and Engine scheduler path compared all `262144` logits per sequence against HF references. Seq 1 max delta was `0.390625`, mean delta about `0.0727`, p99 `0.21875`; seq 2 max delta was `0.578125`, mean delta about `0.1020`, p99 `0.3466797`, all under tolerance `1.0`. This is one batch size and one decode step; no scheduler throughput, ANE, or production batching claim. | Add throughput evidence and broader mixed-length scheduler batches |
| real Gemma 4 E2B batch-two two-step full-vocab logits vs HF | REAL-CHECKPOINT-NUMERIC | `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 cargo test -p rvllm-runtime --features apple real_gemma4_e2b_batch_two_prefill_decode_two_steps_forced_hf_tokens_full_vocab_logits_match_hf_reference -- --ignored --nocapture`; `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 cargo test -p rvllm-runtime --features apple real_gemma4_e2b_engine_batch_two_prefill_decode_two_steps_forced_hf_tokens_full_vocab_logits_match_hf_reference -- --ignored --nocapture` using `/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-steps2.json` and `/tmp/gemma4-e2b-hf-full-logits-prompt-2-17-steps2.json` | Yes, ignored Metal tests | Observed pass for batch-two prompts `[2,4]` and `[2,17]`, direct backend and Engine scheduler paths, two HF-forced decode contexts, and all `262144` logits for each sequence/step. Step 2 max deltas were `0.328125` for seq 1 and `0.33496094` for seq 2. Seq 2 step 2 has an HF exact tie: HF token `38028` and Metal-sampled token `236743` both have HF logit `20.75`; therefore this is batched persistent-context numeric evidence, not an exact tie-break claim. | Add throughput evidence, mixed prompt lengths, and longer scheduler-level decode within the probe cap |
| real Gemma 4 E2B Engine mixed-length batch full-vocab logits vs HF | REAL-CHECKPOINT-NUMERIC | `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 cargo test -p rvllm-runtime --features apple real_gemma4_e2b_engine_batch_two_mixed_prompt_lengths_full_vocab_logits_match_hf_reference -- --ignored --nocapture` using `/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-step1.json` and `/tmp/gemma4-e2b-hf-full-logits-prompt-2-17-42-4-step1.json` | Yes, ignored Metal test | Observed pass for Engine-scheduled mixed prompt lengths `[2,4]` and `[2,17,42,4]`, one decode step, decode positions `[1,3]`, context lengths `[2,4]`, and all `262144` logits per sequence against HF references. Seq 1 max delta was `0.390625`, mean delta about `0.0727`, p99 `0.21875`; seq 2 max delta was `0.2890625`, mean delta about `0.0516`, p99 `0.16796875`, all under tolerance `1.0`. This is mixed-length scheduler numeric evidence for one decode step only; no throughput, ANE, long-context, or production batching claim. | Add mixed-length two-step persistent-context coverage and bucket padding coverage |
| real Gemma 4 E2B Engine mixed-length two-step batch full-vocab logits vs HF | REAL-CHECKPOINT-NUMERIC | `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 cargo test -p rvllm-runtime --features apple real_gemma4_e2b_engine_batch_two_mixed_prompt_lengths_two_steps_forced_hf_tokens_full_vocab_logits_match_hf_reference -- --ignored --nocapture` using `/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-steps2.json` and `/tmp/gemma4-e2b-hf-full-logits-prompt-2-17-42-4-steps2.json` | Yes, ignored Metal test | Observed pass for Engine-scheduled mixed prompt lengths `[2,4]` and `[2,17,42,4]`, two HF-forced decode contexts, decode positions `[1,3]` then `[2,4]`, context lengths `[2,4]` then `[3,5]`, and all `262144` logits per sequence/step against HF references. Max deltas were `0.390625` and `0.2890625` at step 1, then `0.328125` and `0.29296875` at step 2, all under tolerance `1.0`. This is mixed-length persistent-context scheduler numeric evidence only; no throughput, ANE, long-context, or production batching claim. | Add bucket padding coverage and longer scheduler-level decode within the probe cap |
| real Gemma 4 E2B Engine batch-three bucket-padding full-vocab logits vs HF | REAL-CHECKPOINT-NUMERIC | `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 cargo test -p rvllm-runtime --features apple real_gemma4_e2b_engine_batch_three_mixed_prompt_lengths_one_step_full_vocab_logits_match_hf_reference -- --ignored --nocapture` using `/tmp/gemma4-e2b-hf-full-logits-prompt-2-4-step1.json`, `/tmp/gemma4-e2b-hf-full-logits-prompt-2-17-step1.json`, and `/tmp/gemma4-e2b-hf-full-logits-prompt-2-17-42-4-step1.json` | Yes, ignored Metal test | Observed pass for Engine-scheduled prompts `[2,4]`, `[2,17]`, and `[2,17,42,4]`; decode bucket was `4`, positions were `[1,1,3]`, context lengths were `[2,2,4]`, and all `262144` logits for the three live rows matched HF references. Max deltas were `0.390625`, `0.578125`, and `0.2890625`, all under tolerance `1.0`. This is bucket-padding row-mapping evidence only; no throughput, ANE, long-context, or production batching claim. | Add longer scheduler-level decode within the probe cap and measured batching throughput |
| real Gemma 4 E2B probe profiling harness | SCAFFOLD | `cargo test -p rvllm-runtime --features apple metal_probe_pipeline_compilation_counters_do_not_change_after_rollout -- --ignored --nocapture`; `cargo test -p rvllm-runtime --features apple metal_probe_arena_regions_do_not_change_after_rollout -- --ignored --nocapture`; `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 RVLLM_E2B_PROFILE_JSON=/tmp/rvllm-e2b-metal-profile.json cargo test -p rvllm-runtime --features apple real_gemma4_e2b_probe_profile_reports_prefill_and_decode_counters -- --ignored --nocapture`; `cargo test -p rvllm-runtime --features apple real_e2b_probe_profile_artifact_schema_records_unmeasured_slots -- --nocapture`; `RVLLM_E2B_PROFILE_SAMPLE_ID=real-e2b-metal-probe-current-2026-05-18 RVLLM_E2B_PROFILE_JSON=/tmp/rvllm-e2b-metal-profile-current.json cargo test -p rvllm-runtime --features apple real_gemma4_e2b_probe_profile_reports_prefill_and_decode_counters -- --ignored --nocapture`; `python3 tools/apple_profile_gate.py --baseline /tmp/rvllm-e2b-metal-profile.json --current /tmp/rvllm-e2b-metal-profile-current.json --output /tmp/rvllm-e2b-metal-profile-gate.json`; `python3 tools/apple_profile_gate_selftest.py` | Yes, ignored Metal tests | After combining non-debug probe submission into one command buffer per prefill/decode step, fusing the attention residual add with the following pre-FFN RMSNorm, and fusing final softcap with argmax, the baseline host artifact for prompt `[2,4]` plus four decode steps recorded prepare `445091 ms`, prefill `574 ms`, decode `1780 ms`, decode `2.2472 tok/s`, prefill about `3.4843 tok/s`, `5` command buffers, `3187` encoders, `5` forced waits, `1` Metal library compile and `32` PSO compiles during prepare. A second current artifact recorded prepare `456078 ms`, prefill `602 ms`, decode `1899 ms`, decode `2.1064 tok/s`, prefill about `3.3223 tok/s`, and the same command buffer, encoder, forced-wait, library compile, and PSO counts. The local regression gate passed with `10%` timing tolerance and zero tolerance for command-buffer/encoder/forced-wait structural counter increases; observed decode regression was `6.27%` and prefill regression was `4.65%`. The gate self-test verifies that wall-clock regressions fail even when encoder count drops, command-buffer/token changes fail, sample IDs must differ, unmeasured metrics require reasons, and failed runs overwrite stale pass reports. A local rejected fusion experiment reduced encoders but regressed decode `35.71%` and prefill `31.59%`; the gate rejected it and the kernel change was not committed. Artifacts keep memory, CPU, GPU, ANE, and energy explicitly unmeasured/unsupported. This is single-host local regression tracking, not optimized performance, external profiler evidence, ANE, batching throughput, or production readiness. | Reduce more encoders/forced waits and add external profiler artifacts before making performance claims |
| rejected shared-KV tail hot-path skip experiments | SCAFFOLD | Fast guards passed locally: `cargo test -p rvllm-apple-metal --lib kernel_rope_q_only_reference_matches_q_side_without_k_mutation -- --nocapture`, `cargo test -p rvllm-runtime --features apple shared_kv_tail_encoder_estimate_drops_without_extra_command_buffer -- --nocapture`, `cargo test -p rvllm-runtime --features apple tiny_shared_kv_tail_local_kv_poison_does_not_change_logits_but_source_v_does -- --ignored --nocapture`, and `cargo test -p rvllm-runtime --features apple apple_metal_backend::tests:: -- --ignored --nocapture` with `RVLLM_GEMMA4_MODEL_DIR` unset. Real gate failed for both experiments: `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 cargo test -p rvllm-runtime --features apple real_gemma4_e2b_model_backend_prefill_decode_selected_logits_match_hf_reference -- --ignored --nocapture` | Yes, failed real ignored Metal test | A WIP optimization that skipped local K/V projection, K/V norm, and KV-cache writes for shared-KV tail layers reduced estimated encoders in synthetic counters, but real E2B selected-logit parity failed: token 0 logit was `2.7050781` versus HF `-2.765625` (`delta=5.470703`, tolerance `1.0`). A narrower WIP that only skipped the local KV-cache write for shared-KV tail layers produced the same real selected-logit failure after the synthetic poison guard passed. Both WIP changes were reverted and not committed. This is negative evidence against treating tail-local K/V cache writes as dead work in real E2B without a deeper layer trace; it is not a performance improvement. | Do not retry the skip until a paired pre/post-skip trace shows identical layer-13/14 source cache, layer-15/16 tail local cache, attention output, and final residual summaries |
| real Gemma 4 E2B shared-KV tail trace diagnostic | SCAFFOLD | `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 cargo test -p rvllm-runtime --features apple real_gemma4_e2b_shared_kv_layers_13_to_16_decode_trace_explains_tail_cache_use -- --ignored --nocapture` | Yes, ignored Metal test | Observed pass for cached E2B prompt `[2,4]` with one decode step, tracing layers `13`, `14`, `15`, and `16` while skipping final logits. Layers `15` and `16` report `shared_kv_source_layer=13`; their `attention_kv_cache_*` summaries match the layer-13 source cache, while their local `k_projection`, `v_projection`, and `local_kv_cache_*` summaries are finite and nonzero. Example layer-15 decode maxima: local cache K/V `0.4133301`/`2.873047`, source attention cache K/V `0.559082`/`5.0`. This shows current real E2B tail attention reads source K/V while tail-local K/V state is still computed and written. It does not explain why the reverted skip changed selected logits without a paired trace of that WIP, and it does not compare final logits or claim performance. | Add a paired diagnostic that runs the exact skip WIP under the same trace and compares source cache, tail local cache, attention output, and residual before any new optimization attempt |
| real Gemma 4 E2B shared-KV paired skip trace diagnostic | SCAFFOLD | `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 cargo test -p rvllm-runtime --features apple real_gemma4_e2b_shared_kv_baseline_vs_skip_trace_identifies_first_divergence -- --ignored --nocapture` | Yes, ignored Metal test | Observed pass for cached E2B prompt `[2,4]` with one decode step, tracing layers `13..20`, skipping final logits, and comparing baseline against two env-gated debug-only modes: `skip_local_kv_cache_write_only` and `skip_tail_kv_projection_and_cache`. Both modes stayed finite. Both first differed from baseline at layer `15` `local_kv_cache_k`; baseline max_abs was `0.4133300781`, candidate max_abs was `0.0`, delta `0.4133300781`. Neither mode produced a downstream diff through traced layers `13..20` across input, Q projection/norm/RoPE, source attention cache, attention output, post-attention/post-FF summaries, or final residual. This does not explain the earlier selected-logit failure yet; it narrows the gap to later layers or missing trace fields. This is diagnostic evidence only; no skip optimization, final-logit comparison, performance claim, ANE claim, or production claim. | Extend paired traces beyond layer `20` and/or add missing summaries before retrying selected logits or any shared-KV skip |
| real Gemma 4 E2B shared-KV source-map audit | REAL-CHECKPOINT-DRYRUN | `RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca cargo test -p rvllm-loader gemma4_real_model_shared_kv_source_map_is_resolved_by_attention_kind_when_env_is_set -- --nocapture`; `cargo test -p rvllm-loader shared_kv -- --nocapture`; `cargo test -p rvllm-apple-metal --lib shared_kv_source_layers -- --nocapture` | No Metal execution required for the real map audit | The loader owns the shared-KV source-map rule and Metal prepare uses the same implementation. For cached E2B, the explicit 35-layer pattern is 4 sliding + 1 full repeated with 20 shared tail layers: tail sliding layers map to source layer `13`, and tail full layers map to source layer `14`. This is metadata/semantic-map evidence only; it does not rerun logits or prove a shared-KV hot-path optimization. | Add poison diagnostics proving tail-local K/V work is dead before retrying a skip, then re-run a bounded real gate |
| synthetic shared-KV tail poison guard | SYNTHETIC-NUMERIC | `cargo test -p rvllm-runtime --features apple tiny_shared_kv_tail_local_kv_poison_does_not_change_logits_but_source_v_does -- --ignored --nocapture` | Yes, ignored Metal test | Three-layer synthetic shared-KV fixture proves the current Metal attention path reads source-layer K/V for the shared tail: poisoning tail-local K/V weights leaves logits unchanged, while poisoning the source layer's V changes logits. This is synthetic guard evidence only; it does not prove the reverted E2B skip optimization is safe. | Add real E2B layer-13/15 trace or a no-op skip guarded by this synthetic invariant before rerunning selected-logit parity |
| default Metal route toy-disabled guard | SCAFFOLD | `cargo test -p rvllm-runtime --features apple apple_default_metal_route_requires_model_dir_unless_toy_is_explicitly_enabled -- --nocapture` | No real checkpoint execution | Structural runtime-route evidence only: the default macOS Metal route requires `weights_path` and only allows the toy Metal backend when `RVLLM_APPLE_TOY_METAL=1` is explicitly set. This does not prove production serving, E2B throughput, or ANE execution. | Exercise the same route with real E2B model config in an integration harness |
| real checkpoint numeric coverage gaps | SCAFFOLD | current selected-logit and full-vocab E2B tests only | Partial | Real-checkpoint numeric evidence is limited to one cached E2B snapshot, two single-request prompts, one same-length Engine batch-two prompt pair with two forced decode contexts, one direct-backend batch-two prompt pair with two forced decode contexts, one mixed-length Engine batch-two prompt pair with two forced decode contexts, one batch-three bucket-padding decode context, and eight forced-HF-token single-request decode contexts inside the current probe cap. No long-context decode beyond 16 probe tokens, broader prompt distribution, sampling beyond greedy/tie handling, ANE execution, or production performance evidence is recorded. | Broaden HF/Transformers comparisons across more prompts, then address longer context and performance separately |
| HF/Transformers real checkpoint reference hook | REAL-CHECKPOINT-NUMERIC | `scripts/dump_gemma4_hf_reference_logits.py --help`, Python compile check; `/tmp/rvllm-gemma4-hf-ref-venv/bin/python scripts/dump_gemma4_hf_reference_logits.py "$RVLLM_GEMMA4_MODEL_DIR" --prompt-token-ids 2,4 --decode-steps 1 --selected-token-ids 0,1,2,3,4,5 --top-k 16 --output /tmp/gemma4-e2b-hf-reference-logits.json`; `scripts/dump_gemma4_hf_layer_trace.py` layer summaries in `/tmp`; `python3 scripts/dump_gemma4_e2b_hf_reference_suite.py "$RVLLM_GEMMA4_MODEL_DIR" --dry-run --skip-existing --output-dir /tmp/rvllm-e2b-suite-dry-run`; `python3 scripts/dump_gemma4_e2b_hf_reference_suite.py "$RVLLM_GEMMA4_MODEL_DIR" --case 2,4 --case 2,17 --case 2,17,42,4 --decode-steps 1 --top-k 16 --full-logits --output-dir /tmp/rvllm-e2b-reference-suite --dry-run`; `python3 tools/run_apple_metal_e2b_cli_suite.py --help` | No rvLLM hardware execution for artifact generation or runner dry-run | HF/Transformers CPU reference artifacts generated for cached Gemma 4 E2B with greedy token `954`; artifacts live outside repo in `/tmp`; the suite wrapper records both the standard selected/full-vocab prompt, decode-loop, and batch-reference artifact commands and custom repeated `--case` manifests so reference regeneration is reproducible. The CLI suite runner can consume that manifest and existing artifacts to drive the diagnostic Metal CLI, but reference generation itself is not rvLLM, Metal, ANE, or production evidence. | Keep using HF artifacts as external references while broadening rvLLM comparison coverage |
| private ANE tiny compile/load/evaluate diagnostic | SCAFFOLD | `cargo test -p rvllm-apple-ane-sys -- --nocapture`; `cargo test -p rvllm-apple public_coreml_tiny_projection_prediction_cpu_and_neural_engine_smoke -- --ignored --nocapture`; `RVLLM_ENABLE_PRIVATE_ANE=1 cargo test -p rvllm-apple --features private-ane test_hardware_ane_compilation_integration -- --nocapture`; `RVLLM_ENABLE_PRIVATE_ANE=1 cargo test -p rvllm-apple --features private-ane private_ane_tiny_projection_private_compile_boundary_is_reported -- --ignored --nocapture`; `RVLLM_ENABLE_PRIVATE_ANE=1 cargo test -p rvllm-apple --features private-ane private_ane_tiny_projection_load_boundary_is_reported -- --ignored --nocapture`; `RVLLM_ENABLE_PRIVATE_ANE=1 cargo test -p rvllm-apple --features private-ane private_ane_tiny_projection_fp16_and_rank4_boundaries_are_reported -- --ignored --nocapture`; `RVLLM_ENABLE_PRIVATE_ANE=1 cargo test -p rvllm-apple --features private-ane private_ane_tiny_projection_rank3_load_source_boundaries_are_reported -- --ignored --nocapture`; `RVLLM_ENABLE_PRIVATE_ANE=1 cargo test -p rvllm-apple --features private-ane private_ane_tiny_projection_load_options_boundaries_are_reported -- --ignored --nocapture`; `RVLLM_ENABLE_PRIVATE_ANE=1 cargo test -p rvllm-apple --features private-ane private_ane_tiny_projection_evaluate_smoke -- --ignored --nocapture` | Yes, public CoreML and private framework tests | The public CoreML path now compiles a generated rank-3 zero-weight projection, loads it with `MLComputeUnitsCPUAndNeuralEngine`, runs prediction through `MLModel`, and validates finite near-zero output. This proves public CoreML execution with a CPU+NeuralEngine compute-units request, but it does not prove private `_ANEClient` execution or ANE utilization. The private framework and `_ANEClient` are present, public CoreML compile/load succeeds, and the tiny ANE compile path produces a compiled bundle. The private compile-boundary diagnostic runs `_ANEClient compileModel` in child test processes; public `MLModel compileModelAtURL` accepts the generated source, but private `_ANEClient compileModel` aborts the child process for both the source `.mlmodel` and compiled `.mlmodelc`, so the parent records a private compile boundary without taking down the suite. `_ANEClient loadModel` still rejects the generated NN bundle with `_ANEEspressoIRTranslator` / `Cannot serialize ANEC_IR_repr`; a FP16 embedded-weight rank-3 variant compiles as CoreML `storagePrecision = Float16` but hits the same private load rejection, while rank-4 NN feature-shape variants are rejected earlier by `coremlcompiler` because NN MLMultiArray inputs must be rank 1 or 3. A rank-3 source/compiled matrix shows source `.mlmodel` is rejected earlier as `Cannot load network ... model.mlmodel/model.espresso.net`; private and public compiled `.mlmodelc` bundles both hit `Cannot serialize ANEC_IR_repr`; toggling `_ANEModel` `standardizeURL` between `true` and `false` does not change those boundaries. The load-options/client-boundary diagnostic runs `loadModel` in child processes: `sharedConnection` rejects both nil and `ForceEspresso` options with the same ANEC_IR_repr serialization boundary, while `sharedPrivateConnection` rejects both options with an unknown `loadModel` error; no private evaluation is run. The evaluate smoke records skipped execution. No private ANE execution claim is made. IOSurface read/write now locks the surface base address instead of casting the IOSurface object pointer. | Keep public CoreML execution separate from private ANE claims; investigate `_ANEModel` attributes/cache identity or a different private-loadable model package before rerunning private evaluate smokes |
| ANE planning | SCAFFOLD | `ane_partition_selection_models_dense_blocks`, strict unavailable tests | No real ANE execution | Planning and gating only; tiny private ANE compile/load currently stops before evaluation | Produce a loadable tiny ANE program, then add hardware-backed partition execution evidence |
| disaggregated fallback scaffold | SCAFFOLD | `synthetic_one_layer_ane_ffn_*_matches_metal_only_fallback` | No real ANE execution | Fallback scaffold; no production disaggregated inference | Hardware-backed partition execution evidence |
| production and optimization claim-readiness gates | SCAFFOLD | `cargo test -p rvllm-apple current_real_e2b_probe_evidence_records_progress_but_not_production_readiness -- --nocapture`; `cargo test -p rvllm-apple complete_evidence_can_pass_evaluator_in_isolation -- --nocapture`; `cargo test -p rvllm-apple current_incomplete_evidence_fails_with_clear_reasons -- --nocapture` | No | The acceptance evaluator has explicit evidence slots for production inference workflow, tokenizer/text decoding, ANE execution, shared-KV optimization safety, and external performance profiling. Current real E2B evidence now records bounded prepared-serving and three-prompt tokenizer/text session evidence, while still intentionally failing production-candidate status because private ANE execution is not established, current timing evidence lacks external profiler counters, core profile metrics such as peak memory/CPU/GPU utilization are unmeasured, and benchmark coverage is incomplete. This is claim-prevention structure, not support for a production claim. | Fill each remaining slot only with bounded evidence: broaden reference-backed text-tokenizer parity, add private ANE evaluated output comparison if Apple-wide acceptance still requires it, capture external profiler-backed performance samples, and complete the benchmark matrix |
| production acceptance criteria evaluator | SCAFFOLD | `current_incomplete_evidence_fails_with_clear_reasons`, `current_real_e2b_probe_evidence_records_progress_but_not_production_readiness`, `complete_evidence_can_pass_evaluator_in_isolation` | No | Acceptance criteria model only; current real E2B evidence records full-vocab HF parity, Engine batch-two same-length and mixed-length two-step scheduler paths, one batch-three bucket-padding path, one local probe profile comparison, real E2B hot-path arena and pipeline compile invariants, diagnostic raw-token/text CLI reporting, bounded reference-backed text inference smokes, a prepared `rvllm-server --backend metal-direct` completion, a public CoreML CPU+NeuralEngine tiny prediction, and a structural default-toy-disabled guard. It still fails production acceptance because there is no private ANE execution, full benchmark coverage, external profiler counters, complete benchmark categories, broad prompt/long-context coverage, or complete measured profile metrics. | Add private ANE execution evidence if Apple-wide acceptance still requires it, external profiler captures, complete benchmark categories, broader regression baselines, and broader prompt/long-context evidence before production-candidate status |

## Arena-backed multi-layer W4/W8 route (2026-07-29)

The schema-v3 Apple package path can select multiple authenticated, non-MoE
dense `down_proj` W4A16 or W8A16 sidecars for real Metal execution. Packed
values and FP16 group-32 scales live inside the model arena and are charged as
immutable weights before paged-KV capacity is selected. Runtime preparation
sorts and validates the complete replacement set before creating a Metal
context or arena. Duplicate, missing, non-dense, MoE, malformed-payload,
dtype-incompatible, and shape-incompatible inputs fail closed.

The default `HybridFallback` residency policy preserves the earlier behavior:
native F16 projections remain resident while their authenticated low-bit
sidecars are selected as the sole execution source. The internal opt-in
`ReplaceNative` policy omits every selected native projection, subtracts its
exact checkpoint bytes, adds the exact aligned sidecar footprint, and then
sizes paged KV from the reduced fixed-resource budget. Each prepared layer is
validated to expose exactly one execution source to the encoder. Policy,
package identity, sorted tensor names, formats, and shapes participate in
numeric ABI v5, so residency or sidecar changes cannot share prompt KV.

The package is re-authenticated when sidecar bytes are consumed, closing the
mutation window between package validation and Metal upload. Consumption reads
directly into fixed-size arena regions with current-length, EOF, symlink,
regular-file, and SHA-256 checks; production preparation does not create a
second full-size packed-weight staging copy.

Hardware evidence:

- `schema_v3_w4_w8_down_proj_sidecars_are_selected_and_execute_real_layer`
  passed a three-token prefill for W4 and W8 in both default-hybrid and
  native-replacement modes using precompiled metallibs. It verifies package
  selection, arena residency, real decoder-layer dispatch, CPU low-bit
  projection parity across all three rows, removal of the native projection,
  lower reported fixed weight residency, and a result measurably different
  from the native dense fallback.
- `schema_v3_two_layer_mixed_low_bit_package_replaces_both_native_projections`
  passed a real Metal prefill with a reverse-ordered package containing layer-0
  W4 and layer-1 W8. Both native projections were absent, both authenticated
  sidecars were active in canonical order, and the capacity report recorded
  two low-bit projections.
- `metal_low_bit_projection_offsets_match_cpu_reference_for_w4_and_w8` passed
  for both formats using offsets into one shared Metal buffer.
- Portable planner tests verify exact aligned byte displacement, deterministic
  sorting, mixed-format identity separation, and fail-closed duplicate,
  missing, shape, payload, and MoE cases.
- The Apple Metal library suite passed 74 tests with 7 hardware/manual tests
  ignored. The Apple-feature runtime library suite passed 148 tests with 92
  hardware/large-model tests ignored.

This remains an opt-in down-projection replacement route, not whole-model
mobile W4/W8 qualification, an iOS default, a quality promotion, automatic
Core ML routing, measured ANE execution, or a new performance result. A
qualified dense 0.5B–2B package, end-to-end W4/W8 quality gates, physical iOS
memory/thermal qualification, and clean performance measurements remain
required before promotion.

## Continuous T2 warm-cache maintenance (2026-07-29)

Default macOS T2 restore and capture now have an explicit lifecycle inside the
continuous Metal worker. Admission performs exact metadata lookup only and
retains immutable warm pages in an owned `PreparedWarmRestore`; it performs no
Metal page I/O and therefore cannot reject an otherwise valid request merely
because the three-slot submission ring is busy. Prepared bytes remain valid if
T2 metadata is evicted while the request waits.

Pending restore or terminal capture ownership is queued FIFO. Every submission
refill path stops, already-submitted work drains oldest-first, and exactly one
maintenance operation runs at the idle arena barrier. Restore cost is
re-evaluated after the drain; a stale, slower, busy, cancelled, or otherwise
failed optional restore transaction releases all provisional pages and
recomputes the prompt instead of failing admission. Terminal capture failures
are explicitly reported as skipped, and the request-owned chain is always
released. Memory pressure converts queued restores to recomputation, cancels
optional promotions, evicts unpinned T1/T2 state, and preserves active T0.

Evidence:

- `prepared_warm_restore_survives_eviction_and_busy_materialization` and
  `prepared_warm_restore_cancellation_releases_transactional_chain` verify
  owned-byte lifetime, retryability, transactional rollback, and zero leaked
  request chains.
- Continuous-worker tests verify ring-drain gating, maintenance-aware refill,
  FIFO stale-entry cleanup, and the wait-plus-copy versus recomputation cost
  crossover.
- The full Apple-feature runtime library suite passed with 148 tests, 92
  hardware/large-model tests ignored, and zero failures on the audited Mac.

This uses a safe idle-only maintenance barrier. It is correctness evidence for
the default T2 lifecycle, not yet the later optimization that moves page copies
onto preallocated asynchronous Metal blit submissions, nor a T2 throughput or
TTFT promotion result.

## Opt-in encrypted T3 persistent prompt cache (2026-07-29)

The Apple runtime now wires the opt-in `PersistentEncrypted` policy through the
continuous Metal worker, C ABI v3, and Swift host boundary. T3 remains disabled
unless the host supplies explicit consent, a nonzero bounded quota, an absolute
cache root, an engine-owned tenant namespace, and exactly 32 bytes of
installation key material. Legacy ABI configurations cannot enable T3.

Implemented security and correctness:

- Records contain only complete 32-token prompt-prefix KV pages before the
  private writable tail; generated continuation KV is outside the store API.
  AES-256-GCM authenticates the lossless payload and bound identity/token
  locators. Encryption and path-HMAC keys are independently derived with HKDF,
  sensitive host/configuration buffers are redacted or zeroed where owned, and
  decrypted payloads use zeroing storage.
- Tenant directories and record names are keyed, tenant-local locators.
  Restore decrypts and verifies the complete cache identity, exact prompt
  tokens, page count, lengths, checksum, and authentication tag. Corrupt,
  partial, unauthenticated, or mismatched records are discarded from serving
  and may be quarantined; they do not become inference output.
- Tenant directories are verified private (`0700`). Lock, temporary, and
  record files use no-follow opens and are validated as single-link regular
  files with `0600` permissions. A per-tenant OS advisory lock serializes stale
  temporary cleanup, quota accounting, write/sync, and atomic rename across
  processes. Lock waiting is cancellation-safe, and recognizable stale
  temporary records are removed only while that lock is held.
- Continuous serving uses a bounded background T3 I/O worker. Restore is
  attempted only when measured total restore cost is strictly below predicted
  recomputation; authenticated pages are materialized only at the idle Metal
  maintenance barrier. Cancellation and memory pressure advance an I/O epoch.
  An in-process publication gate linearizes epoch invalidation against the
  final atomic rename, so pressure cannot return while an older promotion can
  still publish. Active T0 ownership is preserved.
- ABI v3 is an append-only extension of v2 and copies the exact key,
  normalized absolute root, and bounded engine namespace during creation.
  Swift obtains a random per-install key from the public Keychain API as a
  non-synchronizing `AfterFirstUnlockThisDeviceOnly` item, protects key
  creation/rotation with an installation lock and marker, creates the cache
  under Application Support, excludes it from backup, and on iOS sets and
  verifies `completeUntilFirstUserAuthentication` Data Protection before
  engine creation.

Current automated evidence:

- The full Apple-feature runtime library suite passed 157 tests with 92
  hardware/large-model tests ignored and zero failures.
- The focused persistent-cache suite passed 21 of 21 tests, including encrypted
  round-trip, cost gating, tenant isolation, corruption, no-follow and
  hard-link rejection, quota enforcement, cancellation before publication,
  epoch/publication ordering, advisory-lock serialization, cancellation-safe
  lock waiting, stale-temporary cleanup, hostile root/tenant replacement, and
  symlinked-ancestor rejection.
- The focused continuous-worker suite passed 12 of 12 tests, the request API
  suite passed 17 of 17 tests, the Apple FFI suite passed 17 of 17 tests, and
  the Swift package passed 17 of 17 tests.

These results establish the implemented host boundary and portable
security/correctness behavior on the audited Mac. They do not establish
physical-iOS Keychain or Data Protection behavior, jetsam/thermal robustness,
cross-process soak behavior, persistent-cache energy cost, or a T2/T3 restore
performance win on supported devices. T3 remains opt-in and automatic restore
still requires its measured per-candidate cost win; physical oldest/current
iPhone and iPad qualification plus TTFT, restore-throughput, memory-pressure,
background/foreground, and corruption-recovery measurements remain promotion
gates.

## Sustained pressure control and dirfd-relative T3 storage (2026-07-30)

Memory pressure is now a persistent worker state, not a one-shot purge.
Warning pressure caps admission at `ceil(maximum_concurrency / 2)`, permits
only surviving zero-copy T1 attachment, and suppresses promotion plus T2/T3
restore, capture, and publication. Critical pressure caps admission at one and
disables every new cache attachment. Active T0 requests are not aborted,
excess requests remain queued, and releasing a pinned T0 attachment immediately
re-sweeps the current pressure level. Full policy returns only after the host
sends the public `Normal` recovery level through Rust, C, or Swift.
A live `EngineHandle` worker-loop regression additionally proves that a
Critical acknowledgement remains sticky across request completion, holds later
requests in the real ingress/admission path, and admits them only after an
explicit Normal recovery.

Apple T3 storage pins the absolute root component-by-component from `/` with
`openat`, rejects symlink, dot, parent, and relative traversal, and retains
root/tenant directory descriptors for the store lifetime. Tenant creation,
locks, records, temporary files, quota scans, cleanup, publication, quarantine,
and directory synchronization are descriptor-relative. Directories require the
effective user and exact `0700`; files require that user, regular-file type,
one link, and exact `0600`. Host-side installation lock and marker operations
use the same descriptor-relative pattern and fsync their publication directory.
Hostile root and tenant name replacement therefore cannot redirect an
in-flight store, restore, lock, cleanup, or rename.

Observed automated gates on the audited Mac: the focused request API suite,
including the live pressure regression and strict rejection of persistent-cache
capability under a nonpersistent policy, passed `17/17`. The preceding
hardware-available full-suite checkpoint remains runtime Apple library
`157/157` with `92` hardware/large-model tests ignored; persistent cache
`21/21`; Apple FFI `17/17`; Swift package `17/17`; Rust device and simulator
iOS cross-checks passed. The shipping release gate built macOS, iOS device, and
iOS simulator Rust/Swift artifacts and found zero private-API findings across
all 13 scanned artifacts. These are code and simulator-target build gates, not
physical-device memory-warning, Keychain/Data Protection, jetsam, or thermal
qualification. POSIX advisory locks remain cooperative with other same-user
processes, while encryption/authentication and strict validation continue to
fail closed.

## Exact public Core ML gated-FFN gate (2026-07-29)

The checkpoint exporter now has a bounded public Core ML gated-FFN artifact:
two exact checkpoint-derived 1x1 projections, Gemma's tanh-approximate GELU,
elementwise gate/up multiplication, and the exact checkpoint-derived down
projection. Its schema binds all three tensor names, source dtypes and digests,
static shape, exporter, graph, input KAT, and expected-output KAT. CPU
validation reconstructs each source tensor from the model and rejects graph,
weight, identity, or activation-semantic tampering.

The real public Core ML compiler/runtime gate passed on the audited Mac using
exact rank-4 array mapping. The model was loaded with
`CPUAndNeuralEngine`; the execution report deliberately records this as a
request only and keeps `neural_engine_execution_verified=false`. This is not
automatic routing, a stateful decoder, a cohort, or measured ANE placement.

Compiled Core ML cache manifests are now schema v2 and bind a canonical SHA-256
of the complete compiled bundle. A missing or modified payload, corrupt
manifest, symlink, or non-regular entry is never reused; valid artifacts still
reuse without compilation, while invalid targets are rebuilt under the KAT
gate, quarantine, and atomic rename protocol. Cross-process serialization uses
an OS advisory lock, so process death releases ownership even when the
diagnostic lock path remains. Reuse also requires the current compute-unit
request and device-evidence record to match; stale evidence cannot be reported
for a later caller.

Shipping cache callers can bind an exact known-answer policy consisting of the
versioned case ID, input digest, expected-output digest, and tolerance. Exact
policy hits remain validation-free. A policy change revalidates the existing
authenticated compiled payload without recompilation and atomically replaces
only the manifest; failure preserves the previous payload and validation
record. The exact graph validator additionally rejects nonzero convolution
padding and optional, flexible-shape, or default-valued interfaces, with
refreshed-identity tamper tests for each mutation.
