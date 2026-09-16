# Apple Metal E2B Diagnostic CLI

This workflow is for diagnostic Apple Metal E2B probes only. It defaults to
raw token IDs, and has an optional tokenizer-backed text prompt/text decode
mode for diagnostics. It is not production inference, ANE execution, broad
tokenizer coverage, or an optimization claim.

## Cooperative Microbatch Projection Pass (2026-07-15)

Decode and prompt microbatches (`M <= 19`) now route projection+RMSNorm sites
through the cooperative `gemm_f16_vec8` GEMV followed by an explicit RMSNorm.
The previous single-encoder fused kernel launches only one 256-thread group per
token and serializes large output projections. Larger prefills retain that
fused fallback. Runtime encoder accounting follows the selected policy.

The cutoff is measured rather than assumed. On M4 Max with E2B, matched
repeated-token prefills measured cooperative versus fused at M=17 (365.97 vs
395.78 ms), M=18 (382.60 vs 398.30 ms), and M=19 (400.52 vs 405.37 ms). At
M=20 the ordering reverses sharply (444.40 vs 405.76 ms), and at M=24 the fused
path leads 419.01 vs 543.77 ms. All crossover probes produced the same greedy
token under both policies; they are policy-consistency checks, not external HF
long-context numerical references.

The E2B `Hello` four-token gate generated the checked-in HF sequence
`[236764, 108, 236777, 735]` in three consecutive runs. Decode measurements
were `152.668458`, `159.632292`, and `157.313541` ms (25.1--26.2 tok/s), versus
the prior layer-scale-fused measurement of `908.139792` ms (4.40 tok/s). E4B-it
kept `[236888]` while decode dropped from `459.360709` to `66.006459` ms.
26B-A4B-it kept `[993]` while decode dropped from `234.360792` to `71.849416`
ms. The 26B cold prefill measured `4230.564875` ms with a 50.5 GB arena, so
cold-start/prefill and steady-state decode remain separate performance gates;
memory residency is a likely contributor but was not externally profiled.

Durable E2B reference artifacts now live under
`crates/rvllm-runtime/tests/reference/`; ignored hardware tests no longer depend
on ephemeral `/tmp` reference files. Model-suite cases may set positive
`min_tok_per_s`, `max_prefill_ms`, and `max_decode_ms` values; missing metrics
or threshold misses fail the case. On the measured M4 Max, a conservative E2B
four-token gate is `min_tok_per_s: 20.0`, `max_prefill_ms: 300.0`, and
`max_decode_ms: 250.0`.

The final fail-closed suite run passed with `hf_reference.matched=true`, no
performance failures, `tok_per_s=26.91796090578023`,
`prefill_ms=166.616917`, and `decode_ms=148.599666`. The prepared
`metal-direct` HTTP server returned the same four reference tokens at
`25.4911832369979` tok/s and `156.917` ms decode;
`GET /healthz` reported `ready=true` and `prepared_once=true`.

## Setup

Set the local checkpoint path:

```bash
export RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca
test -d "$RVLLM_GEMMA4_MODEL_DIR"
```

Use the HF reference environment when generating artifacts:

```bash
export RVLLM_HF_REF_PYTHON=/tmp/rvllm-gemma4-hf-ref-venv/bin/python
```

Legacy raw-token Metal probes still require an explicit opt-in:

```bash
export RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1
```

Passing `--large-model-opt-in` to legacy probe CLIs is equivalent for that
process. The public text/session/server/bench Metal paths accept the flag for
compatibility, but the production-candidate pass below no longer requires it for
cached E2B or 31B-it.

Configurable text/session arena sizing:

```bash
export RVLLM_METAL_MAX_TOTAL_TOKENS=128
export RVLLM_METAL_MAX_BATCH_TOKENS=16
export RVLLM_METAL_MAX_BATCH_SEQUENCES=1
```

`RVLLM_METAL_MAX_PROBE_TOKENS` is still accepted as a legacy fallback when
`RVLLM_METAL_MAX_TOTAL_TOKENS` is unset.

## Production-Candidate Configurable Context Snapshot (2026-05-26)

This Metal-only pass removes the old public text/session short probe cap for
accepted scenarios and uses explicit Metal arena sizing. XLA is intentionally
excluded; ANE and shared-KV optimization are untouched. CUDA parity is not
claimed because this macOS host has no runnable CUDA device (`nvidia-smi` was
not found). JSON/report claims remain conservative: this is local default
checkpoint-native Metal execution evidence for cached E2B, E4B-it,
26B-A4B-it MoE, and 31B-it snapshots, not broad production readiness. BF16
Google safetensors now stay BF16-native by default; use
`RVLLM_METAL_DTYPE=f16` only when explicitly requesting the F16 fallback.

Additional Google safetensors snapshots were pulled into the local HF cache:

- `google/gemma-4-E4B-it`:
  `/Users/george/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/d6436b3d62967e1af08bbb046c6300b2a9ae8e85`
- `google/gemma-4-26B-A4B-it`:
  `/Users/george/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/b2a81a03d25f927590a91d84ba43f96e8ef7349f`

E4B-it validates and runs one-token Metal decode. 26B-A4B-it is present
locally, validates its MoE router/expert tensor shapes, and now runs through
the Metal MoE layer path. The Rust model suite rejects `expected_error_contains`
so present snapshots must pass or fail loudly; no 26B expected-error case is
recorded.

Reference-backed E2B commands:

```bash
target/debug/rvllm_metal_infer \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt Hello \
  --max-new-tokens 1 \
  --max-total-tokens 128 \
  --hf-reference /tmp/gemma4-e2b-hf-text-infer-hello-step1.json \
  --json > /tmp/rvllm-e2b-hello-step1-configurable-context.json

target/debug/rvllm_metal_infer \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt Hello \
  --max-new-tokens 4 \
  --max-total-tokens 128 \
  --hf-reference /tmp/gemma4-e2b-hf-text-infer-hello-steps4.json \
  --json > /tmp/rvllm-e2b-hello-step4-configurable-context.json
```

Observed: one-token `[236764]`, `hf_reference.matched=true`,
`decode_ms=220.994541`, `tok_per_s=4.5249986514372775`, `encoders=1063`,
`forced_waits=2`; four-token `[236764,108,236777,735]`,
`hf_reference.matched=true`, `decode_ms=893.389417`,
`tok_per_s=4.477330852465203`.

Session workflow:

```bash
target/debug/rvllm_metal_infer \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --reference-manifest /tmp/rvllm-e2b-text-reference-suite-current/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --session-backend direct \
  --max-total-tokens 128 \
  --case-timeout-seconds 900 \
  --report /tmp/rvllm-e2b-direct-session-configurable-context.json \
  --json > /tmp/rvllm-e2b-direct-session-configurable-context.stdout.json

target/debug/rvllm_metal_infer \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --reference-manifest /tmp/rvllm-e2b-text-reference-suite-current/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --session-backend engine \
  --max-total-tokens 128 \
  --case-timeout-seconds 900 \
  --report /tmp/rvllm-e2b-engine-session-configurable-context.json \
  --json > /tmp/rvllm-e2b-engine-session-configurable-context.stdout.json
```

Observed direct session: status `pass`, all three HF references matched,
`tok_per_s=4.449827912175971`. Observed Engine session: status `pass`, all
three HF references matched, `prefill_ms=433.373875`, `decode_ms=255.023375`,
`tok_per_s=11.763627549827541`, `command_buffers=2`, `encoders=1063`,
`forced_waits=2`.

Profile workflow:

```bash
target/debug/rvllm_metal_infer \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --reference-manifest /tmp/rvllm-e2b-text-reference-suite-current/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --session-backend engine \
  --max-total-tokens 128 \
  --case-timeout-seconds 900 \
  --profile-samples 2 \
  --profile-report /tmp/rvllm-e2b-engine-session-profile-configurable-context.json \
  --json > /tmp/rvllm-e2b-engine-session-profile-configurable-context.stdout.json
```

Median profile values: `tok_per_s=11.925064075849907`,
`latency_ms_per_token=83.85739583333333`,
`command_buffers_per_token=0.6666666666666666`,
`encoders_per_token=354.3333333333333`,
`forced_waits_per_token=0.6666666666666666`,
`cpu_encode_ms_per_token=2.2561531666666665`, and
`command_buffer_wait_ms_per_token=224.52197916666665`.

Model-suite workflow:

```bash
target/debug/rvllm_metal_model_suite \
  --manifest /tmp/rvllm-gemma4-metal-production-candidate-suite-manifest.json \
  --report /tmp/rvllm-gemma4-metal-production-candidate-suite-report.json \
  --json > /tmp/rvllm-gemma4-metal-production-candidate-suite.stdout.json
```

Earlier pre-download observed status was `pass`, `passed=3`, `skipped=2`. E2B one-token and four-token
cases HF matched; 31B-it one-token execution produced `[9259]` with
`decode_ms=3307.8813330000003` and no local HF reference. At that time, missing Google
safetensors snapshots for E4B-it and 26B-A4B-it were explicit
`skip_missing_model` cases.

Updated all-local safetensors suite:

```bash
RVLLM_METAL_MAX_TOTAL_TOKENS=128 target/debug/rvllm_metal_model_suite \
  --manifest /tmp/rvllm-gemma4-metal-all-local-safetensors-suite-f16refs-manifest.json \
  --report /tmp/rvllm-gemma4-metal-all-local-safetensors-suite-f16refs-report.json \
  --json > /tmp/rvllm-gemma4-metal-all-local-safetensors-suite-f16refs.stdout.json
```

Observed status `pass`, `passed=4`, `skipped=0`, with no expected-error
contract. The latest default suite report is
`/tmp/rvllm-gemma4-metal-family-native-bf16-default-report.json`. It records
`metal_compute_dtype=bfloat16`, `metal_weight_dtype=bfloat16`, and
`metal_moe_router_weight_dtype=bfloat16`:

- E2B generated `[236764]`, output `"Hello,"`, `hf_reference.matched=true`,
  `prepare_ms=2400.1242500000003`, `prefill_ms=323.321542`,
  `decode_ms=224.055625`, `tok_per_s=4.463177391774922`.
- E4B-it generated `[236888]`, output `"Hello!"`,
  `hf_reference.matched=true`, `prepare_ms=3966.834417`,
  `prefill_ms=633.378458`, `decode_ms=470.596`,
  `tok_per_s=2.124964938078522`.
- 26B-A4B-it generated `[993]`, output `"Hello there"`,
  `hf_reference.matched=true`, `prepare_ms=10960.55775`,
  `prefill_ms=805.81625`, `decode_ms=239.548375`,
  `tok_per_s=4.174522160711797`, `arena_bytes=50551220556`.
- 31B-it generated `[9259]`, output `"HelloHello"`,
  `hf_reference.matched=true`, `prepare_ms=18911.961834`,
  `prefill_ms=5445.652125`, `decode_ms=3346.544625`,
  `tok_per_s=0.2988156776782859`, `arena_bytes=61739678024`.

Post-fix report paths:

- `/tmp/rvllm-gemma4-metal-family-native-bf16-default-report.json`
- `/tmp/rvllm-gemma4-26b-a4b-it-hello-1tok-default-f16-after-dtype-fix.json`

26B-A4B also passed a four-token F16 gate: generated
`[993,236888,1030,236789]`, output `"Hello there! It'"`,
`hf_reference.matched=true`, `decode_ms=963.987125`, and
`tok_per_s=4.149433012396302`.

The separate 26B-A4B HF BF16 comparison remains a reference-policy blocker:
native Metal BF16 generated `[993]` while BF16 HF/MPS generated `[107]`, with
`hf_reference.matched=false`, `metal_compute_dtype=bfloat16`,
`metal_weight_dtype=bfloat16`, and `metal_moe_router_weight_dtype=bfloat16`.
The diagnostic report is
`/tmp/rvllm-gemma4-26b-a4b-it-hello-1tok-explicit-native-bf16-after-bf16-guard.json`.
An HF CPU `torch_dtype=auto` BF16 reference for the same prompt token IDs
selected `[532]`, while HF MPS BF16 selected `[107]`; this confirms the BF16
comparison surface is still diagnostic only, not a passing correctness gate.
Do not claim CUDA/BF16 parity or broad production readiness from the current
Metal evidence.

## Layer-Scale Fusion Snapshot (2026-05-30)

The current Metal diagnostic CLI now reports the layer-scale fusion counter
added in the runtime path. The optimization folds the final per-layer residual
add and scalar layer-scale multiply into `residual_add_then_scale_f16` where the
Gemma 4 layer shape permits it. The report field is
`layer_scale_encoder_fusions`, with the same value exposed under
`encoder_counts_by_kernel_family.layer_scale_fused`. This is a conservative
encoder-count optimization; it does not change the JSON claim boundary and does
not make the path production-ready.

Reference-backed one-token and four-token commands:

```bash
target/debug/rvllm_metal_infer \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt Hello \
  --max-new-tokens 1 \
  --max-total-tokens 128 \
  --hf-reference /tmp/rvllm-gemma4-metal-family-f16-refs/gemma4-e2b-hello-1tok.json \
  --json > /tmp/rvllm-e2b-hello-one-token-layer-scale-fused-bf16.json

target/debug/rvllm_metal_infer \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt Hello \
  --max-new-tokens 4 \
  --max-total-tokens 128 \
  --hf-reference /tmp/rvllm-e2b-text-reference-suite-layer-scale-fused/gemma4-e2b-hf-text-reference-hello-steps4.json \
  --json > /tmp/rvllm-e2b-hello-four-token-layer-scale-fused-bf16.json
```

Observed one-token result: `[236764]`, `hf_reference.matched=true`,
`metal_weight_dtype=bfloat16`, `decode_ms=225.284125`,
`tok_per_s=4.438839`, `encoders=993`,
`layer_scale_encoder_fusions=70`, and `forced_waits=2`. Observed four-token
result: `[236764,108,236777,735]`, `hf_reference.matched=true`,
`decode_ms=908.139792`, `tok_per_s=4.404608`, `encoders=2487`,
`layer_scale_encoder_fusions=175`, and `forced_waits=5`.

Session workflow:

```bash
target/debug/rvllm_metal_infer \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --reference-manifest /tmp/rvllm-e2b-text-reference-suite-layer-scale-fused/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --session-backend direct \
  --max-total-tokens 128 \
  --case-timeout-seconds 120 \
  --report /tmp/rvllm-e2b-direct-session-layer-scale-fused-bf16.json \
  --json > /tmp/rvllm-e2b-direct-session-layer-scale-fused-bf16.stdout.json

target/debug/rvllm_metal_infer \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --reference-manifest /tmp/rvllm-e2b-text-reference-suite-layer-scale-fused/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --session-backend engine \
  --max-total-tokens 128 \
  --case-timeout-seconds 120 \
  --report /tmp/rvllm-e2b-engine-session-layer-scale-fused-bf16.json \
  --json > /tmp/rvllm-e2b-engine-session-layer-scale-fused-bf16.stdout.json
```

Observed direct session: status `pass`, all references matched,
`prepare_ms=1308.634792`, `prefill_ms=860.986708`,
`decode_ms=680.701083`, `tok_per_s=4.407221`, `encoders=2979`,
`layer_scale_encoder_fusions=210`, `forced_waits=6`. Observed Engine session:
status `pass`, all references matched, `prepare_ms=1309.398833`,
`prefill_ms=435.702084`, `decode_ms=251.777542`,
`tok_per_s=11.915280`, `encoders=993`,
`layer_scale_encoder_fusions=70`, `forced_waits=2`.

All-local Gemma 4 safetensors suite:

```bash
target/debug/rvllm_metal_model_suite \
  --manifest /tmp/rvllm-gemma4-metal-all-local-safetensors-suite-f16refs-manifest.json \
  --report /tmp/rvllm-gemma4-metal-family-layer-scale-fused-bf16-report.json \
  --json > /tmp/rvllm-gemma4-metal-family-layer-scale-fused-bf16.stdout.json
```

Observed suite status `pass`, `passed=4`, `skipped=0`: E2B `[236764]`,
E4B-it `[236888]`, 26B-A4B-it `[993]`, and 31B-it `[9259]`. All four matched
their configured token references with `metal_weight_dtype=bfloat16`. The
recorded `layer_scale_encoder_fusions` values were `70`, `84`, `60`, and `120`
respectively.

Profile workflow:

```bash
target/debug/rvllm_metal_infer \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --reference-manifest /tmp/rvllm-e2b-text-reference-suite-layer-scale-fused/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --session-backend engine \
  --max-total-tokens 128 \
  --case-timeout-seconds 120 \
  --profile-samples 3 \
  --profile-report /tmp/rvllm-e2b-engine-session-layer-scale-fused-profile-bf16.json \
  --json > /tmp/rvllm-e2b-engine-session-layer-scale-fused-profile-bf16.stdout.json
```

Median profile values: `tok_per_s=11.762866`,
`latency_ms_per_token=85.013292`,
`command_buffers_per_token=0.6666666666666666`,
`encoders_per_token=331.0`,
`layer_scale_fusions_per_token=23.333333333333332`,
`forced_waits_per_token=0.6666666666666666`,
`cpu_encode_ms_per_token=2.302570`, and
`command_buffer_wait_ms_per_token=223.704931`.

Public surface checks after the same fusion:

- `rvllm-eval` generated `","` and reported `tok_per_sec=4.424`,
  `prepare_ms=1352.1`, `prefill_ms=320.7`, `decode_ms=226.0`.
- `rvllm-bench` report
  `/tmp/rvllm-e2b-bench-layer-scale-fused-bf16.json` used schema
  `rvllm.apple_metal_bench.v1`, `tok_per_sec=1.7621903566553982`,
  `encoders=993`, `layer_scale_encoder_fusions=70`, `forced_waits=2`.
- `rvllm-ppl` report `/tmp/rvllm-e2b-ppl-layer-scale-fused-bf16.json`
  recorded `perplexity=329.24177482909147`, `tokens=1`, `encoders=996`,
  `layer_scale_encoder_fusions=70`, `forced_waits=3`.

External profiler status: `xcrun xctrace record --template 'Power Profiler'`
reported that Power Profiler is not supported on macOS. `Metal System Trace`
did not terminate cleanly under `xctrace`, and the partial trace exported with
`Document Missing Template Error`. Treat the current evidence as internal
counter/timing evidence only; no external energy metric is claimed.

The E2B three-prompt session gates were also rerun with unset
`RVLLM_METAL_DTYPE`, which now selects BF16 for the cached BF16 checkpoint:

- Direct report `/tmp/rvllm-e2b-direct-session-native-bf16-default-report.json`:
  status `pass`, generated tokens `[236764]`, `[236764]`, `[9079]`, all
  references matched, `prepare_ms=2375.521`, `decode_ms=677.807666`,
  `tok_per_s=4.426034331691963`, `encoders=3189`, `forced_waits=6`.
- Engine report `/tmp/rvllm-e2b-engine-session-native-bf16-default-report.json`:
  status `pass`, generated tokens `[236764]`, `[236764]`, `[9079]`, all
  references matched, `prepare_ms=1498.920708`, `decode_ms=253.98162499999998`,
  `tok_per_s=11.81187812307288`, `encoders=1063`, `forced_waits=2`.

The E2B four-token `Hello` gate also passes under default BF16:
`/tmp/rvllm-e2b-hello-four-token-native-bf16-default-report.json` generated
`[236764,108,236777,735]`, `hf_reference.matched=true`,
`decode_ms=886.9545420000001`, `tok_per_s=4.509813987738731`, `encoders=2662`,
and `forced_waits=5`.

Public surfaces also completed under `RVLLM_METAL_MAX_TOTAL_TOKENS=128`:
`rvllm-eval` generated `","` with `decode_ms=225.0`; `rvllm-bench` emitted
schema `rvllm.apple_metal_bench.v1` with `tok_per_sec=1.823201423191031`;
`rvllm-ppl` emitted `perplexity=504.1798937686529`, `tokens=2`; prepared
`rvllm-server --backend metal-direct` returned `","` with token `[236764]` and
`decode_ms=223.831042`.

External trace:
`/tmp/rvllm-e2b-metal-configurable-context.trace`. Exported
`metal-application-command-buffer-submissions` showed two command buffers with
`495` and `498` encoders, `11.619543 ms` total command-buffer duration, and
`9.736795 ms` total encoder time. This matches the previous fused trace shape;
the accepted performance improvement for this pass is the one-token decode
remaining below the recorded `~226 ms` baseline without correctness regression.

## Current Completion Pass Snapshot (2026-05-26)

The current Metal diagnostic path includes a fused QKV+RoPE+KV-cache-write
decode kernel for supported Gemma 4 shapes, with conservative fallback for
debug/trace or unsupported shapes. Reports now expose CPU encode time,
command-buffer wait time, last-step counters, and encoder counts by runtime
family (`embedding`, `ple_input`, `layer_body`, `final_sample`,
`final_logits_diagnostic`). These counters are diagnostic only and do not make
the path production-ready.

Reference-backed E2B one-token command:

```bash
RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 target/debug/rvllm_metal_infer \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt Hello \
  --max-new-tokens 1 \
  --max-total-tokens 16 \
  --large-model-opt-in \
  --hf-reference /tmp/rvllm-e2b-text-reference-suite-current/gemma4-e2b-hf-text-reference-hello-step1.json \
  --json > /tmp/rvllm-e2b-hello-step1-qkv-rope-cache-family.json
```

Observed result: generated token `[236764]`, text `","`,
`hf_reference.matched=true`, `decode_ms=221.244292`,
`tok_per_s=4.5198906193701935`, `command_buffers=2`, `encoders=1063`
(`embedding=2`, `ple_input=8`, `layer_body=1050`, `final_sample=3`),
`forced_waits=2`, `cpu_encode_ns=8647624`, and
`command_buffer_wait_ns=541699208`.

Four-token E2B quality/perf check:

```bash
RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 target/debug/rvllm_metal_infer \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt Hello \
  --max-new-tokens 4 \
  --max-total-tokens 16 \
  --large-model-opt-in \
  --hf-reference /private/tmp/gemma4-e2b-hf-text-infer-hello-steps4.json \
  --json > /tmp/rvllm-e2b-hello-step4-qkv-rope-cache-family.json
```

Observed result: generated tokens `[236764,108,236777,735]`, generated text
`",\n\nI have"`, `hf_reference.matched=true`, `decode_ms=891.2090000000001`,
`tok_per_s=4.488285015075027`, `command_buffers=5`, `encoders=2662`, and
`forced_waits=5`.

Session reports against
`/tmp/rvllm-e2b-text-reference-suite-current/gemma4-e2b-hf-text-reference-suite-manifest.json`:

- Direct session report `/tmp/rvllm-e2b-direct-session-qkv-rope-cache-family.json`:
  status `pass`, all three HF references matched,
  `prepare_ms=6302.7246669999995`, `prefill_ms=896.1980410000001`,
  `decode_ms=705.6414179999999`, `tok_per_s=4.2514511244293205`,
  `generated_tokens=3`, `command_buffers=6`, `encoders=3189`,
  `forced_waits=6`.
- Engine session report `/tmp/rvllm-e2b-engine-session-qkv-rope-cache-family.json`:
  status `pass`, all three HF references matched,
  `prepare_ms=5631.603333`, `prefill_ms=434.200125`,
  `decode_ms=255.08958399999997`, `tok_per_s=11.760574277309576`,
  `generated_tokens=3`, `command_buffers=2`, `encoders=1063`,
  `forced_waits=2`.
- Engine profile report
  `/tmp/rvllm-e2b-engine-session-profile-qkv-rope-cache-family.json`:
  `sample_count=2`, median `tok_per_s=11.963415507189936`,
  `encoders_per_token=354.3333333333333`,
  `latency_ms_per_token=83.59885416666665`,
  `cpu_encode_ms_per_token=2.2533191666666665`, and
  `command_buffer_wait_ms_per_token=225.88188216666668`.

The Rust model-suite runner recorded
`/tmp/rvllm-gemma4-metal-model-suite-qkv-rope-cache-report.json` from manifest
`/tmp/rvllm-gemma4-metal-model-suite-qkv-rope-cache-manifest.json`. It passed
two local cached snapshots: E2B one-token HF matched `[236764]`, and 31B-it
bounded one-token execution produced `[9259]`. No other
`models--google--gemma-4*` snapshots were present in the local HF cache during
this pass; other sizes are not claimed.

The 31B-it bounded command was:

```bash
RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1 target/debug/rvllm_metal_infer \
  --model-dir /Users/george/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/419b2efe421994fdfd3394e621983d4cc511cd4f \
  --prompt Hello \
  --max-new-tokens 1 \
  --max-total-tokens 16 \
  --large-model-opt-in \
  --json > /tmp/rvllm-gemma4-31b-hello-step1-qkv-rope-cache-family.json
```

Observed result: generated token `[9259]`, text `"Hello"`,
`prepare_ms=44112.850833000004`, `prefill_ms=5031.555125`,
`decode_ms=3287.7048750000004`, `tok_per_s=0.304163554218047`,
`arena_bytes=61863697960`, `command_buffers=2`, `encoders=1325`, and
`forced_waits=2`. This is bounded execution only because no local 31B HF
reference artifact was available.

Public E2B diagnostic surfaces after the same fused path:

- `rvllm-eval`: generated `","`, `decode_ms=226.8`, `tok_per_sec=4.410`.
- `rvllm-bench`: schema `rvllm.apple_metal_bench.v1`,
  `tok_per_sec=1.8191145278124419`, `command_buffers=2`, `encoders=1063`.
- `rvllm-ppl`: `perplexity=504.1798937686529`, `tokens=2`,
  `command_buffers=6`, `encoders=2132`, `final_logits_encoders=6`.
- Prepared `rvllm-server --backend metal-direct`: generated token `[236764]`,
  `decode_ms=228.244917`, `command_buffers=2`, `encoders=1063`.

External profiling artifact:
`/tmp/rvllm-e2b-metal-qkv-rope-cache-fused.trace` (`96M`). Exporting
`metal-application-command-buffer-submissions` showed two submissions:
`495` encoders with `6.67 ms` encoder time and `7.50 ms` duration, then
`498` encoders with `2.92 ms` encoder time and `3.76 ms` duration. This is
bottleneck-localization evidence, not production performance.

## Generate One Reference Artifact

Reference artifacts are generated outside the repo:

```bash
"$RVLLM_HF_REF_PYTHON" scripts/dump_gemma4_hf_reference_logits.py \
  "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt-token-ids 2,4 \
  --decode-steps 1 \
  --selected-token-ids 0,1,2,3,4,5 \
  --top-k 16 \
  --output /tmp/gemma4-e2b-hf-reference-logits.json
```

Do not commit `/tmp` artifacts.

## Run Raw-Token CLI

```bash
cargo run -p rvllm-runtime --features apple --bin probe_apple_metal_decode -- \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt-token-ids 2,4 \
  --decode-steps 1 \
  --top-k 16 \
  --large-model-opt-in
```

The output includes sampled token IDs, per-step top-k logits, timing, arena
bytes, command-buffer and encoder counters, forced waits, debug sync state, and
the diagnostic-only claim.

## Run Text Diagnostic CLI

Text prompts are a diagnostic convenience around the same probe path. By
default the CLI tokenizes text with `tokenizer.json` from the model directory
and prepends BOS token ID `2`. Use `--no-bos` only when the prompt text already
accounts for the desired BOS handling.

```bash
cargo run -p rvllm-runtime --features apple --bin probe_apple_metal_decode -- \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt-text "Hello" \
  --decode-steps 1 \
  --top-k 16 \
  --large-model-opt-in \
  --decode-text
```

The report still includes `prompt_token_ids`. With `--decode-text`, text mode
also reports `tokenizer_json`, `prompt_text`, `sampled_text`, and `output_text`.
This is not a tokenizer/text production workflow or a quality claim.

## Run Text Inference CLI

For a production-facing text-in/text-out command shape, use
`rvllm_metal_infer`. This path does not read debug logits by default; it runs
prefill plus greedy decode, stops on EOS or `--max-new-tokens`, and reports the
current Metal counters. The public text/session path now uses configurable Metal
arena sizing rather than the old 64-token probe cap. Use `--max-total-tokens N`
or `RVLLM_METAL_MAX_TOTAL_TOKENS=N` for per-sequence context capacity; use
`RVLLM_METAL_MAX_BATCH_TOKENS=N` and `RVLLM_METAL_MAX_BATCH_SEQUENCES=N` when a
session or server run needs a larger prefill batch arena. Requests above the
model maximum or configured arena fail with explicit memory/context errors.

```bash
cargo run -p rvllm-runtime --features apple --bin rvllm_metal_infer -- \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt "Hello" \
  --max-new-tokens 1 \
  --max-total-tokens 128 \
  --json
```

The command reports schema `rvllm.apple_metal_text_infer.v1` and keeps an
explicit acceptance boundary in `claim`. If `--hf-reference <JSON>` is supplied,
it compares the tokenizer-derived prompt IDs, requested decode step count, and
generated token IDs with the existing HF artifact and reports
`hf_reference.matched`. Passing this command is workflow evidence. Production
readiness still depends on the accepted gate matrix for the target
model/context/surface and must not be inferred from one prompt.

Generate a text reference artifact outside the repo before running the
reference-backed text command:

```bash
"$RVLLM_HF_REF_PYTHON" scripts/dump_gemma4_hf_reference_logits.py \
  "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt-text "Hello" \
  --decode-steps 1 \
  --top-k 16 \
  --output /tmp/gemma4-e2b-hf-text-infer-hello-step1.json
```

Then run the text inference CLI against it:

```bash
cargo run -p rvllm-runtime --features apple --bin rvllm_metal_infer -- \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt "Hello" \
  --max-new-tokens 1 \
  --max-total-tokens 128 \
  --hf-reference /tmp/gemma4-e2b-hf-text-infer-hello-step1.json \
  --json
```

Do not promote tokenizer/text decoding evidence unless `hf_reference.matched`
is true. The current `"Hello"`, `"Once upon a time"`, and `"The capital of
France is"` one-step smokes plus the `"Hello"` two-step and four-step smokes
are positive checks after correcting Metal layer-scalar ordering, but they are
still narrow prompt/decode coverage, not broad tokenizer/text coverage.

The ignored `rvllm_metal_infer_e2b_engine_reference_backed_text_smoke` test
uses the same tokenizer prompt and HF artifact, but runs prefill/decode through
`Engine` and `ModelMetalBackend` scheduler handoff. It is bounded Engine-path
evidence only, not a serving, batching, throughput, or production claim.

## Generate Text Reference Suite Manifest

```bash
python3 scripts/dump_gemma4_e2b_hf_text_reference_suite.py \
  "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt "Hello" \
  --prompt "Once upon a time" \
  --prompt "The capital of France is" \
  --decode-steps 1 \
  --top-k 16 \
  --output-dir /tmp/rvllm-e2b-text-reference-suite
```

For a mixed-step suite, use repeated `--case` values. A case is either a prompt
or `PROMPT|STEPS`:

```bash
python3 scripts/dump_gemma4_e2b_hf_text_reference_suite.py \
  "$RVLLM_GEMMA4_MODEL_DIR" \
  --case "Hello|4" \
  --case "The capital of France is|1" \
  --top-k 16 \
  --output-dir /tmp/rvllm-e2b-text-reference-suite-mixed
```

Use `--dry-run` first to write the manifest and print the HF commands without
loading Transformers. The generated artifacts stay outside the repo.

## Run Rust Text Session Suite

```bash
cargo run -p rvllm-runtime --features apple --bin rvllm_metal_infer -- \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --reference-manifest /tmp/rvllm-e2b-text-reference-suite/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --session-backend direct \
  --max-total-tokens 128 \
  --case-timeout-seconds 900 \
  --report /tmp/rvllm-e2b-text-session-direct-report.json \
  --json
```

The manifest path is the same HF reference-suite manifest generated above.
`rvllm_metal_infer` validates the manifest schema and cases, rejects missing or
duplicate case names, rejects non-positive decode steps, fails clearly on
missing reference artifacts, prepares one `ModelMetalBackend`, runs all cases
sequentially through the prepared backend, writes schema
`rvllm.apple_metal_text_session.v1`, and fails unless every case reports
`hf_reference.matched: true`. `--case-timeout-seconds` is optional, but should
be used for slow hardware suite runs.

Run the Engine-backed path only after the direct session passes:

```bash
cargo run -p rvllm-runtime --features apple --bin rvllm_metal_infer -- \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --reference-manifest /tmp/rvllm-e2b-text-reference-suite/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --session-backend engine \
  --max-total-tokens 128 \
  --case-timeout-seconds 900 \
  --report /tmp/rvllm-e2b-text-session-engine-report.json \
  --json
```

The Engine session prepares one `Engine`/`ModelMetalBackend`, enqueues the
manifest cases, runs scheduler prefill/decode, and stops each request when the
CLI EOS policy or requested token limit fires. It is scheduler-path evidence,
not production serving or batching readiness.

## Run JSONL Session Mode

Use `--prompts-jsonl` when references are not already grouped in a manifest.
Each line requires `prompt` and may include `name`, `max_new_tokens`,
`max_total_tokens`, `hf_reference` or `reference_path`, and `no_bos`.

```json
{"name":"hello_step1","prompt":"Hello","max_new_tokens":1,"reference_path":"/tmp/gemma4-e2b-hf-text-infer-hello-step1.json"}
{"name":"capital_step1","prompt":"The capital of France is","max_new_tokens":1,"reference_path":"/tmp/rvllm-e2b-text-reference-suite/gemma4-e2b-hf-text-reference-the-capital-of-france-is-step1.json"}
```

Run JSONL direct session mode:

```bash
cargo run -p rvllm-runtime --features apple --bin rvllm_metal_infer -- \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompts-jsonl /tmp/rvllm-e2b-text-session.jsonl \
  --session-backend direct \
  --max-total-tokens 128 \
  --case-timeout-seconds 900 \
  --report /tmp/rvllm-e2b-text-session-direct-report.json \
  --json
```

The session schema keeps one top-level `prepare_ms`, aggregate timing/counters,
per-case generated tokens, decoded text, finish reason, and HF comparison.
The top-level `claim` remains explicit about the current acceptance boundary.

## Run Prepared Server Mode

For a serving-shaped workflow, `rvllm-server` supports an explicit
`metal-direct` backend. Unlike the compatibility `subprocess` backend, this
mode prepares one `ModelMetalBackend` before the server starts listening and
reuses it for completion requests.

```bash
cargo run -p rvllm-serve --features apple --bin rvllm-server -- \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --backend metal-direct \
  --addr 127.0.0.1:18081 \
  --max-new-tokens 1 \
  --max-total-tokens 128
```

Check readiness:

```bash
curl -sS http://127.0.0.1:18081/healthz
```

Run a completion:

```bash
curl -sS -X POST http://127.0.0.1:18081/v1/completions \
  -H 'content-type: application/json' \
  --data '{"prompt":"Hello","max_tokens":1}'
```

The response schema is `rvllm.openai_completions.v1`; the nested backend report
schema is `rvllm.apple_metal_server_completion.v1`. The backend report includes
the prepared `prepare_ms`, per-request prefill/decode timings, generated token
IDs, decoded text, arena bytes, command buffers, encoders, forced waits, and the
same claim boundary. This is prepared serving workflow evidence for the
configured arena; it is not concurrency, streaming, ANE, or broad production
performance evidence.

## Run Public Bench/Eval/PPL Surfaces

The `rvllm-bench`, `rvllm-eval`, and `rvllm-ppl` binaries can route to the
Metal text path with `RVLLM_BACKEND_PROFILE=apple` when built with
`--features apple`. These commands are useful for parity-surface diagnostics and
use the same configurable Metal arena environment as `rvllm_metal_infer`.

Run a one-token eval:

```bash
RVLLM_BACKEND_PROFILE=apple \
RVLLM_MODEL_DIR="$RVLLM_GEMMA4_MODEL_DIR" \
RVLLM_PROMPT='Hello' \
RVLLM_MAX_TOKENS=1 \
RVLLM_METAL_MAX_TOTAL_TOKENS=128 \
cargo run -p rvllm-bench --features apple --bin rvllm-eval
```

Run a one-iteration bench sample:

```bash
RVLLM_BACKEND_PROFILE=apple \
RVLLM_MODEL_DIR="$RVLLM_GEMMA4_MODEL_DIR" \
RVLLM_PROMPT='Hello' \
RVLLM_MAX_TOKENS=1 \
RVLLM_BATCH=1 \
RVLLM_ITERS=1 \
RVLLM_WARMUP=0 \
RVLLM_METAL_MAX_TOTAL_TOKENS=128 \
cargo run -p rvllm-bench --features apple --bin rvllm-bench
```

Run a tiny PPL sample:

```bash
RVLLM_BACKEND_PROFILE=apple \
RVLLM_MODEL_DIR="$RVLLM_GEMMA4_MODEL_DIR" \
RVLLM_PROMPT='Hello world' \
RVLLM_PPL_CHUNK=3 \
RVLLM_PPL_CHUNKS=1 \
RVLLM_METAL_MAX_TOTAL_TOKENS=128 \
cargo run -p rvllm-bench --features apple --bin rvllm-ppl
```

The public-surface claim emitted by these commands is
`Apple Metal bench/eval/ppl path; not production-ready or parity-complete`. A
successful run means the public command surface can execute the configured Metal
E2B path; it does not prove CUDA parity, broad long-context behavior, batching
performance, ANE execution, or broad production readiness.

Current E2B hardware observations after the range-read safetensor loader fix,
native Metal matmul/fusion path, shared scratch arena, and token-only final
sampling path:

- `rvllm-eval`: `prepare_ms=5539.1`, `prefill_ms=333.1`,
  `decode_ms=227.1`, `tok_per_sec=4.403`, generated `,`.
- `rvllm-bench`: schema `rvllm.apple_metal_bench.v1`,
  `prepare_ms=5318.452708`, `ms_per_step=559.057125`,
  `tok_per_sec=1.788725973217853`, `arena_bytes=9321447442`,
  `encoders=1203`.
- `rvllm-ppl`: `perplexity=504.3815019862137`, `tokens=2`,
  `elapsed_s=1.019207042`, `prepare_ms=5203.621333`.

Current four-token E2B text inference check against
`/private/tmp/gemma4-e2b-hf-text-infer-hello-steps4.json`:
`hf_reference.matched=true`, generated tokens `[236764,108,236777,735]`,
generated text `",\n\nI have"`, `prepare_ms=5327.267`,
`prefill_ms=333.035833`, `decode_ms=909.065`, `tok_per_s=4.400125403574002`,
`command_buffers=5`, `encoders=3012`, and `forced_waits=5`.

These numbers are operational diagnostics only. They remain far below a
production performance target.

## Profile Session Mode

Profile mode repeats the same session command and writes a summary artifact.
Samples must pass the same HF matching gate before they are included.

```bash
cargo run -p rvllm-runtime --features apple --bin rvllm_metal_infer -- \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --reference-manifest /tmp/rvllm-e2b-text-reference-suite/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --session-backend direct \
  --max-total-tokens 128 \
  --case-timeout-seconds 120 \
  --profile-samples 1 \
  --profile-report /tmp/rvllm-e2b-text-session-profile-direct-shared-scratch-token-only-summary.json \
  --json
```

The profile schema is `rvllm.apple_metal_text_session_profile.v1`. It reports
prepare, prefill, decode, tokens/sec, command buffers/token,
encoders/token, forced waits/token, and min/median/max summaries. It is a local
profile for the configured arena, not broad production performance. External
profiler evidence is recorded separately with `xctrace`.

Current direct profile sample on cached E2B:
`prepare_ms=5132.582208`, `prefill_ms=866.86025`,
`decode_ms=682.452416`, `tok_per_s=4.39591087915498`,
`command_buffers_per_token=2.0`, `encoders_per_token=1203.0`, and
`forced_waits_per_token=2.0`, with all three HF references matched.

Current Engine profile sample on cached E2B:
`prepare_ms=5156.567083999999`, `prefill_ms=430.395917`,
`decode_ms=249.685042`, `tok_per_s=12.015137054145196`,
`command_buffers_per_token=0.6666666666666666`,
`encoders_per_token=401.0`, and
`forced_waits_per_token=0.6666666666666666`, with all three HF references
matched.

Current Engine session sample against the same manifest:
`prepare_ms=5242.516375`, `prefill_ms=436.179209`,
`decode_ms=249.59758300000001`, `tok_per_s=12.019347158501931`,
`arena_bytes=9321447442`, `command_buffers=2`, `encoders=1203`,
`forced_waits=2`, with all three HF references matched.

External Metal profiling was captured with `xcrun xctrace record --template
'Metal System Trace'` because `xcrun --find metal-capture` was unavailable on
this host. The artifact `/tmp/rvllm-e2b-metal-token-only.trace` (`20M`) records
a one-token E2B `Hello` run on `M4 Max` with target exit status `0`. The target
stdout reported token `[236764]`, `prepare_ms=6101.478292000001`,
`prefill_ms=332.536`, `decode_ms=226.575083`, `tok_per_s=4.413547980472328`,
`arena_bytes=9321447442`, `command_buffers=2`, `encoders=1203`, and
`forced_waits=2`. Exporting `metal-application-command-buffer-submissions`
showed two submissions: `565` encoders with `7.03 ms` encoder time and
`7.96 ms` duration, then `568` encoders with `3.72 ms` encoder time and
`4.87 ms` duration. This localizes current overhead; it is not production
performance evidence.

The cached `google/gemma-4-31B-it` snapshot now passes a bounded one-token
decode, not only dry-run shape planning. The shared-scratch/fused-weight arena
estimate is `61863697960` bytes (`57.62 GiB`), and the actual decode report
used `61681657960` arena bytes. For prompt `Hello`, max total tokens `8`, and
one generated token, it produced token `[9259]`, generated text `"Hello"`,
`output_text="HelloHello"`, `prepare_ms=44562.049332999995`,
`prefill_ms=6318.7690410000005`, `decode_ms=3261.7925840000003`,
`tok_per_s=0.30657988644197615`, `command_buffers=2`, `encoders=1565`, and
`forced_waits=2`. This is larger-checkpoint operability only; no 31B HF
reference or performance gate has passed.

## Run Reference-Backed CLI

```bash
cargo run -p rvllm-runtime --features apple --bin probe_apple_metal_decode -- \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt-token-ids 2,4 \
  --decode-steps 1 \
  --top-k 16 \
  --large-model-opt-in \
  --hf-reference /tmp/gemma4-e2b-hf-reference-logits.json
```

For automation, request one JSON object:

```bash
cargo run -p rvllm-runtime --features apple --bin probe_apple_metal_decode -- \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt-token-ids 2,4 \
  --decode-steps 1 \
  --top-k 16 \
  --large-model-opt-in \
  --hf-reference /tmp/gemma4-e2b-hf-reference-logits.json \
  --json
```

`hf_reference.matched` reports whether selected logits, top-k entries, and the
sampled token match the supplied artifact under the CLI comparison tolerance.

## Generate Reference Suite Manifest

```bash
python3 scripts/dump_gemma4_e2b_hf_reference_suite.py \
  "$RVLLM_GEMMA4_MODEL_DIR" \
  --case 2,4 \
  --case 2,17 \
  --case 2,17,42,4 \
  --decode-steps 1 \
  --top-k 16 \
  --full-logits \
  --output-dir /tmp/rvllm-e2b-reference-suite
```

The manifest records the commands and artifact paths needed to regenerate the
suite.

## Run CLI Suite Runner

```bash
python3 tools/run_apple_metal_e2b_cli_suite.py \
  --manifest /tmp/rvllm-e2b-reference-suite/gemma4-e2b-hf-reference-suite-manifest.json \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --cargo-bin probe_apple_metal_decode
```

Use `--dry-run` first to print the cargo commands without running Metal:

```bash
python3 tools/run_apple_metal_e2b_cli_suite.py \
  --manifest /tmp/rvllm-e2b-reference-suite/gemma4-e2b-hf-reference-suite-manifest.json \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --cargo-bin probe_apple_metal_decode \
  --dry-run
```

The runner writes `/tmp/rvllm-e2b-cli-suite-report.json` by default.

## Interpret Output

The CLIs and Rust session runner are diagnostic evidence. A passing comparison
means the raw-token prompt and decode steps matched the supplied HF artifact for
the reported fields. Text diagnostic output means the CLI loaded
`tokenizer.json`, encoded the supplied text, and decoded the sampled/output
token IDs for inspection. The text inference CLI proves a text-in/text-out
command shape without debug-logit reads for the configured Metal arena. Session
mode proves only that one prepared backend can run many bounded prompts against
reference artifacts through the selected direct or Engine path. No profile
summary is production performance evidence. None of these paths
implies complete production serving, broad prompt coverage, long-context decode,
batching coverage, ANE execution, throughput readiness, or production
acceptance.

## Slow-Test Budget

Every real E2B Metal run can take minutes. Before each run, record:

```text
Hypothesis:
Exact command:
Artifact used:
Expected pass/fail signal:
What I will do if it fails:
```

Run one slow E2B command at a time. Do not run selected/full-vocab parity or
reference-backed suites as a general health check; run them for a concrete CLI,
reporting, or acceptance boundary.
