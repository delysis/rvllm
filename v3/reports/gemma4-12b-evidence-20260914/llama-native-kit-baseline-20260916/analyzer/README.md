# Offline paired-result analyzer

This independent safe Rust tool reads saved receipts only. It depends on `serde_json`; there is no model loader, accelerator library, subprocess, power query, network access or production-crate dependency.

Build/test commands (reuse the existing Cargo target, one standalone package):

```sh
CARGO_TARGET_DIR=/Users/george/Documents/llama-native-kit/target cargo test --offline --locked --release -j 2 --manifest-path /Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/llama-native-kit-baseline-20260916/analyzer/Cargo.toml
CARGO_TARGET_DIR=/Users/george/Documents/llama-native-kit/target cargo build --offline --locked --release -j 2 --manifest-path /Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/llama-native-kit-baseline-20260916/analyzer/Cargo.toml
```

Usage, in actual execution order:

```sh
/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/llama-native-kit-baseline-20260916/rvllm-native-kit-paired-analyzer-v2 DIR_A DIR_B > NEW_ANALYSIS.json
# Or a predeclared ABBA quartet:
/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/llama-native-kit-baseline-20260916/rvllm-native-kit-paired-analyzer-v2 DIR_ANE1 DIR_NATIVE1 DIR_NATIVE2 DIR_ANE2 > NEW_ANALYSIS.json
```

Each distinct directory/job identity must contain the queue's `job.json`, `report.json`, `conditions.jsonl`, and exactly one backend receipt: `backend/report.json` for ANE or `backend.json` for native-kit. Reusing one receipt cannot stand in for another process trial. Inputs are bounded to 64 MiB each. The analyzer emits schema `rvllm.paired_native_ane_analysis.v2` on stdout; a refused comparison has `status: "refused"`, a reason, null ratios, and exit status 1. Successful analysis is still exploratory. Metric-specific drift failures suppress only their associated ratios.

Checks are deliberately strict:

- Queue success, timing purpose, exit zero, sampled-condition eligibility true, no violations/overdue/file drift or validator failure.
- One actual power/processor stratum across a fully covered process interval: age/gaps at most 2500 ms, ordered samples, matching endpoints, no observer journal error. Nominal requires the original raw nominal eligibility flag; Fair requires its separate explicit stable-Fair policy and retains its raw nominal-ineligible flag. The two strata cannot be pooled.
- The actual stratum must match the manifest. All saved condition probes must be ready, have the same controls, meet the disk floor, and show no competing processes. Every configured idle llama server must match PID/port and be idle. Paired runs must have identical sorted idle-server/quiet-process policies. This validates saved sampling, not fixed clocks or absence of every possible competitor.
- Native: complete schema, exactly two marked warmups plus seven measured requests, actual/reference counts 84/10/9, internal counts, valid EOG completion, per-eval timing consistency and repeated exact IDs. The expected Google QAT identity and current native-kit source pins must match.
- ANE: nine serial retained-backend cases, first two discarded under this predeclared policy; 162 programs, context 1024, require-existing cache, zero permitted/used compiles, no ANE diagnostic journal, no layer capture and no decode fallback. Only standard-checkpoint INT8 FFN plans are accepted. Every case has 84 prompt IDs, 10 output IDs, nine steps, and correct position/input/next-token chaining.
- All prompt **and output** IDs must be identical within and across paths. Otherwise matched-work ratios are refused, even though the checkpoint differences might explain the output difference.

The report gives seven-request **process medians** for prefill, ANE import, prefill plus import, decode, and each request's phase sum. Nine actual decode steps determine steps/s. Native internal elapsed timers are preserved separately. For ABBA, each backend's process medians are summarized equally; requests are not treated as independent process repetitions. Job command/input identities must remain constant within each backend.

**Predeclared v2 policy, before timing measurements:** the quartet must have ANE in both outer positions. The primary gate is `abs(ANE2.decode_median - ANE1.decode_median) / ANE1.decode_median <= 0.05`. Decode-ratio eligibility depends only on that decode drift after all shared work/condition checks pass. Prefill-plus-import and phase-sum ratios each use the same <=5% formula on their own outer ANE process medians. An unstable prefill cannot invalidate a stable decode ratio. `per_metric_eligibility` records each observed drift, pass/fail and ratio eligibility; ineligible ratio fields are null individually. `primary_gate` records the predeclared decode decision. `inconclusive_primary_decode_drift` means the primary claim failed, although independently stable secondary ratios remain visible. `exploratory_abba_partial_metric_eligibility` means decode passed while a secondary metric failed. A two-process pair stays exploratory: it can report ratios, but its drift gates are explicitly unassessed/null, never marked passed.

Ratio direction is explicit: native latency divided by ANE-path latency. These are different configurations: **standard BF16-derived INT8 FFNs / other FP16 ANE weights versus Google's native-QAT Q4_0 projections / Q6_K embedding/head**. They cannot establish a hardware-only or quantization-only speedup. ANE prefill includes first-token sampling and KV export; its import is added. ANE decode sums `step.total_ms`, excluding outer streaming/report overhead. Native prefill ends at synchronized logits before first sampling; native decode includes subsequent sampling and loop overhead. Neither phase sum is a complete end-to-end request timer. No device cycles or clock normalization are inferred.

Source mapping: queue acceptance/observations are in `v3/crates/rvllm-runtime/src/bin/rvllm_experiment_queue/queue.rs`; the 2500 ms coverage policy follows `rvllm_compare_disaggregated/fair.rs` and `apple_measurement.rs`. ANE fields follow `rvllm_disaggregated_infer.rs`; native fields follow the adjacent `bench/src/main.rs`. Older preparation/capture results are intentionally rejected. Tests use explicit synthetic values, not claimed hardware results.

The original v1 executable and `../analyzer-build-receipt.json` remain preserved, with SHA256 `1158096441fb95f13e81585350604627d15e9ce281821afc65d424901ae98298`; its source is archived as `policy-v1-main.rs.txt`. No actual paired timing result has been analyzed yet. ABBA order is supplied by the caller and must agree with the queue campaign; this result schema does not supply an absolute cross-process start timestamp from which the analyzer could independently establish chronological order.

V2 receipt: `../analyzer-v2-build-receipt.json`. All ten offline release tests passed, including isolated prefill drift preserving the stable decode ratio and an unbracketed pair leaving drift unassessed. Frozen v2 SHA256: `97270984ecaa8e47a41c89c47a511a9bb68b7b90466fd191146d4d47145178e8` (640,080 bytes). The original executable and archived source were hash-checked against the v1 receipt. No accelerator work was performed during this revision.
