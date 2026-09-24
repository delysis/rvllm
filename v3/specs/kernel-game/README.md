# Kernel game

This directory specifies the evidence-gated kernel evaluation loop implemented by
`rvllm-runtime::kernel_game` and the existing experiment queue.

The design has one rule: performance is considered only after immutable identity,
correctness, exact work, and environment checks pass. A candidate never trades
correctness for speed, and the referee never silently retries or edits an attempted
run.

## Implemented vertical slice

- `kernel_game::SealedSubmission` binds task, candidate/control, source tree,
  executable, generated MSL, metallib, model, reference, workload, oracle, and
  required dispatch counts.
- strict JSON parsing rejects duplicate keys before serde can overwrite them.
- `RouteEvidence::validate_against` requires exact artifact identities,
  successful reference continuation, zero compiler calls, exact positive dispatch
  counts, and no foreign candidate dispatches.
- `TimingPlan::abba` and `score_timing` enforce the declared order, one
  environment stratum, zero compiler calls, the drift ceiling, and a practical
  improvement threshold.
- `reduce_result` keeps blocked/deferred/inconclusive distinct and cannot return
  `Promotable` without independent confirmation.
- `rvllm_experiment_queue` optionally binds a sealed submission pin. Bound jobs
  fail closed unless the queued executable and required input artifact hashes agree
  with the sealed submission.
- normal `rvllm_disaggregated_infer` case receipts now embed
  `rvllm.kernel_game.route_evidence.v1`: executable/metallib/generated-MSL
  hashes, reference/workload identity, per-request compiler-call delta, actual
  prefill dispatch receipt, output counts, and completion.

This PR does **not** install a daemon, change a production selector, clear STOP,
prepare caches, run private accelerator work, or promote any candidate.

## Promotion boundary

A later reviewed action may promote only an independently confirmed result. The
referee itself never changes production defaults.

The first intended native vertical slice is
`metal-mma32-prefetch` versus `off`, using the already established component
oracle and reference route. A/A sanity precedes A/B timing.
