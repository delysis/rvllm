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
- `RouteEvidence::validate_against` requires exact source-tree, executable,
  metallib, generated-source, model, reference, workload, and oracle identities,
  successful reference continuation, zero compiler calls, exact positive dispatch
  counts, and no foreign candidate dispatches.
- `TimingPlan::abba` and `score_timing` enforce the declared order, one
  environment stratum, zero compiler calls, the drift ceiling, and a practical
  improvement threshold.
- `reduce_result` keeps blocked/deferred/inconclusive distinct and cannot return
  `Promotable` without a sealed confirmation naming two distinct evidence hashes.
- `rvllm_experiment_queue` optionally binds a sealed submission pin. Bound jobs
  fail closed unless the queued executable, candidate selector, command arguments,
  and required input artifact hashes agree with the sealed submission.
- normal `rvllm_disaggregated_infer` case receipts contain a non-referee route
  observation. They embed typed `rvllm.kernel_game.evidence.v1` only when invoked
  with a sealed submission plus its pinned source-tree and oracle files; the typed
  receipt is validated against the submission before it is written.

This PR does **not** install a daemon, change a production selector, clear STOP,
prepare caches, run private accelerator work, or promote any candidate.

## Promotion boundary

A later reviewed action may promote only an independently confirmed result. The
referee itself never changes production defaults.

The first intended native vertical slice is
`metal-mma32-prefetch` versus `off`, using the already established component
oracle and reference route. A/A sanity precedes A/B timing.
