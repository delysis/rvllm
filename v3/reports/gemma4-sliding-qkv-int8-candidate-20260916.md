# Sliding QKV INT8 candidate

This explicitly selected full-model candidate changes forty sliding QKV
projections from FP16 constants to the already qualified per-output INT8
encoding. It retains ordinary INT8 FFNs, eight FP16 global packed QKV
projections, FP16 output/head matrices, and the existing single-I/O request
geometry. It uses 162 programs, the same count as the baseline. No model or
service default changes.

Earlier matched-power component trials in
[the projection report](gemma4-int8-linear-projections-20260915.md) motivate
the selection: sliding median baseline/candidate latency ratios were
1.130, 1.112 and 1.186, while global packed INT8 regressed in later runs.
Those observations do not predict a whole-model speedup or establish model
quality. In particular, this candidate changes coefficients and cannot use the
stacked-FFN candidate's bit-identical qualification exception.

Both cache preparation and model loading call the same QKV constructor.
The INT8 branch rejects the 8704-row global geometry before private API access;
the FP16 branch calls the original constructor without changing its MIL or
weights. Host checks verify global exclusion, unchanged baseline selection,
INT8 FFN/cache policy, program count and rejection of invalid weights before
device access. Two focused host tests and the CLI `cargo check` pass. These
checks initialize no accelerator. The first test invocation caught an overly
specific expected error string; the invalid weights were already rejected,
and the assertion now checks the actual quantizer validation message.

The CLI option is `--ane-weights static-int8-ffn-sliding-qkv-cached`; the
separate cache part is `qkv-sliding-int8`. Existing `all-int8` preparation
continues to target the default model. No new cache provisioning, ANE
evaluation, full-model quality qualification or speed claim has occurred for
this candidate. The current native-kit comparison uses its original frozen
baseline binary and is unaffected by source edits.

Next acceptance sequence: freeze one candidate executable; separately prepare
only bounded confirmed misses; require strict zero-compile generation against
the existing 21/84/652/21-token references with layer captures; inspect any
divergence; then stage paired baseline/candidate timing with the experiment
queue. Preserve all failures and reject promotion if quality or matched-power
performance is unresolved. Do not provision this candidate concurrently with
the baseline campaign or enable the quarantined multi-I/O path.
