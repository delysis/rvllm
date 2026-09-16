# Stacked INT8 FFN: full Gemma 4 12B qualification

The stacked gate/up candidate targets the largest measured ANE decode
component. Its one-layer source/backend qualification is recorded in
[the original experiment](gemma4-stacked-int8-ffn-experiment-20260915.md).
This extends checking to real activations throughout all 48 decoder layers.
The default CLI, runtime worker and HTTP plan remain unchanged.

## Explicit research plans

`static-int8-stacked-ffn-checked` retains the original and stacked FFN in each
layer. It quantizes each layer once using the existing per-output INT8
quantizer, then builds both programs from that same packed weight object.
Every FFN call executes both with the same input. The checker requires finite,
bit-identical FP16 outputs, including signed zero, and then continues the
candidate output through the decoder. A mismatch terminates the run; there is
no baseline fallback. Successful comparisons are counted per layer.

The checked route has 210 programs and requires strict cache loading, zero
compile budget, a driver journal, reference inputs and an output directory.
It performs duplicate work and is explicitly excluded from performance
qualification. `static-int8-stacked-ffn-cached` is the candidate-only plan with
162 programs for subsequent ordinary execution and matched timing. Both plans
retain original QKV/output/head weights, attention, CPU pointwise operations,
and Metal prefill. No four-bit work or new private API is involved.

Two focused host tests pass: the checker rejects a one-bit change, signed-zero
change, nonfinite values and wrong lengths; the library rejects a nonzero
compile budget before checkpoint/device access. A frozen release CLI rejects
an invalid checked invocation before reading its nonexistent checkpoint.

## Frozen identity and preparation

Evidence directory:
`gemma4-12b-evidence-20260914/int8-stacked-full-model-20260916/`.
Binary SHA-256:
`aa015ca92c735e36c1da17d258031a4c743823a4b29feacb5ba46cbbc9f3c605`.
Signing identifier: `rvllm_disaggregated_infer-9d1f7275eb5937c9`.
Model: Google Gemma 4 12B IT,
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, capacity 1024.
The frozen binary, source changes, test/build logs and replay manifest are
included. The Metal library identity is unchanged and recorded in the manifest.

`--prepare-ane-cache ffn-int8-stacked` prepared all 48 graphs in a fresh process,
bounded to 48 compiler calls. It reused layer zero and compiled 47 graphs.
All 48 loaded programs successfully unloaded, with zero evaluations and no
failed journal events. The normal `all-int8` cache batch still prepares the
original FFNs. Disk free space was approximately 12 GiB after this preparation;
no build caches or model assets were removed in this experiment.

## Full-model check result

The checked process loaded all 210 programs from cache and made zero compiler
calls. Four successive requests used prompt lengths 21, 84, 652 and 21 tokens.
All 17 output IDs matched the independent references, including the repeated
short request following long recall. Across 13 ANE decode steps, each of the
48 layers completed 13 bit-identical FFN comparisons: 624 comparisons,
covering 2,396,160 output elements. The candidate output continued through
the model in every case. Metal performed prefill only.

The journal contains 3,328 completed evaluations and 210 successful unload
returns, with no cache miss, explicit compile, failed event or unmatched
per-model lifecycle. No owned staging directory remained after exit. See
`checked/report.json`, `checked-lifecycle-audit.json`,
`checked-driver-phases.jsonl` and `staging-remains-after-checked.txt`.

This establishes bit parity on these actual decode activations, not every
possible input or task. It does not establish a throughput improvement.

## Candidate-only execution

The same frozen executable then ran `static-int8-stacked-ffn-cached` on the
84-token copy reference. All ten output IDs matched across nine ANE steps.
The process loaded 162 programs from cache, made zero compiler calls, and
completed exactly 1,872 evaluations (208 per step), confirming that the
additional baseline FFN calls were absent. Its comparison-count field is
null, as expected for the candidate-only plan. All 162 unload returns
completed with no failed or unmatched lifecycle events. No source staging
remained and the boot identity stayed `1789488066` / `242846`.

Receipts: `candidate/report.json`, `candidate-driver-phases.jsonl`,
`candidate-lifecycle-audit.json`, `staging-remains-final.txt`, `boot-after.txt`.
The final power check still reports AC/low-power/Fair and is timing-ineligible.
The candidate remains experimental pending matched, unjournaled timing and
an end-to-end performance comparison. Neither the checked nor candidate-only
driver-journal run supplies such timing evidence.

Power preflight reported AC power, low-power mode enabled and Fair thermal
state. Timing comparisons are ineligible. Preparation and checked execution
qualify construction, numerical behavior and lifecycle, not a speedup.
