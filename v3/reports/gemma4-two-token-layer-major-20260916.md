# Layer-major two-token target reference

The new diagnostic schedule carries two token-major activation states through
each of the 48 layers. It runs both QKV projections, then causal attention in
token order, both output projections and both FFNs before advancing to the next
layer. Vocabulary tiles likewise process both states together. Every device
call still uses an existing S1 single-input/output request; this neither loads
new S2 graphs nor implements a drafter.

This establishes the full-model scheduling boundary at which a qualified S2
projection can later replace two adjacent S1 calls. The existing serial decoder
remains the numerical oracle. Separate scratch preserves both token residuals;
global raw K is copied to V before normalization/RoPE; all existing FP16 norm,
residual, scale and softcap boundaries are retained. The first attention call
completes before the draft's K/V is appended, so the anchor cannot attend to it.

The route shares the existing unwrapped 1024-capacity transaction preflight,
exclusive pending guard, rollback and fail-stop state. It requires journaling
and zero compiler calls. Its diagnostic token timing fields are explicitly
unavailable (zero) and must not be interpreted as performance. No production
route selects it.

Host tests check paired shape validation before projection and stop-after-error
behavior. A bounded ignored live fixture reuses the hash-checked 84-token Metal
prefill and ordinary INT8 decoder. It compares all 96 layer outputs (368,640
FP16 values), both token/top-five signatures, accepted continuation, rejected
draft replacement, one-output stopping, unresolved drop, early/late observer
failure and complete reimport recovery. All 162 cached models must be unloaded;
the driver journal is audited separately. Job 77 builds/runs host tests through
the existing queue; device qualification remains pending at staging time.

The independent S2 FFN pilot jobs 05/06 now precede the longer baseline chains
when their AC/battery Fair controls are ready. Jobs 74/75 remain planned
post-comparison repetitions. A short observation retained in
`int8-s2-fair-timing-20260916/pilot-wait-observations.jsonl` caught transient
llama-server and compiler processes, explaining quiet-window resets that
isolated process snapshots missed. No competing process or OS power setting
was changed.

## Host gate and device submission

Job 77 completed in release mode: five host tests passed and both hardware
fixtures remained ignored. Source pins were unchanged and there was no overrun.
The frozen test executable has SHA-256
`51345c991caab7bf939f800496bcae07a67cbecd633276469ad9275753a0c574`,
with the existing verified client signing identifier. Its test inventory
includes the exact intended fixture. Job 78 now submits that one ignored test,
with a durable driver journal and the original snapshot/config pins. This is
preparation, never a performance result.

An additional upstream cross-check supports retaining the explicit cost gate:
[little-gemma's pinned journal](https://github.com/cortexist/little-gemma/blob/2948f875bd3545ae00a81cccb23e73f80d612265/docs/performance-journal.md#L833-L849)
reports that its Jetson Orin NX Gemma 4 12B assistant costs about 33 ms against
a 121 ms ordinary decode, and can still lose at near-perfect acceptance. That
CUDA/quantized configuration is not this M4/ANE system. It demonstrates why
acceptance alone cannot justify integration: measure `T2 + draft_cost` against
`(1 + acceptance_probability) * T1`. The source snapshot here has SHA-256
`a3dbe004e91849beff9099abc8604100b12ee7fe02a0c733cc6b8592c3d9d772`.

## Live result

Job 78 passed the exact ignored fixture in 170.61 seconds. All 368,640 FP16
values across 96 layer outputs matched serial decoding bit for bit; token and
top-five signatures, accepted continuation, rejected replacement, one-output
stopping, unresolved-drop poisoning, observer failures at callbacks 1/2/95 and
reimport recovery passed. The independent journal count is 162 cache hits,
162 loads, 208 requests, 3,728 completed evaluations and 162 completed unload
returns, with matching per-model load/unload identities and no failed stages,
cache misses or compiler calls. `compile_requested` denotes wrapper entry,
not an actual compilation; all 162 entries resolved to cache hits.

Input pins remained unchanged and no runtime bound was exceeded. Seven sampled
transient llama-server observations made queue timing eligibility false; this
was a preparation job and makes no timing claim. The worker returned to waiting
for the independent timing jobs.

SHA-256 receipts under `baseline-isolated-v7/results/78-two-token-layer-major-qualification`:

- `driver.jsonl`: `14aa2837c63a3dc764b25d2b8ec533e06e5e60e996e51f34d780d29e453795bd`
- `transaction.jsonl`: `5143abca06f9b5101db180756494b52eb2b6232067264c026a0275ae42407cee`
- `report.json`: `39e31bf647094d1af789fb526dfb13dc0226fa59ccaac7b4fcb5bffe1fe54228`
