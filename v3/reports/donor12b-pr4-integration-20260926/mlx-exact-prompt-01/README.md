# Exact-token BF16 MLX / rvLLM comparison at 512 + 64

This packet narrows the workload mismatch in the older random-token MLX
baseline. MLX was fed the **same 512 token IDs** recorded by the real-weight
rvLLM Gemma 4 12B case, and both routes generated 64 tokens. The MLX-LM source
and BF16 checkpoint configuration/index, rvLLM executable/metallib, scripts,
manifests, queue observations, and normal-route reports are sealed in the
manifests and `queue-results.tar.gz`. Subsequent source and tensor-byte
inspection established that the MLX package is a conversion of the **base**
`google/gemma-4-12B` checkpoint while rvLLM used `google/gemma-4-12B-it`;
see [CHECKPOINT_IDENTITY.md](CHECKPOINT_IDENTITY.md). Generated IDs agree at index 0 (`236770`) and differ at index 1
(MLX `236771`, rvLLM `236770`), so this is a speed comparison at matched input
shape, not numerical or output-trajectory equivalence. Dense BF16 execution
should depend mainly on shape and layout, not the checkpoint's weight values;
the mismatch primarily limits correctness claims, not the usefulness of the
performance signal.

| Arm | 512-token prompt phase | 64-token generation | Queue conditions |
| --- | ---: | ---: | --- |
| MLX-LM BF16, exact token IDs, three trials | 546.029 prompt tok/s; implied 937.747 ms mean | 18.979 tok/s mean | all processes exited zero, but observer freshness made sampled conditions ineligible |
| rvLLM BF16 SG8 decode + MMA-only prefill, one trial | 4,820.616 ms | 13.346 tok/s | succeeded; conditions eligible; zero inference-phase compiles |

The orientation is about **5.14× faster MLX prompt phase** and **1.42× faster
MLX generation** in this nearby, serial run. It is *not* a qualification ratio:
the MLX condition sample failed the queue's freshness check (not a thermal or
competing-process gate), the continuations differ, the
sample counts differ, and the timing boundaries differ. MLX's `prompt_tps`
comes from its first generated token after processing the prefix (the reported
milliseconds are derived as `512 / prompt_tps`); rvLLM's phase boundary is
`before_prefill_launch_to_after_prefill_collect`. Likewise the frameworks'
generation-rate denominators are not proven identical. Both recorded AC,
power-mode 2, thermal-state 0 in available host samples; these readings do
not normalize GPU clocks or eliminate changing conditions. No stability wait
was used.

## Bracketed continuation

The queue then ran one more rvLLM arm followed by one more MLX arm, yielding
the serial order MLX–rvLLM–rvLLM–MLX without replaying either completed first
arm. The second rvLLM run measured 4,973.052 ms prefill and 13.399 decode
tok/s. The last MLX run's three trials averaged 540.856 prompt tok/s
(946.700 ms derived first-token phase) and 18.469 generation tok/s. Both new
jobs passed the queue's sampled-condition check and preserved their complete
receipts in `queue-results-abba-tail.tar.gz`; the first MLX arm remains
freshness-ineligible and is not silently discarded.

Across the two arms per framework, the descriptive means are 4,896.834 ms
rvLLM prefill versus 942.223 ms MLX first-token phase, and 13.373 versus
18.724 generation tok/s. That is roughly **5.20× MLX prompt-phase** and
**1.40× MLX generation** orientation. The low within-pair movement is useful,
but the boundary/checkpoint/output/first-condition caveats above still prevent
strict ABBA qualification or a model-quality claim. The fresh last MLX arm
alone is also substantially faster than the old random-prompt baseline;
measurement design and prompt distribution must be examined before assigning
the cause.

The two-token MLX control used the same exact IDs and averaged 547.583 prompt
tok/s, consistent with the 64-token run's prompt phase. Its two-token decode
rate is not a sustained-decode estimate. The older, seven-trial *random-token*
MLX 512/64 benchmark averaged 173.051 prompt tok/s and 5.378 generation
tok/s, far below these exact-token observations. Prompt distribution,
conditions, and measurement sequence are all confounded; the present packet
does **not** isolate which caused the difference. It does invalidate treating
that older result as a settled MLX performance baseline.

The first queued two-token exact-prompt job is preserved as a quarantined
failure: MLX completed measurement, but our reporter tried to get `__file__`
from a function and exited before writing its normal report. The corrected
job used a new ID and source hash; no failed receipt was overwritten. Both
corrected MLX jobs and the rvLLM job are retained in the archive, including
conditions and stdout/stderr.

A later attempt to run MLX-LM directly on the **same original 12B-it
checkpoint** also completed generation trials but failed at receipt writing:
the reporter assumed a sharded safetensors index, while this checkpoint is
single-file. That new failure is retained separately in
`mlx-it-bf16-512-g64-failed-queue-results.tar.gz` (see
`CHECKPOINT_IDENTITY.md`). A repaired reporter validates the weight layout
before timing and uses a new immutable `-03` job/report identity; its result
is recorded below. Do not interpret the failed attempt's process wall time as model
throughput.

## Same-checkpoint 12B-it MLX-LM result

`g4-donor-mlx-it-exact-bf16-512-g64-03` succeeded with exit code zero and
three measured 512+64 trials after one warmup. It loaded the **exact same
original `google/gemma-4-12B-it` model safetensor** pinned in the rvLLM
manifest (`5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d`)
and the exact 512 prompt IDs sealed by SHA-256
`44ee0b95324385ceb5e5823c62d89f62d830f392755ae2665206ba8d6c02aee5`.
All three MLX trials generated identical IDs.

| Same-checkpoint MLX-LM 512+64 trial | Prompt tok/s | Derived prompt phase, ms | Generation tok/s |
| --- | ---: | ---: | ---: |
| 1 | 539.349 | 949.293 | 18.136 |
| 2 | 542.904 | 943.076 | 18.294 |
| 3 | 533.952 | 958.887 | 18.419 |
| Mean | 538.735 | 950.419 | 18.283 |

The earlier two same-prompt rvLLM SG8-decode/MMA-prefill arms averaged
4,896.834 ms prefill and 13.373 decode tok/s. Thus the same-checkpoint
descriptive comparison is approximately **5.15× shorter MLX prompt phase**
and **1.37× higher MLX decode throughput**. This agrees in scale with the
earlier base-conversion MLX orientation (5.20×/1.40×); changing checkpoint
weights did not explain away the performance gap. It is still **not a strict
paired ratio**: the MLX prompt metric derives from its first-token timing,
the rvLLM metric stops after prefill collection, decode denominators are not
proved identical, and this MLX arm ran later rather than interleaved with the
rvLLM arms. Queue sampled conditions were marked ineligible solely because
one power-observer sample was stale (observed age 3.030 s); the preserved
violation recorded AC power, mode 2, thermal state 0, and no competing
process. There was no thermal-stability wait.

The first generated-ID difference between same-checkpoint MLX-LM and the
saved rvLLM SG8/MMA route is at **zero-based index 4**: MLX chose `236761`
while rvLLM chose `236770`. Matching weights and prompt IDs does not prove
either implementation is wrong; internal-tensor/logit/reference evidence is
needed. The old base-conversion run differed at index 1, showing why its
output trajectory was not a suitable correctness control.

The full queue receipt archive SHA-256 is
`6c2056127c600362c17eefafbea2a26f51b99036127431bd6b60d0cb90310793`;
the normal report SHA-256 is
`e79d2caa702d5745ba27f23aa19baf2ba15285ce1b772c5b77917c95ddddf845`.

Next comparisons should use the **same 12B-it checkpoint** for both frameworks,
bracket arms under the same workload, and collect repeated samples without
requiring thermal stability. True
per-role GPU timing remains missing. The opt-in MMA prefill switch must not be
promoted until route-preserving internal-tensor and checkpoint-quality gates
resolve the continuation divergence.

SHA-256: `queue-results.tar.gz`
`57ce2e52d3b16fddf96398728ef77e8ff19b2afe1138767bc926ca411647759d`;
two-token MLX report
`63bf639278e70a1f5aea7e4416612320b890319903c808867d1bcc5e69fe0a9d`;
64-token MLX report
`b0b1e292cb15d97c8ccf62f8d3bfc6270d67045047e4c298630701a422d94627`;
64-token rvLLM report
`689a5874991fa61bbf81bc8a26623443f70f81397197a60a61ad52860097a0d2`.

Bracket-tail SHA-256: `queue-results-abba-tail.tar.gz`
`cde3509fc41376d11e741446811a64894d7565a2b4e3039d4731a0cef2269c89`;
rvLLM repeat report
`02ead4dd4267c6ecc9f7fe04d01f7d9cd0db44ba20e8e193c7be2c12c5f77a2a`;
MLX repeat report
`e71caecf3f96fefa9a159295e1a4ed97785c117fcd4019ad3003aedd61451667`.
