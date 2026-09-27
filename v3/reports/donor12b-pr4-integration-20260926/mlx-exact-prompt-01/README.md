# Exact-token BF16 MLX / rvLLM comparison at 512 + 64

This packet narrows the workload mismatch in the older random-token MLX
baseline. MLX was fed the **same 512 token IDs** recorded by the real-weight
rvLLM Gemma 4 12B case, and both routes generated 64 tokens. The MLX-LM source
and BF16 checkpoint configuration/index, rvLLM executable/metallib, scripts,
manifests, queue observations, and normal-route reports are sealed in the
manifests and `queue-results.tar.gz`. The MLX checkpoint is a separately
packaged conversion; tensor-byte identity with the rvLLM checkpoint is **not**
established. Generated IDs agree at index 0 (`236770`) and differ at index 1
(MLX `236771`, rvLLM `236770`), so this is a speed comparison at matched input
shape, not numerical or output-trajectory equivalence.

| Arm | 512-token prompt phase | 64-token generation | Queue conditions |
| --- | ---: | ---: | --- |
| MLX-LM BF16, exact token IDs, three trials | 546.029 prompt tok/s; implied 937.747 ms mean | 18.979 tok/s mean | all processes exited zero, but observer freshness made sampled conditions ineligible |
| rvLLM BF16 SG8 decode + MMA-only prefill, one trial | 4,820.616 ms | 13.346 tok/s | succeeded; conditions eligible; zero inference-phase compiles |

The orientation is about **5.14× faster MLX prompt phase** and **1.42× faster
MLX generation** in this nearby, serial run. It is *not* a qualification ratio:
the MLX condition sample failed the queue's freshness check (not a thermal or
competing-process gate), the checkpoint bytes and continuations differ, the
sample counts differ, and the timing boundaries differ. MLX's `prompt_tps`
comes from its first generated token after processing the prefix (the reported
milliseconds are derived as `512 / prompt_tps`); rvLLM's phase boundary is
`before_prefill_launch_to_after_prefill_collect`. Likewise the frameworks'
generation-rate denominators are not proven identical. Both recorded AC,
power-mode 2, thermal-state 0 in available host samples; these readings do
not normalize GPU clocks or eliminate changing conditions. No stability wait
was used.

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

Next comparisons should bracket rvLLM and MLX arms under the same workload,
collect repeated samples without requiring thermal stability, and compare
model tensor identities or explicitly keep the checkpoint difference. True
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
