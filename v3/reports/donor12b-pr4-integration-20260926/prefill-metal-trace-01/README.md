# Bounded normal-route Metal System Trace diagnostic

Two serial queue jobs used Xcode 26.2 `Metal System Trace` around one real BF16
Gemma 4 12B 512-token prefill plus two generated tokens. Both completed with
queue-eligible sampled conditions, unchanged declared inputs, and identical
first two rvLLM IDs `[236770, 236770]`. The inference receipts reported
4.949 s and 4.505 s prefill respectively; these are **capture-perturbed**
observations, not comparison scores. The ordinary prefill ledger counted 48
scalar attention, 144 BF16 MMA GEMM, 48 FP32 QKV MMA, and 48 QKV postprocess
dispatches in each. Full queue jobs, conditions, stdout/stderr, and normal
route reports are preserved in `queue-results.tar.gz`.

The first capture launched the shell wrapper. Instruments targeted the shell,
so the `metal-gpu-intervals` export had no rows for the child inference process.
The second capture launched the inference executable directly with the same
one-case prompt fixture and route environment. It recorded 1,644
`metal-application-intervals` rows and 3,360 `metal-driver-intervals` rows,
but **zero `metal-gpu-intervals` rows**. Its table-of-contents says
`Shader Timeline: Disabled` and `Counter Set: (null)`. Application intervals
are CPU-side Metal activity, not per-kernel GPU execution time. Thus neither
capture resolves the projection/FFN/attention GPU-time breakdown; no numbers
were assigned to roles from those rows.

The complete `.trace` directories remain locally under this folder as
`capture.trace` and `capture-direct.trace`; their combined archive is
`traces.tar.gz` (SHA-256
`65b7134038eae53d3777c23b309d48fd9547072f68040650b0406c78dc9c2092`,
52 MiB). It is **not** committed because it has no GPU intervals and would add
large, low-value binary history. The queue receipts and this limitation are
committed. A future capture must explicitly enable a GPU/shader timeline or
use a supported per-encoder GPU timing route, then validate non-empty rows
with kernel labels before reporting per-role GPU time. Apple's [Metal capture
documentation](https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-programmatically)
describes the programmatic capture alternative.

SHA-256: `queue-results.tar.gz`
`c6ee70698840560c0a43fe0d3fca238ef416faff83f93e86950346f67ebcbe1d`;
shell-target normal report
`c4cf7cac2bea3017ae93454ca7df54f1fc9efc3a405d2ceee153f563c7194cb1`;
direct-target normal report
`6ad38ef74a32a621fd6cab9240c42be0559584c2e2883baa019b4b932d7d4bb3`.
