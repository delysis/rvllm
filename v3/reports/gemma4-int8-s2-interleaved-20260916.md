# Interleaved FFN timing staging

The [predeclared protocol](gemma4-int8-s2-interleaved-protocol-20260916.md)
retains the original pilot's total useful work while shortening the separation
between S1 and S2. The original job 04 remains inconclusive under its drift
gate; the original 120-second-lead pilots remain unchanged.

Seventeen probe host tests pass, including the three new checks for exact
counterbalanced work, corrupt/missing observations, and separate occurrence
and temporal drift. Queue jobs 79/80 completed tests and release build with
unchanged input pins and no overrun. Six forbidden parser combinations reject
before device setup. Valid flags reach the missing-model error after the
required report argument is supplied; the missing-report prerequisite was
also checked. Formatting and diff whitespace checks pass.

The frozen signed executable is
`int8-s2-interleaved-20260916/rvllm_ane_int8_probe-interleaved`, SHA-256
`cf1bb39ec914a3749badd9276479ad96f1dd86be9bef45d8784446fc6a11d63e`.
Its verified signing identifier matches the qualified cache owner. The frozen
build receipt is SHA-256
`837855844c0197f684d48c2d134c682b7447685bc939eac6f20914e3df6b8a31`.

The same live v7 queue now contains independent jobs
`07-ac-fair-ffn-s2-interleaved` and `08-battery-fair-ffn-s2-interleaved`.
They require their respective power/low-power/mode controls, exactly Fair
thermal state, 30 seconds of fresh quiet observations, the existing 16 GiB
disk floor and no competing compiler or inference process. Both use strict
cached programs with zero compilation and no driver journal. The CPU-qualified
input subset, prior broad batch qualification, frozen build and protocol are
pinned. There is no automatic retry or accepted speedup at staging time.

The queue's three raw-file JSON readers were independently changed to buffered
reads. Eleven queue tests pass; the prior worker exited cleanly before the
new worker acquired the same lock and resumed the same queue. Status-read
observations improved from 3.64 to 0.30 seconds with identical job summaries.
This concerns host bookkeeping and is not an inference benchmark. See the
[queue report](gemma4-experiment-queue-20260916.md).
