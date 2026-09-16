# Restoring the existing two-token INT8 FFN graph

Job 62's missing descriptor is exactly the descriptor compiled and qualified
in job 30: the MIL component starts `BD9452F7F77F142DA4491C75CC5264542`, and
the weight component starts `4AB3820D3B2F45692B3D45E2310524D1`. All three full
components match. This is recovery of the same graph, not a new shape, weight
format or private selector. The single-input/output restriction remains.

The existing `--restore-int8-cache true` mode now accepts `--int8-batch 2`.
It requires INT8, `--cache reuse`, compile budget exactly one and a driver
journal; CPU diagnosis, capture, comparison, stacked and palette modes remain
incompatible. The restoration branch directly constructs and drops the one
selected graph, with zero evaluations. Its report adds the logical token
count and exact affine-coefficient hash. The existing S1 behavior is retained.

The native timing worker was stopped and reaped while no timing job had
started. Its manifests and pending comparisons remain under
`baseline-isolated-v7`; no measurement was interrupted or replayed. The
`batch-recovery-v8` worker owns the same hardware lock for jobs 63/64: focused
probe host tests, then the binary build. Source pins remain fixed during those
jobs. Device restoration will use a frozen, signed and hashed executable only
after the build and parser checks complete. Resume the untouched native worker
after this bounded diagnostic campaign.

Evidence is under
`gemma4-12b-evidence-20260914/int8-s2-restoration-20260916/` and the v8 queue.
## Completed restoration and capture

Jobs 63/64 passed nine probe host tests and built the release executable with
unchanged source pins. Six parser checks accepted the intended S2 restoration
flags and refused missing journaling, comparison, stacked, S1-batch and zero
budget combinations before model/device access. The frozen executable is
`rvllm_ane_int8_probe-s2-restore`, SHA-256
`341966b7ad6d732d659e9548dae0b3a62a45b1f29017b307696a6815deb7ae53`.

Job 65 restored exactly the known missing S2 descriptor with one compilation,
one successful load/request/unload and zero evaluations. Its affine coefficient
hash is `2078a4984159884e9a0e7fb035d26293e0f961240c0ce44310e0aaa8e51613e9`.
The report SHA-256 is
`67411dc8c404c3074f0f296795b832bf091030ee4614d26b494f65712def13b6`.

Job 66 then used the unchanged frozen capture binary with strict cache loading
and compile budget zero. Nine S1 and nineteen S2 evaluations completed, with
two cache hits, matching successful unloads, no compiler calls and no failed
driver events. All 145,920 candidate FP16 values match their serial references
bit for bit. Lane swaps, zero isolation and repeated use pass the independent
raw-array audit in `66-independent-output-audit.json`.

The old CPU gate remains false: one serial failure and six batch occurrences
all reference the same source-7 coordinate 3627 with identical S1/S2 output.
No tolerance or gate was changed. This establishes component serial equivalence,
not a CPU-gate pass, complete model batching, draft acceptance or acceleration.
The complete capture report SHA-256 is
`b7bcafbe8476dd51ed17d6e006f6419a32c0b1246aeadc77ad174a9014a83ec3`.

The v8 worker was stopped and reaped with exit zero. The native comparison
worker resumed in `baseline-isolated-v7` using the same exclusive hardware lock;
its old STOP marker is preserved as `STOP-before-s2-diagnostics`. Jobs 70/71
add nominal AC/battery S2 timing after each corresponding native comparison.
They pin both the original four-source CPU qualification and the broader
serial-equivalence receipt. The timing binary and original qualified input
subset are unchanged. Fair timing is not supported by that probe and is not
submitted. No timing result or accepted native-kit ratio exists at resumption.

## Disk headroom

The existing safe Rust helper acquired the current Cargo target's profile locks
and removed only regular `.rlib`, `.rmeta` and `.o` artifacts older than 48 hours.
It removed 49,681 files with 9,587,551,200 logical bytes. Available blocks rose
from 16,660,864 to 22,933,940 KiB, clearing the capture's 16 GiB launch floor.
The before/after receipts and removal inventory are retained here. Concurrent
writes and APFS prevent exact physical attribution; sources, executables,
model assets and ANE cache were preserved. This is not evidence of the cause
of the private ANE cache miss.
