# Bounded BF16 prefill GPU capture diagnostic

The existing Xcode Metal System Trace packet recorded CPU-side Metal intervals
but no GPU/shader intervals. This packet adds an **opt-in** programmatic Metal
capture around one completed prefill. The normal build has no capture branch;
the diagnostic build requires `metal-gpu-capture` and the explicit absolute
`RVLLM_METAL_PREFILL_GPU_TRACE` output path. The capture is stopped before
decode and the diagnostic now fails if no nonempty trace appears.

Three serial queue jobs are retained, including two diagnostic mistakes:

| Job | Outcome | Meaning |
| --- | --- | --- |
| `g4-donor-bf16-prefill-gpu-capture-512-01` | Queue succeeded, no trace | Capture was wired into the single-prompt path, but this job used the JSONL session path. No capture ran. |
| `g4-donor-bf16-prefill-gpu-capture-device-512-02` | Queue succeeded, no trace | Device-scoped revision still missed the session path. No capture ran. |
| `g4-donor-bf16-prefill-gpu-capture-session-512-03` | Queue succeeded, trace present | Corrected one-case session path captured the device during prefill only. |

All three jobs used the same real Gemma 4 12B BF16 model, exact 512-token
fixture, SG8 decode selector, and MMA32 prefill selector; all had queue-eligible
sampled conditions, zero reported inference-phase library compiles, and the
same two generated IDs `[236770, 236770]`. The actual capture arm reported
11,489.101 ms prefill versus approximately 4.9 s in the noncapturing routing
mistakes. This is **capture perturbation**, not a kernel slowdown estimate.

The replayable `prefill-session.gputrace` is approximately 3.8 GiB and remains
local at this directory. Its `metadata` file hashes to
`6068816aabbb3363c6bad6cc6f5a11cd59563d4ad046815219a072784826b53c`.
It is not added to Git history. Xcode 26.2's `xctrace export` rejects this
document with `Document Missing Template Error`; this is a GPU debugger
document, not an Instruments trace. The current command-line path therefore
does **not** yet provide labeled per-kernel GPU times. Do not infer role times
from the capture's existence or its perturbed wall time. Analyze it in a
supported GPU debugger, or obtain a separate bounded operator timing capture.

The source and exact job manifests are reviewable here. Complete small queue
receipts, including the two no-capture routing mistakes, are preserved in
`queue-results.tar.gz` (SHA-256
`7b38eb3ce5fe1d02cd0b32178ba17af03cd5cd6dbe1355618bd9da23a297b8d0`).
The successful inference report hashes to
`3d73fdc55a22b45a8b3654f585f0d42d88a6cea3d5e727d8fa37ea26ad4e18dd`.
This is diagnostic evidence only: no kernel promotion, quality claim, or
matched-MLX speed ratio follows from it.
