# Default-off post-layer route observation

This is a diagnostic producer, not a numerical verdict or a speed result. The
older `RVLLM_METAL_DEBUG_TRACE_LAYER` mode supplies intermediate snapshots to
`metal_encode_forward_layer`; that changes optimized kernel selection. The new
test-only `RVLLM_METAL_DEBUG_ROUTE_TRACE_LAYER` mode supplies **no** trace
scratch to the layer encoder. It reads already-existing Q/K/V, attention,
gate/activation, residual, and physical KV-cache buffers after the selected
layer completes. Each summary now includes SHA-256 of its complete little-endian
u16 bit stream, plus selected values and finite statistics. The capture uses a
per-layer command-buffer synchronization, so it changes scheduling and cannot
be used for timing. A normal-route dispatch ledger comparison is still required
before asserting that a particular real-model capture used the intended donor
kernel; this tiny test does not prove that.

Set `RVLLM_METAL_DEBUG_ROUTE_TRACE_LAYER` to a comma-separated list of layer
indices, `RVLLM_METAL_DEBUG_ROUTE_TRACE_POSITION` to the exact first position
of the decode step, and `RVLLM_METAL_DEBUG_TRACE_JSON` to a path containing
`{layer}` for multi-layer capture. Route and detailed trace cannot select the
same layer. A route capture fails rather than overwriting an existing file.
These controls are test-only and inactive in production builds. An invalid or
absent position/path is an error when route capture is requested.

The serial queue ran two immutable Metal smoke jobs. Both exited 0 and had
eligible sampled conditions, without a thermal-stability gate:

| Job | Scope | Result |
| --- | --- | --- |
| `g4-donor-route-trace-tiny-01` | One-layer F16 fixture; trace schema/digest and overwrite refusal | Passed 1/1 |
| `g4-donor-route-trace-tiny-equivalence-01` | Same fixture plus traced/untraced token and exact residual comparison | Passed 1/1 |

The second test compared the complete 128-element residual after one decode
layer, not just the sampled token. It still does not exercise Gemma 4 12B BF16,
SG8 selection, 512/1024-token KV behavior, continuation divergence, or
first-internal-difference localization. Those need a separately queued
same-prompt on/off capture at the first divergent decode position, followed by
full layer-by-layer digest comparison and an independent numerical reference.
This packet does not promote SG8 or the prefill MMA route.

`queue-results.tar.gz` retains the manifests, condition samples, reports, and
stdout/stderr. SHA-256:
`3c80eddc557caba3d3ac50e1dcfc859daaaaeaf733ac43a55678a67688143f4e`.
The producer/test source hashes are
`56fdbb64c4ab837858bc1a6d2027881931006b1ec0f0b6e42ed90d6c707bfea1`
and
`a8db4793d756c692dfeb1e3b702520f7687070527cabcf8b77d348560a5adee9`.
