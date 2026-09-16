# Gemma 4 12B: stacked INT8 gate/up experiment

**2026-09-16 follow-up:** the explicit full-model check now passes 624
bit-identical FFN comparisons across all 48 layers on four successive
requests, with all 17 reference output IDs matching. This extends the
one-layer numerical boundary below; it does not establish a speedup or
promote the candidate. See [the full-model receipt](gemma4-stacked-int8-full-model-20260916.md).

This targets the largest measured decode component: 48 FFNs total about
74.82 ms per token in the characterized HTTP baseline. That component median
does not predict the speedup from changing one graph.

The candidate concatenates the existing gate/up integer rows and their FP16
per-output scales into one `[30720,3840,1,1]` constant projection. Channel
slices recover the two `[1,15360,1,1]` branches before the original tanh-GELU,
multiplication and down projection. The existing single-input/single-output
request ABI stays unchanged. No four-bit weights or activation quantization
are involved.

The source precedent and its limits are recorded in the
[FFN fusion research](gemma4-ffn-fusion-compression-research-20260914.md).
Stacking saves neither weight coefficients nor host submissions: the baseline
already executes the complete FFN in one request. Any benefit must come from
the compiler's internal layout or scheduling and must be measured.

## Host and small-device checks

- All 107 Apple host tests pass (25 explicitly ignored device tests and one
  separately gated device test excluded). The captured original FP16 Gemma
  MIL fixture remains byte-identical after the emitter refactor. The complete
  activation/down tail is shared without changing its operation order.
- The real layer-zero CPU audit decodes the candidate's serialized bytes and
  checks all 176,947,200 reconstructed coefficients against the established
  INT8 representation. Every FP16 bit matches. Reconstructed SHA-256 is
  `9ec1b72edfad27068add34669a960d971f6a52cf982db9170e451cafe158bfe5`.
- Original/reconstructed source hashes, three synthetic input comparisons and
  the saved capital-prompt FFN input exactly match the earlier CPU audit.
  This input has maximum absolute value 171.5.
- Candidate source size is 177,016,640 bytes versus 177,016,768 bytes for the
  baseline. The 128-byte reduction is two fewer blob descriptors, not reduced
  weight traffic or measured device residency.
- A 64→128→64 ANE smoke test uses two compiles and six evaluations, with
  bit-identical baseline/candidate outputs and maximum CPU-reference error
  `5.72304e-6`. Both programs unload successfully; no caller staging remains
  and the boot is unchanged.

Evidence is in `gemma4-12b-evidence-20260914/int8-stacked-ffn/`, including frozen
source, signed executables, CPU results, journals and cleanup receipts.

## Full-size qualification boundary

The first full-size attempt establishes an absent baseline cache entry and
stops before any compilation, load or evaluation. Staging cleanup succeeds
and the boot remains unchanged. It does not exercise the candidate graph.
Restore only the original INT8 layer-zero graph with at most one compile and
zero evaluations, then retry the three-program comparison under the existing
two-compile limit for candidate/control. Keep cache repair separate from
qualification and strict-cache timing. No production FFN route changes here.

The missing baseline identifier is identical to the earlier successful HTTP
decoder's cache-hit identifier; this is not a new identity introduced by the
emitter refactor. The isolated restoration now completes with one compile and
zero evaluations. The subsequent full-size qualification loads the baseline
and dense reconstruction from cache, compiles only the stacked candidate,
and completes twelve evaluations on the three synthetic inputs and actual
capital-prompt input. All candidate outputs are bit-identical to the current
INT8 FFN. Both INT8 forms pass the same CPU-reference tolerance; their relative
L2 execution error ranges 0.0515–0.0862%. Three successful unloads leave no
caller staging and boot identity is unchanged.

A fresh strict-cache timing process compiles nothing and repeats the numerical
checks, then rejects Fair thermal state before warmup. It generates zero timing
trials. No component speedup or full-model benefit is established. The original
INT8 FFN remains the decoder default.

Across journaled smoke, restoration and qualification there are four compiler
calls, eighteen evaluations and six successful unloads. The isolated initial
cache-miss attempt has zero compiles, loads and evaluations. The final host gate
again passes 107 tests. Invalid restoration flags reject before model loading
or journal creation. These checks do not establish long-term daemon cache
persistence.

Use `replay-v2.json` for the immutable signed executable and common arguments.
The first frozen executable lacks the one-graph, zero-evaluation restoration
mode. Timing appends `--compare true --cache require --report NEW_PATH` and
must omit the driver journal. Qualification explicitly supplies its journal,
`--cache reuse --ane-compile-budget 2`; the baseline always requires an
existing cache entry. The restoration path is separate:
`--mode int8 --restore-int8-cache true --cache reuse --ane-compile-budget 1`
with explicit model/report paths and a driver journal, without a layout or
comparison flag.

Eligible timing must include the same input/output transfer path, use the
unchanged INT8 FFN as baseline, preserve raw paired distributions and baseline
drift, and obey the existing power/thermal comparison gate. CPU cycles cannot
normalize ANE time. The [device-timing audit](gemma4-ane-device-timing-research-20260915.md)
finds no verified alternative counter recipe for this M4/macOS target.
