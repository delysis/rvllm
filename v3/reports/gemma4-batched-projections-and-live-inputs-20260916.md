# Batched projection interfaces and live FFN inputs

The two-token INT8 FFN component has passed its first device correctness gate.
The next source changes supply interfaces needed to extend that result to a
whole decode block and improve its input coverage. They do not enable a new
default or establish a speculative decoding speedup.

## Linear projections

`AneLinear` now retains logical width separately from the padded physical
stride. `project_batch` accepts token-major inputs and outputs for 1–8 logical
columns through one external input/output pair. It validates both lengths
before any device work, clears old input columns/padding, and reads every
logical output column. Calling the existing lane-zero `project` on a wider
graph clears other lanes, including values from a previous batch.

The FFN and linear batch paths share the checked column packing/readback
helpers. Single-token MIL, coefficient blobs and request geometry are
unchanged. Eight selected host tests pass; seven private-API hardware tests
remain explicitly ignored. Coverage includes independent physical-column
readback, stale padding, bad shapes/overflow, bounded FFN width, and captured
single-token MIL fixtures. No batched QKV/O/head graph has been compiled or
evaluated by this change.

## Captured inputs

The CLI adds `--capture-ffn-inputs true`. It writes the actual normalized input
passed to each ANE FFN for the first two decode steps of each request, with
layer, absolute token position and SHA-256 in the step receipt. It requires an
output directory and is rejected in cache-maintenance and runtime-worker modes.
An immutable observer sees the vector immediately before the FFN projection;
observer failure leaves the decoder invalid until a complete prefill import.
Ordinary inference supplies a no-op observer and retains its arithmetic.

This is diagnostic execution. Capturing files and the optional driver journal
disqualify it as a performance trial. The initial bounded run will use the
existing 21/84/652-token references, strict zero-compilation cache loading and
the ordinary INT8 FFN plan. Expected output is 15 reference token IDs, 12 ANE
continuation evaluations and 240 FFN input vectors: one captured step for the
short EOS case and two each for the other requests. Successful capture should
then feed a zero-compilation batch qualification using the frozen, already
device-tested FFN probe.

The capture run has now succeeded. All three reference sequences match (15
output IDs, 12 continuation steps). The journal records 162 cache-hit loads,
2,496 completed evaluations, 162 unloads and zero compilations. All 240 files
were independently checked against their recorded SHA-256 and are exactly
7,680 bytes (3,840 FP16 elements). The five captured layer-0 states are all
distinct. Queue job `33-batch-live-inputs` reuses the already qualified frozen
FFN probe with those five vectors, strict existing caches, zero compilation
and a driver journal. It runs correctness checks only.

Capture executable SHA-256:
`498d842ceffb17dced94ff4887ab4284bbfe777bb9c9a220a503c8b9708220b4`.
The complete source/build/hash packet is under
`gemma4-12b-evidence-20260914/ffn-live-inputs-20260916/`; device receipts are in
`experiment-queue-20260916/native-comparison-quiet/results/32-live-ffn-inputs/`.
Concurrent activity and a transition to Fair thermals make its timing
ineligible, as expected for a journaled capture run.

## Broader component check stopped at the serial reference

Job `33-batch-live-inputs` failed its unchanged CPU tolerance on the fifth live
input (case 2, absolute position 653, layer 0), output index 3627. The ordinary
S=1 INT8 program returned -0.55371094 versus CPU -0.52122331. Both graphs were
cache hits and unloaded normally; eight serial evaluations completed and
**zero S=2 evaluations occurred**. The queue stopped without retry. Thus this
failure supplies no new batch correctness evidence and is not a driver failure.

Source review identified a real reference difference: the old CPU GELU keeps
most intermediates/constants in FP32, while MIL declares FP16 after each node.
An offline diagnostic retained that old reference and added explicit FP16-node
emulation with FP32 and FP64 dot accumulation. Both explicit variants give
-0.52246094 at the failing coordinate, so that boundary change alone does not
explain the ANE result. No tolerance or qualification gate has been changed.
The full diagnostic vectors and frozen source are under
`gemma4-12b-evidence-20260914/ffn-oracle-diagnostic-20260916/`.

The existing unstarted native-kit baseline jobs were moved to `baseline-only-v4`
with their original frozen backend binaries and conditions. Failed component
evidence remains in the old queue; it is not replayed. The timing analyzer now
also refuses FFN-input capture metadata or captured step payloads; all ten host
tests pass. No accepted slowdown or batch speedup has been established.

The [detailed discrepancy record](gemma4-ffn-oracle-discrepancy-20260916.md)
adds a fourth offline control: retaining INT8-times-scale coefficients at
higher precision moves the failing output to -0.55029297. This is a lead for
fused dequantization diagnosis, not a proved hardware mechanism or a waived
qualification failure.
