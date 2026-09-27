# Original-checkpoint W4/W8 full text-projection preflight

Source-only audit, 2026-09-27. No new package or device job was built or run.
The existing authenticated W4/W8 packages and teacher-forced results replace
only layer 0's down projection; see `LOWBIT_MMLU_PROTOCOL.md` and
`LOWBIT_MMLU_RESULTS.md`. They do not establish full text-route quality.

The dedicated Apple package exporter accepts repeated `--low-bit-proj`
selectors with group-32 W4A16 or W8A16. Its global `--weight-format w4/w8`
path is not the quantizing exporter. It emits authenticated value and scale
sidecars, checks tensor role/shape and duplicate selectors, copies the source
checkpoint into a new staging package, validates, then renames to the final
path. The BF16 donor route admits the low-bit sidecars used by the existing
one-layer experiment. This is implementation capability, not evidence that
all sidecars dispatch or preserve quality.

Safetensors metadata for the pinned original 12B-it checkpoint contains 328
supported dense language-model projection matrices: q=48, k=48, v=40, o=48,
gate=48, up=48, down=48. Their aggregate is 10,899,947,520 elements, with
all reduction dimensions divisible by 32 and no expert matrices. Four other
two-dimensional tensors (text embedding, audio embedding projection, vision
embedding projection and vision patch dense) are outside this proposed text
projection set. Thus even a 328-selector package is **not** a wholly W4/W8
checkpoint or a multimodal qualification.

At group size 32, estimated sidecar storage for those matrices is 5.710 GiB
for W4 or 10.786 GiB for W8, including FP16 scales, before manifests and
alignment. The pinned source `model.safetensors` resolves to 23,919,549,408
bytes (22.28 GiB). Since the exporter copies it into each package, building
both packages would consume roughly 61 GiB of new disk, before staging,
temporary memory pressure and logs. Available disk was 129 GiB on the data
volume at audit time. The three existing local packages are one-layer only;
none is a full-projection package. These are estimates, not measured output
sizes or a reservation.

Before building, seal the exact 328-selector list, source/config/tokenizer and
donor-library hashes, package paths, output budget and failure receipts. Build
and authenticate one package at a time, retaining an adequate free-space
margin and leaving every existing package and untracked artifact untouched.
Then predeclare distinct correctness inputs and a route/dispatch and
all-position numerical gate against the same BF16 donor route. The current
teacher CLI exposes named low-bit decode dispatch but does not separately
attribute generic low-bit prefill kernels; do not infer prefill selection or
speed from it. No performance or promotion decision follows from this
preflight.
