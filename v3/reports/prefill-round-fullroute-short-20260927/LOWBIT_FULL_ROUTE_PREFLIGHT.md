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

Before building, use the sealed 328-selector list and
source/config/tokenizer/donor-library hashes below, and fix package paths,
output budget and failure receipts. Build
and authenticate one package at a time, retaining an adequate free-space
margin and leaving every existing package and untracked artifact untouched.
Then predeclare distinct correctness inputs and a route/dispatch and
all-position numerical gate against the same BF16 donor route. The current
teacher CLI exposes named low-bit decode dispatch but does not separately
attribute generic low-bit prefill kernels; do not infer prefill selection or
speed from it. No performance or promotion decision follows from this
preflight.

The separate safe-Rust `rvllm_gemma4_lowbit_full_selectors` generator now
seals the first part of that plan. It checks the full model SHA-256
`5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d`,
config `478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9`,
tokenizer `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`,
and donor metallib `21d81a66d31075e9307b64b7cb8eb0eea1f1dee74c0865ff3d0fa8125579dd06`.
It reads only the safetensors header after verifying those files and requires
the exact seven role counts, 328 matrices, BF16/group-32 shapes and element
total above. Its create-new output
`lowbit-full-text-selectors-sealed-v1.json` has SHA-256
`087e5e87f8ef2fd68e6d910af17867ec2cf12ef9344e9b70cf481a6992591d7e`.
Both W4 and W8 CLI selectors are listed for every matrix. Two focused unit
tests and a real pinned-file generation passed; this is still not a built
or authenticated full package.

The existing layer-0 W4 package was revalidated by a separately frozen
`rvllm_apple_package` executable, SHA-256
`c9cc445b139d747548f94c8d6ec141f7059b7111aafd9c5562d4678c3509ae16`.
Validation returned package identity
`d9004e911c7da95f61496a89b21d0d87290ab77833b2a86d957611fcda52a9f1`.
Its `metal/` subtree has all six required macOS/iOS/iOS-simulator ×
F16/BF16 library/manifest pairs; the macOS BF16 library matches the donor
SHA above. The runtime's portable low-bit planner permits a down-only sidecar
or a complete projection set per layer, and treats shared K as V in layers
5, 11, 17, 23, 29, 35, 41 and 47. The sealed plan supplies the complete
set per layer. This is source compatibility, not runtime admission evidence.

A separate safe-Rust one-package runner
`rvllm_gemma4_lowbit_full_w4_build` is frozen at executable SHA-256
`a0c387ed45a20e2fc91eb2b502ba99e39cf6c8acf752b0b91673284482b992d1`.
Its source SHA-256 is
`6f3a77139f471f972b2a35c5cde02d2f330dffc5c10e09185910d3707a44ee16`.
Its `--verify-only` pass checked the pinned plan, builder, model, config,
tokenizer, donor library and unused output path without writing a package.
The intended output is
`v3/target/gemma4-12b-it-fulltext-w4-v1.rvllm` in this PR8 worktree.
The builder source SHA-256 is
`88335d2407689818689f02025a2d849e08b4af378eb1230695fdd4bee19f4306`.
The serial queue caps runs at 3,600 seconds; a future immutable build job
must preserve a failure receipt if the export exceeds that bound. No such
job has yet been submitted, and no full package exists.
