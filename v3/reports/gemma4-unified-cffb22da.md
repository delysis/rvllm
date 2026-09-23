# Gemma 4 unified candidate integration — completed source packet

**Exact base:** `cffb22dabcb57b5f3ecee05acd4ab810d1ce6077`
**Repository/branch:** `delysis/rvllm:codex/gemma4-kernel-candidates`
**Task date:** 2026-09-22; UTC execution receipts extend into 2026-09-23.
**Target:** M4 Max, macOS 15.6 (24G84), Metal 3.1, Metal prefill and single-I/O ANE decode.

## Delivered scope and evidence boundary

This packet completes the previously unfinished unified source delivery. Apply
its entire five-patch series on cff, not on the original faa snapshots. The base
already contains the first wave and the staged second-wave shaders, including
broken references to modules that were not committed. The series reconciles the
actual native preimages instead of replacing them with an older complete file.

It retains the original twelve runtime candidates, connects five distinct
additional candidates, removes the duplicate staged Down4 implementation, and
keeps packed32 source-only. There are eleven Metal selections including `off`,
seventeen append-only Metal entry-point slots, and one twenty-two-arm BF16/FP16
source/compiler matrix. Every performance selection remains explicit/default-off.

**This is not hardware or performance acceptance.** No Cargo/rustfmt/Metal/MIL
compilation, model load, ANE initialization, cache operation, ignored native
fixture, live queue operation, timing, commit, push or promotion was performed.
The current container has no Rust or Apple compiler. Tests executed here exercise
Python tools, CPU/address models, and mocked orchestration. Native Rust code and
MSL remain proposals awaiting the owner's actual compiler and numerical gates.

The branch was freshly observed at cff. Its historical Actions run 35790534598
passed Python checks then failed the first Rust compilation with E0583: missing
`research_wave2.rs`. The related failure archive is preserved. The earlier
`next_tests::` typo was already fixed by cff and is not claimed as this packet's
repair. The local first-wave 45-test/fourteen-library receipt applies to its older
source head, not to cff or this proposed final tree.

## Architecture: one contract for source, selection, dispatch and evidence

`research_catalog.rs` binds stable names, shader sources, entry points, token
bounds, numerical contracts and source-level resource budgets. The safe
`research_projection.rs` creates checked launch descriptions for projections and
post-projection norms. It checks the incumbent full-model/prefill proof, native
BF16, alpha=1/beta=0, supported output roles, checked byte sizes, write aliases
and vector alignment. Unsupported shapes keep the established fallback.

The existing `PipelineCache` separately checks the actual GPU family, pipeline
SIMD width, thread capacity and static/shared allocation. A requested entry must
also agree with its typed source budget; a product name is not a capability
proof. The layer forward boundary consumes the checked launch plan for shader
selection, dispatch geometry and counter recording. No new allocator, device
owner or command queue is introduced. Trace-mode fallbacks remain explicit.

The original ten dispatch slots retain their values; seven are appended.
`selection_exercised` retains its weaker any-entry meaning. The new
`complete_family_exercised` requires every entry of the selected family, rejects
foreign entries, counter reset and overflow, and is required by prefill-only
screens. Receipts retain the distinction between requested selection, encoding,
collection and numerical qualification. **Complete family is not complete layer
coverage.** Normal inference defaults remain unchanged. Prefill-only reference
batches are bounded to 1–16 and preserve receipts before reporting disagreement.

The host exporter has a no-device `--catalog` mode. A reviewed JSON golden is
compared against that Rust export and drives both CI source checks and the native
compiler matrix. A catalog edit cannot silently select a runtime candidate.

## Independent candidates and evaluation priority

Order is investigation value, not predicted speed. No measured winner is chosen.

| Candidate | Changed variable / proper control | Narrow scope and source cost | Qualification requirement |
|---|---|---|---|
| `metal-mma32-f32` | FP32 matrix operands at fixed MMA32 geometry / native-BF16 MMA32 | M6–1024, dense 12B roles, 128 threads, 12,288 shared bytes, two entries | Operand lowering may change arithmetic; original component/tensor gates |
| `metal-mma32-load4` | Four-element cooperative loads / same MMA32 contractions | M6–1024, A/B eight-byte alignment, exact supported N/K, 128 threads, 8,192 shared bytes | Additional bit-exact FP32 output gate; tails/alignment/guard checks |
| `metal-long-mma32x64` | Existing isolated 32×64 tile integrated into runtime / MMA32 | M64–1024, 128 threads, 14,336 shared bytes, two entries | Runtime source shared with component test; no duplicate tile experiment |
| `ane-int8-ffn-interleaved` | Paired gate/up channel order / existing stacked INT8 FFN | H3840/I15360, two conv nodes, one evaluation; 48 replacement FFNs | Raw coefficients/scales preserved; private compiler and actual-input oracle |
| `metal-rmsnorm-simd256` | Hierarchical SIMD reduction / existing post-projection RMS tree | M6–1024/H3840/epsilon1e-6, 256 threads, 32 logical shared bytes | Explicit reduction-order change; **direct isolated native adapter still missing** |

The already-integrated Down4 remains a high-priority decode candidate, not a new
addition: original gate/up and GELU, four complete-K down output-row partitions,
six convolutions, one evaluation. Only the redundant staged implementation is
removed. Its historical proposals and attempted evidence are untouched.

The interleaved graph has the same INT8 coefficients and FP16 scales, permuted
together and streamed directly into one blob. Logical reshape/slices recover gate
and up before the inherited FP16 GELU sequence. Its source blob is 177,016,640
bytes; each ordinary width-one padded external vector remains 245,760 bytes.
These are logical/source sizes, **not measured compiled residency, compression,
physical bandwidth or process-peak memory**. Other projections remain FP16 unless
an existing separately selected INT8-QKV experiment is requested.

Every candidate remains limited to the inspected 12B geometry. Larger models are
not routed through coincidentally matching matrix sizes. The packed32 declaration
still has no runtime consumer or smaller guessed device allocation; compiled
surface descriptor evidence remains missing.

## Correctness work accompanying the integration

The matrix component harness compiles actual runtime shaders and uses the same
intrinsic tile geometry. It retains independent FP32 controls, original relative
L2 and sampled FP64 limits, BF16 output-rounding checks and canaries. Short-MMA
cases explicitly exercise 6/63/64 rows rather than clamping a long prompt.
Prefetch and vector-load fixtures additionally require bit-exact FP32 output.
New selected fixtures are numerical-only; historical long/BK64 fixtures require
the documented qualification-only flag. All remain ignored and were not run.

Three safe ignored ANE fixtures compare chunk4/plain, Down4/plain and
interleaved/stacked on one real layer and 1–3 separately captured activations.
They pin configuration, ordered quantized coefficients/scales and sample bytes
before creating owners. Both programs use RequireExisting; no compile fallback,
timing or retry exists. At most six evaluations occur; outputs and events are
create-only and synced before comparison failure. Driver lifecycle must still be
verified independently. A separate ignored host-only preparer computes these pins
with the existing loader/quantizer and never initializes ANE.

Generated-source tests now share a bounded MIL interpreter instead of permissive
lookalike implementations. It resolves actual convolution arguments, constants,
reshape/slice bounds, concat axis/interleave, declared dtypes and explicit FP16
materializations. Descriptor/header bounds, signed INT8 payload dimensions and
positive finite scales are validated. Mutation tests attack wrong operations,
arguments, return names, slices, dtypes and blobs. This is a host graph model, not
an ANE arithmetic simulator; the Rust tests themselves remain unexecuted here.

The actual offline tensor auditor is integrated into recurring Python checks. It
compares every F16/BF16/F32 scalar, hashes the bytes read, rejects nonfinite values,
requires an externally pinned unchanged oracle policy, and reports worst indices,
coordinates and values. Unknown fields, duplicate JSON, missing policy tensors,
malformed/oversized input and symlink/FIFO leaves fail. Failed CLI input is retained
in a create-only result. Eighteen executed tests cover the real offline tool.
Hash matching does not establish independent reference or driver provenance.

Three staged matrix shaders additionally refuse surprising uniform threadgroup
sizes, out-of-grid groups and bounded-dimension violations before entering any
barrier. Vector transactions retain their strict aligned N/K checks. Valid-path
arithmetic and storage boundaries are unchanged. This source-level review does
not prove native compiler lowering or resource usage.

## Tests and delivery verification

The final installed Python inventory has **108 tests: 107 passed, one skipped**
because real rustfmt is unavailable. Six disjoint batches cover the exact full
inventory once; their IDs, commands and outputs are preserved. Earlier red tests,
an interrupted aggregate run and failed construction attempts are retained.
This 108-test inventory supersedes the earlier unfinished report's 115-test
claim: missing supporting files were recovered/reconstructed and the delivered
inventory was rerun, not inferred from the previous count. The base's 63 checks
remain, with integration assertions updated to the shared catalog/plan.

Nine separate packet-verifier mutation tests pass. They test checksum changes,
missing/extra files, ambiguous metadata, symlinks/special files, exact preimages,
read-only checkout checks, create-only replay and addition/modification/deletion
forward/reverse handling. The final external verification receipt records actual
series replay and archive read-back; its result is not a compiler receipt.

There are **60 reviewed public Rust tests, 22 source exports, twelve native
host-test filters, 32 reviewed Rust formatting paths and 22 Metal 3.1 compile/link
arms** in the installed gates. These counts describe intended gates, **not checks
executed by this researcher**. CI does not run private/ignored accelerator tests.
Native tools must execute locally with the original offline/locked dependency
policy and preserved failure outputs. Formatting has not been certified clean.

All 33 modified/deleted preimages match complete native Git blob identities.
Patches are generated against those bytes, not an imagined source tree. The
selected 200-file reconstruction is nevertheless not a full Cargo checkout.
Replay proves patch applicability and exact resulting selected bytes, not Rust
linking against all workspace dependencies. `verify_packet.py --checkout` is a
read-only preimage/addition check; the owner must also inspect HEAD/status.

## Experiments and local sequence

Five new inert specifications and a tested readonly validator replace the
conflicting wave-specific admission plan. Historical specifications are retained.
The new records have null local input/executable/library/power/process pins and
are **not active queue jobs**. The RMS experiment remains explicitly blocked.
Use a newly reviewed local admission record; never edit attempted manifests.

After source review, apply the complete series and run host/compiler gates. Then
use the supplied actual-source matrix adapters, followed by genuine prefill/KV
and full continuation/tensor oracles. Six tokens cannot exercise GQA, temporal
attention or the long tile; their positive references must be independently
pinned at 64+ tokens. A traced fallback is not an oracle run of a candidate.

Baseline ANE cache recovery remains a separate local operation; short-MMA has no
special ANE cache. Strict baseline inspection requires 162 visits, zero compiler
calls and lifecycle evidence. Provision candidate FFNs separately. Interleaving
uses stacked as control; tiles4 uses INT8 sliding-QKV as control; blocked32 uses
reuse-scratch as control. No candidate is accepted merely because it builds.

Only after those gates pass, use the predeclared ABBA screen: four blocks,
two warmup and seven measured requests each, 36 requests including warmups,
nine decode steps per ten-token output, 324 model steps and at most 67,392 ANE
evaluations under the declared 208-per-step route. Actual work must match; early
EOS is not padded. Retain the 5% control-drift gate, independent confirmation,
16 GiB free-disk floor and single-owner policy. Separate AC/battery, low-power,
pmset and nominal/Fair strata. Never normalize accelerator time with CPU cycles.
No speedup or universal winner is predicted from the historical component medians.

## Remaining risks and strict boundaries

Native type/format/compiler acceptance, actual MSL resource usage, private MIL
acceptance, tensor/full-continuation behavior and power-stratified performance
are unverified. The isolated RMS adapter remains absent. Existing GQA/temporal
attention still needs the original native tensor qualification. Complete-family
counters do not prove per-layer coverage. Filesystem checks assume the required
quiescent owner; they are not atomic attestation against hostile replacement.

No shipping feature gate, model default, numerical tolerance, existing ignored
marker, multi-I/O quarantine, power setting or queue state is weakened. New safe
modules forbid unsafe code. Existing Metal encoder/component unsafe scopes remain
in their boundary crate; no new unsafe scope count is introduced. A native
shipping-symbol scan is still required independently.

## Source provenance

Mechanisms come from the already-inspected preceding research waves, not a new
unverified upstream performance claim. Prior access date: 2026-09-22. The current
pass finishes integration and delivery. Exact prior archives and native preimage
hashes are listed in MANIFEST.json.

- MLX `59d600b5e64c238427d0f8d897ab7c682ef4d3d2`: typed fragments and configurable matrix geometry. https://github.com/ml-explore/mlx/blob/59d600b5e64c238427d0f8d897ab7c682ef4d3d2/mlx/backend/metal/kernels/steel/gemm/mma.h
- llama.cpp `709fe755dfa810d77e2ac386292b29648b536864`: legacy matrix staging and hierarchical norm reduction. https://github.com/ggml-org/llama.cpp/blob/709fe755dfa810d77e2ac386292b29648b536864/ggml/src/ggml-metal/kernels/mul_mm.metal and kernels/norm.metal
- Mistral.rs `d5ae0f18f2170f10d30880cb7d21fb0880410e7b`: specialization/cache identities; inspected code identifies MLX-derived portions. https://github.com/EricLBuehler/mistral.rs/blob/d5ae0f18f2170f10d30880cb7d21fb0880410e7b/mistralrs-quant/src/metal_kernels/mod.rs
- CoreMLTools `01788ff832317a31a14053a05eab70127b14296d`: logical reshape/slice semantics, not private-compiler execution. https://github.com/apple/coremltools/blob/01788ff832317a31a14053a05eab70127b14296d/coremltools/converters/mil/mil/ops/defs/iOS15/tensor_transformation.py
- Apple ANE layout rationale: https://machinelearning.apple.com/research/neural-engine-transformers
- Exact source base: https://github.com/delysis/rvllm/commit/cffb22dabcb57b5f3ecee05acd4ab810d1ce6077
- Retained failed CI: https://github.com/delysis/rvllm/actions/runs/35790534598
