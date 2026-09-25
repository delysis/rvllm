# Attention-atlas candidate suite: exact-base source packet

**Base:** `delysis/rvllm`, branch `astra/gemma4-load4-tiles-20260923`,
commit `2bc5a535b113c0bcc9b1f5e15ab747005e2e3c60`.

This adds default-off Metal **operator candidates** and an explicit staged runner
inside the existing `rvllm-apple-metal` crate. It reuses the eight existing global
D512/g16 cooperative kernels as selectable controls. It emits the current
`rvllm.experiment_job.v1` queue format; it does not create a competing scheduler.

## Status and scope

The implementation is source-delivered. **Cargo, rustfmt, native Metal compilation,
device execution, checkpoint parity and timings have NOT been run in this packet's
Linux authoring environment.** Rust tools and an Apple device were unavailable.
The local owner must format the new Rust files, then run the component gate.
The included Linux C++ checks execute transformed MSL bodies, not native kernels.
They are useful algebra/indexing checks, not substitutes for these missing gates.

Nothing modifies normal `layer_forward` dispatch, existing research selectors,
model loading, INT8 weights, KV ownership, full-route oracles, the kernel-game
referee, STOP markers, power settings, ANE cache compilation or promotion policy.
No remote Git branch, commit, or PR was created. The patch is against the stated
candidate revision, not `main` and not the older `8cf7bea6` source slice.

The ANE addition is **host-only GQA packing, explicit masks and refusal descriptors**.
It is not an executable ANE kernel or a compiled single-I/O graph. The multi-I/O
panic quarantine remains intact. Integrating a qualified single-I/O ANE attention
program and connecting successful Metal candidates to full-model inference remain
separate implementation work. This packet must not be represented as that work.

## Source map

| Path under `v3/crates/rvllm-apple-metal/src` | Role |
| --- | --- |
| `attention_atlas/plan.rs` | Checked geometry, catalog, 64-byte binding ABI, scratch and guarded arenas |
| `attention_atlas/shaders/common.metal` | Vector/cooperative attention, metadata prepass, stable split-KV merge, raw-cache reconstruction |
| `attention_atlas/shaders/matrix.metal` | FP32 SIMD-matrix QK/PV with D panels and output-stationary accumulation |
| `attention_atlas/reference.rs` | Independent full FP64 oracle, BF16 conversion, deterministic synthetic and captured-input contracts |
| `attention_atlas/metal.rs` | The sole new unsafe/native boundary: precompiled PSOs, guarded buffers, synchronous completion, comparison and ABBA trials |
| `attention_atlas/experiment.rs` | Immutable specs, source/executable/metallib pins, compiler receipts, queue jobs, descriptive scores |
| `attention_atlas/ane.rs` | Bit-preserving GQA packing and absolute-position masks; execution is always unavailable |
| `attention_atlas/tests.rs` | 21 portable Rust tests, supplied but unrun |
| `bin/rvllm-attention-atlas.rs` | Explicit catalog/campaign/prepare/compile/oracle/bench/queue/ane-layout stages |

Only `src/lib.rs`, the crate's `Cargo.toml`, and one dependency line in `v3/Cargo.lock`
are modified existing files. Everything else is new and feature-gated by
`attention-atlas-research`; the binary requires that feature.

## Kernel families

The bounded catalog has **158 configurations**, not 158 measured winners:

| Family | Count | Configurations |
| --- | ---: | --- |
| Fixed-order vector | 6 | R1/K1/P64/T32; splits 1, 2, 4, 8, 16, 32 |
| Cooperative per-key softmax | 96 | R2/4/8/16, K8, P64/128, T64/128; the six split choices |
| Cooperative per-tile softmax | 30 | R8/16/32, K16/32/64, P64/128, T64/128; unsplit; source scratch at most 32 KiB |
| FP32 SIMD-matrix per-tile softmax | 26 | Same tile grid and 32-KiB source admission; unsplit |

R is packed query rows, K is the KV tile, P is the head-dimension panel, and T is
threads. R2/R4 cooperative variants are admitted only for local D256 decode or
verification. Split-KV is limited to Q <= 8. Queried PSO thread width, maximum
threads and static threadgroup memory are checked independently of source estimates.
Source estimates are not register counts and do not establish occupancy or speed.

The local geometry is Hq16/Hkv8/D256/window1024; global is Hq16/Hkv1/D512/window0.
Batch is one, queries are a contiguous absolute-position chunk, Q <= 4096, and
live KV <= 262144. Non-power-of-two page sizes, reordered physical pages, negative
page holes and invisible future page-table poison are represented explicitly.
Packed row `t*gqa + head_in_group` is never used as a causal position. Scale is
exactly 1.0. Local visibility is `position - 1024 < key <= position`.

The vector/per-key cooperative implementation completes fixed 64-coordinate dot
leaves across all D panels before softmax. Per-tile and matrix families deliberately
change arithmetic association and require separate numerical qualification.
Storage is BF16 (explicit `ushort` bits), with FP32 scores, exponentials, softmax
state, probabilities and accumulation. There is no explicit FP16 probability or
matrix-operand conversion. Native FP32 matrix behavior is still an unrun device gate.

Split partitions store **unnormalized `(maximum, denominator, numerator)`**.
A common maximum and fixed balanced reduction tree merge these states. Empty
partitions are identities. There are no floating-point atomics and no averaging of
independently normalized outputs. Each operation includes a metadata prepass and,
for split variants, a merge dispatch. Those costs remain in the measured interval.

Controls are `existing_default`, `existing_prefill_simd`, `atlas_vector`, and all
eight names `existing_global_r{8,16}p{64,128}t{64,128}`. Global cooperative controls
use their existing checked plan and buffer ABI. `existing_default` reproduces the
selected scalar global decode, local online decode or scalar prefill operator;
it is not a claim that the isolated control reproduces the whole runtime schedule.
A control that fails its own oracle stops the comparison; it is not waived.

## Reconstructible global cache: separate experimental ABI

`base_bf16_factor_f32_rope_v1` stores raw BF16 z, a positive FP32 normalization
factor per physical token, BF16 learned K gamma, and compact FP32 RoPE coefficients.
V is `BF16(widen(z)*factor)`; unrotated K is
`BF16((widen(z)*factor)*widen(gamma))`. The global rotation covers coordinate pairs
(i, i+256) for i=0..63. Multiplication, explicit FMA and BF16 rounding boundaries are
spelled out identically in the shader and the host representation helper.

**This is not the current fused producer ABI.** That producer retains FP32 raw
projection values. Narrowing those values to BF16 and treating this as layout-only
would be an unsupported arithmetic change. This packet neither performs nor
silently authorizes it. K is never reconstructed from already rounded V.

The runner can compare reconstruction against materialized K/V of this exact new
ABI. It materializes the control before timing, without mutating caller caches.
The measured candidate includes reconstruction, gamma and coefficient reads.
No shared raw tile is retained across the QK and PV phases: the raw base is read
again for PV, and rotated coordinates need their paired operand. Thus this version
is a concrete recomputation candidate, not an assertion of halved DRAM traffic.
The 1028-byte raw base/factor payload replaces a 2048-byte separate K/V payload,
but this operator snapshot also carries 512 coefficient bytes per logical token,
plus gamma, page metadata and padding. All are included in arena accounting.
A future shared-coefficient/cache-resident implementation needs separate evidence.

Repeated physical pages with different logical RoPE phases cannot be materialized
into one common K buffer; the control preparation explicitly refuses that case.

## Input capture ABI

The usual input is a deterministic synthetic fixture. A captured input uses
`{"kind":"captured","buffers":[...]}` with exactly 12 `{path, sha256}` objects.
Files are exact little-endian bytes, pinned before and after loading; loaded bytes
are hashed as well. No model checkpoint identity is inferred from a filename.

| Index | Buffer | Shape |
| ---: | --- | --- |
| 0 | Q BF16 | `[queries,16,D]` |
| 1 | K BF16, or raw z BF16 | `[physical_blocks,page_size,kv_heads,D]` |
| 2 | V BF16 | Same as K; exactly 2 dummy bytes in raw mode |
| 3 | Page table i32 | `[max_blocks]` |
| 4 | Absolute positions i32 | `[queries]` |
| 5,6,7 | Reserved output/partial/status | **Empty files**, never trusted oracle output |
| 8 | Normalization factors FP32 | `[physical_blocks,page_size]` raw; 4 dummy bytes otherwise |
| 9 | Learned K gamma BF16 | `[512]` raw; 2 dummy bytes otherwise |
| 10,11 | Cosine/sine FP32 | Each `[live_keys,64]` raw; 4 dummy bytes otherwise |

Unused allocated K/V padding must be finite. Negative page IDs represent missing
keys, not pointers. No cache is mutated. The caller must capture only committed,
producer-complete state, retaining prompt/checkpoint/producer provenance outside
these operator input pins. This packet does not add a model-state capture exporter.

## Local-owner workflow

First verify/apply the root packet on its exact base. Then, from the repository:

```sh
bash v3/tools/attention-atlas/format-owned.sh
# Review only the new-file formatting diff; do not format unrelated workspace code.
bash v3/tools/attention-atlas/local-gate.sh /absolute/new/atlas-host-gate
```

The gate requires 21 named tests and 21 passes, checks the default-off build and
builds the release runner with `--offline --locked`. Logs are write-once. It does
not run ignored device tests, invoke inference, spawn a queue owner or time kernels.
The packet's original bytes are hashed separately from subsequent owner formatting.

After that gate succeeds, freeze/hash/sign the resulting binary under the existing
local-owner policy. The commands below use `BIN` for that exact absolute binary
path, and `ROOT` for this repository. All output directories must be new and their
parents must already exist. `prepare` pins the executable itself.

```sh
"$BIN" catalog /absolute/new/catalog.json
"$BIN" campaign /absolute/new/specs
"$BIN" prepare "$ROOT/v3/tools/attention-atlas/examples/global-decode.json" /absolute/new/prepared
```

`campaign` writes **776 synthetic workload specs**, without compiling or running
any of them: 110 global-decode, 158 local-decode, 110 global-verification,
158 local-verification, 65 local-prefill, 65 global-prefill and 110 raw-global-decode.
These are an initial screening matrix, not exhaustive long-context/device coverage.
The current generator uses mixed inputs; other adversarial fixtures are selectable
in pinned specs. Counts were checked from the source catalog during authoring; the
Rust campaign command itself remains unrun.

The next stages require the existing exclusive accelerator owner and fresh
readiness/power/process observations, even for compilation. Conditions are
recorded rather than held stable: use `stable_seconds=0`, leave thermal and power
mode fields unpinned, and compare only matching observed strata. Do not run them
alongside other accelerator work:

```sh
"$BIN" compile /absolute/new/prepared /absolute/new/build
"$BIN" oracle /absolute/new/prepared /absolute/new/build /absolute/new/oracle
"$BIN" bench /absolute/new/prepared /absolute/new/build \
  /absolute/new/oracle/oracle.json /absolute/new/bench
```

Compilation uses xcrun/macOS SDK, Metal 3.1 and `-fno-fast-math`. It preserves
compiler/linker stdout, stderr, command lines, SDK identity and pinned tools.
Oracle/bench load the frozen metallib; they never compile shaders or ANE graphs.
The identities cover named inputs/tools, not a transitive attestation of the SDK.

For the actual existing harness, edit the explicitly marked path/condition
placeholders in `examples/queue-request.json`, then:

```sh
"$BIN" queue /absolute/edited/queue-request.json /absolute/new/job.json
# The LOCAL OWNER submits this JSON through the existing rvllm_experiment_queue.
# This command only emits the job; it does not submit it or start a worker.
```

Compile/oracle jobs are preparation; benchmark jobs are exploratory timing and
require a successful exact-workload oracle receipt and frozen build. Their
`{output}/atlas` directory belongs to the existing queue attempt. Stage jobs are
created sequentially after their prerequisites exist, not with invented hashes.
No kernel-game submission or full-route qualification receipt is forged. The example
blocks any `llama-server` by default. Declare the actual PID/port through the
existing idle-server policy only after inspection; never stop an unrelated server.

## Native correctness and timing contract

The native oracle evaluates the full input in FP64 with an explicit work budget;
it never samples a few coordinates. It checks candidate FP32 accuracy, repeat-bit
identity, BF16 accuracy and BF16 equality to RNE(candidate FP32), the control's
accuracy, read-only input/arena guards and a refused-position no-write test.
Absolute and relative-L2 tolerances are fixed in the spec before execution.
The example tolerances are **synthetic operator screening tolerances**, not changes
to the existing driver/tensor/model oracles. Captured model tensors need their
unchanged model-oracle criteria and complete continuation gates as well.

The benchmark re-runs the full operator oracle before warmup. The example uses
two warmups per arm and seven ABBA blocks: **14 measured invocations per arm**.
This exploratory operator protocol is separate from the established full-model
campaign. No sample is dropped or retried. Partial blocks and errors are preserved.
The symmetric early/late-control drift ratio must be at most 1.05. `score.json`
contains descriptive ratios, never a promotion verdict or significance claim.

GPU intervals include the complete metadata/main/merge command. Host intervals
include command allocation/encoding/submission/completion wait. No cache readback,
reset, hashing or file writes occur between samples of one ABBA block; block
journaling occurs afterwards. Dispatch completion counts mean encoders recorded
in a command that completed, not separate hardware timestamps per kernel.
Readiness, power and thermal observations belong to the existing parent harness.

These are warm, repeatedly reused operator snapshots. There is no cache flushing,
streamed whole-model weight traffic, overlapped request scheduler, KV producer,
output projection, GPU/ANE crossing or end-to-end generation in these timings.
A faster score does not establish faster Gemma token generation. `device.json`
records the actual device and queried PSO/arena sizes when the local runner executes.

## Checks completed in this source packet

Linux C++ emulation compiled transformed copies of the actual MSL bodies and
executed **32 positive cases** plus **128 rejected-metadata/no-write checks**.
Maximum absolute difference from its independent FP64 oracle was
**1.97548e-7**. It also checked 25 BF16 rounding ties/boundary values.

Five separately compiled, deliberately wrong shader mutants were rejected by
FP64 comparisons: overwriting D-panel sums, averaging normalized split outputs,
leaking future keys, an off-by-one window and aliasing raw V. Compilation failure
was not counted as successful detection. The host fiber shim models barrier
participants and whole 8x8 matrices; it does not model Apple's fragment layout,
compiler register allocation, hardware floating-point semantics or GPU races.
The long-window host fixture is intentionally sparse across boundary pages;
it is not evidence of dense long-window native correctness or bandwidth.

Reproduce these OPTIONAL Linux host checks with:

```sh
bash v3/tools/attention-atlas/emulate.sh /absolute/new/emulation
bash v3/tools/attention-atlas/mutation-check.sh \
  /absolute/new/emulation /absolute/new/mutations
```

They are secondary tools; Rust remains the runner/reference/planning implementation,
and MSL is the device implementation. No Python is part of the delivered suite.
The supplied evidence preserves earlier bounded-tool timeouts separately; they
are incomplete attempts, not native failures or successful tests.

## Remaining gates and integration work

The immediate gate is local Rust formatting/typechecking and Metal compilation.
Then require native adversarial cases (including dense window wrap boundaries,
all-hole controls, high dynamic range, multiple page sizes, Q=1..8 and prefill
remainders), nonzero matching dispatch evidence, captured tensors, the unchanged
checkpoint continuation oracle and separately declared matched power-stratum
comparisons. Cover long global contexts through 262144 with explicit memory and
full-reference budgets; compilation or a tiny test is not that evidence.

Do not replace a selector on the strength of this packet. A winning operator
still needs producer/cache-owner/normal-route integration and its own full-route
qualification. Raw-cache producer support and an executable, qualified single-I/O
ANE attention path are explicitly not included.
