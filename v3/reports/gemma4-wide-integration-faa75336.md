# Wide candidate integration — source proposal, not accelerator acceptance

Base: `faa753360e30ce59540d1d113994c711a1d19675` (visible research branch).
Origin: `rvllm-gemma4-wide-research-226dbaad-20260922`, eight candidate/staging
patches. This integration series supersedes that delivery, not its arithmetic
or experiment constraints. Do not apply both series.

## Runtime source now included

| Candidate | Runtime boundary | Independent control |
|---|---|---|
| `metal-mma32-prefetch` | Existing BF16 MMA prerequisites, 12B shapes, 6–1024 rows; two new PSOs and ledger slots | Existing MMA32 |
| `metal-attn-q4` | Four query positions share K/V; 64–1024 rows and bounded prefix; D256/D512 | Existing SIMD attention |
| `metal-rms-simd32` | Post-projection RMS only; real gamma, FP32 statistics; separate reduction-order experiment | Existing reduction |
| `ane-int8-ffn-down4` | Full gate/up, unchanged GELU; four full-K down output partitions; one single-I/O graph | Ordinary INT8 FFN |
| `ane-attention-transpose-flags` | Same packed I/O/roundings; matmul flags instead of two operand transposes | Original attention MIL |
| `cpu-head-softcap-prune` | Cached baseline only; preserves rounded-softcap ties; host timing separate | Original host ranker |
| `ane-int8-ffn-packed32` | BLOCKED source/codec only; physical I/O descriptors unknown | No runtime admission |

The existing six candidates are not renamed, replaced or enabled. The new
selectors remain explicit/default-off and use the archived dense 12B geometry.
No bigger model is admitted based on coincidentally matching matrix dimensions.
There is no multi-I/O or four-bit ANE addition.

## Integration corrections

All fourteen modified Rust preimages are complete native Git-blob matches. The
previous seven context-only/older-formatting files were reconciled before any
candidate changes were merged. The native gate-review changes at `faa75336`
are included as the base, not reverted. A partial context still cannot substitute
for checking application/builds in the full native checkout.

The head-ranking setter previously rejected a non-30 baseline softcap even when
both experiment/timing were disabled. A pure validation helper now preserves the
baseline, and the owner enforces zero-compile-budget/baseline-cached eligibility
for direct API callers too. One new portable Rust regression joins the 23 tests
from the wide packet (24 new Rust tests in total; unrun by the reviewer).

## CI and native gate

`.github/workflows/gemma4-candidate-host.yml` is ordinary push/PR/dispatch CI on
Ubuntu. Actions are pinned; repository permissions are read-only; credentials
are not persisted. Dependency setup explicitly fetches Cargo.lock and installs
the pinned NumPy test dependency in a venv. Test/export commands then use offline,
locked Cargo with no private Apple feature and no ignored tests.

`run_gemma4_candidate_ci.py` checks 45 fully qualified expected Rust test names,
not just a positive total. It also runs the real Rust source exporter for seven
selectors in BF16/F16 and checks the declared research entry points. These are
fourteen MSL source exports, NOT fourteen Metal compilations. Errors and partial
stdout/stderr/status are retained. The workflow never calls a device or queue.

`run_gemma4_python_checks.py` runs the checked-in CPU design models, source guards,
staging checks and gate/CI contract tests. Each required module must contain
actual tests. CI requires rustfmt so the independent formatter semantics check
cannot silently skip. Python models are not Rust or accelerator execution.

The native delivery script is the single compile matrix: fourteen Metal 3.1
compile/link arms, the original plus new native host-test filters, 24 reviewed
Rust formatting inputs and six shader input hashes. The old `research::tests`
filter covered only three of the twelve research tests; it is now `research::`.
The corrected blocked32 selector remains intact. Per-build binary identities,
per-export exporter checks, final source checks, native-host refusal and fresh
output requirements remain. The proposal renderer prints this one gate call;
it no longer maintains a second six-arm command pipeline.

Formatting remains local: no rustfmt executable was available to the reviewer.
Format only reviewed paths before demanding a clean native gate. Do not format
historically drifting unrelated workspace files to make this packet pass.

## Evidence and limits

The reviewer collected 63 Python tests: 62 passed and one real-rustfmt semantics
test skipped. The tests include independent half-domain ranking, attention and
matrix/layout models, plus mocked child-process/gate fault injection. They do
not imply execution of the 45 Rust tests or either compiler.

No CI job, Rust/Metal build, private MIL compile, inference, cache operation,
power change, active queue submission, timing or promotion was performed by the
reviewer. The local owner's earlier eight-arm native receipt still describes
only that earlier revision, not this expanded source.

Follow root `handoff.md`: integrate source/tests first, run CI and the native
compiler gate, then seek only the next authorized hardware phase. Preserve the
new local documentation commits rather than overwriting them with this base's
historical handoff. A local integration error should produce a concrete receipt,
not another conclusion that the packet contains no runtime implementation.
