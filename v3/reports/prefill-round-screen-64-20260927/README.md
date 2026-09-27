# First native screen of Astra's exact-`fb5f169c` prefill delivery

The supplied ZIP (`sha256:f7b949a1bb046f8437167874da7d40401afb2b63e8eb314c529fb288f1437827`) was applied only to this separate exact-base worktree. The patched source is not integrated into PR #6. Full Metal-crate host gates passed, including 186 library tests; the strict Apple build compiled all eight source arms, metallibs, and the Swift driver. That is compiler admission, not device correctness or speed.

All native trials below used the existing serial rvLLM experiment queue with immutable manifests, `stable_seconds:0`, observed conditions, the same frozen referee/driver/libraries, and a distinct inner referee lock. No failed job was retried or overwritten.

| Job | Result |
| --- | --- |
| `prefill26-screen64-20260927-s00-m64`, normal down-norm | Passed five fixtures through M64; sampled conditions eligible. |
| `prefill26-screen64-20260927-s01-m64`, matched load4 down-norm control | Failed the independent FP64 raw-intermediate check at M6 periodic, index 25: FP32 result 347.1277160644531 versus 347.12372314184904. The declared allowance is about 0.00231225756; observed delta is about 0.00399292260. The M6 structured fixture passed. |
| `prefill26-pipeline64-20260927-s00-m64`, new lookahead down-norm | Independently failed at the same M6 periodic element with the same values. The complete raw intermediate buffer is byte-identical to the load4 control (`sha256:d8755f87ea818e0015822fc2457f63539333c5f82ad71b99ab469cc7ecf8623d`). M6 structured passed. |
| `prefill26-q4k1664-20260927-s00-m64`, new global Q4K16 paged attention | Passed all 18 fixtures through M64, including periodic, newest, holes, and all-holes; sampled conditions eligible. |
| `prefill26-q4k16g256-20260927-s00-m256`, new global Q4K16 paged attention | Passed all 22 fixtures through M256; sampled conditions eligible. |
| `prefill26-q4k16l64-20260927-s00-m64`, new local Q4K16 paged attention | Passed all 18 fixtures through M64; sampled conditions eligible. |

The original generator chained the four arms with `after` dependencies. The load4 failure therefore prevented its later two arms from running; their submitted manifests are preserved. Independent new campaign IDs ran the lookahead and global-attention arms without depending on the failed control. The later v2 generator emits independent correctness arms while retaining serial dependencies for ordered timing; it did not rewrite old manifests or change the frozen referee that ran those jobs.

A read-only check of all 23,040 M6 periodic raw projection elements found 2,820 above the packet's declared `1e-5 + 2e-6 * sum_abs_products` allowance; the worst observed excess was 3.38 times that allowance. An independent CPU calculation using the exact BF16 operands and a sequential FP32 accumulator matched **all 23,040 raw FP32 output bits** from both Metal arms. At failed index 25 the CPU sequential result is exactly 347.1277160644531, while grouping eight products before adding to the accumulator gives 347.1239929199219 and the FP64 dot is 347.12372314184904. This strongly identifies ordinary long sequential-FP32 rounding, not a lookahead-specific numerical defect, as the reason for the synthetic gate failure. It does **not** make the original FP64 acceptance threshold pass. A versioned referee should separate an exact declared-FP32-arithmetic check from a mathematically justified FP64 error/quality bound; old failed receipts remain failures. Do not silently relax the old gate or claim a projection winner.

A separate, freshly built v2 referee now makes that distinction explicitly for the periodic raw-norm projection fixture: it reads the actual sealed BF16 fixture bytes, computes a sequential FP32 `mul_add` reference, requires bitwise equality, and independently enforces the standard `gamma_K * sum_abs_products + 1e-5` FP32 error bound against FP64, with `gamma_K = (K * 2^-24)/(1 - K * 2^-24)`. BF16 products are exactly representable in FP32 for these bounded inputs. This is a new synthetic arithmetic contract, **not** a retroactive pass under v1's stricter empirical threshold or a change to production quality policy. A unit test pins the observed 15,360-term example. The generator v2 independently emits correctness arms; a host-only three-arm manifest test verified empty `after` arrays. The newly frozen v2 referee and all eight strict metallibs built successfully. Its normal, load4-control, and lookahead down-norm arms then independently passed all five correctness fixtures through M64 under eligible sampled conditions; the two tiled arms produced byte-identical M6 periodic raw and normalized outputs. Old failures are untouched.

The two tiled v2 down-norm arms subsequently passed all required fixtures
through M256, independently and under eligible sampled conditions. The normal
fused down-norm arm separately passed M256. The v2 global-attention scalar and
Q4K16 arms both passed their correctness ladders through M512. These jobs
retain distinct source/executable/driver/library identities from the v1
failures.

## Further paired timings: a clear attention result, inconclusive projection

At M512, eight serial ABBA/BAAB global-attention jobs all passed their
pre-timing correctness and repeat-stability screens and were queue eligible.
The referee's 5% within-arm drift gate passed: 0.205% scalar and 0.417%
Q4K16. Per-process median GPU times were 71.093–71.239 ms for the scalar
control and 11.756–11.806 ms for Q4K16. The descriptive median ratio was
**6.0357×**. This independently extends the M256 *isolated operator* result,
not a full-route speedup or production promotion. Adjudication SHA-256:
`7e3cd446734847b133424d3783637dcc8dc9ca3e3a0ee65fe3f7bb4d62f91073`.

At M256, eight serial ABBA/BAAB down-norm jobs compared the load4 control to
the lookahead pipeline. Both arms had passed their separate correctness
screens; all eight timing jobs succeeded and were queue eligible. Their
per-process median GPU times ranged 2.978–3.748 ms for load4 and
2.372–3.229 ms for the pipeline. The within-arm drift was **25.85% and
36.12%**, respectively, far above the fixed 5% gate. The referee therefore
reported `timing-screen-rejected-drift`, retaining all samples. Its descriptive
1.1375× control/pipeline median ratio is **not a speed qualification or a
winner**. We will inspect the variance and choose a fresh, predeclared trial
if warranted; these receipts will not be rewritten or selectively discarded.
Adjudication SHA-256:
`8e39027dc82e50f905a7ae177864ffb241aefc8638e64e48fd7ac27babc6d8d4`.

The frozen v1 executable is SHA-256
`227d6e45a90060ca7b47f723ed9fefb6c030d9d3a117c18939468e61356c2107`.
The frozen v2 executable is SHA-256
`b0a12c27c1c0e28a7e486448f6737e544f7bf2cb20fcdddd20f3eec9e6087924`.
The branch's `cd80315ea5536ab3967c7f74de3eb4e3b1127bce` commit contains
the v2 source used to build and freeze the latter; `7f7fa13a` is a subsequent
rustfmt-only commit. The complete small queue manifests, reports, conditions,
and failed receipts are in `queue-results.tar.gz` (SHA-256
`fd30211ea8f948f121eae9719a26fcbd28e363b025cc756b3799042fe8d30431`).
The much larger device buffers and per-fixture output remain local in their
immutable paths; the archive is not a substitute for them.

All screen files, original failed outputs, condition samples, and queue records remain in their immutable paths. No production selector changed and no real-checkpoint full-route quality was established.

## First paired attention timing

The `normal` global-attention arm independently passed its M256 correctness
screen. Eight later serial-queue jobs then compared it to Q4K16 at identical
global-attention M256 shape in ABBA/BAAB order. All eight succeeded and were
queue-eligible, with no observed competing processes. Every condition sample
reported AC power, power mode 2, thermal state 0; there was no request for a
stable thermal period. The referee checked full-output correctness and repeat
stability before timing each process. Its adjudication passed the 5% drift gate:
2.211% for the scalar control and 3.974% for Q4K16. Per-process median GPU
times were 30.369–31.040 ms for the scalar control and 4.159–4.324 ms for
Q4K16. The descriptive median ratio is **7.2708×**, with first/second order
blocks at 7.353×/7.158×. This is a promising **isolated operator** result,
not full-model prefill throughput, MLX parity, or a promotion. The adjudicator
deliberately leaves `conditions_qualified:false` until external review; this
report records the above condition audit without changing its output. A second
independent campaign and longer shapes remain necessary. Adjudication JSON
SHA-256: `464d76c422e73219c850d9917d974dbcbf6a03c52879c8f448e7da21c97628d4`.
