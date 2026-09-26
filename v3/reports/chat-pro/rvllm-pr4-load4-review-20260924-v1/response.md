## Verdict: request changes before native qualification

I reviewed the pinned comparison from **`d391eea9acaca47653e19bb7d6e58e81dbd51b7a`** to **`90003d0f5f9e8d2173c65fddd4447a996e7094b3`**, including the shared shader, all seven specializations, admission/resource checks, dispatch accounting, and the new qualification harness.

**I found one high-confidence source-level compilation blocker, two qualification-evidence defects, and a smaller failure-capture gap.** I did **not** find a separate concrete barrier-divergence, fragment-overlap, or threadgroup-allocation overrun in the seven supplied geometries. The compilation failure below is predicted from the source and documented type rules—not an observed compiler diagnostic.

All repository line references below refer to the pinned head.

## 1\. Prioritized findings

### P1 — The BF16 GEMM instantiation contains an invalid implicit `float` → `bfloat` conversion

**Locations**

`v3/crates/rvllm-apple-metal/src/research_shaders/load4_tiled_common.metal:85–90,110–113`

`v3/crates/rvllm-apple-metal/src/research_shaders/load4_m16n32k64.metal:4,9–10`, with the same instantiation pattern in the other six leaves.

`v3/crates/rvllm-apple-metal/src/kernels.rs:2579–2601`

The shared epilogue contains:

C++

```
if (FP32) C[output] = value;
else C[output] = f16_sat(value);
```

For each BF16 GEMM export, the source rewriting makes the output pointer `device bfloat *`, so template deduction gives `OUT = bfloat` and the leaf supplies `FP32 = false`. Nevertheless, this is an **ordinary `if`**, not a discarded template branch: the assignment in its first arm must still be well-formed. `value` is `float`, and Metal’s `bfloat` does not accept that implicit conversion. Apple’s MLX implementation explicitly documents this restriction. The exporter only rewrites `half` and `f16_sat`; it does not repair this assignment.

**Failure scenario:** exporting any new selector as BF16 instantiates its GEMM helper and should fail semantic compilation. Because each selected library includes both entry points, this can prevent obtaining the QKV pipeline as well—even though its own `OUT = float` instantiation is not the offending case. F16-only compilation would not expose this defect.

**Minimal proposed patch—not applied:**

C++

```
if (FP32) C[output] = OUT(value);
else C[output] = f16_sat(value);
```

The explicit conversion makes the otherwise-unused assignment legal. In the FP32 specialization, `OUT(value)` is still `float(value)`; in the BF16 specialization, the executed branch remains the existing `bf16_sat(value)` after export. Thus this does not introduce another executed rounding boundary.

**Required verification:** retain the existing complete **36-arm BF16/F16 compile/link gate**. A source-string assertion is not a substitute for instantiating both output types. Preserve the exact FP32-bit and once-rounded BF16 numerical gates afterward.

**Blocks native execution? Yes, through compilation.** This is the immediate blocker to resolve before candidate device qualification.

* * *

### P2 — The repeated-use test can pass skipped writes and transient corruption

**Location**

`v3/crates/rvllm-apple-metal/src/load4_tile_tests.rs:377–391`

The repeat phase executes:

Rust

```
for path in [3, 2, 1, 0, 0, 1, 2, 3] {
    run(path)?;
}
```

It then reads each output only once and compares it with the original captured output. The repeat phase neither re-poisons outputs nor validates each invocation before the next invocation can overwrite it.

**Two concrete false-pass scenarios:**

-   A repeated dispatch omits some or all stores. The previous correct output remains in the buffer, so the final comparison passes.
    
-   The first repeated invocation corrupts output, but the second invocation of that same path repairs it. Only the repaired state is inspected.
    

The initial poisoned-output checks remain useful; this finding specifically limits what the additional repeat phase establishes.

**Minimal patch/test:** validate and preserve output **after every invocation**. Before each positive repeat, re-poison its payload while retaining/checking the surrounding canaries. Expected-refusal arms must likewise begin poisoned and remain untouched. Give captures invocation-specific names so `create_new` continues to protect previous evidence.

Add harness regression cases for a skipped repeat and for “bad first repeat, good second repeat.” Both must fail. Alternating two distinguishable inputs is a further useful check, but re-poisoning plus per-invocation validation addresses the immediate false-pass windows.

**Blocks native execution? No.** It blocks treating the current repeat phase as adequate repeated-use qualification.

* * *

### P2 — The new component-qualified artifact does not bind the model, executable, and run identity

**Locations**

`v3/crates/rvllm-apple-metal/src/load4_tile_tests.rs:46–57,402–408`

`v3/reports/gemma4-load4-tiles-20260923.md:136–159`

The test obtains its model directory from an environment variable and saves generated shader source. Its final `qualification.json` identifies the selector and case outcomes, but does not bind the actual test executable, checkpoint/tensor contents, device identity, or the saved source and captures through a digest-bearing run manifest. The documented direct invocation does not supply such a manifest.

**Failure scenario:** a shape-compatible checkpoint from another revision can produce a passing report. Candidate, baseline, and independent sampled CPU dots all consume that same alternate data, so mathematical agreement does not detect the wrong checkpoint identity. Similarly, a report from another build cannot be reliably attributed to this head from `qualification.json` alone.

This does **not** make the arithmetic comparisons invalid. It makes the component receipt insufficient, by itself, to establish qualification of the intended immutable candidate/model combination.

**Minimal patch/test:** use a runner-written immutable manifest, or extend the component artifact, to bind the actual test executable, generated source, actual tensor bytes used, relevant device/compiler configuration, and the final report/capture digests. An existing trusted enclosing manifest is sufficient; a second provenance system is unnecessary. The consumer must verify that identity join rather than accept the status string alone.

Add a receipt-consumer mutation test that substitutes a different executable, source, or tensor identity while retaining a successful numerical report; qualification must be rejected. The report already asks the local owner to return tested-commit and build/source hashes, so making that relationship machine-verifiable completes rather than relaxes the stated boundary.

**Blocks native execution? No.** It blocks standalone attribution of the resulting component-qualified artifact unless an external immutable manifest supplies the missing bindings.

* * *

### P3 — Command-buffer errors exit before diagnostic captures are saved

**Location**

`v3/crates/rvllm-apple-metal/src/load4_tile_tests.rs:297–320`

After `waitUntilCompleted()`, `command.error()` becomes an immediate `Err`. The caller propagates that error while running the initial four paths, before reaching the output-read and capture section. The code preserves subsequent **numerical assertion** failures, but not command-buffer failures.

**Failure scenario:** a candidate command errors after earlier paths completed. The attempt directory survives, but the failing case’s guarded outputs and structured command status are not written. That loses useful evidence for distinguishing unchanged poison, partial writes, guard damage, and an execution failure.

**Minimal patch/test:** after the command reaches a terminal state, preserve its status/error and available guarded bytes **before** propagating failure. Never write `qualification.json` for that attempt. This can share the per-invocation capture/check helper recommended for finding 2.

A host-side injected-error test should assert that failure artifacts survive and no success artifact is emitted.

**Blocks native execution? No.** This is a failure-diagnostics defect, not an acceptance false-pass.

## 2\. Synchronization, bounds, resources, rounding, and accounting

These are source-review conclusions, not device qualification.

### Barrier uniformity and scratch reuse

I found no divergent path to the barriers in the supplied specializations. The early returns depend on threadgroup-uniform dimensions, scale, group coordinates, and threadgroup size. The K-loop trip count is also uniform. Importantly, the barrier at **`load4_tiled_common.metal:70` includes the final K iteration**, so operand readers finish before output stores reuse the allocation. The barrier at **line 82** precedes scalar reads of the staged output.

The allocation formula is correct for the implemented phase reuse:

The leaves and Rust budgets agree for all seven profiles. In particular, **32×64×32 correctly reserves 8,192 bytes**, not merely its 6,144-byte operand staging requirement; output staging is larger. The largest source allocation is 24,576 bytes for 32×64×128. These are source allocations, not measured compiler resource use or occupancy. The runtime additionally checks queried static memory, maximum threads, device capacity, and SIMD width.

### Bounds, tails, and fragment ownership

For the admitted shapes, the missing per-vector K-tail mask is **not** a defect: Rust and the shader require K divisibility by BK, and the admitted strides satisfy vector alignment. N must divide the N tile. M tails are zero-filled during A staging and suppressed during final output stores. The shader’s bounded role dimensions also keep the group-coordinate arithmetic within its intended range. See **`load4_tiled_common.metal:32–46,93–109`** and **`research_projection.rs:92–115`**.

Under the enforced 32-lane SIMD width, the `sg / GN` and `sg % GN` mapping assigns nonoverlapping output rectangles to SIMD groups. The 8×8 fragments cover those rectangles, and each accumulator receives ascending K/8 contributions. I found no transposition or fragment-ownership mismatch in the supplied geometries. This preserves the source-level accumulation sequence; it does not prove the compiler’s exact numerical result.

Apart from finding 1’s type-checking problem, the intended epilogue preserves FP32 output or one BF16 storage conversion. The existing exact-bit comparisons should remain the authority after compilation, especially because matching source-level K order alone is not sufficient qualification.

### Rust admission and dispatch receipts

I found no new admission mismatch: the plan retains model/prefill eligibility, native-BF16 classification, identity scale, role-specific shapes, input alignment, checked spans, output alignment, and write/read disjointness. The PSO boundary checks the selected kernel identity and its resource budget rather than accepting an arbitrary pipeline by name.

The unchanged production encoder obtains the thread count from the selected kernel’s limits: **the 64-thread candidate is not accidentally launched with 128 threads**. It records the selected dispatch after encoding, not merely after a successful routing predicate. The registry appends fourteen entries without renumbering the existing seventeen, and the delivery source manifest includes the common shader.

One important limit is explicit in **`research_evidence.rs:211–221`**: `complete_family_exercised` means every selected entry point has a nonzero count. It does **not** establish complete per-layer work. For a two-entry family, counts of one GEMM and one QKV satisfy that predicate. This is consistent with its documented purpose, but it cannot replace an exact-work requirement in a timing receipt.

## 3\. Can the tournament support causal speed claims?

**Yes for the total effect of a selected candidate against an identical-workload control, provided the inherited qualification and timing gates are actually enforced. No for attributing the improvement specifically to fewer barriers or a single layout mechanism.**

The proposed ABBA/BAAB comparison against both unchanged `metal-mma32-load4` and `off`, with matched hardware/power/thermal conditions and retained paired observations, is a reasonable foundation. It explicitly rejects promotion from exploratory samples. See **`v3/reports/gemma4-load4-tiles-20260923.md:118–125`**.

There are three distinctions to preserve:

**Whole-candidate effects are not mechanism effects.** The incumbent uses a K=32 loop and separate operand/output arrays—**`mma32_load4.metal:11–39,68–73,107–112`**. The common implementation changes allocation reuse and code organization as well as, for most comparisons, tile geometry. A faster new 32×32×64 candidate therefore would not demonstrate that halving the K-block barriers caused the improvement. Within the new 32×64 family, varying BK is a cleaner parameter comparison, but still changes staging memory and potentially compiler resource use.

For mechanism research, a **32×32×32 common-implementation control** would help separate the shared-helper/scratch-reuse change from the BK change. That is an additional ablation, not a prerequisite for reporting a properly controlled whole-candidate effect, and it should not replace either existing control.

**Family coverage is not equal-work accounting.** Timing admission must require the workload’s exact expected dispatch work and successful completion, not just both counters being positive. Otherwise partial fallback can be included in a result described as candidate-wide behavior. Exact prompt/reference identity must also match within each pair: the 64×64 candidate’s larger minimum M must not lead to comparing its longer workload against another candidate’s short workload. The base campaign already distinguishes short and long workloads and requires exact dispatch evidence.

**Selecting the fastest of seven is not independent confirmation.** Keep the existing predeclared replication, control-drift ceiling, immutable identities, workload separation, and independent-confirmation/promotion requirements. Freeze the primary latency metric, warmup/order policy, and stopping rule before examining timing results; confirm a selected winner with fresh evidence rather than presenting its best selection sample as an unbiased estimate. The existing campaign calls for at least five valid counterbalanced pairs and a 5% control-drift ceiling, while the governing handoff requires independent confirmation and fixed work counts. The new report’s shorter timing paragraph should not be treated as replacing those requirements.

## 4\. Execution decision and review limitations

**Finding 1 blocks proceeding to native candidate qualification at this head.** Fix the type-safe epilogue and obtain the complete native compile/link result first. I found no additional concrete source-level safety defect that independently requires disabling these seven geometries. Findings 2 and 3 should be resolved—or, for provenance, satisfied by a verified enclosing manifest—before subsequent results are accepted as qualification evidence.

Useful additional qualification coverage would include resource-legal wrong threadgroup shapes, excess grid coordinates, and rejected scale/dimension cases against poisoned outputs, without bypassing the host resource limits. That would exercise the shared shader’s refusal paths rather than relying solely on host calculations.

This was a **read-only source review**. I did not compile Rust or Metal, run tests, initialize a Metal device, execute inference, inspect native command-buffer behavior, or collect timings. The BF16 compilation diagnosis is a source/type-rule finding awaiting the local compiler’s diagnostic. I did not independently verify historical native receipts, compiler lowering, register pressure, occupancy, or the complete tournament referee implementation. Historical campaign results are not evidence for this head.

**No files or PR state were changed.** The immediate unblocker is the type-safe epilogue; the next qualification repairs are per-invocation capture/checking and an immutable identity binding for the component receipt.