# INT8 ANE projections: source and small device qualification

2026-09-15. The working Gemma 4 12B path still uses INT8 FFNs and FP16
QKV, output and vocabulary projections. This change adds an experimental
INT8 projection constructor; it does not switch the model to it.

The characterized HTTP baseline spends median 31.19 ms in QKV, 21.34 ms in
output projections and 18.91 ms in the vocabulary projection per decode step.
That makes these projections a useful next optimization target. Smaller source
weights alone do not establish faster ANE execution or smaller runtime residency.

`AneInt8LinearWeights` reuses the existing per-output FFN quantizer: FP16 stored
scale, signed coefficients in [-127,127], ties-to-even rounding and affine
dequantization on axis zero. Activations remain FP16. The new constructor shares
the existing single-input/output convolution and aligned I/O implementation.
The captured 4096-to-3840 FP16 projection MIL remains byte-identical, preserving
its cache identity. The existing FP16 blob construction is unchanged.

## Verification

Six host quantizer tests pass, including nonsquare blob round trips and exact
agreement with the existing FFN quantizer. The Apple host suite passes 105 tests,
with 24 ignored device tests and one explicitly excluded device integration test.
The initial unfiltered suite reached that existing integration test and failed
at its missing private-API opt-in; it did not compile or evaluate a device graph.
The corrected host command excludes that test without enabling additional APIs.

An isolated 64-input/128-output projection smoke test passed three synthetic
inputs, including one containing 171.5, against an independently accumulated
FP32 CPU dot product of the stored INT8 reconstruction and a dense ANE control.
The declared numerical tolerance was `0.002 + 0.01 * abs(cpu_reference)`.
There were zero violations across both backends and all three inputs. The INT8
maximum CPU errors were 0.00006676, 0.00006294 and 0.01226425 respectively.
Outputs were not bit-identical between the two ANE encodings.

The journal records two compiler calls, two loads, six evaluations and two
successful unloads. Both owned staging directories are absent and the boot
identity is unchanged. The signed development-profile test binary SHA-256 is
`3cbf139cd4e12f66d1b24368e52bd10f34e0339abcd207f977c51e202b1959fc`.

Evidence: `gemma4-12b-evidence-20260914/int8-linear-smoke/`, including captured
source, original host logs, the corrected host suite, result, journal and
before/after verification receipts. The laptop switched from low battery to AC
before the device test. This test establishes numerical backend behavior only;
there is no latency comparison or normalization claim.

## Remaining evidence

Actual QKV weights have now been checked as described below. Output/vocabulary
projections, matched-power timing and full-model reference checks remain before
promotion. No new full-model cache provisioning has occurred.

The user's native-QAT exception for any future four-bit experiment is recorded
in [the source audit](gemma4-native-qat-four-bit-viability-20260915.md).

## Actual Gemma QKV qualification

The release-profile probe in `ane_int8_projection_tests.rs` loads only one
validated checkpoint layer's QKV and input norm. It derives normalized QKV
inputs from saved post-layer states of three successful INT8-FFN baseline
requests, at positions 21, 84 and 652. Input files, normalized values, original
weights and reconstructed weights are hashed. CPU and hardware runs agree on
these hashes and all quantization-error results.

| Target | Graph shape | Q relative L2 error | K relative L2 error | V relative L2 error |
| --- | --- | --- | --- | --- |
| Sliding layer 1 | 3840 → 8192 | 0.754–0.864% | 0.499–0.595% | 1.366–1.469% |
| Global layer 5 | 3840 → 8704 | 0.730–0.849% | 0.789–1.095% | Shared raw K |

These are quantization changes in individual projections, not model-level
quality scores. The `backend_tolerance_violations` field inside the CPU
quantization diagnostics simply applies the same reporting threshold; its
nonzero counts are not ANE backend failures. Actual backend comparisons use
each encoding's own CPU reference, with `0.01 + 0.02 * abs(reference)` tolerance.
All three encodings pass every captured input: zero backend violations.
INT8 backend relative L2 error is 0.0302–0.0344%; dense controls are
0.0203–0.0227% against their respective CPU references.

The original QKV baseline for layer 1 was missing from the daemon cache under
the stable signing identifier. Its exact model identifier matches an earlier
cache-hit receipt. The first strict attempt compiled, loaded and evaluated
nothing, and left no staging directory. An explicit `restore-original` mode
then restored each target baseline separately, with a one-compilation budget
and zero evaluations. It remains distinct from qualification (at most two new
compiles) and timing (strict cache, zero compiles). This does not diagnose cache
eviction or establish persistence.

Across both restores and qualifications, journals record six compiler calls,
eight successful loads and unloads, and 18 completed evaluations. All owned
staging paths are absent and boot identity remains unchanged. Two subsequent
strict timing processes compiled nothing and repeated the backend checks, but
both timing preflights found Fair thermal state. They performed no warmups or
timing trials, so there is no performance result to promote. AC/Low Power Mode
and complete raw observations are retained. Checks after source review and a
45-second cooldown still reported Fair; no additional accelerator timing was
attempted. The three focused runtime host tests pass (one checkpoint audit is
ignored); the new ignored probe itself passed both CPU and hardware runs.

Evidence: `gemma4-12b-evidence-20260914/int8-qkv-real-probe/`. This includes the
frozen signed release probe, source, original failed strict-cache attempt,
repair and qualification journals, cleanup checks, CPU reports and rejected
timing preflights. The normal inference path still uses FP16 QKV.

The frozen `replay-v2.json` records the immutable executable path, verified
SHA-256 and test name. The older `binary-v2.json` also records the original
Cargo output path, which later builds replace; use the frozen replay path.
For a later timing retry, use a new output directory, the same model and input
paths from `intent.json`, `RVLLM_INT8_QKV_MODE=time`, and no driver journal.
Set `RVLLM_INT8_QKV_LAYER` to 1 or 5 and `RVLLM_INT8_QKV_INPUTS` to that layer's
JSON input array. `RVLLM_GEMMA4_MODEL_DIR` and `RVLLM_INT8_QKV_OUTPUT` supply the
remaining paths. Run the named test with `--exact --ignored --nocapture
--test-threads=1`. A cache miss remains a failure; a thermal/power eligibility
failure records a skipped measurement without warmup. Compare at least five
eligible pairs for each control before drawing a timing conclusion.

## Full vocabulary CPU audit

The CPU-only `ane_int8_head_tests.rs` probe now checks all 262,144 vocabulary
rows (1,006,632,960 coefficients) using the existing row INT8 quantizer. It
streams sixteen 16,384-row tiles, so it does not retain a full second model.
It verifies each saved final-layer state's hash, applies the checkpoint's final
normalization and the decoder's explicit FP16 projection/softcap boundaries,
and ranks the full vocabulary independently.

Four captured states cover positions 21, 84, 230 and 652. Original CPU winners
and the recorded ANE top logits agree within the declared backend tolerance.
INT8 preserves all four winning tokens. Raw projection relative L2 error ranges
0.269–0.447%; clipped-logit relative L2 error ranges 0.119–0.380%. Original winner
margins are 14.58–23.56 logits. In the 652-token case, the fifth-ranked alternative
changes from token 236771 to 236743; the winning token remains 236778.

These are four high-margin states, not evidence about close decisions or
multi-token quality. No INT8 head graph was loaded or evaluated on ANE, and no
performance claim follows. The default head remains FP16. The test completes
with zero ANE compiler calls and zero accelerator evaluations.

Evidence: `gemma4-12b-evidence-20260914/int8-head-host-quality/`, including exact
capture inventory, source, frozen signed release test binary, per-tile source
hashes, every comparison and the CPU-only test log. The combined INT8 source
blobs would contain 1,007,160,320 bytes; this is source storage, not measured
runtime residency or bandwidth.

## Matched-power QKV timings and next decision

After thermal state returned to nominal, the frozen QKV probe completed three
fresh strict-cache runs per layer. All six runs shared sampled AC power, Low
Power Mode enabled, pmset mode 1 and nominal thermal state. They compiled
nothing, passed the backend checks, left no owned staging paths and retained
the same boot. Each run has six ABBA/BAAB pairs against original FP16 and six
against FP16 reconstruction of the same INT8 weights: 144 trials and 72
eligible pairs total. Eligibility concerns sampled controls, not constant
clocks or absence of external contention.

Median paired baseline/candidate wall ratios are below. A value above one
favors INT8. These are component observations, not whole-model speedups.

| Layer / control | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| Sliding / original FP16 | 1.130 | 1.112 | 1.186 |
| Sliding / reconstructed FP16 | 0.895 | 1.270 | 1.365 |
| Global / original FP16 | 1.103 | 0.639 | 0.548 |
| Global / reconstructed FP16 | 1.057 | 0.572 | 0.608 |

Individual pair variation is substantial: the first sliding run ranges from
0.818 to 2.205 against original FP16. The global candidate is slower in every
pair of both later runs against both controls. CPU instruction ratios remain
near one while CPU cycle ratios also move; those are host counters and do not
normalize ANE time. The data do not support blanket INT8 QKV promotion.

**Decision:** keep global packed QKV FP16. Sliding INT8 remains promising but
still needs representative full-model quality and performance evidence.
The next bounded global experiment should separate its 8192-row query matrix
from the 512-row shared K/V matrix: INT8 Q plus original FP16 K/V, both using
the already qualified single-I/O linear path. This reuses the promising query
shape and removes additional K/V quantization error, at the cost of another
submission. It is a hypothesis; neither the output-channel count nor a
particular compiler lowering has been established as the cause of regression.
Qualify the concatenated result against an exact CPU reconstruction and compare
its complete two-submission latency with original packed FP16 before expanding.

Raw results, per-run summaries and cleanup checks are under
`int8-qkv-real-probe/time-layer-*-nominal-*`; the compact aggregate is
`int8-qkv-real-probe/timing-repetitions-summary.json`. The frozen signed probe
is SHA-256 `5bc44687c51bdce1b1ac0ce50a940f0f0ce6e47f8b60e30df5bcaa39e815ab84`.

## Split global query and shared K/V

The isolated layer-5 candidate now uses two existing single-I/O linear
requests: INT8 Q with 8,192 rows, then the original FP16 shared K/V with 512
rows. The baseline remains one packed FP16 projection with 8,704 rows. No new
multi-I/O request or full-model route was introduced. Refactoring the capture
loader preserved the earlier complete CPU QKV report exactly; query
quantization errors also match the earlier packed INT8 query errors, and the
mixed reconstruction leaves shared K/V unchanged.

Qualification used two compiles, nine evaluations and three successful
unloads. All three saved inputs pass the backend tolerance, and the split
FP16 K/V outputs are bit-identical to the packed baseline. Owned staging paths
are absent and boot identity is unchanged. INT8 query execution error versus
its own CPU reference is 0.0332–0.0343% relative L2. These are component
numerical results, not full-generation quality evidence.

The first strict-cache timing run completed six eligible ABBA/BAAB pairs under
AC power, Low Power Mode enabled, pmset mode 1 and nominal thermal state.
The median paired baseline/candidate wall ratio is 1.066, ranging from 0.559
to 1.435. Median complete projection times are 0.742 ms packed and 0.675 ms
split; the candidate includes both submissions. The spread and single run
do not justify promotion. The next process compiled nothing but rejected
Fair thermal state before warmup, producing no timing trials. CPU cycles
remain host counters, not ANE clock normalization.

Evidence: `gemma4-12b-evidence-20260914/int8-global-split/`, including unchanged
CPU report verification, both source files, lifecycle journal, per-run power
samples, cleanup receipts and `timing-summary.json`. The immutable signed
probe SHA is `0b1741140318c7b91db5f9e03d1331b1c626f9c0726cdedbfb2b42d279140592`.
The production global projection remains packed FP16. Further repetition is
useful only when nominal controls permit a meaningful comparison.

## Actual vocabulary tile on ANE

The existing single-I/O linear constructor now qualifies vocabulary tile zero
(rows 0–16,383, input width 3,840) with the same four authenticated states as
the complete CPU vocabulary audit. The shared capture-loader refactor produces
identical normalized-input hashes and receipts for all four states; original
and reconstructed tile hashes and source-byte counts match the earlier audit.
Three focused host tests pass; the full-checkpoint host audit remains explicitly
ignored in ordinary test runs.

Original FP16, INT8 and reconstructed FP16 all pass the backend tolerance on
all four inputs. INT8 relative L2 execution error versus reconstruction is
0.0297–0.0311%, separate from quantization error. All eleven recorded top logits
that belong to this tile agree with original FP16 within tolerance. The
within-tile winners are unchanged, but the 652-token state's actual winning
token belongs to tile 14, so this is not a full-vocabulary ANE ranking result.

The qualification journal records two compiles, twelve evaluations and three
successful unloads; owned staging is absent and boot is unchanged. The first
strict-cache timing process repeats the numerical checks with zero compiles,
then rejects Fair thermal state before warmup. It produces no timing pairs,
so no ANE head speedup is established and the production head stays FP16.
Independent rootless power checks immediately afterward and after a quiet
45-second interval also report Fair thermal state. No additional timing work
was launched from those checks.

Evidence: `gemma4-12b-evidence-20260914/int8-head-device/`. The immutable signed
probe SHA is `63f6859d8a42f94082370464815462629817465b84ce60ecd3d4c41da4b0ba00`.
Its `intent.json` pins the executable, test and input records. Set
`RVLLM_INT8_HEAD_TILE=0`, `RVLLM_INT8_HEAD_DEVICE_MODE=time` and a new
`RVLLM_INT8_HEAD_DEVICE_OUTPUT` to repeat timing under eligible controls;
`RVLLM_GEMMA4_MODEL_DIR` and the JSON `RVLLM_INT8_HEAD_STEPS` array come from
that intent. Timing forbids the driver journal and all compilation. The
separate `restore-original` mode allows at most one compile and zero
evaluations if a future strict baseline load establishes a cache miss.
