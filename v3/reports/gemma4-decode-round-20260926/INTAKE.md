# Gemma 4 Metal decode round intake — 2026-09-26

## Identity and scope

- Integration branch: `astra/gemma4-load4-tiles-20260923`
- Integration parent: `5dacd6bf1f316fe3e9612c14c8aa75806bdcb088`
- Pro patch base: `a52bdf46627219561efa6610e88ffd5c37a821d9`
- Chat Pro job: `rvllm-gemma4-metal-next-route-20260926-v1`
- Delivered patch SHA-256: `9800ee4f20605b3d70984442d2ddbd61448464502957f700ade12112e2d6fc37`
- Delivered source archive SHA-256: `986155ae8694c3429a306e62441ec0c1ad3bc767810e4e6052bf90c31dc53d46`
- Delivered package SHA-256: `661ea9e1aa28bd8e18bde0c32c3a467dcd5e3df8bf845d514d84f401006958e2`

The patch adds four default-off research selectors: fused BF16 FFN, W4 group-32
QMV, W8 group-32 QMV, and a bounded D=512 short global-attention decode arm.
No selector is promoted by this intake.

Independent application review found all thirteen modified preimages identical
to the Pro base, all fifteen new paths absent, and no overlap with intervening
commits. A semantic review found that the initial QMV integration could also
intercept prefill. Integration now requires `MetalPhase::Decode` in both
low-bit route helpers, preserving incumbent prefill behavior.

## Local qualification

The exact integration tree passed:

- Rust formatting and `git diff --check`;
- offline locked library type checking;
- 159 library tests, with 27 explicitly ignored device tests;
- focused decode, attention, evidence, catalog, queue-generator and report tests;
- the instrumentation-feature type check;
- strict Metal 3.1 compilation and metallib linking for all four generated
  candidate sources.

The checked-in workspace retains pre-existing warnings in unrelated crates;
this is not a claim of workspace-wide warning cleanliness.

## Native queue campaign

Campaign `g4-next-live-01` uses the existing persistent experiment queue and
the shared accelerator lock. The exact release test executable, generator,
retainer, compiler, source and metallib identities are sealed into receipts.
All five compile jobs and all four native correctness oracles succeeded.

Thermal state, power mode, low-power mode and competing processes are recorded
but unconstrained. Every manifest uses `stable_seconds: 0`; no stability dwell,
sample pruning or automatic retry is permitted. AC is retained as an analysis
stratum. Measurements below retain forty samples across ten alternating
ABBA/BAAB blocks.

| Candidate / cell | Baseline ms | Candidate ms | Ratio | Drift | Status |
|---|---:|---:|---:|---:|---|
| BF16 fused FFN, K=3840 | 1.0038 | 0.5766 | 1.741x | 1.78% | passed screen |
| short global attention, L=256 | 15.5139 | 0.5077 | 30.559x | 0.73% | passed screen |
| short global attention, L=512 | 30.9743 | 1.0138 | 30.552x | 0.55% | passed advancement |
| W4 down QMV, K=15360 | 0.2423 | 0.1846 | 1.313x | 13.90% | promising, drift-inconclusive |
| W8 output QMV, K=8192 | 0.1558 | 0.1411 | 1.104x | 5.99% | promising, drift-inconclusive |
| W8 output QMV, K=4096 | 0.0832 | 0.1002 | 0.830x | 134.10% | likely reject; order-sensitive |

The FFN paired-block median ratio is 1.743x with bootstrap diagnostic interval
`[1.735, 1.750]`. The L=256 attention paired-block median is 30.529x with
interval `[30.506, 30.615]`. These are comparisons to rvLLM incumbent operator
routes, not MLX or end-to-end model ratios. Both remain screening evidence
until real-checkpoint/full-route qualification and independent confirmation.

The predeclared L=512 attention advancement also passed: paired-block median
30.552x, bootstrap diagnostic interval `[30.539, 30.568]`, agreement across
order strata, and 0.55% control drift. The near-identical L=256 and L=512
ratios are encouraging scaling evidence within the candidate's deliberately
bounded capacity, but remain operator-only comparisons against rvLLM.

## Next decisions

1. Retain the FFN and short-attention arms for real-checkpoint/full-route work.
2. Confirm W4 and W8 K=8192 in new independent campaign IDs; never overwrite or
   retry these drift-inconclusive receipts.
3. Reject or redesign W8 K=4096 unless generated-code evidence identifies a
   correctable shape-specific defect.
4. Compare qualified routes against the same-operation MLX captures before any
   production-default change.
5. Do not infer ANE qualification from any Metal result in this report.

## Independent low-bit confirmation

Fresh campaign `g4-next-confirm-02` rebuilt both low-bit candidates, reran their
native oracles, and collected new ABBA/BAAB samples under new immutable job IDs.
It did not replay, overwrite, or selectively discard the first campaign.

| Candidate / cell | Baseline ms | Candidate ms | Ratio | Drift | Status |
|---|---:|---:|---:|---:|---|
| W4 down QMV, K=15360 | 0.2455 | 0.1859 | 1.321x | 38.61% | repeated speed signal, drift-inconclusive |
| W8 output QMV, K=8192 | 0.1565 | 0.1396 | 1.122x | 3.20% | passed independent screen |
| W8 output QMV, K=4096 | 0.0768 | 0.0798 | 0.962x | 117.22% | reject/inconclusive; order-sensitive |

For W8 K=8192, the paired-block median ratio is 1.120x with bootstrap
diagnostic interval `[1.115, 1.128]`; ABBA and BAAB strata agree. This is now a
prospective operator winner, still pending real-weight/full-route and external
framework comparison. W4 reproduces the approximate 1.3x advantage, but the
incumbent control is too unstable in both campaigns to qualify the result. Its
next experiment should change the sampling design or diagnose that control,
not repeat the same campaign until a favorable drift result appears.
