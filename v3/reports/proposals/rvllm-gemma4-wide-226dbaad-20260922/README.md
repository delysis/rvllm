# Second-batch proposals — staged, not submitted

Source lineage: `226dbaad8dec9bdd3b9684a224747384b6e47319`.
Integration base: `faa753360e30ce59540d1d113994c711a1d19675`. Six explicit runtime candidates
and one blocked source-only ANE I/O declaration. These are additive to the
original six experiments, not replacements or promotions.

## Local sequence

Apply the complete rebased integration series rather than layering the older
226dbaad source packet over it. The local owner checks/formats only the 24
explicit Rust paths in `tools/gemma4_candidate_rustfmt.paths`, then runs the
single expanded `tools/check_gemma4_candidate_delivery.sh` gate. That gate
contains the original eight plus six new Metal compile/link arms (14 total),
all candidate source hashes and the newly required native host test filters.

`tools/run_gemma4_python_checks.py` runs the installed CPU design models and
orchestration/staging contracts. `tools/run_gemma4_candidate_ci.py` compiles and
runs exactly 45 reviewed public Rust tests and exports fourteen source files;
it does not compile Metal or run a device. The checked-in CI workflow runs these
on Ubuntu, fetches locked dependencies separately, uses no private Apple feature,
and preserves failures. A missing or unexpected test name fails the Rust suite.

`tools/gemma4_wide_proposals.py THIS_DIRECTORY` validates only these inert
proposals. `--render-compile-plan /absolute/v3 /absolute/target /absolute/new-output`
prints one invocation of the expanded native delivery gate; it executes nothing.
It does not print an additional six-arm pipeline that would duplicate that gate.
A plan, a Python model, a public Rust test and native compilation are distinct
levels of evidence. No host result supplies device/tensor/performance acceptance.

After native compilation, run candidate-specific component oracles and only
then the existing prefill/full-continuation checks. The old projection layer
trace disables the new prefetch and RMS routes: do not call a traced fallback
a candidate oracle. Target component adapters remain a local integration gate
where the existing fixture does not select the new entry point.

Full-route timing is four blocks A/B/B/A, each with two warmups and seven
measured requests. Pin an unchanged 84-token/10-output reference: nine ANE steps
per request, 208 evaluations per step,36 requests and 67,392 evaluations per
complete campaign. Warmups are recorded but excluded from estimates. Control
drift greater than 5%, changed strata/work, any failed oracle, or nonpositive
paired gain rejects the screen. Require an independent confirmation before
considering promotion. All source costs and dispatch counts are hypotheses or
static accounting, not measured speed or residency.

## Queue bridge

The seven `.json` files use `rvllm.kernel_proposal.v1`, deliberately not the
live queue schema. `native-job.template.json.in` follows the repository's
`rvllm.experiment_job.v1` field structure but contains null required identities
and conditions; native deserialization/validation must reject it. Do not submit
it. The local owner must create fresh IDs and populate exact paths/hashes,
args, source/input/oracle pins, observed power controls, a current quiet-process
policy, and dependencies. Empty process lists here are unfilled worksheets,
not permitted exemptions. Never reuse stale PIDs or edit attempted jobs.

Create separate preparation/component/full-route/timing jobs. Preparation is
not a performance result. A/B/B/A dependencies enforce order but do not confer
statistical acceptance. Do not clear STOP or start workers from this packet.
The packed32 declaration has no device selector and must stay outside the
live queue until compiled descriptor facts and a separate boundary review exist.

## Independent controls

Metal candidates use `RVLLM_METAL_PREFILL_GEMM=mma32` and
`RVLLM_METAL_PREFILL_ATTENTION=simdgroup`, with `RVLLM_METAL_RESEARCH=off`
for their primary control. Short-MMA, rounded-gate and GQA remain separate
secondary comparisons. ANE down4 and transpose-flags compare to ordinary
`static-int8-ffn-cached`, never FP16 FFNs. Host pruning uses the same cached ANE
plan and explicit baseline/prune head selectors, with timing enabled on both
sides. Do not combine candidates until each independent effect is understood.

Safety constraints in `AGENTS.md` and `v3/HANDOFF.md` remain intact. No changes
to shipping/private gates, multi-I/O quarantine, numerical tolerances, ignored
fixtures, old shaders, active queue files, attempted evidence or power settings.

The integrated gate checks the exporter before each arm, retains the hashes
established immediately after each binary build, and verifies Rust/manifest and
six candidate shader input hashes before final success. These are consistency
checkpoints, not a lock or transitive source attestation. The checkout and shared
Cargo target still need a quiescent local owner. Staging tests ensure the renderer
uses this one gate and does not silently resurrect the superseded command plan.
