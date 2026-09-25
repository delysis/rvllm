# Gemma 4 native-BF16 low-bit seven-role campaign

Date: 2026-09-25

This is a real-checkpoint **projection-operator** campaign. It is not evidence
for checkpoint quality, a full model route, ANE execution, or promotion.

The campaign covers layer-0 Q, K, V, O, gate, up, and down projections. Every
job evaluates both `W4ABF16` and `W8ABF16` at decode-like `M=1` and bounded
prefill `M=4`. Activations, dense reference weights, and outputs remain BF16;
group-32 scales are FP16 and accumulation is FP32.

For each role, the referee completes the CPU low-bit reference comparison,
dense-BF16 reference comparison, two exact candidate correctness dispatches,
bitwise repeatability check, output-guard check, and dispatch-ledger check
before timing begins. A three-block ABBA screen is followed only after success
by a separate nine-block ABBA confirmation. All fourteen jobs are serialized.

The first Metal job depends on
`gemma4-ane-stacked-baseline-exact-v2-12-a6-stacked-20260924`, so this campaign
cannot interrupt or interleave with the active ANE sequence. `stable_seconds`
is zero. Thermal state, power mode, and observed competing processes are
recorded without selecting a preferred stratum; exploratory observations are
retained even when sampled conditions change. The existing queue still treats
AC power, a working observer, sufficient disk, and valid hardware-control
readings as launch prerequisites.

`campaign.json` seals the executable, model configuration, validator, complete
matrix, ordering, and source manifests. The independent validator rejects a
successful process unless all four cases have the requested role and ABI,
exact dispatch counts, intact guards, repeated output bits, finite reference
errors, complete positive ABBA samples, and SHA-256 identities.

Status at staging: all fourteen manifests were accepted by the existing
experiment queue and are waiting behind the ANE tail. Results belong under
`reports/gemma4-global-decode-local-20260924/queue/results/<job-id>/`; every
attempt is preserved by queue identity and is never silently replayed.

Recreate the source manifests with:

```sh
python3 tools/stage_gemma4_low_bit_bf16_campaign.py
```

