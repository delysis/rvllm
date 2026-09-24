# Gemma 4 Metal and ANE candidate campaign, 2026-09-23

## Evidence boundary

This is a preparation/correctness campaign on Apple hardware, not a promotion
campaign.  It used checkout `360a00fc1ef3435acb16f798b6d56b6dbe9f13d5`,
Gemma 4 12B revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`,
and `rvllm_disaggregated_infer` SHA-256
`5c604784063a4cb82a95c9adc54ba81496781fe625228e527ffa28bd4b50dbf5`.
The host was on AC power, low-power mode enabled, `powermode=1`, and reported
nominal thermals.  The queue jobs declared `purpose=preparation`, so their
latencies are exploratory observations only.  No ABBA timing campaign,
candidate tensor oracle, independent confirmation, or promotion occurred.

The native delivery gate passed after correcting the Rust module declaration
order required by the checked-in rustfmt manifest.  Its result remains exactly
`compiled-only; no accelerator acceptance`.  It compiled and linked the BF16
and F16 Metal 3.1 libraries for the control and all ten Metal candidates and
ran the reviewed host suites, including 15 ANE candidate contract tests.

## Metal prefill screen

Every row below is one fresh-process prefill-only run.  `correct` means the
first generated token exactly matched the pinned CPU reference.  `dispatches`
lists the nonzero candidate-family encoded-dispatch counters.  Complete and GPU
times are single observations, not speed estimates.  The short workload has 21
prompt tokens; the long workload has 84.  Long-workload candidates cannot be
compared to the short control.

| selector | workload | correct | nonzero dispatches | complete ms | GPU ms | vs short control |
|---|---:|:---:|---|---:|---:|---:|
| `off` | 21 | yes | none (control selection exercised) | 1839.875 | 1064.114 | control |
| `metal-short-mma16x64` | 21 | yes | GEMM 144, QKV 48 | 2625.902 | 1848.388 | +42.7% |
| `metal-rounded-gate32` | 21 | yes | rounded gate 48 | 1518.644 | 1094.860 | -17.5% |
| `metal-gqa-kv8` | 84 | yes | d256 40, d512 8 | 2194.191 | 1639.329 | not comparable |
| `metal-mma32-prefetch` | 21 | yes | GEMM 144, QKV 48 | 1512.344 | 1018.195 | -17.8% |
| `metal-attn-q4` | 84 | yes | d256 40, d512 8 | 3902.144 | 3090.825 | not comparable |
| `metal-rms-simd32` | 21 | yes | RMS 96 | 1796.708 | 1119.573 | -2.3% |
| `metal-mma32-f32` | 21 | yes | GEMM 144, QKV 48 | 2212.305 | 1418.824 | +20.2% |
| `metal-long-mma32x64` | 84 | yes | GEMM 144, QKV 48 | 3879.114 | 3045.085 | not comparable |
| `metal-mma32-load4` | 21 | yes | GEMM 144, QKV 48 | 1269.277 | 342.829 | -31.0% |
| `metal-rmsnorm-simd256` | 21 | yes | RMS 96 | 2207.388 | 1469.895 | +20.0% |

The screen makes `metal-mma32-load4` the strongest timing-campaign candidate,
with `metal-mma32-prefetch` and `metal-rounded-gate32` also worth controlled
measurement.  It is evidence against spending immediate timing capacity on
`metal-short-mma16x64`, `metal-mma32-f32`, or `metal-rmsnorm-simd256`.  These are
prioritization statements, not causal speed claims: process startup, host work,
DVFS, and order were not controlled by an ABBA design.

Raw receipts are in
`v3/reports/gemma4-12b-evidence-20260914/kernel-game-campaign-20260923-screen-queue`
and its `-1` continuation.  The first queue was explicitly stopped between
jobs after entries 00--05, preserving its STOP record; entries 06--10 ran in a
fresh queue.  SHA-256 values for the eleven `prefill-result.json` files, in
table order, are:

```
b00fecc6240c3c82a7b889a7877ce9c5a0895635a03c9aabb02fe999b5028ad3
9035eead91ba3a82cbccc6889ce4c728bfacb1bc19425219f11c1696b1ce5f51
5014e7ce25bfff425de23731779c57b7b8b39c693a191944464fbacf36681221
80f963c2301f2f3cd5cdd5bdeb07abac2d3ecb6f704dab69cf8fbccdee9d5997
059f89df91d5982bbadede4a250e0b200d5923d2efbb36b80b1b64402bb6f36e
6b37a1f5b98f26771bf7c87f7ecaaa298af3c490929861d289c47e9daa9e8ec7
a3dd1ea1e494d73a1bdf81e411707e871812c67f1cc8ba2ac8532ff91106ff57
aadfc4de7803ce339f9b04f520e7c3f19406e950a9d7047bd9a1051f25acf771
9017b39716358fa56a06689b8c3d36c525a74b912af743e94fe2973b1f7e57ab
f80ca65af4f4702bd5f49cc1a930c4405d702f212c358d4b7e3dbd207def21eb
41a8348679bb849a9cba9958f1853968d4d9b669b2e29417b468bf7fcc928e3e
```

## ANE full-route screen

Each plan was attempted with the same pinned two-token `Paris` reference,
control Metal prefill, and `--ane-compile-budget 0`.  Missing cache entries were
not repaired.  The baseline, `chunk4`, and `down4` completed one real ANE decode
step, generated `[50429, 106]`, matched the full reference, reported ANE
execution verified, and used zero compiler calls.

| ANE plan | result | prepare ms | decode-step ms | interpretation |
|---|---|---:|---:|---|
| `static-int8-ffn-cached` | pass | 53967.260 | 382.862 | zero-compile full route |
| `static-int8-ffn-sliding-qkv-tiles4-cached` | blocked | -- | -- | compiled cache absent |
| `static-int8-ffn-sliding-qkv-cached` | blocked | -- | -- | compiled cache absent |
| `static-int8-chunk4-ffn-cached` | pass | 32763.277 | 437.080 | zero-compile full route |
| `static-int8-down4-ffn-cached` | pass | 31240.085 | 463.877 | zero-compile full route |
| `static-int8-interleaved-ffn-cached` | blocked | -- | -- | compiled cache absent |
| `static-int8-ffn-transpose-attention-cached` | blocked | -- | -- | compiled cache absent |
| `static-int8-stacked-ffn-cached` | blocked | -- | -- | compiled cache absent |

The three successful runs establish this narrow reference and route only; one
decode step is not broad numerical qualification.  Their one-shot decode
latencies suggest no advantage for `chunk4` or `down4` over the baseline, but
the jobs were preparation runs and some queue envelopes observed transient
competing-process violations.  Therefore these values must not be used as a
speed score.  The five cache-blocked plans have no device correctness or speed
result from this campaign; their host contracts passed in the delivery gate.

Successful inference-report SHA-256 values are baseline
`55083ba912f83b7548a1ba3f205e38e86430c82c330dc8be4959521f1bb624af`,
chunk4 `801d794f390e33a0cbe1db6ed5e8bdea21982510f8aaf158f07585f92b9327c3`,
and down4 `ff5b7e23fc091ee3833a083acae836bd3d3ec6948a9166aedfacac3baca712ed`.
Raw successful and failed attempts are under the `...ane-screen-queue` and
`...ane-independent` directories beside the Metal receipts.

## Next experiments

1. Seal and run at least five valid counterbalanced pairs for the short control
   versus `metal-mma32-load4`; require exact output, exact dispatch, zero compile
   calls, a single sampled stratum, and the 5% control-drift ceiling.
2. If capacity remains, repeat for `metal-mma32-prefetch` and
   `metal-rounded-gate32`.  Obtain a matched long-workload control before making
   any speed statement about GQA, attention-Q4, or long-MMA.
3. Run independent tensor/original-driver oracles before treating token equality
   as numerical acceptance.
4. Re-provision missing ANE caches only as an explicit preparation campaign,
   preserve compiler-call receipts, then rerun full-route correctness at zero
   compile budget.  Do not reinterpret cache absence as a candidate failure.
5. For ANE speed, use warm repeated requests and the repository's paired
   comparator.  The present cold prepare and single decode observations are not
   a timing campaign.

No selector was promoted and no production default changed.
