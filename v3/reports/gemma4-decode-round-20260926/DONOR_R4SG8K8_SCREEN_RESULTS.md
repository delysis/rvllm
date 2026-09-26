# Donor-schedule W4/W8 operator screens, first campaigns

Both source-bound native oracles passed on the M4 Max with synthetic full
operator shapes. W4 covered dense Down `M1,N3840,K15360`; W8 covered Output
`M1,N3840,K4096` and `K8192`. Each case matched its independent scalar-FP64
bound, repeated bit-exactly three times, retained surrounding guards, and
exercised four host refusals plus three encoded shader-guard negatives. These
are not dense checkpoint-weight, full-route, or model-quality tests.

The [W4 oracle](g4-donor-w4-r4sg8k8-02/oracle-receipt.json) SHA-256 is
`764c525bbab33b84bc99b55cb607b8e0616a8b5d2d721425c5bdbd8d0acacbd2`;
the [W8 oracle](g4-donor-w8-r4sg8k8-01/oracle-receipt.json) SHA-256 is
`ea7c1e598be2baeaff17ab0463c1553cec5add4fcdc06478f9ac4f615540671b`.
Both queue jobs succeeded with unchanged inputs. Their strict-math compiler
receipts reproduce the preflight metallib hashes, W4
`5bbda735656a5502a3c7f7394f0692f8f74ce53ffa1aad089a1a674fd14747af`
and W8
`e8f3cf094d2b7ace1022fc98067620ee613645d66ced90e2fca14565ddceb777`.

Paired GPU-timestamp operator screens used ten balanced ABBA/BAAB blocks,
100 operator iterations per sample. A is rvLLM's group-32 BF16 N4 control;
B is the donor-schedule candidate. The table reports all-block medians in
milliseconds per iteration and the median of the ten *within-block* A/B
ratios, not a throughput claim for a full model.

| Candidate | K | A median ms | B median ms | Paired ratio median | Ratio range | Control drift |
|---|---:|---:|---:|---:|---:|---:|
| W4 r4/sg8/k8 | 15360 | 0.23777 | 0.09431 | 2.520x | 2.417–2.599x | 7.09% |
| W8 r4/sg8/k8 | 4096 | 0.10463 | 0.03326 | 2.272x | 1.748–4.040x | 136.31% |
| W8 r4/sg8/k8 | 8192 | 0.15501 | 0.05881 | 2.650x | 2.486–3.684x | 79.78% |

Raw receipts: [W4 K15360](g4-donor-w4-r4sg8k8-02/timing-k15360-receipt.json)
SHA-256 `7d5631d017aecdbc2e491e29711e57c1b797a8fed5dcba01ff2a5a6e9af100f1`;
[W8 K4096](g4-donor-w8-r4sg8k8-01/timing-k4096-receipt.json)
SHA-256 `c578173b82bf232c883985abe64ddd7859bbe77e4bce9643585cb373b3173308`;
[W8 K8192](g4-donor-w8-r4sg8k8-01/timing-k8192-receipt.json)
SHA-256 `72c7da1e6fae82a35333d88c91673a00c75ff1dbb7ce190dd52ee8ecee704745`.
W8 used an isolated debug host executable, W4 a release executable. Thus
even the operator controls are not a cross-W4/W8 precision comparison.

**Evidence correction.** The first-campaign oracle and timing `identity`
objects incorrectly report the *old* 16-row/64-thread/240-group launch for
both new candidates. The new shader and host dispatch use 32 rows, 256
threads, 120 groups; the per-kernel PSO entry in the same receipts says 256
threads, exposing the inconsistency. Numerical and ledger checks did run,
but the launch metadata is not a valid exact-dispatch receipt. Additionally,
every screen exceeds the referee's 5% control-drift limit (W8 substantially),
and the timing receipts mark `timing_eligible:false` and `promotion:false`.
These are promising **exploratory** signals, not qualified speedups.

The source now derives receipt geometry from the candidate's actual dispatch
contract and rejects a mismatched oracle before timing generation. A fresh
campaign must re-establish correct identity and independent timing; then
dense real-weight/full-route testing decides whether the schedule helps Gemma
4 inference and how it compares with MLX. The old receipts remain unmodified.
