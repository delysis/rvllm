# Independent W4/W8 donor-schedule operator repeat

The serial queue completed independent campaign
[`g4-donor-r4sg8k8-confirm-04`](g4-donor-r4sg8k8-confirm-04/campaign.json)
without manual stage advancement. Both strict-math Metal compilations and
native oracles succeeded; the queue's [compile](g4-donor-r4sg8k8-confirm-04/advance-compile-receipt.json)
and [oracle](g4-donor-r4sg8k8-confirm-04/advance-oracle-receipt.json)
continuations submitted the next jobs. The oracle receipts are
[W4](g4-donor-r4sg8k8-confirm-04/w4-oracle-receipt.json), SHA-256
`86f1435aed5a19d5c00d8bd475bee7f0929ec84b069700e0187385b49af2541c`,
and [W8](g4-donor-r4sg8k8-confirm-04/w8-oracle-receipt.json), SHA-256
`5e4c63307c965c27ffda44b1c76435c171f60c2de1732cf1df21177eae6d0cbd`.
Both report the actual 32-row, 256-thread, `[120,1,1]` launch, the same
strict-math metallib identities as confirmation `03`, all full synthetic
shapes within the scalar-FP64 bound, exact repeated results, untouched
guards and arena, and exact dispatch accounting. There were zero source
compilations in the timing samples.

The paired operator screens used 40 GPU-timestamp samples in ten balanced
ABBA/BAAB blocks, 100 actual operator iterations per sample. A is the rvLLM
group-32 BF16 N4 control; B is the donor-schedule candidate. The within-block
A/B ratio remains in the same direction as [confirmation `03`](DONOR_R4SG8K8_CONFIRM_03.md),
but every control-drift check again exceeds the referee's 5% limit.

| Candidate | K | Campaign | A median ms | B median ms | Paired ratio median | Ratio range | Control drift |
|---|---:|---|---:|---:|---:|---:|---:|
| W4 r4/sg8/k8 | 15360 | 03 | 0.241354 | 0.094493 | 2.558x | 2.492–2.774x | 17.94% |
| W4 r4/sg8/k8 | 15360 | 04 | 0.246223 | 0.094456 | 2.604x | 2.558–3.289x | 58.62% |
| W8 r4/sg8/k8 | 4096 | 03 | 0.096845 | 0.029666 | 2.956x | 2.422–3.973x | 102.52% |
| W8 r4/sg8/k8 | 4096 | 04 | 0.075609 | 0.026646 | 2.861x | 2.713–4.743x | 122.15% |
| W8 r4/sg8/k8 | 8192 | 03 | 0.156686 | 0.061590 | 2.636x | 2.453–2.984x | 46.66% |
| W8 r4/sg8/k8 | 8192 | 04 | 0.154863 | 0.059825 | 2.609x | 2.494–2.806x | 23.06% |

The `04` raw timing receipts are
[W4 K15360](g4-donor-r4sg8k8-confirm-04/w4-timing-k15360-receipt.json)
SHA-256 `49b1f209f6d5182f368ae2f9037f45b0cb3ff05dde5d51375b2e8678b7fe5158`,
[W8 K4096](g4-donor-r4sg8k8-confirm-04/w8-timing-k4096-receipt.json)
SHA-256 `966d022ad585684caf1e3c5da43f8764c4613a6b420724ebebce5215938cd2ce`,
and [W8 K8192](g4-donor-r4sg8k8-confirm-04/w8-timing-k8192-receipt.json)
SHA-256 `870d3c8a19253a931c9fb564025ddd759aa0e3bb3cc6feb52fdd057a26f1fa2a`.
All six `03`/`04` timing receipts correctly say `timing_eligible:false` and
`promotion:false`. The repeated paired ratio is a useful *operator research*
signal despite changing host conditions, not a qualified speedup, full-route
result, or MLX comparison. Further repetition without a new experimental
question has diminishing value; the next useful gates are real checkpoint
weights, selected full routes, quality, and comparative end-to-end timing.
