# SG8 global-attention sustained decode at 1024-token context

The serial queue ran one donor-global-on and one donor-global-off arm on the
same real BF16 Gemma 4 12B model path and the same 1024-token synthetic prompt.
Both retained the donor projection, gate, and local-attention components;
only global attention changed. Unlike the earlier two-token timing screen,
both arms kept the inference CLI's **default prefill policy** and generated
64 tokens. The queue manifests seal exact commands and declared inputs.

| Arm | Prefill ms | Decode ms, 64 tokens | Decode tokens/s | Donor global-attention dispatches |
| --- | ---: | ---: | ---: | ---: |
| SG8 global on | 47,686.20 | 5,373.54 | 11.910 | 512 |
| SG8 global off | 47,904.78 | 42,983.66 | 1.489 | 0 |

The single-pair sustained-decode ratio is **8.00x**. Both queue jobs succeeded
in the same sampled power/thermal stratum, with eligible conditions, no queue
violations, and zero reported inference library/pipeline compiles. Both
recorded identical donor counts for the fixed components: 2560 local
attention, 3072 gates, and 9216 projections. Both generated arrays contain
exactly 64 copies of token ID 236770, so they match completely. This repetitive
`a a a ...` prompt is appropriate for a fixed-length speed screen but is a
**weak numerical/semantic quality challenge**. There is no independent
teacher-forced logit, per-layer, diverse-prompt, or checkpoint-quality gate.
This pair is exploratory rather than an independently confirmed speed or
promotion result.

For orientation only, the earlier retained MLX-LM BF16 benchmark reported
6.276 sustained decode tokens/s at nominal 1024-token context over 64 tokens
(`../../gemma4-rvllm-good-enough-matrix-20260924/README.md`). The 11.910
tokens/s on arm is numerically about **1.90x** that rate. The runs were not
interleaved and did not seal identical MLX prompt tokens, model-file identities,
or host conditions; this does **not** establish rvLLM superiority or parity
with MLX. A same-workload matched framework run remains required.

The mechanism inference stays limited: the control's Gemma global D512 heads
fall back to one-thread-per-head serial-KV attention, while SG8 maps one head
over eight SIMD groups and merges FP32 sufficient statistics. The paired
two-token ABBA result in `../global1024-01/` supplies a separate repeat
control. Neither receipt isolates per-kernel GPU time or the first internal
numeric difference.

The queue-owned executable SHA-256 is
`bf149a5bfd2835fe8668008988f24b69f0331526750cff5712340112e9ec3dc5`;
the metallib SHA-256 is
`21d81a66d31075e9307b64b7cb8eb0eea1f1dee74c0865ff3d0fa8125579dd06`.
The executable predates the later diagnostic numeric-ABI mask fingerprint
repair; these are direct sessions, not imported/shared KV qualification.
Complete unpruned queue receipts are in `queue-results.tar.gz` (SHA-256
`69f1de685f7cf3b50c1b8551123d0a71509537593328b258083370cb08c0fb10`).
