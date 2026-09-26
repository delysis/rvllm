# Donor QMV real-checkpoint BAAB repeat

The serial Apple experiment queue executed these two immutable jobs against the
same pinned layer-0 Gemma 4 12B BF16 tensors and the same `research-r4-sg8-k8`
executable as round 01. Unlike round 01, each job included the exact research
receipt validator in the queue. Both jobs and both validators exited zero;
the queue preserved execution conditions, stdout, validation output, and full
reports under `results/`. There was no stable-condition dwell or quiet-process
gate. The queue daemon returned to idle after both jobs.

| Cell | Candidate vs quantized CPU reference | Native BF16 vs dense CPU reference | Native / candidate median wall time | Ratio |
| --- | --- | --- | --- | --- |
| W4 down `[3840,15360]`, M1 | max abs 0.00390625; relative L2 0.0008470 | max abs 0.001953125 | 0.997000 / 0.310958 ms | 3.206x |
| W8 output `[3840,4096]`, M1 | max abs 0.00390625; relative L2 0.0008003 | max abs 0.000061035 | 0.328458 / 0.173500 ms | 1.893x |

Each receipt records two exact correctness dispatches, 19 timing dispatches,
bitwise repeatability, unchanged output guards, packed-weight and activation
hashes, source-tensor hash, generated-MSL hash, and the exact candidate kernel.
The pinned executable SHA-256 is
`8f2f93b98bcbc76ff6c8079703ebcf9778c1ffdc6f05495e065eaffe6788467b`;
the validator SHA-256 is
`0857816135c4fb576ee6007c50124452997df71ec9f5b1ff1f155bfa3f99dcc0`.

The earlier ABBA run and this BAAB repeat agree in direction (W4 2.876x vs
3.206x; W8 1.946x vs 1.893x), but absolute W8 times changed by roughly 2x.
These are wall-clock, hot-cache, single-operator comparisons against a
**dense BF16** projection, not against a same-format incumbent low-bit kernel
or MLX. They do not qualify a kernel-game promotion, quantized model quality,
full-route behavior, or a broad 4/8/16-bit claim. The prior synthetic
GPU-timestamp confirmation rounds failed the 5% drift gate; those failures
remain in force. The next speed gate needs like-for-like low-bit controls,
GPU timestamps, and full-route correctness on additional real tensors.
