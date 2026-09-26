# Fused ANE attention plus output projection: first timing screen

## Outcome

The corrected, real-weight, component-qualified fused graph is a strong prospective latency winner in the first complete-pair screen.

- Separate attention plus `o_proj`: median `19.9215126953125 ms/token`
- Fused attention plus `o_proj`: median `10.1311015625 ms/token`
- Median control/candidate ratio: `1.966371827625482`
- Warmup outputs: bit-identical between arms (`0.0` maximum difference)
- Compiler calls during warmup and measurement: `0`
- Setup compiler calls: `2` within a declared limit of `3`
- Measured tokens per arm: `384`
- Sequence: `ABBA/BAAB/ABBA`

The control includes both complete accelerator operations: attention input updates, attention evaluation/read, the intermediate projection input write, projection evaluation, and projection read. The candidate uses the same incremental newest-Q/K/V/mask updates and one fused evaluation/read. Both use persistent KV surfaces.

## Variance boundary

The host conditions were logged without a stability wait. The full observed ranges were `9.25%` for the control and `10.86%` for the fused arm, exceeding the kernel-game's final 5% drift threshold. Nevertheless every individual control block was substantially slower than every fused block: the slowest fused observation was `10.7051 ms/token`, while the fastest control observation was `18.5931 ms/token`.

This is therefore a prospective winner, not promotion evidence. It requires an independently ordered confirmation and then a production-selected full-route oracle/timing campaign.
