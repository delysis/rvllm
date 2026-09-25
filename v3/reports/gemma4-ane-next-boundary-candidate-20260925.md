# Next ANE boundary candidate: fused attention plus output projection

## Finding

Another FFN layout is not a credible next boundary experiment.  The qualified
stacked and interleaved FFN sources already use one program and two
convolutions.  The first convolution produces gate and up together; the second
must consume the input-dependent `GELU(gate) * up` value.  A one-convolution
exact Gemma FFN is therefore not representable by the current MIL convolution
graph.  Chunk4 and down4 change tiling while retaining or increasing internal
convolutions, so neither tests this hypothesis.

The smallest exact boundary-reducing candidate is instead
`ane-attention-output-fused`: append the layer's `o_proj` constant convolution
to `PackedAttentionLayout::mil()` and return 3,840 channels.  This replaces the
current attention evaluation followed by output-projection evaluation with one
single-input/single-output evaluation.  Per layer it changes two programs and
two evaluations to one program and one evaluation; it does not alter attention
math, quantization, normalization, residuals, or shipping defaults.

## Implemented compile-source boundary

The repository deliberately quarantines ANE attention to a single external
input/output after the former four-input graph caused an AppleH16ANEInterface
kernel panic (`crates/rvllm-apple/src/ane_attention.rs`). The default-off
compile-source implementation preserves that ABI, appends exactly one constant
1x1 convolution, reuses the existing FP16 linear-weight blob format, and seals
the MIL/blob hashes and padded I/O byte counts. It deliberately exposes no
request or evaluation method.

An ignored layer-0 device probe now permits exactly one compiler attempt and
zero evaluations in a fresh process, requires fresh receipt and driver-journal
paths, and preserves either success or the exact compiler failure. Host MIL and
shape tests pass. The private compiler probe has not yet run, so acceptance of
the 3,840-channel output and cache stability remain unproved; provision must
still precede runtime integration.

## Serial queue stages

1. `compile-source`: build only layer 0, compile budget 1, driver journal on;
   require one compiler call and zero evaluations.  Preserve any failure.
2. `component-oracle`: fresh process, compile budget 0; load the fused cache
   entry and the existing attention/output controls.  Feed the same captured
   real activation and packed KV state to both, compare all 3,840 FP16 output
   bits (also report max absolute/relative error), and require verified ANE
   execution.
3. `provision-48`: compile each layer serially, bounded to 48 compiler calls and
   zero evaluations.  Record cache identity per layer.
4. `full-route`: fresh process, compile budget 0, require all 48 fused entries,
   exact route selection, complete continuation/reference equality, and ANE
   execution.  Record loaded-program accounting: the candidate removes the 48
   standalone output programs and replaces the two shared attention programs
   with 48 fused programs, so the existing 162-program route should report 160.
5. Only after those gates, run matched ABBA timing.  No earlier receipt is a
   speed or promotion claim.

`rvllm_ane_boundary_referee` is the safe-Rust, fail-closed final gate.
It rejects tiling/layout-only submissions, nonzero compilation, incomplete
cache residency, synthetic-only or empty activation comparisons, route
fallback, missing ANE verification, and token mismatch.
