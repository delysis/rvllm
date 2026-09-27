# Gemma 4 12B donor-schedule implementation

**Base:** `delysis/rvllm@593e1d6fb25088608198f2248dcac038d19fe761`.
**Donor:** `john-rocky/coreai-model-zoo@a2a664e84ee4807cf4f6441944bd0ac6d047224d`, app-host-matched raw-Metal resources.
**Status of the original Astra packet:** implemented, default-off research source; no Apple compilation, native execution, checkpoint qualification, or performance promotion was claimed there. Subsequent exact-base native results are recorded separately in `reports/donor12b-exact-base-20260926/README.md`; they do not qualify the integrated tree or a checkpoint.

## Implemented

Two explicit selectors, `metal-donor12b-sg8` and `metal-donor12b-sg4`, own twelve entry points each. SG8 preserves the donor's R4/eight-SIMDgroup decode geometry; SG4 is a lower-threadgroup-size comparison. Both retain the same packed weight and FP16 scale bytes.

| Operation | Actual implementation |
| --- | --- |
| W4/W8 decode projections | Four output rows per SIMDgroup; eight consecutive K values per lane; shared activation loads; signed group-32 weights; FP16 scales; FP32 accumulation. |
| Small-batch projections | Two rows per SIMDgroup, eight tokens per internal tile; one weight load/dequantization feeds eight token accumulators; M=2..128 with masked tails. |
| Low-bit fused FFN | Gate and up share activation loads; each projection rounds to BF16 in registers before the existing GELU cutoff/tanh rule and product; one output store. |
| Native BF16 fused FFN | Exact M1/K3840/I15360 contract; intermediate BF16 rounding retained; trace requests decline fusion. |
| Low-bit QKV | Local Q4096/K2048/V2048; global Q8192/K512 with raw K reused for V only when authenticated storage aliases exactly. K and V destinations remain distinct. |
| Native BF16 projections | Decode and real M8 wide projection schedules; explicit BF16 or FP32 output. Native projected QKV keeps four-byte scratch into existing normalization/RoPE/cache code. |
| Paged local/global decode | Local D256/16Q/8KV and global D512/16Q/1KV; online softmax/PV and in-threadgroup partial merge; page holes, physical-page validation, window bounds, speculative suffix visibility and empty partitions. |

The local attention query-to-KV mapping is `query_head / 2`, not modulo eight. Global V derives from the raw K projection, never the rotated/learned-normalized K cache. Existing norm, proportional RoPE, independent V normalization, cache writes, residual and layer scalar remain authoritative. Cache writes stay in a prior encoder; there is no cross-threadgroup cache-write dependency.

A changed dot/reduction schedule is **not** bitwise equivalent to the incumbent. The code preserves storage formats and explicit rounding boundaries; oracle tolerances and checkpoint/full-model gates remain required.

## Integration, not shader-only delivery

`donor12b.rs` supplies allocation-free checked plans. `donor12b_metal.rs` binds those plans through the existing typed pipeline cache and buffer arena. `layer_forward.rs` invokes the new projection, QKV, FFN and attention paths; unsupported requests decline before encoder creation and use existing routes. BF16 low-bit fallback is explicitly BF16, not an F16 reinterpretation.

Admission checks full matrix spans, alignment, overflow, write/read aliasing, phase and dtype. Full-layer fusions additionally require the supported dense 48-layer, hidden3840/intermediate15360, 16-head 12B geometry without MoE/PLE. Pipeline availability and queried thread/shared-memory limits are still mandatory. The existing **Apple9** gate is unchanged; another device family is not automatically admitted.

No ordinary token-path pack transformation, allocation, environment lookup, compilation, clock, wait or CPU readback is added by the adapter. Physical candidate dispatches use the existing research ledger. Logical low-bit role participation is recorded separately; fused operations do not pretend to be several physical launches. A bounded per-layer correction keeps the runtime encoder estimate consistent with actual new fusions.

`gemma4_model.rs` and the runtime supplement fix global low-bit completeness: a global layer has six physical projection tensors, not a separate seventh V tensor. The validated global K name is retained as the V source. After authenticated sidecar installation, a Value-role descriptor aliases those exact K bytes without a second allocation or requantization. Conflicting aliases fail closed.

The runtime supplement admits BF16 activations with the existing group-32/FP16-scale package only for the explicit donor selector with quantized BF16 accumulation disabled. Other selectors retain the previous package dtype restriction. The explicit embedded-native-options path still rejects packages as before. Numeric ABI version advances from 10 to 11.

## Pins and branch integration

The handoff was originally based on `a52bdf46`; this implementation intentionally builds on the subsequently supplied `593e1d6f` source, which already contains the preceding FFN/QMV/short-attention round.

The original patch used research slots **60..83** on `593e1d6f`. The PR #4 integration preserves the two incumbent donor-QMV slots **60..61** and appends this family at **62..85** with schema v5. The original apply helper still deliberately requires its exact base; the integration must be checked on its own source and executable identities.

## Validation actually performed in Chat

The literal new shader bodies, with MSL address-space/entrypoint annotations erased, were compiled as C++20 and executed through a threaded CPU SIMD/barrier shim. **136 cases passed** for both variants, covering signed W4/W8, all target K widths, M8 tails, output padding/guards, fused rounding, raw K/V reuse, native FP32 output and paged local/global attention. The same cases passed **AddressSanitizer and UndefinedBehaviorSanitizer** with no reported error.

This is execution of the shader arithmetic under a CPU surrogate, **not Metal compilation, Metal memory-model proof or a GPU benchmark**. Many projection cases use selected output rows with full K, not a whole checkpoint.

Supplementary checks cover source/ABI invariants, unchanged shipping defaults and shader source, append-only registry, contiguous kernel buffer arguments, resource calculations, shell syntax and Rust lexical delimiter balance. Delimiter checks are not a Rust parser/type checker. Twelve portable Rust test functions, plus loader/alias tests, were authored but **not run**.

The core patch is checked by normal `git apply --check`, applied to a pristine copy of the supplied base subset, and byte-compared to the resulting source. The source archive omits the full runtime file: its eight supplement hunks were written from exact pinned GitHub excerpts and checked on a shifted fragment fixture. **Full runtime/workspace application is not verified here.** The apply helper checks the runtime blob SHA `f7f8bfc92e005b7289ba78edc896ac4d36bbf5d9` and performs the complete combined `git apply --check` on the real checkout before changing files.

No Rust/rustfmt/Cargo toolchain, Apple SDK, Apple device or target model was available. Do not substitute these local checks for the pending gates below.

## Native gates and trial runner

From a clean, exact-base **complete repository**, apply the combined patch with the bundle's `apply.sh`. Then, from `v3/`:

```sh
cargo fmt -p rvllm-apple-metal -p rvllm-runtime
cargo check -p rvllm-apple-metal --lib --locked
cargo test -p rvllm-apple-metal --lib donor12b --locked -- --nocapture
cargo test -p rvllm-apple-metal --lib key_as_value_alias --locked
cargo test -p rvllm-apple-metal --lib --locked
cargo check -p rvllm-runtime --lib --features apple --locked
cargo clippy -p rvllm-apple-metal --lib --tests --locked -- -D warnings
```

Record any formatting/compiler repair commit before freezing source/executable hashes. Some surrounding source/tests depend on files omitted from the supplied subset; run these commands in the full repository, not `source/` in the bundle. These commands are **pending**, not receipts.

Under the existing exclusive Apple experiment queue/accelerator ownership, run:

```sh
sh tools/run-donor12b-native.sh /absolute/fresh/donor12b-trial
```

The script is a native operator runner, not a replacement queue owner. It does not obtain the project's shared accelerator lock or enqueue itself. Do not run concurrently with other Metal/ANE trials. Retain failures and raw outputs.

`rvllm-donor12b-source` exports the exact selected BF16 library source and invokes `xcrun metal -std=metal3.1 -fno-fast-math` followed by metallib linking. The existing source-bound native Setup checks source/metallib/build-receipt identities, executable identity, actual PSO limits and the selected family. Neither environment-timing data nor a compiler receipt can promote the candidate.

The new ignored device oracle exercises all twelve entry points per selector using full-dimension dense **synthetic** inputs, FP64 reference samples, all-output finite/padding checks, guards and repeated scratch reuse. Attention compares all output components. This is a checked-plan host-adapter operator test, not a whole-model forward test.

The ignored timing referee compares W4 down K15360, W8 output K8192 and K4096, and M9 local W8 output against the existing native-BF16 n4 same-weight control, from the same strict library. It retains twenty alternating ABBA/BAAB blocks, eight repeated dispatches per command and all samples. Separate five-percent drift and order-sensitivity gates suppress the admitted speedup field when unstable. This is a **hot-cache synthetic operator** test; a positive ratio cannot be reported as end-to-end, MLX, real-weight, checkpoint-quality or ANE evidence.

For a development model path, the existing selector mechanism is:

```sh
export RVLLM_METAL_RESEARCH=metal-donor12b-sg8
# Run the existing rvLLM model/package route and capture its dispatch identities.
```

Use `metal-donor12b-sg4` for the second candidate and `off` for the native default control. BF16 sidecar packages require the matching candidate metallib and checkpoint-authenticated sidecars; this bundle does not fabricate or requantize a model package.

Before promotion, run the existing real-weight/checkpoint oracle, teacher-forced per-layer and KV comparisons, prompt-length/window/prefix/rollback tests, logits/continuation quality gates and paired end-to-end timing with actual cache/thermal/power/queue conditions. Fresh GPU timing is required for both variants.

## Deliberate remaining scope

No IL4 transformed sidecar, activation quantization, INT2 format, fused newest-KV write, norm/residual prologue rewrite, cross-token GPU chaining/EOS/cancellation rewrite, ANE/MIL route change, or new multimodal encoder is implemented. Existing large-prefill attention and final sampler remain in place; the native global QKV weight concatenation still includes both logical raw K/V segments. The optimized one-projection global reuse is the authenticated low-bit route.

The implementation is intended to make comparative trials possible while retaining correct fallbacks. It does not establish that either candidate is faster, compiler-clean, checkpoint-correct, or production-ready.
