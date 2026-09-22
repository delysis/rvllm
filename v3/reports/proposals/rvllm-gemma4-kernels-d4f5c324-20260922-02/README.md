# Non-live local experiment specifications

These are **untrusted, unpinned proposals**, not queue manifests. Nothing here
launches, schedules, retries, resumes, provisions, or changes power settings.
The campaign stays paused. Every local pin is null; the local maintainer must
review the full series, explicitly resume eligible work, obtain real pins and
apply the repository's normal admission process. No local model, executable,
metallib, process exemption, or cache hash has been invented.

Apply the entire series through patch0008 before any qualification; some final
safety/test fixes are deliberately visible as a separate review patch. Select
only one research candidate at a time. Metal selectors are mutually exclusive;
the two new ANE weight plans are also mutually exclusive. Do not combine CPU,
Metal and ANE candidates in initial measurements.

## Preparatory checks (not executed here)

Run from a complete checked-out `v3/` workspace with the locally pinned toolchain.
The supplied source packet omits workspace crates and the older MIL fixtures.
These examples are **host-only filters or compile-only checks**, not permission
to run ignored accelerator fixtures:

```text
cargo fmt --all -- --check
cargo test --offline --no-default-features -p rvllm-apple --lib ane_int8_candidates::tests
cargo test --offline --no-default-features -p rvllm-apple --lib ane_attention_layout::blocked32_tests
cargo test --offline --no-default-features -p rvllm-apple-metal --lib research::
cargo clippy --offline --no-default-features -p rvllm-apple -p rvllm-apple-metal --all-targets -- -D warnings
cargo check --offline --no-default-features -p rvllm-runtime --features apple
cargo check --offline --no-default-features -p rvllm-runtime --features macos-private-ane-research --bin rvllm_disaggregated_infer
```

The last command is macOS/aarch64 research-only. Compile-only and filtered host
checks do not establish private API safety or target numerical acceptance.
A separate shipping-feature build and symbol scan are still required; unchanged
Cargo gates plus source scans are not a substitute for inspecting a built binary.
No broad unfiltered macOS test run is prescribed here.

## Metal source export / compilation boundary

The new `rvllm-metal-research-source` binary prints source and never creates a
Metal device. Example **unrun** host command after compilation:

```text
cargo run --offline --no-default-features -p rvllm-apple-metal --bin rvllm-metal-research-source -- bf16 metal-short-mma16x64
```

Use `off` for the baseline, or the other exact Metal candidate name. Save each
output to a unique reviewed file and hash it. The archive's prior accepted
compiler recipe uses these commands, not a guessed Metal4 API:

```text
xcrun --toolchain Metal -sdk macosx metal -std=metal3.1 -c ${SOURCE} -o ${AIR}
xcrun --toolchain Metal -sdk macosx metallib ${AIR} -o ${METALLIB}
```

Re-pin the actual installed compiler/SDK locally; no such command was executed
here. A library missing a candidate PSO falls back and is **not a candidate
performance sample**. Query SIMD width, maximum threads, static threadgroup
allocation and device limit. The code only admits a typed Apple9 PSO with width32
and enough measured limits; it does not infer those facts from “M4 Max.”

The old ignored Metal fixtures do **not** automatically exercise the new names.
A local component adapter must dispatch the exported candidate function, expose
its output, preserve the existing tolerance gates and record its source hash.
The new full-runtime dispatch paths and source export are implemented; a new
standalone accelerator acceptance runner is **not** supplied. This remaining
adapter/qualification step must not be mistaken for a passing hardware test.

## ANE qualification boundary

The pure builders return `Int8CandidateSource { mil, blob, budget }`. They do not
call the driver. The explicit component entry points are
`AneGatedFfn::compile_int8_chunk4_with_cache_policy` and
`AneLinear::compile_int8_tiles4_with_cache_policy`. They use the existing
single-input/single-output compile boundary. Before full provisioning, the local
maintainer must pin one graph's actual activation/weight fixtures and compare its
outputs with the corresponding unchanged baseline. These new API calls have
not been compiled or invoked on an ANE here.

The two cache part names are `ffn-int8-chunk4` and `qkv-sliding-int8-tiles4`.
Provisioning is a separate, explicitly authorized future operation. Do not use
`all-int8` to provision a candidate: that alias deliberately remains the original
baseline. Inference plans require existing entries and compile budget0. No
fallback retry or multi-I/O route is added.

## Measurement protocol

Each prompt length is a separate case. One screening block comprises baseline
before, candidate, baseline after; each arm has2 warmups and7 measured requests,
27 requests total. For ten generated tokens, require exactly nine ANE steps per
request; early EOS invalidates a work-matched comparison. Retain all raw results.
Never turn a failed/noisy block into a success by deleting samples or changing
power strata. Bracket drift over5% rejects the block. The templates propose a
conservative5% stage-improvement screen and a2% end-to-end non-regression screen,
not an assertion that the hardware can meet either threshold.

The first stratum is battery / low-power-off / pmset mode0 / nominal, observed
rather than forced. AC, Fair, low-power-on and other pmset modes are separate
experiments. Record wall/completed GPU times with their real boundaries; never
normalize them with CPU cycles. Compile/load time, persistent memory, source
bytes, resident memory and transient memory are different measurements.

For C6 the primary control is **reuse-scratch**, not the older allocating path.
Measure CPU packing separately, then complete import wall including the original
surface writes. For C5 use the untiled INT8 sliding-QKV branch as the tiling
control and evaluate the active FP16-QKV baseline separately for quality. The
historical6.0859steps/s does not replace any local matched control.
