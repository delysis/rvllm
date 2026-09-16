# BF16 checkpoint conversion: release measurement

The direct integer BF16-to-FP16 converter is now the default for ANE checkpoint
preparation. Its test-only scalar reference retains the preceding implementation.
The implementation uses safe Rust, no scratch allocation and no CPU feature
override. Normal values adjust exponent bias and widen the fraction; subnormals
round to nearest, ties to even. A reduction detects overflow and nonfinite inputs;
callers discard the output on error.

Every one of the 65,536 encodings was compared with the preceding converter for
both BF16 and FP16. Successful outputs match bit for bit, including signed zero;
error results match. A batch containing every accepted encoding additionally
checks the compiler's vectorized loop. Empty inputs, length errors, unsupported
types and representative matrix-row lengths pass. These tests passed in both
development and release profiles before promotion and in the development
profile after promotion.

The release ABBA experiment used the actual layer-zero gate projection, four
conversions per trial (235,929,600 values). Source loading, output allocation and
parity checks occurred outside timing. Twelve alternating trials formed six
eligible pairs: battery, low-power mode off, pmset power mode zero, nominal
thermal state, no reported limits. Median paired baseline/candidate ratios:

| Measurement | Ratio | Candidate reduction |
|---|---:|---:|
| Wall duration | 1.5551 | 35.69% |
| Process CPU cycles | 1.5447 | 35.26% |
| Process CPU instructions | 3.3801 | 70.41% |

These are checkpoint-conversion measurements on one host and one actual tensor,
not full initialization, prefill, decode or energy savings. Power sampling does
not guarantee fixed clocks or eliminate contention. Process counters include
all process threads and observer overhead; they exclude accelerators.

Evidence: `gemma4-12b-evidence-20260914/bf16-integer-release-abba/`, containing
raw trials, power observations, pairs, archived source and build/binary identity.
The release test harness used opt-level 3, fat LTO and one codegen unit. The
production binary uses the same optimization settings with aborting panics.

Earlier slice-scratch and integer experiments used the development profile
(opt-level 1). They showed slower results there and are explicitly labeled
exploration, not production evidence. The integer release result reversed the
development result. The slice-scratch candidate was not promoted; its release
performance has not been measured. No half-crate feature settings were changed.
