# Gemma 4 D512 split-KV r8s256 correctness receipt

Date: 2026-09-24

Commit under test: `ffd0ba40`

Candidate: `metal-global-d512-split-r8s256t128`

Status: **native operator correctness passed; timing remains deliberately
ineligible until the two-dispatch v2 timing receipt and verifier exist.** This
is not model-route qualification or a promotion result.

## Native evidence

- Device: Apple M4 Max, Apple9
- Partial grid: `2 x 16 x 1`, 128 threads, 10,016 bytes static threadgroup memory
- Merge: 32 threads, zero static threadgroup memory
- Dedicated partial scratch: 526,336 bytes
- Strict source SHA-256: `3c6251c6af87d651baf055681acbc755e720cae1a5b35a9dc98f50c50466d7e6`
- Metallib SHA-256: `4ffdaa1d3b78de3fa09e1b7f5917d3667a5bdaa2030c8f33ef2a5578bb9da276`
- Build receipt SHA-256: `137201aab9ebdbadf71b2abc12b4d52fa1d9fdac324a56e96fef602b544a5d13`
- Test executable SHA-256: `d6d31c434ff5d7c7e511ae47ff3541614435d832091a7b20da4b757b859985b0`
- Native receipt SHA-256: `44fe484664e4ff66a8abd7b47913f3b71012f23bc551c07f2130848a697e9188`

The ignored native gate passed lengths 1, 255, 256, 257, 511, 512, 513,
1023, 1024, 1025, 2047, 2048, and 2049, plus entirely absent first,
middle, and last partitions in a 4096-token table. All outputs were finite,
guard regions held, and BF16 was rounded once after the merge. Because split
reduction changes FP32 association, the gate compares against the independent
CPU reference with an explicit absolute bound rather than claiming false
bitwise parity. The worst observed FP32 absolute error was
`4.291534423828125e-6`, below the sealed `5e-5` bound.

## Corrections discovered by the native gate

The first execution exposed two harness defects and no shader correctness
failure:

1. correctness-job generation selected the diagnostic `oracle` source even
   though this split gate exercises the production partial and merge entry
   points directly; split correctness jobs now select the exact `core` source;
2. the 4096-token hole fixture retained two deliberately reserved future
   logical table entries, exceeding this family's exact 4096-token admission
   cap; the hole cases now trim only those unused logical entries while
   retaining poisoned physical padding.

## Next gate

Implement `rvllm.global-decode.abba.v2` so receipts separately bind partial,
merge, and total GPU work. Advancement must use total time. First timing cell:
L512 against the current shared-KV leader, then L1024 and L2048 only if useful.

