# Gemma 4 12B W4 donor-schedule trial

## Source and hypothesis

The schedule comes from the pinned `john-rocky/coreai-model-zoo` application
shader at commit `a2a664e84ee4807cf4f6441944bd0ac6d047224d`,
`apps/CoreAIChat/Resources/g4msl/gemma4_matvec.metal.txt`. Its decode matvec
uses four output rows per SIMD group, eight SIMD groups per threadgroup, and
eight adjacent K values per lane. The application shader is the scheduling
reference; its affine group-64 codes, optional activation quantization and
E2B dimensions are **not** the rvLLM tensor ABI.

Research selector `metal-qmv-w4-g32-r4-sg8-k8` applies that schedule to the
authenticated rvLLM W4A16 group-32 sidecar at Gemma 4 12B dense down shape
`M=1,N=3840,K=15360`. It reads the existing signed two's-complement nibbles
and one native FP16 scale per 32 weights, accumulates in FP32, and writes BF16
once. Each lane loads one four-byte packed word and reuses eight activations
across four rows. No repack, requantization, affine bias, production default,
or checkpoint format is changed. Its new reduction association requires the
independent numerical oracle before timing claims.

The comparator `metal-qmv-w4-g32-r8-sg2` produced a repeated approximate
1.3x operator signal against the incumbent N4 route, but its controls drifted
13.90% and 38.61% in two campaigns. That is a reason to change the schedule
and sampling design, not to call the older variant qualified.

## Current gates

- Host library check and focused decode tests passed: 6 passed, 2 native tests
  remained intentionally ignored.
- Queue-generator tests passed: 9 passed.
- The generated source compiled and linked with Metal 3.1 and
  `-fno-fast-math`: source SHA-256
  `f74e67aa911b101575a770b7803e59b0fda95b07c1b127c135113e905b61b918`;
  preflight metallib SHA-256
  `5bbda735656a5502a3c7f7394f0692f8f74ce53ffa1aad089a1a674fd14747af`.
- The first immutable campaign `g4-donor-w4-r4sg8k8-01` submitted a compile
  job before review found a host launch-width mistake: it would launch 240
  groups for a kernel covering 32 output rows per group, leaving half the
  groups empty. The generated shader remains a valid compile probe, but no
  oracle or timing will be advanced from that executable.
- The corrected campaign `g4-donor-w4-r4sg8k8-02` pins a rebuilt test
  executable with 120 groups at `N=3840` and has submitted its compile job.
  The device oracle and timing must be generated from its completed compile
  and oracle receipts, respectively. No native result or speed claim exists.

The compiler preflight is outside the queue and is only a source gate; the
queued build will create its own sealed compiler and metallib receipt.

## ANE all-sliding confirmation in parallel

The earlier four-token full-route ABBA receipt
`ac035b490a8e9de6863c1121c7933e3c5387cc6ccc5f5173bd7272851db473ae`
measured 2130.43 versus 1752.51 ms/token (1.216x), with 17.57% candidate
drift. The reverse-order receipt
`734ef8532ba034624b5f7c789cbcaaeaaa243e0a549107ab6a2fe1247e70ef29`
measured 2132.80 versus 1767.26 ms/token (1.207x), with 9.97% baseline and
6.06% candidate drift. Both retained exact outputs and zero compiler calls in
warmup/timing. Their common direction is promising; neither clears the 5%
drift gate.

The persistent queue now has independent eight-token ABBA and BAAB manifests,
`g4-ane-all-sliding-confirm-t8-{abba,baab}-20260926`. They retain all
observations and record concurrent audio restoration and build processes.
Changing thermal conditions are observational, with `stable_seconds: 0`.
These trials can reduce fixed-overhead noise and test the signal at a longer
dependent-token batch; they do not erase the earlier inconclusive receipts.

## Next decision

After the queued W4 source compiles, run its native oracle, then paired timing
against the same-shape incumbent. Preserve failures. A passing operator screen
would justify real-weight/full-route work and generated-code inspection; it
would not establish an MLX-relative or whole-model win. For the ANE route,
read both eight-token orders before any promotion decision.
