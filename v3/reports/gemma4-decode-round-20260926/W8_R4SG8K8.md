# W8 output-projection donor-schedule arm

`metal-qmv-w8-g32-r4-sg8-k8` is a default-off decode candidate for Gemma 4
12B `OutputProjection`, `M=1, N=3840, K=4096 or 8192`. It compares the
four-output-row/eight-SIMD-group/eight-adjacent-K schedule to the already
selectable `metal-qmv-w8-g32-r8-sg2` arm. The W8 weights remain signed
two's-complement bytes with one FP16 scale per 32 weights. The new shader
loads one `uint2` (eight contiguous codes) and eight BF16 activations per
lane, reusing the activations across four rows, accumulating in FP32 and
rounding once to BF16. No layout permutation, affine conversion, scale
change, or production default is involved.

## Preflight only

- `cargo check -p rvllm-apple-metal --offline --locked`: passed.
- `cargo test -p rvllm-apple-metal --lib research_decode --offline --locked`:
  six passed; the two device tests intentionally require an explicit run.
- `cargo test -p rvllm-apple-metal --bin rvllm-global-decode-jobs --offline
  --locked`: nine passed.
- Generated BF16 Metal 3.1 source compiled with `-fno-fast-math` and linked.
  Shader SHA-256 `ba784fecdbe0bf4bc19512a82ac88e5462d8af55ff232f67aa3e77f1051efb1a`;
  generated-source SHA-256
  `ea4cfd77da67ad656e7e6d5f531d51ffb898540defe56f4e276c095443359482`;
  preflight metallib SHA-256
  `e8f3cf094d2b7ace1022fc98067620ee613645d66ced90e2fca14565ddceb777`.
  The preflight output is local at
  `/tmp/rvllm-w8-r4sg8k8-preflight.KgmNKD/build`; it is not a queue receipt.

No native oracle, dense-weight correctness, paired timing, full-route, or
MLX-relative result exists for this arm. The W4 v02 campaign already pins
the current release generator and test executable by absolute path and hash.
Rebuilding those paths for W8 before W4 completes would invalidate W4's
later stages. Campaign `g4-donor-w8-r4sg8k8-01` instead pins separate copies
of the checked debug generator, native-test binary, queue submitter, and ABBA
retainer under `v3/target/campaign-binaries/`, leaving W4's release binaries
untouched. Its compile job is submitted to the persistent serial queue.
These local binary copies are rebuildable artifacts, not checked-in source;
their SHA-256 identities are in `campaign.json` and its job manifest.

After compile success, run the native oracle for both K shapes, then paired
operator timing and real-weight/full-route selection, keeping each failed
receipt. Debug-host timing can screen a GPU kernel, but any production speed
claim must be rechecked with the release route and exact executable identity.
