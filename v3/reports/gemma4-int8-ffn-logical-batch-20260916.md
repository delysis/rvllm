# INT8 FFN with independent logical token columns

The source now includes an explicit component constructor for 2–8 logical
tokens and a token-major `project_batch` interface. The first device experiment
must use two tokens. This is not integrated speculative decoding and does not
change any runtime/CLI/server default.

The three convolution projections, stored INT8 coefficients/scales and FP16
tanh-GELU ordering remain unchanged. Only logical activation tensor width
changes. Physical channel stride remains 32 FP16 elements, so two-token
H3840 input and output allocations are each 245,760 bytes. A single external
input/output is retained; the quarantined multi-I/O route is not involved.
Source arithmetic and proposed layout do not establish device compatibility
or useful weight reuse.

Host validation covers distinct column placement, clearing old values/padding,
shape and overflow rejection, bounded width before private API access, and
the unchanged three-convolution/single-I/O graph topology. Three new tests
pass. The existing captured Gemma FP16 FFN MIL fixture still matches byte for
byte in a separate passing test; ordinary INT8 and stacked S=1 also retain
their existing generation paths. The normal single-token `project` method
rejects batched instances instead of silently using only column zero.

The component probe now selects this path with `--int8-batch 2 --mode int8
--compare true`. The baseline must already exist in the cache. Qualification
allows at most one candidate compilation, requires the driver journal and
does no timing. With the current input fixture there are three synthetic
vectors and **one** authenticated captured layer-0 input, not two captured
tokens. Five serial calls and eleven batched calls cover each input in both
lanes, swapped columns, zero isolation and repeated use after changing inputs.
Every output is checked against the independent reconstructed-weight CPU
oracle, using the existing absolute-plus-relative tolerance. Swaps and repeat
use additionally require exact FP16 bit parity; serial-versus-batch bit parity
is reported separately. Source and assembled-input hashes are retained.

After successful qualification, a separate strict-cache, zero-compilation run
without the journal can compare one S=2 call with two S=1 calls. Three rotating
ABBA/BAAB blocks each use 128 repetitions per phase, counting 256 useful FFN
outputs per phase. Packing, evaluation and readback are included. Eligible
matched-power pairs are observations, not a full-model speedup claim. More
captured inputs and reproducible block savings remain necessary before
provisioning more shapes.

The one-graph qualification may be submitted as an independent preparation
job while unrelated CPU builds prevent timing. It retains exclusive queue
ownership of the accelerator, bounded work, benign thermal state and the
16 GiB disk floor. It may coexist with CPU builds because it makes no timing
claim; the ordinary timing jobs continue to require their quiet window.

## First device qualification

The queue's `native-comparison-quiet/results/30-batch-ffn-qualification/`
contains the successful device receipt, full driver journal and raw activity
observations. The S=1 baseline was a cache hit; exactly one S=2 graph compiled.
Five S=1 and eleven S=2 evaluations completed, and both programs unloaded.
All 84,480 batched output values match the serial outputs bit for bit. Lane
swaps, zero isolation and repeated use also pass. Maximum absolute error
against the independent CPU oracle is 0.03875351 for the captured input, with
relative L2 error 0.0005147551; this is the same error as the serial path and
passes the declared per-element tolerance. Synthetic-input maximum absolute
errors are 0.00205136, 0.00296402 and 0.00219173.

The probe binary SHA-256 is
`b89c68308f5e10b636e606ab2315225a3be42bee2379d46737809a1c7a0c5bf0`;
the reconstructed FP16 weight hash is
`9ec1b72edfad27068add34669a960d971f6a52cf982db9170e451cafe158bfe5`.
The source blob is 177,016,768 bytes for either graph. No timing was recorded.
A transient external llama server appeared during preparation; the queue
preserves that ineligibility and makes no performance inference. The dependent
strict-cache timing job is queued behind the normal quiet/power conditions.

Subsequent broader qualification on five actual ANE decoder inputs stopped
at the serial program's CPU-reference check, before any S=2 evaluation. The
original four-input device result above remains valid, but broader acceptance
and component timing are suspended while the reference discrepancy is diagnosed.
See [the follow-up evidence](gemma4-batched-projections-and-live-inputs-20260916.md).

See [the feasibility report](gemma4-ane-multitoken-verification-feasibility-20260916.md)
for causal attention, transactional KV, target numerical parity and the exact
Google 12B assistant requirements. A batched FFN alone does not verify or
accept draft tokens and has no established full-model speedup.
