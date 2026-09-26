# Gemma 4 checkpoint-quality referee

This is a default-off, fail-closed experiment for W4/W8 checkpoint quality. It
does not infer model quality from projection correctness and does not turn a
partial route into a checkpoint acceptance claim.

`rvllm_gemma4_quality_referee` binds, by SHA-256:

- checkpoint repository, immutable revision, manifest, and config;
- quantizer implementation revision, bit width, group size, scale dtype,
  zero-point and rounding contracts, and optional calibration dataset;
- BF16 control and candidate source tree, executable, model package, and
  full-route declarations;
- a fixed token fixture and independently generated observation artifacts;
- an optional independently justified held-out calibration contract; and
- an independent operator-correctness receipt when one exists.

It compares the same representative logit token set at every fixture position,
target-token negative log likelihood, perplexity ratio, and top-1 agreement.
Strict JSON rejects duplicate and unknown keys. Non-finite values, missing
positions, changed token sets, hash mismatches, and identity mismatches fail.

## Decisions

- `calibration_required`: metrics were recorded, but no threshold artifact was
  supplied. The referee deliberately has no built-in permissive threshold.
- `rejected`: at least one independently calibrated gate failed.
- `bounded_slice_evidence`: calibrated metrics passed, but execution was not a
  complete model route, all seven roles were not quantized, or an independent
  operator-correctness receipt was absent.
- `full_model_accepted`: calibrated gates passed on a declared full BF16 and
  candidate route, all seven projection roles cover the checkpoint, and an
  operator-correctness receipt is bound.

Thresholds must come from a separately sealed calibration artifact whose
protocol and rationale explain the acceptable BF16 run-to-run envelope and the
quality-loss budget on a calibration corpus distinct from this held-out
fixture. The fixture hash must be declared as held out. This repository does
not currently contain that empirical calibration; inventing values merely to
make a candidate pass is forbidden.

## Running

Build without an Apple feature or model load:

```sh
cargo build -p rvllm-runtime --bin rvllm_gemma4_quality_referee --release
target/release/rvllm_gemma4_quality_referee input.json
```

Queue acceptance jobs should add one of these gates:

```sh
target/release/rvllm_gemma4_quality_referee input.json \
  --require bounded_slice_evidence
target/release/rvllm_gemma4_quality_referee input.json \
  --require full_model_accepted
```

The receipt is written before exit status 3 when the required decision is not
met, so failed evidence remains in `trial.stdout`. The template in this
directory is intentionally not a live queue job: an author must replace every
`REQUIRED_*` token, seal all artifacts, and copy it to a queue `jobs/`
directory. This prevents an incomplete scaffold from occupying the active
hardware queue.

## Current boundary

The referee is runnable as soon as BF16 and candidate observation generators
emit the documented schema. The present Metal low-bit work has projection-level
real-weight correctness and timing, but no full model using W4/W8 packages.
Therefore its maximum honest result today is `bounded_slice_evidence`, and only
after empirical calibration. No ANE or Metal W4/W8 checkpoint has acquired a
full-model quality acceptance through this scaffold.

## Observation and calibration formats

Each observation artifact uses `rvllm.gemma4_quality_observations.v1` and
contains the sealed checkpoint-manifest, config, fixture, and route identities.
Its `cases` exactly match fixture case IDs. Every position contains
`target_token_id`, full-vocabulary-derived `target_negative_log_likelihood`, and
a nonempty `logits` object mapping the same representative token IDs to logits
for both arms. Representative logits should include the reference and candidate
top-k union plus the target token; because this is not necessarily the complete
vocabulary, the receipt calls its ordering metric *representative* top-1.

Calibration uses `rvllm.gemma4_quality_calibration.v1` with `protocol`,
`calibration_corpus_sha256`, `held_out_fixture_sha256`,
`minimum_target_positions`, `max_absolute_logit_delta`,
`max_mean_absolute_logit_delta`, `max_mean_nll_increase`,
`max_perplexity_ratio`, `minimum_representative_top1_agreement`, and a nonempty
`rationale`. Calibration thresholds are data, never command-line defaults.
