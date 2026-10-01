# Prospective 32 GiB disk guard for later W4 correctness jobs

Historical planning note: the user's later explicit instruction authorized a
live amendment of pending local manifests. The controlling provenance record
is `MMLU_BALANCED16_W4_LIVE_GUARD_AMENDMENT.md`; the restrictions below describe
the earlier prospective-only plan, not the subsequently authorized action.

This is a new protocol decision, not an amendment to the submitted balanced16
v1 manifests. Those manifests remain immutable with their 64 GiB guard. The
q01-W4 receipt remains condition-ineligible, and the frozen 32-arm v1 referee
cannot accept the v1 campaign. No v2 job has been generated or submitted.

For a future distinct immutable job version, set
`conditions.minimum_free_bytes` to exactly **34,359,738,368** (32 GiB). Keep
the source/checkpoint/package/executable hashes, sampled-condition journaling,
AC and process checks, zero thermal dwell, result preservation, and the
all-position numerical/dispatch gates. A job is ineligible if any applicable
sample violates its own recorded guard. A local manifest amendment is permitted
when its original bytes/hash, amended bytes/hash, and operational transition
are preserved; completed receipts must not be overwritten. A new threshold
alone cannot supply condition observations that were skipped by the old guard.

The four completed v1 outer condition journals gave the following read-only
disk observations. This sensitivity check supports the resource choice only;
it is **not** a new referee result, speed result, or acceptance claim.
For q01-W4, 332 samples below the original 64 GiB guard skipped activity
sampling; a 32 GiB disk-only sensitivity check cannot prove those missing
conditions.

| v1 arm | Journal SHA-256 | Samples | Minimum free bytes | Samples below 32 GiB |
| --- | --- | ---: | ---: | ---: |
| q00 native | `8a74cd0ce17345c2197624f24141522992292453aadba1fba77dfac3f417c7bd` | 27 | 107,943,297,024 | 0 |
| q00 W4 | `9bfcff3bfc604e6373a4737820061a88ee7df058691c2dc75f698ee7115a8675` | 313 | 106,675,728,384 | 0 |
| q01 native | `84bc0070a773856a4d258c551e6b8e214111a3aca8bc4e8b8abab8176a2784dc` | 26 | 95,252,774,912 | 0 |
| q01 W4 | `bd3da3cac9d5b04e8dcad602763197b8e59fdb02da14d6b34b3a112fb78ec98b` | 367 | 49,231,007,744 | 0 |

The live amendment of pending q02/q03 under their existing IDs is documented
separately. For any additional submission, freeze its ID, output path, source
selection, exact manifest hash and a fail-closed referee appropriate to the
resulting cohort. Do not silently substitute completed cases or select
replacements using inspected model results. Record disk capacity and verify
every new output path is unused immediately before serial submission. The
existing q01-W4 archive and disqualification report remain authoritative for
what the queue observed.
