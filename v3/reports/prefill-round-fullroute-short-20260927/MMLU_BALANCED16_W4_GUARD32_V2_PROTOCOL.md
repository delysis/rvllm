# Prospective 32 GiB disk guard for later W4 correctness jobs

This is a new protocol decision, not an amendment to the submitted balanced16
v1 manifests. Those manifests remain immutable with their 64 GiB guard. The
q01-W4 receipt remains condition-ineligible, and the frozen 32-arm v1 referee
cannot accept the v1 campaign. No v2 job has been generated or submitted.

For a future distinct immutable job version, set
`conditions.minimum_free_bytes` to exactly **34,359,738,368** (32 GiB). Keep
the source/checkpoint/package/executable hashes, sampled-condition journaling,
AC and process checks, zero thermal dwell, result preservation, and the
all-position numerical/dispatch gates. A job is ineligible if any applicable
sample violates its own prospectively sealed guard. Do not use this threshold
to reclassify v1 receipts or edit an accepted queue manifest.

The four completed v1 outer condition journals gave the following read-only
disk observations. This sensitivity check supports the resource choice only;
it is **not** a new referee result, speed result, or acceptance claim.

| v1 arm | Journal SHA-256 | Samples | Minimum free bytes | Samples below 32 GiB |
| --- | --- | ---: | ---: | ---: |
| q00 native | `8a74cd0ce17345c2197624f24141522992292453aadba1fba77dfac3f417c7bd` | 27 | 107,943,297,024 | 0 |
| q00 W4 | `9bfcff3bfc604e6373a4737820061a88ee7df058691c2dc75f698ee7115a8675` | 313 | 106,675,728,384 | 0 |
| q01 native | `84bc0070a773856a4d258c551e6b8e214111a3aca8bc4e8b8abab8176a2784dc` | 26 | 95,252,774,912 | 0 |
| q01 W4 | `bd3da3cac9d5b04e8dcad602763197b8e59fdb02da14d6b34b3a112fb78ec98b` | 367 | 49,231,007,744 | 0 |

Before any v2 submission: decide the disposition of still-pending v1 q02/q03
without replaying or overwriting them; freeze genuinely new IDs, output paths,
source selection, exact manifests and hashes, and a separate fail-closed
referee appropriate to the resulting cohort. Do not silently substitute
completed v1 cases or select replacements using inspected model results.
Record disk capacity and verify every new output path is unused immediately
before serial submission. The existing v1 archive and disqualification report
remain the authoritative evidence for q01-W4.
