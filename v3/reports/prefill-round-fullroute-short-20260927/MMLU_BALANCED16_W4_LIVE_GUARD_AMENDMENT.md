# Balanced16 W4 live disk-guard amendment

The user explicitly directed changing this local harness from a 64 GiB to a
32 GiB free-disk guard. This record distinguishes the original condition
policy, the amended pending manifests, and the unchanged completed receipts.
It does not claim that the original q01-W4 queue report passed its original
guard, or that any numerical result has been accepted.

The serial queue was in `waiting` state with no active trial. Its STOP control
was recorded and `state.json` became `stopped` before editing. q02/q03 had no
result directories. The exact four original pending queue manifests were
archived create-new in `mmlu-balanced16-w4-pending-pre-amendment-64g.tar.gz`,
SHA-256 `229ca4cdf26e9859b5dfb47fffcde8c3caa5060244d1070bea1131ed11b5edc3`.
The previous authoritative repository revision is `a6d8e9095f1f430bcc3b108696467a7cf41bbaf6`.

Only `conditions.minimum_free_bytes` changed, from 68,719,476,736 (64 GiB)
to 34,359,738,368 (32 GiB), in the four pending queue jobs q02/q03
native/W4 and their authoritative repository mirrors. The same single-field
amendment was made to q04–q15 authoritative manifests, which have **not** been
submitted. No ID, dependency, command, output path, model/package/executable
pin, prompt/target, run/wait limit, or other condition changed. Completed
q00/q01 queue result files and their authoritative manifests were not edited.

| Pending arm | Original queue SHA-256 | Amended queue SHA-256 |
| --- | --- | --- |
| q02 native | `11a8eb03ea096aa7cdc017b3e80c8a41a4eefa14292a1fdc05041fc9e299acdf` | `ff439f16548f43c4127b5a7dab63618cc62c0e25db1290486877ec601eb8ae50` |
| q02 W4 | `875257eca6b26b8cfcf4cfc55e3ef41e0220d73b1ab63a864746bbaf1b105ad4` | `05a3a9614dd5771926a3750a848266b43a67ca6e72547ab368a1aae32d1c58b4` |
| q03 native | `e57938a06905bdcd107d9e384bb556fae48804a5a98b1656877be00fbea971a3` | `90b82176d2b2bdc193cb4f0b8fbd52e722bc768bb91870dabef5184329fe49ee` |
| q03 W4 | `0b382d36b8d675003b7af9fb67051a847b928d369ffdd81bdbcd68f4f3e290b1` | `1a45cb06c8ba67f3a3d551ddece4506492b23e4011c5f35e1e8305134fc565eb` |

The queue-added null `kernel_game_submission` and `validator` fields explain
why queue and authoritative file hashes differ even before amendment; their
JSON content otherwise matches. All 28 authoritative q02–q15 manifests and
four pending queue mirrors were parsed and verified at exactly 32 GiB. The
pending mirrors matched their authoritative files after ignoring only those
two queue-added null fields. The frozen original generator and original
referee are not silently rewritten: their original source/manifest hashes
remain in Git history. A separately identified amended-policy referee must
verify mixed historical/amended lineage and re-evaluate the original q01
condition journal explicitly before any 32 GiB aggregate admission.

The temporary STOP marker was then removed, and the existing queue supervisor
started a new daemon process. Its first read-only state showed q02-native
waiting with 53,299,462,144 bytes free—above 32 GiB—but not ready because the
machine was on battery in low-power mode. The AC requirement was not changed;
no q02/q03 trial had started at that snapshot.

That historical re-evaluation cannot establish full eligibility from the
existing q01-W4 receipt: 332 of its 367 journal entries were below the old
64 GiB guard, and the queue consequently recorded `activity_sampled=false`
for exactly those 332 entries. None was below 32 GiB, but the missing process
activity observations cannot be reconstructed by changing a threshold. A
fail-closed amended referee must retain q01-W4 as condition-ineligible; it
may report those positions descriptively, not call the 16-case aggregate
accepted.

No completed job was replayed or overwritten. The old q01-W4 report and
five-file receipt archive remain authoritative for what the original queue
recorded. The amended policy cannot create evidence of unobserved hardware
conditions or justify a timing, quality, or promotion claim.
