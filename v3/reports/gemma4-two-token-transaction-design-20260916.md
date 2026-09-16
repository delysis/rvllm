# Two-token target verification: transaction boundary

The reference transaction is now implemented for diagnostic qualification;
it is not a batched verifier or a speed result. It preserves Metal prefill and
ANE target decode. Production callers are unchanged. The eventual batched FFN
still needs its numerical and performance gates.

## Meaning of the two inputs

At committed KV frontier `p`, `anchor` is the target-selected pending token
which has not yet entered KV. Evaluating `[anchor, draft]` produces `t0` after
anchor and `t1` after draft. This verifies one draft proposal, not two.

| Resolution | Retained inputs | New frontier | Published predictions |
|---|---|---:|---|
| draft differs from t0 | anchor | p+1 | t0 |
| draft equals t0, both outputs permitted | anchor, draft | p+2 | draft, t1 |
| first prediction ends the request or only one output is permitted | anchor | p+1 | t0 |

The last published token remains pending outside KV, matching serial decode.
If `t0` is EOS, consuming the speculative EOS input must be undone even when
the draft matched. If `t1` is EOS, both input positions remain committed;
the predicted EOS itself has not been consumed. Stop-string and output-budget
handling must choose the retained prefix before exposing tokens to callers.

## Smallest implementation sequence

Start with a diagnostic reference transaction executing the existing S1
decoder twice. This adds no graph or driver selector and establishes rejection
semantics before changing arithmetic. It is deliberately not a speed path.
Keep `GemmaAneDecode.next_position=None` from the start of tentative execution
until every layer and resolution operation has succeeded. A borrow-owning
pending result can prevent another decode while the caller resolves the prefix;
dropping it unresolved leaves the decoder unusable until full prefill import.

Preflight both token IDs, capacity 1024, checked `p+2<=1024`, and all 48 layer
frontiers before mutation. For this bound, tentative slots append without
overwriting committed sliding or global KV. Do not support a wrapped ring in
the first implementation.

On rejection or one-output termination, each attention request must restore
frontier `p+1`. Rebuild its mask using `PackedAttentionLayout::encode_mask`.
Do not clone the host mask vector: `AneAttention::import_cache` writes a packed
resident surface without synchronizing that scratch vector. Clear the rejected
slot's K/V through the existing strided-write API, write the rebuilt mask, then
update that request's `tokens_seen`. Across 48 layers the clear writes total
336 KiB of K/V plus 96 KiB of masks; no full-cache readback is needed.

Restore the decoder's usable frontier only after all layers succeed. A partial
layer, observer, vocabulary or rollback-write failure leaves it poisoned.
Never restore `Some(p)` merely because an error occurred; the layers may
already disagree. Recovery remains a complete successful prefill import.

After this reference transaction is exercised, replace arithmetic layer by
layer: keep QKV/O/head as two S1 calls initially, run attention lane 0 before
lane 1, and batch the FFN using the existing single-I/O S2 API. Preserve two
residual vectors, absolute RoPE positions, and the global V copy before K
normalization. An S2 wrapper and physical padding alone do not implement this.

## Required evidence

- Preflight rejects position 1023, overflow, wrong capacity, invalid tokens or
  unequal layer frontiers without mutation; position 1022 is admitted.
- Acceptance, rejection, EOS and output-budget cases produce the exact retained
  prefix and pending token. A mismatched draft can never commit both inputs.
- Both attention geometries restore masks correctly immediately after import.
- Tentative append/clear preserves every committed KV byte, and the replacement
  token overwrites the rejected slot at its original absolute position.
- Injected execution and rollback errors prevent further decode until import.
- Live strict-cache reference comparisons establish accepted and rejected
  continuation parity against ordinary serial decode, with journaled lifecycle
  checks and zero compilation. Only then test the batched replacement.

## Implemented reference and qualification fixture

`ane_two_token_reference.rs` exposes an exclusive pending reference transaction.
It checks both tokens, capacity and all 48 frontiers before setting the decoder
frontier to `None`, then executes the existing S1 decoder twice. Resolution
returns only committed predictions; a mismatched draft or caller-specified
one-output boundary clears the rejected append. Unresolved drop and partial
execution/rollback errors leave the decoder poisoned. The method requires a
driver journal and zero prior compiler calls. No graph or private selector was
added, and no production request is routed through this diagnostic API.

The layout test compares every packed byte against an independently imported
retained prefix for both actual attention geometries, including position 1023,
stale mask scratch, replacement writes and invalid wrap/overflow boundaries.
The preflight tests cover all 48 possible mismatched layers and capacity/token
limits. Source review found no blocking defect.

The ignored live fixture validates all 96 captured KV file hashes from an
identified 84-token Metal prefill, then uses the ordinary cached INT8 plan.
It compares token IDs and top-five logit bits for acceptance, rejection followed
by replacement, and a one-output boundary. It exercises unresolved drop,
observer errors in each tentative step and a host-injected error between layer
rollback writes after one layer succeeded. These injections do not deliberately
fail a driver call. Successful reimport must recover ordinary serial results.
The fixture keeps a flushed JSONL receipt; successful unloads require independent
driver-journal validation. It does not measure performance or claim live Metal
prefill execution from replaying a captured snapshot.

Job 60 completed compilation and the three focused host tests in `cache-audit-v6`,
serializing them with hardware experiments. The live test was ignored by the
host gate, then selected explicitly by job 61 using a frozen executable.

Job 61 passed the full captured-prefill transaction fixture. Accepted and
rejected/replaced continuations matched all expected token IDs and top-five
logit bits. The one-output boundary, unresolved drop, both execution-error
injections, partial rollback and reimport recovery all passed. The independent
driver-journal audit accounts for 162 cache hits, 208 requests, 3,960 completed
evaluations and 162 matching successful unload returns, with no cache miss,
compiler invocation or failed driver event. The boot identity remained
`1789488066` / `242846`. This verifies the reference transaction for that
captured 84-token prefix; the position-1023 boundary remains host-tested only.

The frozen executable SHA-256 is
`fad1ff59d073794974676dcdfdd1d7321cf24d7886fd01faddd5b4ec87b038ca`.
Source snapshots, manifests, build receipt and per-model journal audit are in
`gemma4-12b-evidence-20260914/two-token-reference-20260916/`.
The live receipt SHA-256 is
`da9db4674d5c3a04c8f3ef48f1a336af37385f29e8696b53c71827103c99023e`.
No S2 arithmetic, drafter, throughput improvement or production integration is
implied by this qualification. Ordinary decoding remains the active path.

The initial source audit used `gemma_ane_decode.rs`, `ane_attention.rs`,
`ane_attention_layout.rs`, `gemma_decode_math.rs` and the existing safe batch
interfaces in `ane_linear.rs`. No new private selector, multi-I/O request,
assistant download, cache provisioning or device transaction was performed
during this design review.
