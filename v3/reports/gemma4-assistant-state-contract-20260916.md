# Assistant state ownership after two-token verification

Pinned vLLM revision `3bb782621492711485dc86791b5978128783814a` was fetched
directly for this source check; no upstream code was executed. Its
[proposer](https://github.com/vllm-project/vllm/blob/3bb782621492711485dc86791b5978128783814a/vllm/v1/spec_decode/gemma4.py)
holds draft positions and sequence lengths constant, and maps assistant layers
to the last non-shared target layer of each attention type. The local standard
12B configuration has zero shared layers: those targets are sliding layer 46
and global layer 47. Only those committed KV mirrors need consideration for
the assistant; this is not yet an implemented adapter.

The [assistant forward](https://github.com/vllm-project/vllm/blob/3bb782621492711485dc86791b5978128783814a/vllm/model_executor/models/gemma4_mtp.py#L405-L436)
combines target-width token embeddings with feedback hidden state. Its normalized
draft-width output feeds logits, while a separate projection creates target-width
feedback. These are distinct tensors, so matching dimensions alone is insufficient.
The [target's main output](https://github.com/vllm-project/vllm/blob/3bb782621492711485dc86791b5978128783814a/vllm/model_executor/models/gemma4.py#L1378-L1389)
is normalized before return. The additional runner trace below establishes
which returned tensor reaches the proposer for the ordinary one-draft path.

Local implementation consequence: the current per-layer observer runs before
final normalization. Also, after rejected-draft rollback, `decoder.hidden` still
contains the rejected step's scratch result. This is harmless for ordinary
decode, which overwrites it from the next embedding, and the live continuation
qualification passed. A future assistant must not consume that scratch buffer.
Retain hidden/KV mirror updates per tentative input and publish only the resolved
prefix. Keep the second prediction pending outside KV. This ownership requirement
applies even if the first target verifier still uses two S1 calls.

The fetched sources and SHA-256 values are retained under
`gemma4-12b-evidence-20260914/int8-s2-restoration-20260916/upstream-gemma4-*.py`.
No assistant weights were downloaded and no drafter performance is established.

## One-draft runner trace

At the same revision, the [runner's Gemma branch](https://github.com/vllm-project/vllm/blob/3bb782621492711485dc86791b5978128783814a/vllm/v1/worker/gpu_model_runner.py#L623-L695)
does not enable auxiliary hidden outputs. Its
[proposal dispatch](https://github.com/vllm-project/vllm/blob/3bb782621492711485dc86791b5978128783814a/vllm/v1/worker/gpu_model_runner.py#L5207-L5289)
therefore selects the target's ordinary returned hidden states, using accepted
token indices or the padded equivalent after verification. The optional target
override method is absent from the inspected Gemma target implementation.
Consequently the initial assistant input is the final-normalized target hidden
state, not the last layer observer's pre-final-normalized residual.

For ordinary sequential drafting, the [base proposer](https://github.com/vllm-project/vllm/blob/3bb782621492711485dc86791b5978128783814a/vllm/v1/spec_decode/llm_base_proposer.py#L837-L857)
shifts token IDs by one but retains target positions and hidden states. At the
last committed position this pairs its hidden state with the already sampled
next token, which remains outside target KV. A local initial call should
therefore supply `position = committed_tokens - 1`, that normalized hidden
state, the pending token's scaled embedding, and KV through the committed
prefix. Advancing the assistant's RoPE position to the pending token's eventual
target position would disagree with this source contract.

With exactly one proposal, the [proposer returns after sampling draft logits](https://github.com/vllm-project/vllm/blob/3bb782621492711485dc86791b5978128783814a/vllm/v1/spec_decode/llm_base_proposer.py#L618-L650).
The assistant's projected feedback hidden state is needed for subsequent draft
steps, not this initial one-draft verifier. Skipping that unused projection is
a possible local optimization; it has not been implemented or measured.

The two additional source snapshots are `upstream-llm-base-proposer.py`
(SHA-256 `5bd11c41f0a8cf1272b5880d9878043f876bee6c7e04fb781e613f92930b15ab`)
and `upstream-gpu-model-runner.py`
(`87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51`).
This is an implementation contract derived from upstream source, not local
assistant execution or proof of acceptable draft acceptance under INT8 target
arithmetic.
