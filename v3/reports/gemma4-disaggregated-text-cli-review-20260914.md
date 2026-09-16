# Ordinary text prompts on the qualified disaggregated path

Read-only code/design review, 2026-09-14. No implementation edits, builds, hardware execution, or cache operations. Parent reports INT8 passing 7 prompts / 31 IDs / 24 ANE steps at 5.203 tok/s; global-1024 qualification is still separate.

**Recommendation:** extend the existing binary with a deliberately bounded, SHA-checked **single-user, non-thinking Gemma 4 text mode**. Keep one prefill/import/decode implementation and its phased residency. A full Jinja renderer is unnecessary for this first usable CLI, but the bounded mode must not claim general chat-template, history, tools, or multimodal support.

## Prompt formatting: bounded mode versus full rendering

The actual checkpoint is revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. Its standalone [chat_template.jinja](/Users/george/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7/chat_template.jinja:183) is 18,683 bytes, SHA256:

`ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`

For exactly one `{role:"user", content:<string>}`, no tools, `enable_thinking=false`, and `add_generation_prompt=true`, its output is:

```text
<bos><|turn>user\n{trimmed_user_text}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
```

Here `\n` denotes an actual newline. The template emits BOS at line 188, maps roles/opens turns at 230–234, trims user text at 328, closes the turn at 373, and emits the non-thinking model prefix at 381–385. The existing [HF fixture generator](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/fresh-chat-oracle-source.txt:34) uses precisely this request shape.

| Choice | Integration and limits |
|---|---|
| **Bounded single-user mode — recommended** | Read and hash the checkpoint template using existing `std::fs` / `sha2`; reject a mismatching template before loading weights. Resolve `bos_token` from `tokenizer_config.json`, verify it resolves with `Tokenizer::token_to_id`, build the exact prefix/text/suffix, then `Tokenizer::encode(rendered, false)`. No template dependency. Advertise `gemma4-single-user`, not arbitrary chat. |
| Full checkpoint renderer | No Rust chat-template renderer was found in v3. MiniJinja 2.24.0 supplies `Environment::new`, `add_template`, `get_template`, and `Template::render`; add `minijinja-contrib`'s `pycompat` callback for the template's `.get()` / `.split()` calls. Supply messages, BOS/EOS strings, tools, thinking flags, and generation flag explicitly; disable autoescaping and match Hugging Face whitespace settings. Register `raise_exception`. This adds compatibility/parity testing across a 390-line template and should accompany an actual requirement for history/system/tool messages. [Pinned renderer](https://github.com/mitsuhiko/minijinja/blob/0ca749f7ba507514fa6b052c74130ae6ae472e03/minijinja/src/environment.rs), [Python-method adapter](https://github.com/mitsuhiko/minijinja/blob/0ca749f7ba507514fa6b052c74130ae6ae472e03/minijinja-contrib/src/pycompat.rs#L49). Its declared Rust minimum is 1.70, within this workspace's 1.80; compatibility was not executed. |

For exact bounded-mode parity, account for Jinja/Python `strip()` semantics: ordinary Rust `trim()` covers ordinary whitespace, but Python additionally strips U+001C–U+001F. An explicit trim predicate can include those four characters. Do not escape or normalize content differently from the checkpoint template. A whitespace-only message can be rejected as a CLI input policy.

Two reuse traps:

- [Existing Metal tokenization](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/bin/rvllm_metal_infer.rs:390) unconditionally prepends token 2 unless `no_bos`. Reusing it on rendered chat duplicates BOS. Its tokenizer loading pattern at line 376 is reusable, but those helpers are private to that binary.
- This checkpoint's `tokenizer.json` postprocessor is identity: even `encode(text, true)` **does not add BOS**. An explicit raw mode needs `encode(text, false)` plus the metadata-resolved BOS policy. The existing package [template fingerprint helper](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/model_package.rs:821) only considers embedded `tokenizer_config.json.chat_template`; it does not authenticate this standalone Jinja file.

## Streaming, stopping, and context

Load one `tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))` before backend preparation; avoid the current per-case reload at [lines 325–329](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/bin/rvllm_disaggregated_infer.rs:325).

Extract/reuse the owned [IncrementalTextDecoder wrapper](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/apple_continuous_worker.rs:575), including its `read_index = prefix_index` reset. It calls safe `tokenizers::tokenizer::step_decode_stream` with `skip_special_tokens=true`. The locked tokenizers version is 0.20.4; its plain `DecodeStream::step` does not perform that reset, so treat switching wrappers as a behavior change requiring its own tests. Never concatenate `decode(&[one_id], true)` calls: spacing and byte-fallback Unicode require state. At completion, decode the complete generated-ID vector once and append any remaining suffix after verifying it starts with already-emitted text; this handles buffered incomplete byte sequences and supplies a consistency check. Write chunks to a locked stdout and flush; send timing/status to stderr.

Reuse [load_eos_token_ids](/Users/george/Downloads/rvllm/v3/crates/rvllm-loader/src/generation.rs:7): generation config first, then model config, then nested text config. Here EOS is **[1, 106, 50]**, not only `<eos>`: 106 ends a turn and 50 is `<|tool_response>`. Include terminal IDs in diagnostic token counts, skip them in visible text, and never feed an EOS token into ANE. `max_new_tokens` includes the first token sampled by Metal. If that token is EOS, or the limit is one, skip ANE loading entirely.

For prompt length `P`, requested outputs `N >= 1`, and operational capacity `C`, validate with checked arithmetic **`1 <= P <= C` and `P + N - 1 <= C`**, before either backend loads. This matches [existing admission](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/bin/rvllm_disaggregated_infer.rs:147) and [ANE's position bound](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/gemma_ane_decode.rs:511). The output sampled after consuming position `C-1` need not itself be consumed. Never silently truncate the rendered prompt. Source currently admits only capacities 64/1024; the checkpoint's 262,144 positional limit and enormous tokenizer `model_max_length` do not establish supported runtime capacity.

Generation remains **greedy**: [ANE selects top_five[0]](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/gemma_ane_decode.rs:630). Loading EOS metadata does not apply the checkpoint's sampling defaults or `suppress_tokens`; do not describe the CLI as reproducing all Hugging Face generation settings.

## Minimal integration and residency

1. Add mutually exclusive reference and text inputs to the current binary: `--prompt` or UTF-8 `--prompt-file`, an explicit prompt-format choice, and positive `--max-new-tokens`. Parse/format/tokenize/validate entirely before hardware setup. Keep the existing reference mode and optional evidence output; ordinary text mode should not require reference JSON, KV files, or per-step JSON files.
2. Generalize the reference record into a prepared request with optional expected IDs and a generation limit. Share [existing Metal prefill](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/bin/rvllm_disaggregated_infer.rs:196) and [existing ANE loop](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/bin/rvllm_disaggregated_infer.rs:262); reference matching becomes an optional observer/check, not a second inference engine. Keep the explicit `GemmaAneDecode` owner and existing cache policy; do not route through another generic backend that loads a second model.
3. For ordinary mode retain only one `AneDecodeStart`. Stream its first token immediately after synchronous prefill, drop Metal as [already done at line 257](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/bin/rvllm_disaggregated_infer.rs:251), then lazily load ANE, `import_prefill(&start.cache)`, and **drop the imported host snapshot** before the decode loop. Import copies into each layer's owned attention state. Repeatedly feed the actual previous output to `decode`; never re-feed the prompt's last token. [Import/ownership contract](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/gemma_ane_decode.rs:465), [first-token contract](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/ane_prefill.rs:39).
4. Preserve phased residency for any later interactive mode. A new turn requiring Metal prefill must release ANE before preparing Metal, then reverse phases again. Keeping both resident reintroduces the measured memory problem. Bounded one-shot text support can ship independently of efficient multi-turn serving; report initialization separately from warm token throughput.

Before promotion, CPU-only formatter/tokenizer tests should match every existing chat fixture's prompt IDs and cover non-ASCII text, whitespace, literal delimiter strings, missing/wrong template hash, BOS duplication, EOS as first token, `N=1`, and capacity boundaries. Streaming tests should cover long output and fragmented UTF-8. Parent-controlled end-to-end qualification should then exercise the real new text CLI once with an existing fixture prompt and once with an ordinary unseen prompt. This review performed none of those tests.
