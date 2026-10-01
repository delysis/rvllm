# Same-boundary independent reference plan

The existing original-12B-it CPU/Hugging Face script
`v3/scripts/dump_gemma4_hf_reference_logits.py` evaluates the full prompt
with `AutoModelForCausalLM` and selects `outputs.logits[0, -1, :]` at its
first step. The Metal direct CLI prefills **all** prompt IDs, then the
default-off `--teacher-prefill-last-logits` probe runs final normalization,
LM head, and logit softcap for all prefill rows, selecting the last row.
Those two readbacks refer to the same logical next-token position with the
same pinned prompt IDs. They do not imply bitwise parity: HF and Metal may
choose different arithmetic and output precision, and the installed CPU
Transformers 5.14.1 differs from the checkpoint's 5.10.0.dev0 config.

The ordinary Metal teacher-forced **step zero is a different operation**.
After the full prefill, `rvllm_metal_infer` calls `launch_rollout` with the
last prompt token again, `position = prompt_len - 1`, and
`context_len = prompt_len`; its logit readback follows this one-token decode
route. The CPU script's first full-prompt output is therefore not a direct
reference for the Metal teacher step-zero logits. Comparing their token IDs
or NLL without this distinction could falsely attribute a prefill/decode
route difference to a candidate kernel.

For the two already frozen MMLU natural-question cases, the next independent
numerical trial should:

1. Seal the exact original 12B-it checkpoint, tokenizer, MMLU source,
   prompt IDs, first target ID, script/executable, environment and output
   paths. Use a fresh immutable queue ID for each arm; never replay or
   overwrite prior results.
2. Run the unchanged CPU/HF script for **one full-prompt step** with exact
   prompt IDs and full logits, retaining its raw result and conditions.
3. Run separately frozen Metal native and combined prefill-last probes on
   those same prompt IDs and first target, requiring actual candidate
   prefill dispatch in the latter. Treat all probe timing as invalid.
4. Compare full-vocabulary finite logits, target rank, target NLL, greedy
   ID, and top-token margins at this one aligned logical position. Preserve
   both favorable and unfavorable differences and version/precision caveats.

That trial would provide a narrow, same-token-boundary independent reference
for *prefill-final* quality. It would not validate the post-prefill teacher
decode route, every continuation position, full-checkpoint quantization,
general language quality, performance, or promotion. A reference for the
teacher decode route must separately reproduce its cache/last-token replay
semantics; the unchanged HF script does not do that.
