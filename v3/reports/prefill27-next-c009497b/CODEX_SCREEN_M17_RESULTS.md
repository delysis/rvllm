# Codex PR27 second-rung screen (2026-09-27)

This extends the [M=6 component screen](CODEX_SCREEN_M6_RESULTS.md), not production or full-model qualification. Four fresh dependent `prefill27-first17-20260927-s00-m17` through `s03-m17` correctness jobs ran once in the existing serial queue. All four terminated `succeeded`, exit 0, sampled-condition eligible, with no violations. Their screens all report `status: pass` through M=17; the packet's higher-rung command rechecks earlier fixtures inside these new job IDs, without resubmitting an old queue job.

The new wide64 down-norm arm passed its same-process fragment layout/product probe. At M=17, it and old pipeline32x64 have identical complete output and FP32 raw-intermediate hashes for structured and periodic fixtures: structured `ef98e0c42d0574ec0d3a80615a43ac31a3df3d5566f495fbceeb78162784c236` / `1a782ede5c009b4336bb14c4909a42d658bf1661c0dc0eddf90d0b9186289688`; periodic `7b8732ece85d7edbc0229b47bacde539afdb7509398e4492186831e4ba08d6aa` / `fdef78476eb776db4e425bbaf1fe5357756e289197e39cb0c7a667453f940088`.

The new MMA8K32 local-attention arm passed its layout/product probe, the shared structured/periodic fixtures and its extra dense-V/masked-NaN-V fixtures. At M=17, its shared output hashes equal old Q4K16: structured `6660ec68ccdbb56c95bd0efe2b38e04651a97a12ea9954ab3f8e445ec98d4ad4`, periodic `b91345a28e7093ed8698fc6a817f1510188b5557dd1f62df8bc48574ae8d20ed`. Negative metadata/page/shape/thread cases remain in the per-cell JSON.

Original operator artifacts, including binary readbacks, remain unmodified under `screen-m17-20260927/screens/` (approximately 954 MiB). Complete compact queue receipts are in `prefill27-first17-queue-results.tar.gz`, SHA-256 `2f5dbe3190579e43ee55bc73e780799955b12c8eb5c0b6d969109c22b5e5ac1d`. All per-screen JSON is in `prefill27-first17-screen-json.tar.gz`, SHA-256 `5dddaa61deeb4fb95b0caa8701fb72ac7e3fa2dfa4534c418aa8f6e1f97c53f6`. The config SHA-256 is `0a78d2b9ff9c9cdaa23e5fcc51891c09bf7426386c38d1ac7be205511aae0929`.

No timing or full-model test was performed. These two short synthetic lengths do not establish longer-context or checkpoint-wide correctness, speed, or promotion.
