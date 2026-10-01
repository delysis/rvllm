# E2B qualification referee: host-test path correction

At PR #8 head `1725da358196e2d6809bea5fa860a10b8e9d26b8`, hosted `cargo test --workspace` failed only `sealed_manifests_match_source_prompt_targets` in `rvllm_gemma4_e2b_multicase_qual_summary`. The test read absolute prompt paths embedded in the immutable queue manifests from George's worktree; those paths do not exist on a GitHub runner. The three manifest-hash checks and the other referee tests passed.

The correction is confined to the host test. It remaps each prompt path in an in-memory job copy to the corresponding checked-in file under the current checkout, first requiring that the file's SHA-256 matches the manifest's original pin. It then exercises the unchanged production `verify_prompt_inputs` function and its negative target-mismatch case. No manifest, submitted queue job, result, production referee gate, model or prior failure receipt was modified. The five focused default-feature tests pass locally after this correction.

The source hash at original submission was `06236b4fa29325c0c99cb2d617c965e2a93b65f894c2cba7ec9b666577024e3f`; the test-only corrected source hash is `db04ace8e3d90c9ded0c89cc1d117b06460708eab888aaefa1a775343c460d6e`. The E2B cohort had already failed before inference because its manifest omitted the BF16 metallib, so this correction does not rescue or rerun it.
