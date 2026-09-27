# BF16 checkpoint identity correction

The prior exact-token MLX–rvLLM comparison used **different checkpoints**.
The cached `mlx-community/gemma-4-12B-bf16` model card identifies its
`base_model` as `google/gemma-4-12B` and says the MLX package was converted
from that base model. The rvLLM run used
`google/gemma-4-12B-it` snapshot `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
This is stronger evidence than the previous statement that tensor-byte
identity was merely unproved. It does **not** establish that the checkpoint
difference is the sole cause of generated-token divergence.

Independent raw safetensors-header inspection found equal BF16 dtype and
shape, but different tensor-byte SHA-256 for three corresponding layer-0
weights. The SHA values cover only the tensor data ranges named in the
respective safetensors headers, not the enclosing file. MLX package keys
start `language_model.model.layers.0`; original checkpoint keys start
`model.language_model.layers.0`.

| Layer-0 tensor | Shape | Google 12B-it tensor SHA-256 | MLX base-conversion tensor SHA-256 |
| --- | --- | --- | --- |
| `self_attn.q_proj.weight` | 4096 × 3840 | `3ae6a6f7f56408ec8fc7a78ec9cb9d29054af0a717cf92552af09591f63d6444` | `3dc94af99b585e5641f06ef5c89c594065eea9dd7d24b8e65d82d965b517388e` |
| `mlp.down_proj.weight` | 3840 × 15360 | `39c7935f5f0d2b491af13596579eadd58101a2af128ffec922b20f0a3ef29bab` | `22e853e90de75fb592489f189e7923295da44550943b14848ca37302f92e2740` |
| `mlp.gate_proj.weight` | 15360 × 3840 | `382d11f00312d1f2f00729252250be971b17372e3f1cf73ef7b9498db806d9a6` | `f0ed2b2ca18eb996da4bf6ed611a30dfce933626c2decfc044e60911492e4d62` |

Model-card SHA-256 is
`788c3afc2674a8932ff553db0af6a2dd788df8e4cec3ae7f71d9423f0c760673`;
MLX conversion config SHA-256 is
`95ac51f934f85c9243f970e6077ad5ff6056b50de7bb4c4be866205f4a7901b6`;
MLX weight index SHA-256 is
`5a2037525ab516767d2a213bf7cb74f7d05940bbccb64e1de281f93b40953577`.
The original 12B-it config SHA-256 is
`478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9`;
its single model safetensor SHA-256 is
`5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d`.

For dense BF16 kernels, these checkpoints have the same relevant projection
shapes and storage type. Tensor *values* should not materially change the
amount of matrix work, so the ~5.20× prompt and ~1.40× decode framework
figures remain useful **planning-grade performance signals**. Different
checkpoint weights do, however, invalidate numerical/output-equivalence
claims and prevent attributing the continuation mismatch to an implementation
bug. The ratio is still not a strict timing qualification: MLX's derived
first-token prompt phase and rvLLM's collected prefill phase have different
boundaries, their decode-rate denominators are not proven identical, and the
first MLX host-condition sample failed the freshness check. MLX-LM's pinned
`gemma4.Model.sanitize` maps the original checkpoint's
`model.language_model.*` keys into MLX's expected
`language_model.model.*` keys, so the next queued stage arm points MLX-LM
directly at the original `google/gemma-4-12B-it` snapshot. A successful
load and source inspection are still not a full output-equivalence proof;
that arm is a stage microbenchmark only. A same-checkpoint exact-token full
route comparison and internal-tensor/reference checks remain necessary. A
same-checkpoint 512+64 exact-token MLX-LM run was therefore queued as
`g4-donor-mlx-it-exact-bf16-512-g64-01`, depending on a direct-load stage
arm. The first stage arm failed in the harness identity collector before
device loading because it required a sharded index; this checkpoint has a
single safetensor. Both original manifests and the failed stage receipt are
preserved. A focused single-file identity repair passed 10/10 Python tests;
new stage and dependent exact-token arms use IDs ending `-02`. Their results
are pending and must not inherit the earlier base-checkpoint claims.
