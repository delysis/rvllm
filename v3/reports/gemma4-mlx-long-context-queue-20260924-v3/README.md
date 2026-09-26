# Gemma 4 MLX long-context queue v3

This queue supersedes, but does not overwrite, v1 and v2.

- v1 failed before model load because Homebrew MLX-LM 0.30.0 did not support
  `gemma4_unified`.
- v2 used the corrected pinned MLX-LM source and completed its first command,
  but live sampling observed competing Cargo/Rust processes and rejected the
  timing block. That rejected stdout remains useful only as diagnostic data.
- v3 uses a referee that preserves condition-rejected timing blocks as
  `rejected` terminal evidence and advances to subsequent predeclared jobs.
  It never retries or promotes the rejected block.

The v3 matrix and sealed MLX identities otherwise match v2. The resident
referee executable SHA-256 at admission was
`1658c8475987abc8da89991878fc9a9168281e38a7f13a90e1161c95c82eddec`.
