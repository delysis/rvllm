# Campaign executable snapshots

The global-decode campaign preparer now snapshots four executable inputs into
`QUEUE/executables/` before publishing its campaign and compile-job manifests:
the generator, native test binary, queue submitter, and ABBA retainer. Names
include the SHA-256 of the bytes and a fixed role. An existing snapshot is
accepted only if it has the expected hash; it is never overwritten or
"repaired" in place. A source change during copying fails preparation.

This removes a continuity hazard: prior campaigns pinned a mutable
`target/{debug,release}` path, so an unrelated rebuild could invalidate a
later oracle or timing job even though the candidate source and manifest had
not changed. Snapshots let the next campaign build while the current one is
still queued. They do not relax the queue's executable and input-hash checks.

Evidence: the focused generator suite passes ten tests, including creation,
reuse, and corrupt-snapshot refusal. A separate preparation probe produced
four SHA-pinned paths under a temporary queue, pinned the copied generator in
its compile job, and successfully launched the copied native test binary with
`--list`. This is a host workflow check, **not** a kernel/device result.

Already-submitted campaigns are immutable. W4 v02 still references its
original release paths; W8 v01 uses manually isolated debug binary copies.
Do not rebuild or delete either campaign's pinned files until its dependent
jobs finish. New campaigns prepared with this version use queue-owned
snapshots automatically. The snapshot directory is a local rebuildable cache
whose contents must remain available for the lifetime of its queued jobs;
campaign manifests and receipts carry the identities that can be pushed.
