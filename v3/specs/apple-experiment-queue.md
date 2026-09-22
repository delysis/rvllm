# Local Apple experiment queue

`rvllm_experiment_queue` runs independently while development continues. It
watches a local directory for immutable job submissions, serializes trials
with a cooperative accelerator lock, and saves stdout, stderr, raw power
observations, process activity and a JSON result. It never invokes a shell,
changes power settings, repairs a cache implicitly, or retries a failed job.

Build the single binary with `cargo build -p rvllm-runtime --offline
--features macos-private-ane-research --bin rvllm_experiment_queue`.

```
rvllm_experiment_queue submit /absolute/queue /absolute/job.json
rvllm_experiment_queue run /absolute/queue /absolute/shared-accelerator.lock 600
rvllm_experiment_queue status /absolute/queue
rvllm_experiment_queue stop /absolute/queue
rvllm_experiment_queue audit /absolute/queue /absolute/shared-accelerator.lock
rvllm_experiment_queue qualify-host /absolute/new-evidence /absolute/shared-accelerator.lock /absolute/conditions-job.json
```

Launch `run` as a background process; it needs no interactive terminal. The
last argument is idle timeout in seconds, not a pending-job deadline. New jobs
can be submitted while a job waits or runs. `status` is compact; full evidence
lives in `results/<id>/`. A STOP marker prevents further launches. An active
child is joined naturally, including on an error or a handled parent signal;
there is no forced accelerator termination. A child's runtime overrun is
recorded immediately as `overdue-awaiting-safe-child-exit`, disqualifies the
job, and does not cause a kill or retry. This cannot recover from kernel panic,
unhandled termination, or hardware hangs. An incomplete attempt is never
silently replayed on restart.

## Manifest

Schema is `rvllm.experiment_job.v1`. Unknown fields are rejected. Every job has:

- `id`: up to 96 ASCII letters, digits, dashes or underscores. IDs are unique.
- `purpose`: `timing` or `preparation`. Preparation can succeed despite power
  changes, but its timing eligibility stays false and it makes no performance
  claim. Timing requires eligible observations throughout the child lifetime.
- `command`: `executable: {path, sha256}`, absolute `cwd`, `args` array and
  optional `env` map. Arguments expand only the literal `{output}` token to
  the exclusive attempt directory. No shell interpretation occurs. Environment
  inheritance is cleared; PATH and LANG are fixed, plus explicit manifest env.
- `inputs`: optional `{path, sha256}` pins for prompts, model config, kernels
  or reference reports. Submission checks pin syntax and file presence without
  scanning model-sized inputs. Full hashes are checked before and after
  execution, outside timed child phases. Queued status is not hash
  qualification. Unlisted files are not certified by these pins.
- `validator`: optional pinned command with the same structure, run only after
  a successful eligible timing trial (or successful preparation). Its output is
  separate. Use a bounded, read-only validator, never an accelerator workload.
- `after`: previously submitted job IDs which must have succeeded.
- `conditions`: explicit `power_source` (`ac`/`battery`), `low_power_mode`,
  `pmset_power_mode` (0–2), `thermal_state` (0 nominal or 1 exploratory Fair),
  `minimum_free_bytes`, absolute `disk_path`, and `quiet_process_names`.
  Optional `idle_llama_servers: [{pid, port}]` allows explicitly identified
  persistent localhost llama servers only while all `/slots` entries report
  `is_processing: false`. The PID must also own that IPv4 loopback or wildcard
  listening port, checked rootlessly with `lsof`; an unrelated idle server cannot
  exempt a busy PID. Unknown, busy, unreachable or unowned endpoints block launch.
  Preparation jobs may use `thermal_state: null` to accept either benign state;
  timing jobs must name exactly one state. Unknown/Serious/Critical is refused.
- `stable_seconds` (1–600), `max_wait_seconds` (at most one day), and
  `max_run_seconds` (1–3600).

Jobs are selected in lexical ID order among satisfied dependencies and stable
launch conditions. The quiet window restarts after any trial and after an
observation gap over 2.5 seconds; equal controls on either side of an
unobserved interval do not establish continuity. Waiting deadlines are retained
when a window restarts. An unready job does not block an independent ready job;
use dependencies to enforce ABBA order. Input hashes are checked again before
launch, followed by a fresh condition check. Before probing or selecting any
candidate, the entire queue is inspected.
A failed, incomplete, orphaned or mismatched attempt stops the queue, including
after restart; a lexically early ready job cannot hide a later failure.
A successful result must identify its job and have consistent exit, pin,
overrun, purpose and validator outcomes. Its saved manifest must match the
submitted manifest. Missing/cyclic dependencies and filename/ID mismatches
are rejected before selection. Review the
evidence and submit a revised attempt to a new queue; no automatic cache repair
or statistical retry policy is inferred. Dependency submission order prevents
cycles through normal submission. Edit a source manifest and submit a new ID;
do not edit submitted jobs in place.

## Builds and offline analysis

Preparation jobs may run bounded host tests or Cargo builds as well as device
qualification. Running builds through this same worker prevents them from
overlapping its hardware timing, while the coordinator can continue reviewing
and editing other work. Use an absolute pinned Cargo executable, explicit
toolchain paths, `--offline --locked`, a reused target directory, and source /
manifest pins for the work being built. Set a realistic runtime bound.
Preparation success is never a performance result.

Pinned sources must remain unchanged until their queued tests and builds
finish. If review requires a new revision, record STOP, let the active child
finish, preserve and withdraw only the never-started superseded manifest, then
submit a new job ID and resume. A failed or partially executed job remains
untouched. Freeze and hash the resulting executable before submitting device
trials. A pin list certifies only listed files; it is not automatically a
complete transitive source or toolchain attestation.

## Measurement boundaries

Waiting jobs share raw disk, process and idle-server observations within one
queue pass. Each job independently applies its own policy and child exclusions;
policy decisions are never cached. Observations expire after 2.5 seconds,
including probe duration, and are never reused across passes. Prelaunch and
active-child checks take fresh observations. JSON reports use buffered reads
so completed-job metadata does not consume the observation window.

The gate requires fresh known power controls, no reported CPU restriction,
stable controls for the configured interval, enough available disk space and
no matching competing process. Name matching uses process executable names;
the current trial and its descendants are excluded. Include `cargo`, `rustc`,
other inference executables and test executables when those can contaminate
the experiment. The cooperative lock only covers participating workers.
Unlisted applications and brief activity between samples remain uncontrolled.
The optional localhost check also runs during trials; a busy or unavailable
slot disqualifies timing without interrupting either service. It reads no
prompts and changes no server state. Idle model memory remains resident and
is an ambient condition to record across both sides of a comparison.

Raw backend measurements still decide prompt/decode work counts and numerical
acceptance. Queue wall time includes process startup and is **not** decode
latency. Parent CPU counters exclude child, GPU and ANE work. CPU cycles are
never used to normalize device time. Fair data remain explicitly exploratory,
and are not pooled with nominal or a different power stratum. Use repeated
counterbalanced comparisons and baseline drift checks before changing defaults.

Each backend invocation must bound its own work. A waiting deadline prevents
an indefinite pending launch; a runtime deadline identifies overruns while
preserving safe child ownership. File hashes and cooperative locks prevent
ordinary artifact drift and overlapping queue workers, not adversarial local
filesystem changes.

## Resume inspection and live host qualification

`audit QUEUE GLOBAL_LOCK` acquires both worker and shared accelerator locks.
It inspects all attempts and dependencies, then hashes every pending command,
validator and input. Each file is read once per audit; every expected hash is
checked independently, including conflicting expectations for the same path.
The JSON report includes all pin mismatches and the running worker's digest.
A mismatch exits nonzero. No STOP marker or manifest is changed, no observer
or trial is started, and a successful audit is not reused to skip launch-time
pin checks. Malformed/quarantined queue state fails before pin hashing.

`qualify-host NEW_OUTPUT GLOBAL_LOCK CONDITIONS_JOB.json` is an explicit live
host-only exercise. NEW_OUTPUT must not exist. It copies only the supplied
job's conditions; **it never executes that job's command**. Under one continuous
shared hardware lock it stages an impossible-disk negative control, a later
ready `/usr/bin/true` job, and a dependent `/usr/bin/touch` job that sets STOP
only in the new evidence directory. Each stock executable is pinned. These
are preparation jobs with two-second stability windows, 30-second waiting
bounds and five-second runtime bounds; the original campaign durations and
16 GiB floor are unchanged. Failed runs retain their evidence without retry.
The conditions file's exact bytes and digest are saved and checked after use.

Success requires that the blocked job was never attempted, both independent
and dependent jobs succeeded, and neither claimed timing eligibility. The
normal worker loop, observations, dependency resolution, pin verification and
child ownership are exercised. This is not inference, numerical, throughput,
S2 or KV-import acceptance, nor a substitute for an actual timing job's quiet
window. Run it from the exact worker binary being qualified and retain its
digest; hosted tests cannot freeze a worker in the target machine's campaign.

During a timing child, activity-observation gaps over 2.5 seconds disqualify
the result even when power continues to be sampled and both endpoint probes
are ready. Preparation reports always set `sampled_conditions_eligible=false`;
`raw_conditions_eligible` separately preserves their actual observed gate result.
Historical receipts are not rewritten.
