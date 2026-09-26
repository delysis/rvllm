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
- `purpose`: `timing`, `exploratory_timing`, `correctness`, or `preparation`.
  Preparation and correctness may succeed despite ineligible sampled
  conditions and make no performance claim. Exploratory timing retains all
  observations but cannot promote from an ineligible run. Strict timing
  requires eligible observations throughout the child lifetime.
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
  `is_processing: false`. Unknown, busy or unreachable servers block launch.
  Preparation jobs may use `thermal_state: null` to accept either benign state;
  timing jobs must name exactly one state. Unknown/Serious/Critical is refused.
- `stable_seconds` (legacy compatibility field, 0–600; ignored by the current
  queue), `max_wait_seconds` (at most one day), and `max_run_seconds`
  (1–3600). New manifests must write zero.

Jobs are selected in lexical ID order among satisfied dependencies and current
launch conditions. The queue does not wait for thermal, clock, or power
conditions to remain unchanged: it records their changes and relies on
counterbalanced sampling, drift gates, and later confirmation. A currently
unready job does not block an independent ready job; use dependencies to
enforce ABBA order. Input hashes are checked again before launch, followed by
a fresh condition check. A failed
or incomplete attempt stops the queue, including after restart. Review its
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

The launch check requires fresh known power controls, no reported CPU
restriction, enough available disk space and no matching competing process.
It does not require a dwell interval or unchanged conditions. Name matching
uses process executable names;
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
