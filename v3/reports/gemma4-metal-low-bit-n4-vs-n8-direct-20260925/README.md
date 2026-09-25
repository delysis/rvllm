# Gemma 4 direct N4-vs-N8 Metal referee

This packet contains 22 queue-ready jobs: ABBA and independently ordered BAAB
runs for each of the 11 cells that was a stable operator win in at least one of
the sealed N4 or N8 reports. No other role/format/M cell is admitted by the
binary or represented by a manifest.

The referee uses the layer-0 real BF16 checkpoint tensor, independently
quantizes W4/W8 group-32 weights, compares both schedules with the CPU BF16
oracle, requires bitwise repeatability and bitwise equality between N4 and N8,
checks output guards, and verifies exact per-role dispatch counts and kernel
names. Timing is direct N4 against N8; native BF16 is not an arm. Each manifest
runs nine four-dispatch blocks (18 samples per arm). The BAAB job depends on its
cell's ABBA job so the two orderings cannot overlap.

Host conditions are retained by the queue observer, but `stable_seconds` is
zero: thermal or idle observations are evidence, never a wait gate. These jobs
produce projection-operator evidence only. They do not establish a full-route,
model-quality, shipping-selector, or promotion result.

The executable and validator hashes in the manifests bind the release binary
and strict receipt validator built from this tree. Any source or tool change
requires rebuilding and regenerating every manifest; the queue fails closed on
an identity mismatch.
