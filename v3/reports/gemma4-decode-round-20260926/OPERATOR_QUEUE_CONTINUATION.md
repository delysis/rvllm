# Operator campaign queue continuation

New campaigns containing only exact-shape decode operators now publish a
compile-continuation job alongside their compile jobs. Its dependency list is
the complete compile family. Once all compile receipts succeed unchanged, the
same pinned generator validates each build, creates the native-oracle jobs,
and submits them plus an oracle-continuation job. That second continuation
depends on the complete oracle family; only after their exact source,
executable, numerical, and output-hash checks pass does it create and submit
paired timing jobs. No stage selects a winner or promotes a route.

The queue still owns serial execution, dependency ordering, immutable job
manifests, condition observations, and failure receipts. The controller
requires the complete campaign candidate family and its own exact result
directory, verifies its generator and submitter pins, and writes an immutable
stage receipt listing the jobs it submitted. A failed predecessor prevents
the dependent stage from executing; no failed receipt is silently retried.

Host verification: the focused generator suite passed twelve tests. A fresh
isolated preparation probe for `metal-qmv-w8-g32-r4-sg8-k8` produced one
compile job, a compile continuation depending on it, and an oracle
continuation depending on the future oracle job. The real queue submitter
accepted the compile job and its dependent continuation in that isolated
queue. No daemon or accelerator was started for that probe.

The subsequent [joint W4/W8 campaign](g4-donor-r4sg8k8-confirm-03/campaign.json)
ran under the persistent device queue and verified both transitions end to
end. The [compile advancement](g4-donor-r4sg8k8-confirm-03/advance-compile-receipt.json)
submitted both oracle jobs only after both compile receipts succeeded; the
[oracle advancement](g4-donor-r4sg8k8-confirm-03/advance-oracle-receipt.json)
submitted three paired timing jobs only after both native oracles passed.
All submitted jobs then ran. The advancement receipts have SHA-256
`7becac1d59bff742b649effc23ec56a6741a63fcda5f0630fe4d1e9faa904175`
and `77dfbb31bd9b61ba3421bad7da2a86a443b5aa154c4a6ec5244cead34ad6c0e7`.
This verifies the queue continuation mechanism, not a speedup or promotion;
the three timing receipts all failed the 5% control-drift gate.

Previously prepared W4 v02 and W8 v01 campaigns remain immutable and retain
their manual advancement path. The new controller applies only to campaigns
prepared with this version of the generator. These trial gates must still
be followed by real-weight/full-route confirmation and the project's referee
promotion requirements.
