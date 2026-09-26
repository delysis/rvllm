# Donor QMV real-checkpoint operator trial

These two queue jobs exercise the default-off `research-r4-sg8-k8` selector
added at source commit `8e4635ce`. They read one BF16 tensor each from the
pinned Google Gemma 4 12B checkpoint and quantize it to rvLLM's native
group-32 W4/W8 format. The selector admits only W4 Down `M1,N3840,K15360`
or W8 Output `M1,N3840,K4096/8192`. It uses the same authenticated packed
descriptor and output/ledger checks as the existing real-weight referee.

The queue owns the device slot. Inputs, executable, source tensor bytes,
generated MSL, packed values, scales, activations, and CPU reference are
hashed in the manifests or produced receipts. Nine ABBA samples per job are
diagnostic wall-clock operator timings, **not** GPU-timestamp qualification.
There is no thermal-stability dwell or quiet-process gate. These jobs cannot
qualify model quality, full-route selection, or an MLX-relative speedup.

The immutable queue jobs were submitted with `validator: null` before the
research-cell validator was built. Queue `succeeded` therefore means only
that the executable exited successfully. The copied receipts were subsequently
checked with `rvllm-low-bit-receipt-validate` (release executable SHA-256
`0857816135c4fb576ee6007c50124452997df71ec9f5b1ff1f155bfa3f99dcc0`):
both passed the exact research shape, selected-kernel, dispatch-ledger,
correctness, guard, repeatability, and derived-timing checks. This is
post-hoc validation, not a validator run by the queue at execution time.

| Tensor / candidate | Candidate vs quantized CPU reference | Native BF16 vs dense CPU reference | Median native BF16 / W4 or W8 wall time | Exploratory ratio |
| --- | --- | --- | --- | --- |
| Layer 0 down, W4 | max abs 0.00390625; relative L2 0.0008470 | max abs 0.001953125 | 0.872666 / 0.303417 ms | 2.876x |
| Layer 0 output, W8 | max abs 0.00390625; relative L2 0.0008003 | max abs 0.000061035 | 0.723917 / 0.372000 ms | 1.946x |

These ratios compare a dense BF16 projection with quantized-weight operator
execution, not equivalent-format incumbent kernels, MLX, or a full model.
There is only one real tensor per shape; neither job establishes a general
quality or speed claim. Both jobs passed two exact candidate-dispatch ledger
checks, 19 timing dispatches, unchanged guards, and bitwise repeated output.
The earlier GPU-timestamp synthetic confirmation
rounds showed a consistent direction but failed the 5% control-drift gate,
so no candidate is promotable from these results alone.

Copied receipt SHA-256: W4
`b6069ac62578ae0921d88f5bc2845c8f7385941ab156c2c90d40d2db3239b71b`;
W8 `c0a9b26805dac1e968881060934c2ac1d3b5ddcfc8fee32872c966b394354953`.
