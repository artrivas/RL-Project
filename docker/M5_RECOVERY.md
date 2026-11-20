# Continue the interrupted overnight batch

User-reported status: budget_exhausted after 8 hours, 20 successful jobs,
S0_seed5 terminated by the controller with SIGTERM (-15) after 977 seconds,
and V1/V2/V3/N0 seed5 never started. The five missing manifests do not
represent five algorithm errors.

Run on the Pod:
```bash
bash scripts/launch_m5_recovery.sh
```

This is a completion of the same development experiment, not a new budget.
The added scripts do not modify any src/retail Python file, preserving the
checkpoint's code fingerprint. The existing suite, settings and assumptions
are used verbatim. The model runner also checks device, torch version,
configuration, data and code compatibility before resuming.

The recovery:
- Validates finished results and raw file hashes.
- Skips jobs with validated complete manifests.
- Appends --resume only for an incomplete deep job with a saved checkpoint.
- Refuses to restart an existing output without a checkpoint.
- Detects active trainers and serializes recovery controllers with a Linux lock.
- Keeps original status.json, logs and analysis unchanged.
- Saves its plan, logs and new status in m5-night-01/recovery-01.
- Runs a fresh complete/partial aggregate in recovery-01/analysis.
- Uses a four-hour training limit and 90-minute per-job ceiling.
- Does not evaluate test, stop the Pod, or retry failed jobs automatically.

Estimated remaining time is about 3 hours based on the measured seed3/4 jobs.
Actual progress in S0 is read from checkpoint.pt, not inferred from wall time.
Only the last completed episode is recoverable; an interrupted episode is repeated.
A checkpoint already at the target but missing its manifest is refused for review.

Read:
```bash
tail -n 30 /workspace/rl-project-validation/m5-night-01/recovery-01.log
cat /workspace/rl-project-validation/m5-night-01/recovery-01/status.json
cat /workspace/rl-project-validation/m5-night-01/recovery-01/analysis/summary.json
```

The old analysis/summary.json remains the original incomplete snapshot.
The new summary must show complete=true before treating the matrix as complete.
