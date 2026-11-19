# Overnight development batch

The eight-episode M5 pilot completed on the user's Pod. H0 return 88129.7128,
V0 63394.1952, P0 48514.3596, V3 82035.9312 and T0 -15314.1038.
These are validation results from one development seed, not final evidence.
V3 combines PBRS and Dyna; it cannot isolate either effect.

Run `bash scripts/launch_m5_overnight.sh` on the Pod, after fetching this commit.
It tests the orchestration layer and launches with nohup, no stdin, and a log
outside the output directory. SSH may close once BATCH RUNNING is printed.
This repository checkout must remain unchanged during execution.

## Fixed scope

- Same SKU FOODS_3_090, raw file hashes, simulator assumptions as m5-pilot-01.
- H0 once; T0/V0/P0/S0/V1/V2/V3/N0 at development seeds 3,4,5: 25 jobs.
- 300 episodes of 365 days: 109500 real regional steps per learner;
  2628000 real training steps overall, plus Dyna synthetic steps.
- DDQN exploration decays to 0.05 across the first 50000 real steps.
  T0 retains its documented episode-based schedule. All DDQN ablations share settings.
- Validation every 10 episodes. Test is never evaluated. Report seeds 0,1,2 remain unused.
- Sequential execution on one GPU. Initial estimate 5–8 hours extrapolated from
  the small pilot, uncertain because checkpoint/replay costs grow.
- One-hour limit per job; eight-hour global training/preflight budget, plus short
  shutdown and analysis overhead. A job failure is recorded and later jobs continue.
- Deep checkpoints preserve the last completed episode. T0 only saves final tables.
  There is no automatic batch retry/resume; inspect artifacts before continuing.
- Same output cannot be launched twice. Logs, status, plan, protocol, input hashes,
  checkpoints and validation summaries remain under /workspace.
- The process exits at completion/budget. It does NOT stop or terminate the Pod;
  the provisioned Pod remains billable until the user stops it.

## Inspect

```bash
tail -n 30 /workspace/rl-project-validation/m5-night-01.log
cat /workspace/rl-project-validation/m5-night-01/status.json
cat /workspace/rl-project-validation/m5-night-01/analysis/summary.json
```

Summary completeness and missing_runs must be checked before comparing methods.
Even a complete batch is development evidence, not a confirmatory test.
The raw completed validation curves are analysis/validation_curves.csv.
No plotting dependency is required for the overnight summary.
