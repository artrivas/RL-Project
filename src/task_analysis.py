"""Summaries of ``src/task_train.py`` runs for the comparison tables (saved as JSON artifacts).

Per condition and evaluation variant, over training seeds:

* ``final``: greedy success at the last evaluation (mean, min, max, bootstrap 95% CI of the mean
  over seeds);
* ``auc``: mean greedy success over all evaluations (sample efficiency: higher = learned sooner);
* ``online_final``: epsilon-greedy success of the last 10% of training episodes (the behaviour
  while exploring; on-policy methods optimise this, off-policy ones the greedy policy);
* ``seed_std``: standard deviation of the final greedy success across seeds.

Two conditions differ "detectably" only when their bootstrap CIs do not overlap; with 5 seeds
this is a coarse test and is reported as such.

    python -m src.task_analysis --runs notebooks/artifacts/runs_shooting --out notebooks/artifacts/shooting_summary.json
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

BOOT_SEED = 20260925
N_BOOT = 10_000


def bootstrap_ci(values, n_boot: int = N_BOOT, seed: int = BOOT_SEED, level: float = 0.95):
    """Percentile bootstrap CI of the mean over seeds."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return [float(lo), float(hi)]


def _stats(values) -> Dict[str, Any]:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
            "ci95": bootstrap_ci(v), "per_seed": [float(x) for x in v]}


def load_condition(cond_dir: str) -> List[Dict[str, Any]]:
    runs = []
    for sd in sorted(glob.glob(os.path.join(cond_dir, "seed_*"))):
        with open(os.path.join(sd, "eval.json")) as f:
            run = json.load(f)
        with open(os.path.join(sd, "config.json")) as f:
            run["config"] = json.load(f)
        run["history"] = dict(np.load(os.path.join(sd, "history.npz")))
        run["Q"] = np.load(os.path.join(sd, "q.npy"))
        runs.append(run)
    return runs


def summarize_condition(runs: List[Dict[str, Any]], metric: str = "success_rate") -> Dict[str, Any]:
    cfg = runs[0]["config"]
    out: Dict[str, Any] = {"algorithm": cfg["algorithm"], "alpha": cfg["alpha"],
                           "schedule": cfg["schedule"], "q_init": cfg.get("q_init", 0.0),
                           "n_episodes": cfg["n_episodes"], "seeds": [r["seed"] for r in runs],
                           "variants": {}}
    for v in runs[0]["final_eval"]["variants"]:
        curves = np.array([[e["variants"][v][metric] for e in r["evals"]] for r in runs])
        final = curves[:, -1]
        out["variants"][v] = {"final": _stats(final), "auc": _stats(curves.mean(axis=1)),
                              "seed_std": float(final.std()),
                              "episodes": [e["episode"] for e in runs[0]["evals"]],
                              "curve_mean": curves.mean(axis=0).tolist()}
    tail = max(1, cfg["n_episodes"] // 10)
    out["online_final"] = _stats([r["history"]["success"][-tail:].mean() for r in runs])
    out["online_final_return"] = _stats([r["history"]["return"][-tail:].mean() for r in runs])
    return out


def summarize_dir(runs_dir: str) -> Dict[str, Any]:
    return {os.path.basename(c): summarize_condition(load_condition(c))
            for c in sorted(glob.glob(os.path.join(runs_dir, "*"))) if os.path.isdir(c)}


def overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True when two ``_stats`` CIs overlap (no detectable difference)."""
    return not (a["ci95"][1] < b["ci95"][0] or b["ci95"][1] < a["ci95"][0])


def value_accuracy(runs: List[Dict[str, Any]], v_star, pi_star) -> Dict[str, Any]:
    """Against an exact reference: mean over states of max_a Q - V*, and the share of states
    where the greedy action equals the reference policy (states with a reference only)."""
    v_star = np.array([np.nan if x is None else x for x in v_star], dtype=float)
    pi_star = np.asarray(pi_star)
    mask = ~np.isnan(v_star)
    bias = [float(np.mean(r["Q"].max(axis=1)[mask] - v_star[mask])) for r in runs]
    agree = [float(np.mean(r["Q"].argmax(axis=1)[mask] == pi_star[mask])) for r in runs]
    return {"value_bias": _stats(bias), "policy_agreement": _stats(agree)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="one or more run directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--optimal", default=None,
                    help="JSON with representation_start_values / representation_policy_actions")
    args = ap.parse_args()
    ref = None
    if args.optimal:
        with open(args.optimal) as f:
            ref = json.load(f)
    out: Dict[str, Any] = {"method": __doc__.splitlines()[0], "boot_seed": BOOT_SEED,
                           "n_boot": N_BOOT, "conditions": {}}
    for d in args.runs:
        for c in sorted(glob.glob(os.path.join(d, "*"))):
            if not os.path.isdir(c):
                continue
            runs = load_condition(c)
            s = summarize_condition(runs)
            s["runs_dir"] = d
            if ref is not None:
                s.update(value_accuracy(runs, ref["representation_start_values"],
                                        ref["representation_policy_actions"]))
            out["conditions"][os.path.basename(c)] = s
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    for name, s in out["conditions"].items():
        cells = "  ".join(f"{v}: {m['final']['mean']:.3f} [{m['final']['ci95'][0]:.3f}, "
                          f"{m['final']['ci95'][1]:.3f}]" for v, m in s["variants"].items())
        print(f"{name:55s} {cells}")


if __name__ == "__main__":
    main()
