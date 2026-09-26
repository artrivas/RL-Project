"""Task-generic training and evaluation for the four catalog tasks.

Same experiment contract as ``src/train.py`` (which keeps the reported Ball Pursuit pipeline
unchanged): several training seeds per condition, periodic greedy evaluation (epsilon = 0,
learning off) on fixed starts with the environment noise seeded identically (common random
numbers), and per run ``q.npy``, ``history.npz``, ``eval.json`` and ``config.json``.

A task is a ``TaskSpec``: environment factory, state representations, evaluation starts per
variant, and how to summarise an episode (``success`` plus task-specific fields). The
``pursuit_kit`` spec reproduces ``train.py``'s starter-kit runs exactly (regression gate).

    python -m src.task_train --task shooting --preset matrix --out notebooks/artifacts/runs_shooting
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.agents import AGENTS, TabularAgent
from src.exploration import decay_reaching, make_schedule
from src.train import git_commit, source_fingerprint

METHODS = ("qlearning", "sarsa", "mc_first_visit", "mc_first_visit_alpha")
ALPHA_METHODS = ("qlearning", "sarsa", "mc_first_visit_alpha")   # mc_first_visit uses 1/N


@dataclass
class TaskSpec:
    name: str
    action_names: List[str]
    make_env: Callable[..., Any]                     # (seed, **env_kwargs) -> env
    representations: Dict[str, Callable[[], Any]]    # name -> discretizer factory
    eval_starts: Callable[[int, int, str], list]     # (n, seed, variant) -> starts
    eval_variants: List[str]                         # evaluated periodically; first is primary
    summarize: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]  # (out, last info)
    history_keys: List[str] = field(default_factory=list)   # numeric summary fields to log
    trajectory_keys: List[str] = field(default_factory=lambda: ["step"])
    extra_metrics: Callable[[List[Dict[str, Any]]], Dict[str, float]] = lambda results: {}
    final_variants: List[str] = field(default_factory=list)  # evaluated once, at the end
    default_t_max: int = 40


# ------------------------------------------------------------------ registry
def _pursuit_kit_spec() -> TaskSpec:
    from src.discretizer import Discretizer
    from src.env_base import ACTION_NAMES, T_MAX
    from src.kit_env import KitBallPursuitEnv
    from src.sampler import evaluation_starts
    from src.train import KIT_DISCRETIZATIONS

    def summarize(out, info):
        captured = out["terminated"]
        return {"success": captured and out["steps"] <= 39,
                "capture_step": out["steps"] if captured else None}

    def extra(results):
        steps = np.array([r["capture_step"] or 10**6 for r in results])
        return {"capture_rate_lt40": float(np.mean(steps <= 39)),
                "capture_rate_le40": float(np.mean(steps <= 40))}

    return TaskSpec(
        name="pursuit_kit", action_names=list(ACTION_NAMES),
        make_env=lambda seed, t_max=T_MAX: KitBallPursuitEnv(t_max=t_max, seed=seed),
        representations={k: (lambda c=c: Discretizer(c)) for k, c in KIT_DISCRETIZATIONS.items()},
        eval_starts=lambda n, seed, variant: evaluation_starts(n, seed=seed, law=variant),
        eval_variants=["uniform"], final_variants=["kit"], summarize=summarize,
        extra_metrics=extra,
        trajectory_keys=["step", "player", "body_dir", "ball", "true_d", "true_theta"],
        default_t_max=T_MAX)


def _shooting_spec() -> TaskSpec:
    from src import shooting_env as se
    return se.task_spec(TaskSpec)


TASKS: Dict[str, Callable[[], TaskSpec]] = {"pursuit_kit": _pursuit_kit_spec,
                                            "shooting": _shooting_spec}


def get_task(name: str) -> TaskSpec:
    return TASKS[name]()


# ------------------------------------------------------------------ config
@dataclass
class TaskConfig:
    name: str
    task: str
    algorithm: str = "qlearning"
    n_episodes: int = 20_000
    alpha: float = 0.1
    gamma: float = 0.99
    schedule: Dict[str, Any] = field(default_factory=lambda: {"kind": "constant", "eps": 0.1})
    q_init: float = 0.0                  # initial Q value (0 = the reported runs)
    representation: str = "default"
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    t_max: Optional[int] = None          # None: the task's default
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    eval_every: int = 1000
    eval_episodes: int = 500
    eval_seed: int = 12345
    snapshot_episodes: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def decay_schedule(n_episodes: int, eps_start: float = 1.0, eps_min: float = 0.1,
                   fraction: float = 0.6) -> Dict[str, Any]:
    """Geometric decay reaching ``eps_min`` at ``fraction`` of the budget (as in ``train.py``)."""
    return {"kind": "decay", "eps_start": eps_start, "eps_min": eps_min,
            "decay": decay_reaching(eps_start, eps_min, int(fraction * n_episodes))}


def make_env(spec: TaskSpec, cfg: TaskConfig, seed: int):
    t_max = cfg.t_max if cfg.t_max is not None else spec.default_t_max
    return spec.make_env(seed, t_max=t_max, **cfg.env_kwargs)


# ------------------------------------------------------------------ episodes
def run_episode(spec: TaskSpec, env, agent: TabularAgent, disc, epsilon: float,
                learn: bool = True, start=None, record: bool = False) -> Dict[str, Any]:
    """One episode. Consumes random numbers exactly as ``train.run_episode`` does."""
    obs, info = env.reset(start)
    trajectory = [{k: info.get(k) for k in spec.trajectory_keys}] if record else None
    actions = []
    s = disc(obs)
    a = agent.act(s, epsilon)
    ret, disc_ret, discount = 0.0, 0.0, 1.0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, r, terminated, truncated, info = env.step(a)
        if record:
            trajectory.append({k: info.get(k) for k in spec.trajectory_keys})
            actions.append(a)
        s2 = disc(obs)
        done = terminated or (truncated and not agent.needs_next_action)
        a2 = None if done else agent.act(s2, epsilon)
        if learn:
            agent.observe(s, a, r, s2, a2, terminated, truncated)
        ret += r
        disc_ret += discount * r
        discount *= agent.gamma
        s, a = s2, a2
    if learn:
        agent.end_episode()
    out = {"return": ret, "discounted_return": disc_ret, "steps": env.step_count,
           "terminated": terminated, "truncated": truncated}
    out.update(spec.summarize(out, info))
    out["success"] = bool(out["success"])
    if record:
        out["trajectory"], out["actions"] = trajectory, actions
    return out


def evaluate(spec: TaskSpec, env, agent: TabularAgent, disc, starts: Sequence,
             noise_seed: int) -> Dict[str, Any]:
    """Greedy policy, no learning, fixed starts and a fixed environment noise stream."""
    env.rng = np.random.default_rng(noise_seed)
    q_before = agent.Q.copy()
    results = [run_episode(spec, env, agent, disc, 0.0, learn=False, start=s) for s in starts]
    assert np.array_equal(q_before, agent.Q)
    metrics = {"success_rate": float(np.mean([r["success"] for r in results])),
               "mean_return": float(np.mean([r["return"] for r in results])),
               "mean_discounted_return": float(np.mean([r["discounted_return"] for r in results])),
               "mean_steps": float(np.mean([r["steps"] for r in results]))}
    metrics.update(spec.extra_metrics(results))
    return metrics


# ------------------------------------------------------------------ training
def train_one(cfg: TaskConfig, seed: int, out_dir: Optional[str] = None) -> Dict[str, Any]:
    spec = get_task(cfg.task)
    disc = spec.representations[cfg.representation]()
    env = make_env(spec, cfg, seed)
    eval_env = make_env(spec, cfg, cfg.eval_seed)
    starts = {v: spec.eval_starts(cfg.eval_episodes, cfg.eval_seed, v) for v in spec.eval_variants}
    agent = AGENTS[cfg.algorithm](disc.n_states, len(spec.action_names), cfg.alpha, cfg.gamma,
                                  seed=seed, q_init=cfg.q_init)
    schedule = make_schedule(cfg.schedule)

    hist_keys = ["return", "discounted_return", "steps", "success", "epsilon"] + spec.history_keys
    hist = {k: np.zeros(cfg.n_episodes) for k in hist_keys}
    evals, t0 = [], time.time()
    snapshots, snapshot_q, train_trajs = set(cfg.snapshot_episodes), {}, []
    for ep in range(cfg.n_episodes):
        eps = schedule(ep)
        if ep in snapshots:
            snapshot_q[f"ep_{ep}"] = agent.Q.copy()
        r = run_episode(spec, env, agent, disc, eps, record=ep in snapshots)
        if ep in snapshots:
            train_trajs.append({"episode": ep, "epsilon": eps, **r})
        for k in hist_keys:
            v = eps if k == "epsilon" else r[k]
            hist[k][ep] = np.nan if v is None else v
        if (ep + 1) % cfg.eval_every == 0:
            evals.append({"episode": ep + 1, "variants": {
                v: evaluate(spec, eval_env, agent, disc, starts[v], cfg.eval_seed)
                for v in spec.eval_variants}})

    result = {"name": cfg.name, "task": cfg.task, "seed": seed, "evals": evals,
              "final_eval": evals[-1] if evals else None, "train_seconds": time.time() - t0}
    for v in spec.final_variants:
        extra = spec.eval_starts(cfg.eval_episodes, cfg.eval_seed, v)
        result[f"final_eval_{v}"] = evaluate(spec, eval_env, agent, disc, extra, cfg.eval_seed)
    if out_dir:
        run_dir = os.path.join(out_dir, cfg.name, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)
        np.save(os.path.join(run_dir, "q.npy"), agent.Q)
        if cfg.n_episodes in snapshots:
            snapshot_q[f"ep_{cfg.n_episodes}"] = agent.Q.copy()
        if snapshot_q:
            np.savez_compressed(os.path.join(run_dir, "q_snapshots.npz"), **snapshot_q)
            with open(os.path.join(run_dir, "train_trajectories.json"), "w") as f:
                json.dump(train_trajs, f, default=_json_default)
        np.savez_compressed(os.path.join(run_dir, "history.npz"), **hist)
        with open(os.path.join(run_dir, "eval.json"), "w") as f:
            json.dump(result, f, indent=2)
        config = {**cfg.to_dict(), "seed": seed, "actions": spec.action_names,
                  "n_states": disc.n_states,
                  "representation_detail": (disc.as_dict() if hasattr(disc, "as_dict") else
                                            disc.config.as_dict()),
                  "git_commit": git_commit(), "source_sha256": source_fingerprint()}
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    return result


def _json_default(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if hasattr(o, "as_dict"):
        return o.as_dict()
    raise TypeError(f"not JSON serializable: {type(o)}")


def run_experiment(cfg: TaskConfig, out_dir: Optional[str], workers: int = 8) -> List[Dict]:
    with ProcessPoolExecutor(max_workers=min(workers, len(cfg.seeds))) as pool:
        return list(pool.map(train_one, [cfg] * len(cfg.seeds), cfg.seeds,
                             [out_dir] * len(cfg.seeds)))


# ------------------------------------------------------------------ presets
def schedule_name(schedule: Dict[str, Any]) -> str:
    if schedule["kind"] == "constant":
        return f"eps_const_{schedule['eps']:g}"
    return f"eps_decay_{schedule['eps_start']:g}_to_{schedule['eps_min']:g}"


def matrix_configs(task: str, n_episodes: int, alphas: Optional[Dict[str, float]] = None,
                   methods: Sequence[str] = METHODS, **overrides) -> List[TaskConfig]:
    """Every method x {constant eps 0.1, decaying eps 1.0 -> 0.1}, same budget, seeds and starts.
    ``alphas``: step size per method (chosen by ``alpha_sweep_configs`` on other seeds)."""
    alphas = alphas or {}
    out = []
    for method in methods:
        for sched in ({"kind": "constant", "eps": 0.1}, decay_schedule(n_episodes)):
            alpha = alphas.get(method, 0.1)
            out.append(TaskConfig(name=f"{task}_{method}_{schedule_name(sched)}", task=task,
                                  algorithm=method, n_episodes=n_episodes, alpha=alpha,
                                  schedule=sched, **overrides))
    return out


def alpha_sweep_configs(task: str, n_episodes: int, alphas: Sequence[float] = (0.03, 0.1, 0.3),
                        seeds: Sequence[int] = (5, 6, 7, 8, 9), **overrides) -> List[TaskConfig]:
    """Step-size selection on seeds disjoint from the reported ones (0-4), decaying epsilon."""
    return [TaskConfig(name=f"{task}_sweep_{m}_alpha{a:g}", task=task, algorithm=m,
                       n_episodes=n_episodes, alpha=a, schedule=decay_schedule(n_episodes),
                       seeds=list(seeds), **overrides)
            for m in ALPHA_METHODS for a in alphas]


def pick_alphas(sweep_dir: str, task: str, metric: str = "success_rate",
                last_k: int = 5) -> Dict[str, Any]:
    """Best alpha per method from a finished sweep: mean over seeds of the last ``last_k``
    greedy evaluations of ``metric`` (primary variant, or the mean over all variants when a
    task has several). Returns ``{"alphas": {...}, "table": {...}}`` for saving as JSON."""
    table: Dict[str, Dict[str, float]] = {}
    prefix = f"{task}_sweep_"
    for cond in sorted(os.listdir(sweep_dir)):
        if not cond.startswith(prefix):
            continue
        method, alpha = cond[len(prefix):].rsplit("_alpha", 1)
        scores = []
        for sd in sorted(os.listdir(os.path.join(sweep_dir, cond))):
            with open(os.path.join(sweep_dir, cond, sd, "eval.json")) as f:
                evals = json.load(f)["evals"][-last_k:]
            scores.append(np.mean([np.mean([m[metric] for m in e["variants"].values()])
                                   for e in evals]))
        table.setdefault(method, {})[alpha] = float(np.mean(scores))
    alphas = {m: float(max(t, key=t.get)) for m, t in table.items()}
    return {"metric": metric, "last_k": last_k, "alphas": alphas, "table": table}


def shooting_diagnostic_configs(n_episodes: int, alphas: Optional[Dict[str, float]] = None
                                ) -> List[TaskConfig]:
    """Why does constant epsilon trail decaying epsilon on shooting (seeds 0-4, same starts)?

    * D1 step size: constant epsilon with alpha 0.1 and 0.3 (alpha was selected with the
      decaying schedule only), for the constant-alpha methods;
    * D2 initial values: constant epsilon, selected alpha, optimistic ``q_init = 100`` (the goal
      reward) instead of 0, to test slow recovery from the zero initialisation;
    * D3 budget: constant epsilon, selected alpha, 4x the episodes (slow vs stuck).
    """
    alphas = alphas or {}
    const = {"kind": "constant", "eps": 0.1}
    out = []
    for m in ALPHA_METHODS:
        a0 = alphas.get(m, 0.1)
        for a in (0.1, 0.3):
            out.append(TaskConfig(name=f"shooting_diag_step_{m}_alpha{a:g}", task="shooting",
                                  algorithm=m, n_episodes=n_episodes, alpha=a, schedule=const))
        out.append(TaskConfig(name=f"shooting_diag_optinit_{m}", task="shooting", algorithm=m,
                              n_episodes=n_episodes, alpha=a0, schedule=const, q_init=100.0))
        out.append(TaskConfig(name=f"shooting_diag_budget4x_{m}", task="shooting", algorithm=m,
                              n_episodes=4 * n_episodes, alpha=a0, schedule=const))
    out.append(TaskConfig(name="shooting_diag_optinit_mc_first_visit", task="shooting",
                          algorithm="mc_first_visit", n_episodes=n_episodes, schedule=const,
                          q_init=100.0))
    return out


DIAGNOSTICS = {"shooting": shooting_diagnostic_configs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--preset", choices=["matrix", "alpha_sweep", "diagnostics"], default="matrix")
    ap.add_argument("--episodes", type=int, default=20_000)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--alphas", default=None,
                    help="JSON file from pick_alphas (matrix preset): step size per method")
    ap.add_argument("--only", nargs="*", help="run only these config names")
    ap.add_argument("--seeds", nargs="*", type=int, help="override the training seeds")
    ap.add_argument("--snapshots", nargs="*", type=int, default=[])
    ap.add_argument("--representation", default="default")
    args = ap.parse_args()

    overrides: Dict[str, Any] = {"representation": args.representation}
    if args.eval_every:
        overrides["eval_every"] = args.eval_every
    alphas = None
    if args.alphas:
        with open(args.alphas) as f:
            alphas = json.load(f)["alphas"]
    if args.preset == "matrix":
        cfgs = matrix_configs(args.task, args.episodes, alphas, **overrides)
    elif args.preset == "diagnostics":
        cfgs = [replace(c, **overrides) for c in DIAGNOSTICS[args.task](args.episodes, alphas)]
    else:
        cfgs = alpha_sweep_configs(args.task, args.episodes, **overrides)
    for cfg in cfgs:
        if args.only and cfg.name not in args.only:
            continue
        if args.seeds:
            cfg = replace(cfg, seeds=args.seeds)
        if args.snapshots:
            cfg = replace(cfg, snapshot_episodes=args.snapshots)
        t0 = time.time()
        results = run_experiment(cfg, args.out, args.workers)
        for v in results[0]["final_eval"]["variants"]:
            final = [r["final_eval"]["variants"][v]["success_rate"] for r in results]
            print(f"{cfg.name} [{v}]: greedy success final = {np.mean(final):.3f} "
                  f"(min {np.min(final):.3f}, max {np.max(final):.3f}; seeds {cfg.seeds}, "
                  f"{time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
