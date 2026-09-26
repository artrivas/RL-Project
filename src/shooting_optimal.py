"""Exact optimum of the Goal Shooting task (``src/shooting_env.py``) and the parameter
calibration check.

Outcome probabilities are exact: the crossing point ``y_c = y + d tan(phi + eps)`` is monotone in
the angular error ``eps ~ N(0, sigma^2)``, so each outcome is an interval of ``eps`` and its
probability is a difference of normal CDFs.

Two optima, both computed per start without sampling the shot noise:

* **true optimum**: the policy that sees the continuous state (d, y, y_k). Only ``d`` changes
  within an episode (CONDUCIR), so the Bellman equation is a backward pass over the chain
  ``d_i = max(11, d_0 - 2 i)``, ``i < T_max``.
* **representation optimum**: the best deterministic policy that sees only our 36 discrete states.
  The lateral and keeper bins never change within an episode, so each (lateral, keeper) group is
  independent and its policy is a map from the 3 distance bins to the 6 actions: all 216 are
  evaluated exactly, and the one with the best mean discounted return over a large sample of
  training starts is kept.

Both maximise the expected *discounted return* (what the agents optimise); the goal rate of that
policy is reported next to it.

Calibration (fixed before any training; it checks structure, not a target percentage):
(a) the representation optimum's shot differs across keeper bins in some (distance, lateral) bin;
(b) CONDUCIR is optimal in at least one state; (c) its goal rate with a keeper is below 100%.

    python -m src.shooting_optimal --calibrate notebooks/artifacts/shooting_calibration.json
    python -m src.shooting_optimal --out notebooks/artifacts/shooting_optimal.json
"""

import argparse
import itertools
import json
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.special import ndtr

from src import shooting_env as se

GAMMA = 0.99


def _interval_prob(d, y, phi, sigma_rad, lo, hi):
    """P(lo < y_c < hi) for shots with nominal direction ``phi`` (radians)."""
    a = np.arctan2(lo - y, d) - phi
    b = np.arctan2(hi - y, d) - phi
    return np.clip(ndtr(b / sigma_rad) - ndtr(a / sigma_rad), 0.0, 1.0)


def outcome_probs(d, y, target, keeper_y, params: se.ShootingParams):
    """Exact (P_goal, P_off, P_saved) for arrays ``d, y, keeper_y`` (NaN = no keeper)."""
    d, y, ky = np.broadcast_arrays(np.asarray(d, float), np.asarray(y, float),
                                   np.asarray(keeper_y, float))
    sigma = np.radians(params.sigma_deg)
    phi = np.arctan2(target - y, d)
    p_on = _interval_prob(d, y, phi, sigma, -se.POST_Y, se.POST_Y)
    has = ~np.isnan(ky)
    lo = np.where(has, np.maximum(ky - params.keeper_reach, -se.POST_Y), 0.0)
    hi = np.where(has, np.minimum(ky + params.keeper_reach, se.POST_Y), 0.0)
    p_saved = np.where(has & (hi > lo), _interval_prob(d, y, phi, sigma, lo, hi), 0.0)
    return p_on - p_saved, 1.0 - p_on, p_saved


class Chains:
    """Per-start chain of reachable distances and exact shot values at each of them."""

    def __init__(self, starts: Sequence[se.ShootStart], params: se.ShootingParams,
                 t_max: int = se.T_MAX):
        self.params, self.t_max = params, t_max
        self.d0 = np.array([s.d for s in starts])
        self.y = np.array([s.y for s in starts])
        self.ky = np.array([np.nan if s.keeper_y is None else s.keeper_y for s in starts])
        self.has_keeper = ~np.isnan(self.ky)
        self.d = np.maximum(se.D_MIN, self.d0[:, None] - se.CONDUCIR_STEP * np.arange(t_max))
        n, T, A = len(starts), t_max, len(se.TARGETS)
        self.shot_r = np.zeros((n, T, A))
        self.shot_goal = np.zeros((n, T, A))
        for a, target in enumerate(se.TARGETS):
            pg, po, ps = outcome_probs(self.d, self.y[:, None], target, self.ky[:, None], params)
            self.shot_r[:, :, a] = se.R_GOAL * pg + se.R_OFF * po + se.R_SAVED * ps
            self.shot_goal[:, :, a] = pg
        p_out = np.where(self.has_keeper, params.p_keeper_out, 0.0)
        self.cond_r = se.R_CONDUCIR + p_out * se.R_SAVED       # immediate, per CONDUCIR
        self.cond_survive = 1.0 - p_out

    # --------------------------------------------------------------- true optimum
    def true_optimum(self, gamma: float = GAMMA) -> Dict[str, np.ndarray]:
        n, T = self.d.shape
        V, G = np.zeros(n), np.zeros(n)                 # after T actions: truncated
        first = np.zeros(n, dtype=int)
        for t in range(T - 1, -1, -1):
            best = self.shot_r[:, t].argmax(axis=1)
            shoot_v = self.shot_r[np.arange(n), t, best]
            shoot_g = self.shot_goal[np.arange(n), t, best]
            cond_v = self.cond_r + self.cond_survive * gamma * V
            cond_g = self.cond_survive * G
            take = cond_v > shoot_v
            V, G = np.where(take, cond_v, shoot_v), np.where(take, cond_g, shoot_g)
            first = np.where(take, se.CONDUCIR, best)
        return {"value": V, "goal": G, "first_action": first}

    # ------------------------------------------------ fixed policy on the 3 distance bins
    def d_bins(self) -> np.ndarray:
        return np.digitize(self.d, [15.0, 20.0])        # same bins as default_discretizer

    def policy_value(self, policies: np.ndarray, rows: np.ndarray, gamma: float = GAMMA,
                     start_index: int = 0, first_action: Optional[int] = None):
        """Exact expected discounted return and goal probability of each policy in
        ``policies`` (shape (P, 3): action per distance bin) from each start in ``rows``,
        beginning at chain index ``start_index``; ``first_action`` overrides the first action.
        Returns arrays of shape (P, len(rows))."""
        P, n = len(policies), len(rows)
        bins = self.d_bins()[rows]
        v, g = np.zeros((P, n)), np.zeros((P, n))
        mass, disc = np.ones((P, n)), 1.0
        cond_r, surv = self.cond_r[rows], self.cond_survive[rows]
        for k, t in enumerate(range(start_index, self.t_max)):
            if k == 0 and first_action is not None:
                a = np.full((P, n), first_action)
            else:
                a = policies[:, bins[:, t]]                           # (P, n)
            shoot = a != se.CONDUCIR
            a_idx = np.where(shoot, a, 0)
            r = self.shot_r[rows, t][np.arange(n), a_idx]              # (P, n)
            pg = self.shot_goal[rows, t][np.arange(n), a_idx]
            v += np.where(shoot, mass * disc * r, mass * disc * cond_r)
            g += np.where(shoot, mass * pg, 0.0)
            mass = np.where(shoot, 0.0, mass * surv)
            disc *= gamma
        return v, g


def _groups(chains: Chains) -> np.ndarray:
    """(lateral bin, keeper bin) of each start, as one index 0..11."""
    y_bin = np.digitize(chains.y, [-3.5, 3.5])
    k_bin = np.where(chains.has_keeper, 1 + np.digitize(np.nan_to_num(chains.ky), [-1.0, 1.0]), 0)
    return y_bin * 4 + k_bin


ALL_POLICIES = np.array(list(itertools.product(range(len(se.ACTION_NAMES)), repeat=3)))


def representation_optimum(chains: Chains, gamma: float = GAMMA) -> Dict[int, np.ndarray]:
    """Best 3-bin policy for each (lateral, keeper) group by mean discounted return."""
    groups = _groups(chains)
    best = {}
    for gidx in range(12):
        rows = np.flatnonzero(groups == gidx)
        v, _ = chains.policy_value(ALL_POLICIES, rows, gamma)
        best[gidx] = ALL_POLICIES[int(v.mean(axis=1).argmax())]
    return best


def evaluate_policy_table(chains: Chains, table: Dict[int, np.ndarray], gamma: float = GAMMA):
    groups = _groups(chains)
    v, g = np.zeros(len(groups)), np.zeros(len(groups))
    for gidx, pol in table.items():
        rows = np.flatnonzero(groups == gidx)
        if len(rows):
            vv, gg = chains.policy_value(pol[None, :], rows, gamma)
            v[rows], g[rows] = vv[0], gg[0]
    return v, g


def state_table(table: Dict[int, np.ndarray]) -> np.ndarray:
    """Representation optimum as an action per discrete state of ``default_discretizer``."""
    disc = se.default_discretizer()
    out = np.zeros(disc.n_states, dtype=int)
    for s in range(disc.n_states):
        d_bin, y_bin, k_bin = disc.decompose(s)
        out[s] = table[y_bin * 4 + k_bin][d_bin]
    return out


def start_values(chains: Chains, table: Dict[int, np.ndarray], gamma: float = GAMMA):
    """Per discrete start state: mean over starts in that state of V^pi (start value) and of
    Q^pi(s, a) for every first action a (then following the representation optimum pi)."""
    disc = se.default_discretizer()
    groups = _groups(chains)
    s0 = np.array([disc({"d_g": d, "y": y, "keeper": None if np.isnan(k) else k})
                   for d, y, k in zip(chains.d0, chains.y, chains.ky)])
    n_a = len(se.ACTION_NAMES)
    V = np.full(disc.n_states, np.nan)
    Q = np.full((disc.n_states, n_a), np.nan)
    count = np.zeros(disc.n_states, dtype=int)
    for s in range(disc.n_states):
        rows = np.flatnonzero(s0 == s)
        count[s] = len(rows)
        if not len(rows):
            continue
        pol = table[int(groups[rows[0]])][None, :]
        V[s] = chains.policy_value(pol, rows, gamma)[0].mean()
        for a in range(n_a):
            Q[s, a] = chains.policy_value(pol, rows, gamma, first_action=a)[0].mean()
    return V, Q, count


def analyse(params: se.ShootingParams, n_train: int = 120_000, train_seed: int = 2026,
            eval_n: int = 500, eval_seed: int = 12345) -> Dict:
    rng = np.random.default_rng(train_seed)
    train = Chains([se.sample_start(rng, None, params.p_keeper) for _ in range(n_train)], params)
    table = representation_optimum(train)
    actions = state_table(table)
    disc = se.default_discretizer()
    V, Q, count = start_values(train, table)

    out = {"params": params.as_dict(), "gamma": GAMMA, "t_max": se.T_MAX,
           "selection_sample": {"n": n_train, "seed": train_seed},
           "evaluation": {}, "representation_policy": {}}
    for variant in ("open", "keeper"):
        ev = Chains(se.evaluation_starts(eval_n, eval_seed, variant), params)
        opt = ev.true_optimum()
        v_rep, g_rep = evaluate_policy_table(ev, table)
        out["evaluation"][variant] = {
            "n": eval_n, "seed": eval_seed,
            "true_optimum_goal_rate": float(opt["goal"].mean()),
            "true_optimum_mean_discounted_return": float(opt["value"].mean()),
            "true_optimum_first_action_share": {
                se.ACTION_NAMES[a]: float(np.mean(opt["first_action"] == a))
                for a in range(len(se.ACTION_NAMES))},
            "representation_optimum_goal_rate": float(g_rep.mean()),
            "representation_optimum_mean_discounted_return": float(v_rep.mean()),
            # a greedy evaluation draws one shot per start: its goal rate has this sampling std
            "sampling_std_of_goal_rate": float(np.sqrt(np.sum(g_rep * (1 - g_rep))) / eval_n)}
    for s in range(disc.n_states):
        out["representation_policy"][disc.state_to_label(s)] = {
            "state": s, "action": se.ACTION_NAMES[actions[s]],
            "start_value": None if np.isnan(V[s]) else float(V[s]),
            "start_q": None if np.isnan(V[s]) else [float(q) for q in Q[s]],
            "n_starts_in_selection_sample": int(count[s])}

    # calibration checks
    shots_differ = False
    for d_bin in range(3):
        for y_bin in range(3):
            shots = {actions[disc.compose((d_bin, y_bin, k))] for k in range(4)}
            shots.discard(se.CONDUCIR)
            shots_differ |= len(shots) > 1
    out["calibration"] = {
        "a_shot_differs_across_keeper_bins": bool(shots_differ),
        "b_conducir_optimal_somewhere": bool(np.any(actions == se.CONDUCIR)),
        "c_keeper_goal_rate_below_1": out["evaluation"]["keeper"]["representation_optimum_goal_rate"] < 1.0}
    out["calibration"]["passed"] = all(out["calibration"].values())
    out["representation_policy_actions"] = actions.tolist()
    out["representation_start_values"] = [None if np.isnan(v) else float(v) for v in V]
    return out


# Calibration grid, in order of preference: closest to the provisional parameters first
# (sigma 4 deg, reach 2.0 m, p_out 0.1). p_out is lowered before sigma is raised; reach is kept.
CALIBRATION_GRID = [(sigma, 2.0, p_out) for sigma in (4.0, 6.0, 8.0) for p_out in (0.1, 0.05, 0.02)]


def calibrate(**analyse_kwargs) -> Dict:
    """Analyse every grid point in order; the first one that passes all checks is selected."""
    rows, selected = [], None
    for sigma, reach, p_out in CALIBRATION_GRID:
        res = analyse(se.ShootingParams(sigma, reach, p_out), **analyse_kwargs)
        rows.append({"params": res["params"], "calibration": res["calibration"],
                     "evaluation": res["evaluation"]})
        if selected is None and res["calibration"]["passed"]:
            selected = res["params"]
    return {"rule": "first grid point (ordered by closeness to sigma 4, reach 2.0, p_out 0.1; "
                    "p_out lowered before sigma raised) that passes checks a-c",
            "selected": selected, "grid": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="notebooks/artifacts/shooting_optimal.json")
    ap.add_argument("--calibrate", default=None,
                    help="write the calibration grid to this JSON file instead")
    ap.add_argument("--sigma-deg", type=float, default=se.ShootingParams.sigma_deg)
    ap.add_argument("--keeper-reach", type=float, default=se.ShootingParams.keeper_reach)
    ap.add_argument("--p-keeper-out", type=float, default=se.ShootingParams.p_keeper_out)
    args = ap.parse_args()
    if args.calibrate:
        cal = calibrate()
        with open(args.calibrate, "w") as f:
            json.dump(cal, f, indent=2, ensure_ascii=False)
        for row in cal["grid"]:
            p, e = row["params"], row["evaluation"]
            print(f"sigma {p['sigma_deg']:g} reach {p['keeper_reach']:g} p_out {p['p_keeper_out']:g}: "
                  f"checks {row['calibration']}, keeper optimum "
                  f"{e['keeper']['representation_optimum_goal_rate']:.3f}")
        print("selected:", cal["selected"])
        return
    params = se.ShootingParams(args.sigma_deg, args.keeper_reach, args.p_keeper_out)
    out = analyse(params)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for v, e in out["evaluation"].items():
        print(f"{v}: true optimum goal {e['true_optimum_goal_rate']:.3f}, "
              f"representation optimum goal {e['representation_optimum_goal_rate']:.3f} "
              f"(sampling std {e['sampling_std_of_goal_rate']:.3f})")
    print("calibration:", out["calibration"])


if __name__ == "__main__":
    main()
