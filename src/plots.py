"""Plotting helpers shared by the report notebooks (static matplotlib figures for the LaTeX report).

Colour roles (validated reference palette): conditions use fixed categorical
slots (never cycled); magnitudes use a single-hue blue ramp; reference lines are
muted gray. Policy maps label every cell, so identity never relies on colour.
Pitch plots invert the y-axis to match the rcssserver monitor (SPEC §1).
"""

import json
import os
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from src.discretizer import Discretizer

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
REF = "#8a8984"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
ACTION_SHORT = ["D100", "D50", "T+35", "T−35"]
ACTION_COLORS = ["#2a78d6", "#c3c2b7", "#eb6834", "#1baf7a"]  # DASH 50 muted: rarely chosen


def style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)


# ----------------------------------------------------------------- loading
def load_runs(root: str) -> Dict[str, List[Dict]]:
    """{condition: [ {eval.json fields, "history": npz, "Q": array, "config": dict}, ... ]}."""
    runs: Dict[str, List[Dict]] = {}
    for cond in sorted(os.listdir(root)):
        cdir = os.path.join(root, cond)
        if not os.path.isdir(cdir):
            continue
        for sd in sorted(os.listdir(cdir)):
            rdir = os.path.join(cdir, sd)
            with open(os.path.join(rdir, "eval.json")) as f:
                run = json.load(f)
            with open(os.path.join(rdir, "config.json")) as f:
                run["config"] = json.load(f)
            run["history"] = dict(np.load(os.path.join(rdir, "history.npz")))
            run["Q"] = np.load(os.path.join(rdir, "q.npy"))
            run["dir"] = rdir
            runs.setdefault(cond, []).append(run)
    return runs


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


# ----------------------------------------------------------------- curves
def _band(ax, x, ys: np.ndarray, color: str, label: str, direct_label: bool = True) -> None:
    mean, std = ys.mean(axis=0), ys.std(axis=0)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=2, label=label)
    if direct_label:
        ax.annotate(label, (x[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                    color=TEXT, fontsize=8, va="center")


def plot_eval_curves(runs: Dict[str, List[Dict]], metric: str = "capture_rate_lt40",
                     refs: Optional[Dict[str, float]] = None, ax=None, ylabel: str = "",
                     title: str = "", direct_labels: bool = True,
                     labels: Optional[Dict[str, str]] = None):
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    x0 = 0
    for i, (cond, rs) in enumerate(runs.items()):
        x = np.array([e["episode"] for e in rs[0]["evals"]])
        ys = np.array([[e[metric] for e in r["evals"]] for r in rs])
        _band(ax, x, ys, SERIES[i], (labels or {}).get(cond, cond), direct_labels)
        x0 = x[0]
    for label, y in (refs or {}).items():
        ax.axhline(y, color=REF, linewidth=1, linestyle="--")
        ax.annotate(label, (x0, y), xytext=(0, 4), textcoords="offset points", color=TEXT_2,
                    fontsize=8)
    style(ax, title or f"Evaluación greedy ({metric}), media ± desv. est. entre semillas",
          "episodios de entrenamiento", ylabel or metric)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return ax


def plot_training_curve(runs: Dict[str, List[Dict]], key: str, window: int = 500, ax=None,
                        title: str = "", ylabel: str = "", labels: Optional[Dict[str, str]] = None,
                        direct_labels: bool = True):
    ax = ax or plt.subplots(figsize=(8, 3.5))[1]
    for i, (cond, rs) in enumerate(runs.items()):
        ys = np.array([moving_average(np.nan_to_num(r["history"][key].astype(float)), window)
                       for r in rs])
        x = np.arange(window, window + ys.shape[1])
        _band(ax, x, ys, SERIES[i], (labels or {}).get(cond, cond), direct_labels)
    style(ax, title or f"Entrenamiento: {key} (media móvil de {window} episodios)",
          "episodios de entrenamiento", ylabel or key)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return ax


# ------------------------------------------------------------ policy maps
SPEED_NAMES = {0: "en reposo (v < 0.2)", 1: "en movimiento (v ≥ 0.2)"}


def _angle_label_es(label: str) -> str:
    return (label.replace("front |θ|≤", "frente |θ|≤").replace("right ", "der. ")
            .replace("left ", "izq. "))


def _grid(disc: Discretizer, values_by_state: np.ndarray, speed_bin: int) -> np.ndarray:
    grid = np.zeros((disc.n_dist, disc.n_angle))
    for d in range(disc.n_dist):
        for a in range(disc.n_angle):
            grid[d, a] = values_by_state[(d * disc.n_angle + a) * disc.n_speed + speed_bin]
    return grid


def _angle_order(disc: Discretizer) -> List[int]:
    """Angle bins ordered left-to-right: back-left ... front ... back-right."""
    n_rings = len(disc.config.angle_edges)
    left = [2 * r for r in range(n_rings - 1, 0, -1)]
    right = [2 * r - 1 for r in range(1, n_rings)]
    return left + [0] + right


def plot_policy(Q: np.ndarray, disc: Discretizer, visits: Optional[np.ndarray] = None,
                speed_bin: int = 0, ax=None, title: str = "Política greedy aprendida π̂(s)"):
    ax = ax or plt.subplots(figsize=(8, 3.6))[1]
    greedy = Q.argmax(axis=1).astype(float)
    if visits is not None:
        greedy[visits == 0] = np.nan
    order = _angle_order(disc)
    grid = _grid(disc, greedy, speed_bin)[:, order]
    cmap = ListedColormap(ACTION_COLORS)
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    for (r, c), v in np.ndenumerate(grid):
        ax.text(c, r, "—" if np.isnan(v) else ACTION_SHORT[int(v)], ha="center", va="center",
                fontsize=9, color=TEXT)
    labels = disc.angle_labels()
    ax.set_xticks(range(len(order)), [_angle_label_es(labels[i]) for i in order], rotation=25, ha="right")
    ax.set_yticks(range(disc.n_dist), disc.distance_labels())
    style(ax, title + (f" — {SPEED_NAMES.get(speed_bin, speed_bin)}" if disc.n_speed > 1 else ""),
          "rumbo del balón (izq. ← frente → der.)", "distancia")
    ax.grid(False)
    ax.text(1.0, 1.02, f"DESCONOCIDO → {ACTION_SHORT[int(Q[disc.unknown_state].argmax())]}",
            transform=ax.transAxes, ha="right", fontsize=8, color=TEXT_2)
    return ax


def plot_value(Q: np.ndarray, disc: Discretizer, speed_bin: int = 0, ax=None,
               title: str = "Valor aprendido V̂(s) = max_a Q(s, a)"):
    ax = ax or plt.subplots(figsize=(8, 3.6))[1]
    order = _angle_order(disc)
    grid = _grid(disc, Q.max(axis=1), speed_bin)[:, order]
    im = ax.imshow(grid, cmap=BLUE_RAMP, aspect="auto")
    for (r, c), v in np.ndenumerate(grid):
        ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=9,
                color="white" if v > grid.min() + 0.6 * np.ptp(grid) else TEXT)
    labels = disc.angle_labels()
    ax.set_xticks(range(len(order)), [_angle_label_es(labels[i]) for i in order], rotation=25, ha="right")
    ax.set_yticks(range(disc.n_dist), disc.distance_labels())
    style(ax, title + (f" — {SPEED_NAMES.get(speed_bin, speed_bin)}" if disc.n_speed > 1 else ""),
          "rumbo del balón (izq. ← frente → der.)", "distancia")
    ax.grid(False)
    ax.figure.colorbar(im, ax=ax, fraction=0.03)
    return ax


# ------------------------------------------------------------ trajectories
# Ordinal blue ramp (validated steps 250 -> 700): light = early, dark = late.
ORDINAL_BLUES = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]


def draw_pitch(ax, light: bool = False) -> None:
    """Pitch lines; ``light`` uses the chart surface with gray lines (for ordinal colours)."""
    ax.set_facecolor(SURFACE if light else "#2e7d32")
    kw = dict(color=GRID if light else "white", linewidth=1.2)
    ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], **kw)
    ax.plot([0, 0], [-34, 34], **kw)
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, **kw))
    for sx in (-1, 1):
        x0 = sx * 52.5
        ax.plot([x0, x0 - sx * 16.5, x0 - sx * 16.5, x0], [-20.16, -20.16, 20.16, 20.16], **kw)
    ax.set_xlim(-56, 56)
    ax.set_ylim(37, -37)  # +y downward, as on the rcssserver monitor
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_trajectories(records: Sequence[Dict], ax=None, title: str = "", zoom: bool = True):
    """``records``: episode dicts with ``trajectory`` (from rollout/live_eval)."""
    ax = ax or plt.subplots(figsize=(8, 5.4))[1]
    draw_pitch(ax)
    xs, ys = [], []
    for rec in records:
        traj = rec["trajectory"]
        px = [t["player"][0] for t in traj]
        py = [t["player"][1] for t in traj]
        bx, by = traj[0]["ball"]
        ax.plot(px, py, color="#cde2fb", linewidth=1.6)
        ax.plot(px[0], py[0], "o", color="#cde2fb", markersize=5)
        ax.plot(bx, by, "o", color="white", markeredgecolor=TEXT, markersize=7)
        if not rec["captured"]:
            ax.plot(px[-1], py[-1], "x", color="#fab219", markersize=8, markeredgewidth=2)
        xs += px + [bx]
        ys += py + [by]
    if zoom and xs:
        # Fixed square window (same metres per panel across calls) centred on the episode.
        half = 23.0
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy + half, cy - half)
    ax.set_title(title or "Trajectories (dot: start, white: ball, ✕: not captured)",
                 fontsize=11, loc="left", color=TEXT)
    ax.figure.set_facecolor(SURFACE)
    return ax


def plot_trajectory_pair(live: Dict, sim: Dict, ax=None, half: float = 23.0):
    """Same start, same policy: live server (slot 1) vs simulator (slot 2)."""
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    draw_pitch(ax)
    xs, ys = [], []
    for rec, color, name in ((live, SERIES[0], "servidor"), (sim, SERIES[1], "simulador")):
        traj = rec["trajectory"]
        px = [t["player"][0] for t in traj]
        py = [t["player"][1] for t in traj]
        ax.plot(px, py, color="white", linewidth=3.2)          # surface ring for overlaps
        ax.plot(px, py, color=color, linewidth=2,
                label=f"{name}: " + (f"captura en {rec['steps']}" if rec["captured"] else "sin captura"))
        if not rec["captured"]:
            ax.plot(px[-1], py[-1], "x", color=color, markersize=8, markeredgewidth=2)
        xs += px
        ys += py
    bx, by = live["trajectory"][0]["ball"]
    ax.plot(bx, by, "o", color="white", markeredgecolor=TEXT, markersize=7)
    xs.append(bx)
    ys.append(by)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy + half, cy - half)
    s = live["start"]
    ax.set_title(f"d₀={s['distance']:.1f} m, θ₀={s['bearing']:.0f}°", fontsize=9, loc="left", color=TEXT)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
    ax.figure.set_facecolor(SURFACE)
    return ax


def plot_snapshot_trajectories(snapshots: Dict[str, Dict], idx: int, env: str = "sim", ax=None,
                               half: float = 16.0, legend: bool = True):
    """One start, the greedy policy of every training snapshot (light = early, dark = late).

    ``snapshots``: ``{"ep_<k>": {"sim": [records], "live": [records]}}`` as saved by
    ``src/live_snapshots.py``; ``idx`` selects the start.
    """
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    draw_pitch(ax, light=True)
    keys = sorted(snapshots, key=lambda k: int(k.split("_")[1]))
    colors = ORDINAL_BLUES[-len(keys):] if len(keys) <= len(ORDINAL_BLUES) else \
        [BLUE_RAMP(x) for x in np.linspace(0.2, 1.0, len(keys))]
    xs, ys = [], []
    for key, color in zip(keys, colors):
        rec = snapshots[key][env][idx]
        px = [t["player"][0] for t in rec["trajectory"]]
        py = [t["player"][1] for t in rec["trajectory"]]
        ep = int(key.split("_")[1])
        ax.plot(px, py, color=color, linewidth=2,
                label=f"ep {ep:>5}: " + (f"captura en {rec['steps']}" if rec["captured"] else "sin captura"))
        if not rec["captured"]:
            ax.plot(px[-1], py[-1], "x", color=color, markersize=7, markeredgewidth=2)
        xs += px
        ys += py
    first = snapshots[keys[0]][env][idx]
    bx, by = first["trajectory"][0]["ball"]
    ax.plot(first["trajectory"][0]["player"][0], first["trajectory"][0]["player"][1], "o",
            color=TEXT_2, markersize=6)
    ax.plot(bx, by, "o", color="white", markeredgecolor=TEXT, markersize=8)
    xs.append(bx)
    ys.append(by)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(half, (max(xs) - min(xs)) / 2 + 2, (max(ys) - min(ys)) / 2 + 2)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy + half, cy - half)
    s = first["start"]
    ax.set_title(f"d₀={s['distance']:.1f} m, θ₀={s['bearing']:.0f}°", fontsize=9, loc="left", color=TEXT)
    if legend:
        ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
    ax.figure.set_facecolor(SURFACE)
    return ax


# ------------------------------------------------------ catalog tasks (src/task_train.py)
def plot_task_eval_curves(runs: Dict[str, List[Dict]], variant: str, metric: str = "success_rate",
                          refs: Optional[Dict[str, float]] = None, ax=None, ylabel: str = "",
                          title: str = "", labels: Optional[Dict[str, str]] = None,
                          colors: Optional[Dict[str, str]] = None, direct_labels: bool = False):
    """Greedy evaluation curves of ``task_train`` runs for one variant: mean over seeds, band =
    min-max over seeds (stays inside [0, 1] for rates, unlike mean ± std).
    ``runs``: ``{condition: [run, ...]}`` from ``src.task_analysis.load_condition``."""
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    x0 = 0
    for i, (cond, rs) in enumerate(runs.items()):
        x = np.array([e["episode"] for e in rs[0]["evals"]])
        ys = np.array([[e["variants"][variant][metric] for e in r["evals"]] for r in rs])
        color = (colors or {}).get(cond, SERIES[i % len(SERIES)])
        label = (labels or {}).get(cond, cond)
        ax.fill_between(x, ys.min(axis=0), ys.max(axis=0), color=color, alpha=0.15, linewidth=0)
        ax.plot(x, ys.mean(axis=0), color=color, linewidth=2, label=label)
        if direct_labels:
            ax.annotate(label, (x[-1], ys.mean(axis=0)[-1]), xytext=(6, 0),
                        textcoords="offset points", color=TEXT, fontsize=8, va="center")
        x0 = x[0]
    for label, y in (refs or {}).items():
        ax.axhline(y, color=REF, linewidth=1, linestyle="--")
        ax.annotate(label, (x0, y), xytext=(0, 4), textcoords="offset points", color=TEXT_2,
                    fontsize=8)
    style(ax, title, "episodios de entrenamiento", ylabel or metric)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return ax


def plot_policy_grid(actions: np.ndarray, disc, row: str, col: str, facet: str,
                     action_short: Sequence[str], action_colors: Sequence[str],
                     values: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None,
                     title: str = "", figsize=None, names: Optional[Dict[str, str]] = None):
    """Small multiples of a policy over a ``ProductDiscretizer`` with three features: one panel
    per ``facet`` bin, ``row`` x ``col`` cells labelled with the action (and ``values`` if
    given). Every cell is labelled, so identity never relies on colour. ``mask``: states to
    blank (e.g. never visited). ``names``: display name per feature key."""
    names = names or {}
    keys = [f.key for f in disc.features]
    fi = {k: keys.index(k) for k in (row, col, facet)}
    feats = {k: disc.features[i] for k, i in fi.items()}
    n_f = feats[facet].n_bins
    fig, axes = plt.subplots(1, n_f, figsize=figsize or (3.1 * n_f, 2.6), squeeze=False)
    cmap = ListedColormap(list(action_colors))
    for p, ax in enumerate(axes[0]):
        grid = np.full((feats[row].n_bins, feats[col].n_bins), np.nan)
        vals = np.full_like(grid, np.nan)
        for r in range(feats[row].n_bins):
            for c in range(feats[col].n_bins):
                bins = [0] * len(keys)
                bins[fi[row]], bins[fi[col]], bins[fi[facet]] = r, c, p
                s = disc.compose(bins)
                if mask is not None and mask[s]:
                    continue
                grid[r, c] = actions[s]
                if values is not None:
                    vals[r, c] = values[s]
        ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-0.5,
                  vmax=len(action_colors) - 0.5, aspect="auto")
        for (r, c), a in np.ndenumerate(grid):
            text = "—" if np.isnan(a) else action_short[int(a)]
            if values is not None and not np.isnan(vals[r, c]):
                text += f"\n{vals[r, c]:.0f}"
            ax.text(c, r, text, ha="center", va="center", fontsize=8, color=TEXT)
        ax.set_xticks(range(feats[col].n_bins), feats[col].bin_labels, fontsize=8)
        ax.set_yticks(range(feats[row].n_bins), feats[row].bin_labels if p == 0 else [""] * feats[row].n_bins,
                      fontsize=8)
        style(ax, f"{names.get(facet, facet)}: {feats[facet].bin_labels[p]}", names.get(col, col),
              names.get(row, row) if p == 0 else "")
        ax.grid(False)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=TEXT)
    fig.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


SHOT_OUTCOME_COLORS = {1: "#1baf7a", 2: "#eb6834", 3: "#2a78d6", 4: "#8a8984", 0: "#8a8984"}


def plot_shooting_episodes(records: Sequence[Dict], ax=None, title: str = "",
                           outcome_names: Optional[Dict[int, str]] = None, reach: float = 2.0,
                           show_reach: bool = True):
    """Shooting episodes near the rival goal: start (ring), CONDUCIR path, shot to the goal line
    coloured by outcome, keeper as a bar of its reach (or, with ``show_reach=False``, a tick at
    its position, for many overlapping episodes). +y downward, as on the monitor."""
    from src import shooting_env as se
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    ax.set_facecolor(SURFACE)
    ax.plot([se.GOAL_X, se.GOAL_X], [-12, 12], color=GRID, linewidth=1.2)
    ax.plot([se.GOAL_X] * 2, [-se.POST_Y, se.POST_Y], color=TEXT, linewidth=3)
    for y in (-se.POST_Y, se.POST_Y):
        ax.plot(se.GOAL_X, y, "s", color=TEXT, markersize=4)
    seen = set()
    for rec in records:
        traj = rec["trajectory"]
        bx = [t["ball"][0] for t in traj]
        by = [t["ball"][1] for t in traj]
        ax.plot(bx[0], by[0], "o", color=TEXT_2, markerfacecolor="none", markersize=6)
        ax.plot(bx, by, color=TEXT_2, linewidth=1.2)
        keeper = traj[0]["keeper"]
        if keeper is not None and show_reach:
            ax.plot([se.GOAL_X + 0.6] * 2, [keeper - reach, keeper + reach], color="#2a78d6",
                    linewidth=5, alpha=0.35, solid_capstyle="butt")
        elif keeper is not None:
            ax.plot(se.GOAL_X + 0.8, keeper, "_", color="#2a78d6", markersize=9, markeredgewidth=2.5)
        end, outcome = traj[-1]["shot_end"], traj[-1]["outcome"]
        color = SHOT_OUTCOME_COLORS[outcome]
        label = (outcome_names or se.OUTCOME_NAMES)[outcome]
        if end is not None:
            ax.plot([bx[-1], end[0]], [by[-1], end[1]], color=color, linewidth=1.6,
                    label=None if label in seen else label)
            ax.plot(*end, "o", color=color, markersize=4)
            seen.add(label)
    ax.set_xlim(25, 55)
    ax.set_ylim(13, -13)
    ax.set_aspect("equal")
    style(ax, title, "x (m)", "y (m)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    return ax


DRIBBLE_OUTCOME_COLORS = {1: "#1baf7a", 2: "#eb6834", 3: "#eb6834", 0: "#8a8984"}


def plot_dribbling_episodes(records: Sequence[Dict], ax=None, title: str = "",
                            outcome_names: Optional[Dict[int, str]] = None, full_pitch: bool = False):
    """Dribbling episodes on the pitch (light surface): ball path coloured by outcome, player
    path in gray, start as a ring, 30 m progress line per episode. +y downward, as on the
    monitor."""
    from src import dribbling_env as de
    ax = ax or plt.subplots(figsize=(8, 5.4))[1]
    draw_pitch(ax, light=True)
    names = outcome_names or de.OUTCOME_NAMES
    seen, xs, ys = set(), [], []
    for rec in records:
        traj = rec["trajectory"]
        px = [t["player"][0] for t in traj]
        py = [t["player"][1] for t in traj]
        bx = [t["ball"][0] for t in traj]
        by = [t["ball"][1] for t in traj]
        outcome = traj[-1]["outcome"]
        ax.plot(px, py, color=REF, linewidth=1.0)
        label = names[outcome]
        ax.plot(bx, by, color=DRIBBLE_OUTCOME_COLORS[outcome], linewidth=1.8,
                label=None if label in seen else label)
        seen.add(label)
        ax.plot(px[0], py[0], "o", color=TEXT_2, markerfacecolor="none", markersize=6)
        ax.plot([bx[0] + de.GOAL_ADVANCE] * 2, [by[0] - 1.5, by[0] + 1.5], color=TEXT_2,
                linewidth=1, linestyle=":")
        xs += px + bx + [bx[0] + de.GOAL_ADVANCE]
        ys += py + by
    if not full_pitch and xs:
        pad = 3.0
        x0, x1 = min(xs) - pad, max(xs) + pad
        y0, y1 = min(ys) - pad, max(ys) + pad
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    ax.set_title(title, fontsize=11, loc="left", color=TEXT)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.figure.set_facecolor(SURFACE)
    return ax


PASS_OUTCOME_COLORS = {0: "#1baf7a", 1: "#eb6834", 2: "#eb6834", 3: "#eda100", 4: "#8a8984"}


def plot_passing_episode(record: Dict, ax=None, title: str = "",
                         outcome_names: Optional[Dict[int, str]] = None):
    """One 2v1 episode in the 30 x 20 m zone: ball path (passes are straight jumps) coloured by
    the outcome, pass receptions as dots, defender path dashed gray. +y downward, as on the
    monitor."""
    from src import passing_env as pe
    ax = ax or plt.subplots(figsize=(5, 3.6))[1]
    ax.set_facecolor(SURFACE)
    ax.plot([-pe.HALF_X, pe.HALF_X, pe.HALF_X, -pe.HALF_X, -pe.HALF_X],
            [-pe.HALF_Y, -pe.HALF_Y, pe.HALF_Y, pe.HALF_Y, -pe.HALF_Y], color=GRID, linewidth=1.2)
    traj = record["trajectory"]
    outcome = traj[-1]["outcome"]
    color = PASS_OUTCOME_COLORS[outcome]
    hx = [t["holder"][0] for t in traj]
    hy = [t["holder"][1] for t in traj]
    ax.plot([t["defender"][0] for t in traj], [t["defender"][1] for t in traj], color=REF,
            linewidth=1.2, linestyle="--", label="defensor")
    ax.plot(hx, hy, color=color, linewidth=1.6,
            label=f"balón: {(outcome_names or pe.OUTCOME_NAMES)[outcome]}")
    passes = [i for i, t in enumerate(traj) if t.get("event") == "pase"]
    ax.plot([hx[i] for i in passes], [hy[i] for i in passes], "o", color=color, markersize=3.5)
    ax.plot(hx[0], hy[0], "o", color=TEXT_2, markerfacecolor="none", markersize=7)
    ax.set_xlim(-pe.HALF_X - 1, pe.HALF_X + 1)
    ax.set_ylim(pe.HALF_Y + 1, -pe.HALF_Y - 1)
    ax.set_aspect("equal")
    style(ax, title, "x (m)", "y (m)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    return ax
