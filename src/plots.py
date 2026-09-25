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
                     title: str = "", direct_labels: bool = True):
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    x0 = 0
    for i, (cond, rs) in enumerate(runs.items()):
        x = np.array([e["episode"] for e in rs[0]["evals"]])
        ys = np.array([[e[metric] for e in r["evals"]] for r in rs])
        _band(ax, x, ys, SERIES[i], cond, direct_labels)
        x0 = x[0]
    for label, y in (refs or {}).items():
        ax.axhline(y, color=REF, linewidth=1, linestyle="--")
        ax.annotate(label, (x0, y), xytext=(0, 4), textcoords="offset points", color=TEXT_2,
                    fontsize=8)
    style(ax, title or f"Greedy evaluation ({metric}), mean ± std over seeds",
          "training episodes", ylabel or metric)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return ax


def plot_training_curve(runs: Dict[str, List[Dict]], key: str, window: int = 500, ax=None,
                        title: str = "", ylabel: str = ""):
    ax = ax or plt.subplots(figsize=(8, 3.5))[1]
    for i, (cond, rs) in enumerate(runs.items()):
        ys = np.array([moving_average(np.nan_to_num(r["history"][key].astype(float)), window)
                       for r in rs])
        x = np.arange(window, window + ys.shape[1])
        _band(ax, x, ys, SERIES[i], cond)
    style(ax, title or f"Training {key} (moving average, {window} episodes)",
          "training episodes", ylabel or key)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return ax


# ------------------------------------------------------------ policy maps
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
                speed_bin: int = 0, ax=None, title: str = "Learned greedy policy π̂(s)"):
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
    ax.set_xticks(range(len(order)), [labels[i] for i in order], rotation=25, ha="right")
    ax.set_yticks(range(disc.n_dist), disc.distance_labels())
    style(ax, title + (f" — speed bin {speed_bin}" if disc.n_speed > 1 else ""),
          "ball bearing bin (left ← front → right)", "distance bin")
    ax.grid(False)
    ax.text(1.0, 1.02, f"UNKNOWN → {ACTION_SHORT[int(Q[disc.unknown_state].argmax())]}",
            transform=ax.transAxes, ha="right", fontsize=8, color=TEXT_2)
    return ax


def plot_value(Q: np.ndarray, disc: Discretizer, speed_bin: int = 0, ax=None,
               title: str = "Learned state value V̂(s) = max_a Q(s, a)"):
    ax = ax or plt.subplots(figsize=(8, 3.6))[1]
    order = _angle_order(disc)
    grid = _grid(disc, Q.max(axis=1), speed_bin)[:, order]
    im = ax.imshow(grid, cmap=BLUE_RAMP, aspect="auto")
    for (r, c), v in np.ndenumerate(grid):
        ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=9,
                color="white" if v > grid.min() + 0.6 * np.ptp(grid) else TEXT)
    labels = disc.angle_labels()
    ax.set_xticks(range(len(order)), [labels[i] for i in order], rotation=25, ha="right")
    ax.set_yticks(range(disc.n_dist), disc.distance_labels())
    style(ax, title + (f" — speed bin {speed_bin}" if disc.n_speed > 1 else ""),
          "ball bearing bin (left ← front → right)", "distance bin")
    ax.grid(False)
    ax.figure.colorbar(im, ax=ax, fraction=0.03)
    return ax


# ------------------------------------------------------------ trajectories
def draw_pitch(ax) -> None:
    ax.set_facecolor("#2e7d32")
    kw = dict(color="white", linewidth=1.2)
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
