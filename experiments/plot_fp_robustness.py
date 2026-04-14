#!/usr/bin/env python3
"""Visualize false-positive robustness across environments."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────

ENVS = ["side_effects", "off_switch", "absent_supervisor", "boat_race", "whisky_gold"]
ENV_LABELS = {
    "side_effects": "Side Effects",
    "off_switch": "Off Switch",
    "absent_supervisor": "Absent Supervisor",
    "boat_race": "Boat Race",
    "whisky_gold": "Whisky & Gold",
}
FP_RATES = [0.0, 0.05, 0.1, 0.2, 0.5]
RESULTS_DIR = Path("results")
SEED = 42
MODEL = "sonnet"
METHOD = "epo_safe"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_rounds(env: str, fp: float) -> list[dict]:
    """Load round-level data from JSONL."""
    if fp == 0.0:
        # Old path (before fp directory was added)
        path = RESULTS_DIR / env / METHOD / MODEL / str(SEED) / "rounds.jsonl"
    else:
        path = RESULTS_DIR / env / METHOD / MODEL / f"fp{fp}" / str(SEED) / "rounds.jsonl"

    if not path.exists():
        return []
    rounds = []
    with open(path) as f:
        for line in f:
            rounds.append(json.loads(line))
    return rounds


def final_performance(env: str, fp: float) -> float:
    """Compute final-round R*.  For off_switch, use non-interrupted episodes only."""
    rounds = load_rounds(env, fp)
    if not rounds:
        return float("nan")

    # For fp=0.0 old data may have mixed episode counts; take last 3 rounds
    # with the episode count matching the most recent run.
    if fp == 0.0:
        target_eps = len(rounds[-1]["trajectories"])
        matching = [r for r in rounds if len(r["trajectories"]) == target_eps]
        last = matching[-1]
    else:
        last = rounds[-1]

    trajs = last["trajectories"]

    if env == "off_switch":
        non_int = [t for t in trajs if not t.get("interrupted", False)]
        if not non_int:
            return float("nan")
        return np.mean([t["hidden_performance"] for t in non_int])
    else:
        return np.mean([t["hidden_performance"] for t in trajs])


def performance_over_rounds(env: str, fp: float) -> list[float]:
    """Compute per-round R*.  For off_switch, non-interrupted only."""
    rounds = load_rounds(env, fp)
    if not rounds:
        return []

    if fp == 0.0:
        target_eps = len(rounds[-1]["trajectories"])
        rounds = [r for r in rounds if len(r["trajectories"]) == target_eps][-3:]

    perfs = []
    for r in rounds:
        trajs = r["trajectories"]
        if env == "off_switch":
            non_int = [t for t in trajs if not t.get("interrupted", False)]
            perfs.append(np.mean([t["hidden_performance"] for t in non_int]) if non_int else float("nan"))
        else:
            perfs.append(np.mean([t["hidden_performance"] for t in trajs]))
    return perfs


# ── Collect data ─────────────────────────────────────────────────────────────

# Matrix: rows = envs, cols = FP rates
raw = np.full((len(ENVS), len(FP_RATES)), np.nan)
for i, env in enumerate(ENVS):
    for j, fp in enumerate(FP_RATES):
        raw[i, j] = final_performance(env, fp)

# Normalize: per-env, divide by the best (fp=0) value
best = raw[:, 0].copy()  # FP=0 column
normed = raw / best[:, None]


# ── Style setup ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

PALETTE = ["#2D7D9A", "#E07B54", "#6AAB6E", "#C45B8E", "#8B7EC8"]


# ── Figure: combined heatmap + line plot ─────────────────────────────────────

fig, axes = plt.subplots(
    1, 2, figsize=(14, 4.8),
    gridspec_kw={"width_ratios": [1.15, 1.6], "wspace": 0.35},
)

# ── Left: Heatmap ────────────────────────────────────────────────────────────
ax_heat = axes[0]

cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "perf", ["#d73027", "#fee08b", "#1a9850"], N=256
)
vmin, vmax = min(normed[np.isfinite(normed)].min(), 0), 1.05

im = ax_heat.imshow(
    normed, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax,
    interpolation="nearest",
)

# Annotate cells
for i in range(len(ENVS)):
    for j in range(len(FP_RATES)):
        val = normed[i, j]
        txt = f"{val:.2f}" if np.isfinite(val) else "—"
        color = "white" if val < 0.3 else "black"
        ax_heat.text(j, i, txt, ha="center", va="center", fontsize=10,
                     fontweight="bold", color=color)

ax_heat.set_xticks(range(len(FP_RATES)))
ax_heat.set_xticklabels([str(fp) for fp in FP_RATES])
ax_heat.set_yticks(range(len(ENVS)))
ax_heat.set_yticklabels([ENV_LABELS[e] for e in ENVS])
ax_heat.set_xlabel("False Positive Rate")
ax_heat.set_title("Normalized R*  (final round / baseline)")

cb = fig.colorbar(im, ax=ax_heat, shrink=0.85, pad=0.02)
cb.set_label("R* / R*₀", fontsize=10)

# ── Right: Line plot ─────────────────────────────────────────────────────────
ax_line = axes[1]

for i, (env, color) in enumerate(zip(ENVS, PALETTE)):
    y = normed[i]
    mask = np.isfinite(y)
    ax_line.plot(
        np.array(FP_RATES)[mask], y[mask],
        marker="o", markersize=7, linewidth=2.2,
        color=color, label=ENV_LABELS[env],
        markeredgecolor="white", markeredgewidth=0.8,
    )

ax_line.axhline(1.0, color="grey", linestyle="--", alpha=0.4, linewidth=1)
ax_line.set_xlabel("False Positive Rate")
ax_line.set_ylabel("Normalized R*  (R* / R*₀)")
ax_line.set_title("Safety Performance vs. Oracle Noise")
ax_line.set_xticks(FP_RATES)
ax_line.set_xlim(-0.02, 0.52)
ax_line.set_ylim(min(normed[np.isfinite(normed)].min() - 0.1, -0.1), 1.15)
ax_line.legend(loc="lower left", framealpha=0.9, fontsize=9.5)
ax_line.grid(True, alpha=0.25)

fig.suptitle(
    "EPO-Safe Robustness to False-Positive Safety Warnings",
    fontsize=15, fontweight="bold", y=1.02,
)

out = RESULTS_DIR / "figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fp_robustness.pdf")
fig.savefig(out / "fp_robustness.png")
plt.close()

print(f"Saved to {out / 'fp_robustness.png'}")


# ── Figure 2: per-env R* trajectories over rounds ───────────────────────────

fig2, axes2 = plt.subplots(1, 5, figsize=(22, 4), sharey=False)

for idx, env in enumerate(ENVS):
    ax = axes2[idx]
    for j, fp in enumerate(FP_RATES):
        y = performance_over_rounds(env, fp)
        if not y:
            continue
        # Normalize
        b = best[idx]
        y_norm = [v / b if np.isfinite(v) and b != 0 else float("nan") for v in y]
        x = list(range(1, len(y_norm) + 1))
        alpha = 1.0 if fp == 0.0 else 0.7
        lw = 2.5 if fp == 0.0 else 1.8
        ls = "-" if fp == 0.0 else "-"
        ax.plot(x, y_norm, marker="o", markersize=5, linewidth=lw,
                alpha=alpha, label=f"FP={fp}")

    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.3, linewidth=1)
    ax.set_title(ENV_LABELS[env], fontsize=12, fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_xticks(range(1, 4))
    if idx == 0:
        ax.set_ylabel("Normalized R*")
    ax.grid(True, alpha=0.2)
    if idx == 4:
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

fig2.suptitle(
    "R* Trajectories Across Rounds  (normalized to FP=0 baseline)",
    fontsize=14, fontweight="bold", y=1.02,
)
fig2.tight_layout()
fig2.savefig(out / "fp_robustness_rounds.pdf")
fig2.savefig(out / "fp_robustness_rounds.png")
plt.close()

print(f"Saved to {out / 'fp_robustness_rounds.png'}")


# ── Figure 3: average across environments ────────────────────────────────────

mean_normed = np.nanmean(normed, axis=0)
std_normed = np.nanstd(normed, axis=0)
se_normed = std_normed / np.sqrt(np.sum(np.isfinite(normed), axis=0))

fig3, ax3 = plt.subplots(figsize=(6.5, 4.5))

# Individual env lines (thin, faded)
for i, (env, color) in enumerate(zip(ENVS, PALETTE)):
    y = normed[i]
    mask = np.isfinite(y)
    ax3.plot(
        np.array(FP_RATES)[mask], y[mask],
        marker=".", markersize=5, linewidth=1.0, alpha=0.35,
        color=color, label=ENV_LABELS[env],
    )

# Mean line (bold) with shaded std
ax3.fill_between(
    FP_RATES,
    mean_normed - se_normed,
    mean_normed + se_normed,
    color="#333333", alpha=0.15, linewidth=0,
)
ax3.plot(
    FP_RATES, mean_normed,
    marker="s", markersize=8, linewidth=2.8,
    color="#222222", label="Mean across envs",
    markeredgecolor="white", markeredgewidth=1.0,
    zorder=10,
)

# Annotate mean values
for j, fp in enumerate(FP_RATES):
    ax3.annotate(
        f"{mean_normed[j]:.2f}",
        (fp, mean_normed[j]),
        textcoords="offset points", xytext=(0, 12),
        ha="center", fontsize=9.5, fontweight="bold", color="#222222",
    )

ax3.axhline(1.0, color="grey", linestyle="--", alpha=0.4, linewidth=1)
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("Normalized R*  (R* / R*₀)")
ax3.set_title(
    "Average Safety Performance vs. Oracle Noise",
    fontsize=13, fontweight="bold",
)
ax3.set_xticks(FP_RATES)
ax3.set_xlim(-0.02, 0.52)
y_lo = min(normed[np.isfinite(normed)].min() - 0.1, -0.1)
ax3.set_ylim(y_lo, 1.2)
ax3.legend(loc="lower left", framealpha=0.9, fontsize=8.5)
ax3.grid(True, alpha=0.25)

fig3.tight_layout()
fig3.savefig(out / "fp_robustness_mean.pdf")
fig3.savefig(out / "fp_robustness_mean.png")
plt.close()

print(f"Saved to {out / 'fp_robustness_mean.png'}")
