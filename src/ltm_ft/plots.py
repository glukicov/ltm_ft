"""Figures for the README and write-up, rendered only from the JSON files under outputs/.

    uv run ltm-ft plot          # -> docs/figures/*.png

Colours are the first two slots of a colour-vision-deficiency-validated categorical palette
(zero-shot orange, fine-tuned blue); the ceiling is a neutral dashed line. Every series is
labelled directly, so nothing depends on colour alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ZERO_SHOT = "#eb6834"
FINE_TUNED = "#2a78d6"
CEILING = "#8a8983"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def validation_curve(tune_json: Path, out: Path, title: str) -> Path:
    """Validation accuracy against fine-tuning step, with the zero-shot start and the ceiling marked."""
    run = json.loads(tune_json.read_text())
    steps = [h["step"] for h in run["history"]]
    acc = [100 * h["val_accuracy"] for h in run["history"]]
    ceiling = 100 * run["data"]["val_ceiling"]["accuracy"]

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.axhline(acc[0], color=ZERO_SHOT, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.axhline(ceiling, color=CEILING, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.plot(steps, acc, color=FINE_TUNED, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE)
    best = steps.index(run["best_step"])
    ax.annotate(
        f"ceiling {ceiling:.1f}%",
        (steps[-1], ceiling),
        xytext=(0, 6),
        textcoords="offset points",
        ha="right",
        color=INK,
        fontsize=10,
    )
    ax.annotate(
        f"zero-shot {acc[0]:.1f}%",
        (steps[1], acc[0]),
        xytext=(0, -16),
        textcoords="offset points",
        ha="left",
        color=INK,
        fontsize=10,
    )
    ax.set_ylim(acc[0] - 3, ceiling + 2)
    ax.scatter([steps[best]], [acc[best]], s=90, color=FINE_TUNED, edgecolor=INK, linewidth=1.5, zorder=4)
    ax.annotate(
        f"kept: step {steps[best]}, {acc[best]:.1f}%",
        (steps[best], acc[best]),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        color=INK,
        fontsize=10,
    )
    ax.set_xlabel("fine-tuning step", color=MUTED, fontsize=10)
    ax.set_ylabel("validation accuracy (%)", color=MUTED, fontsize=10)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def before_after(run_jsons: dict[str, Path], out: Path, title: str) -> Path:
    """One row per task: zero-shot and fine-tuned test log loss (lower is better), with the ceiling as a tick."""
    rows = []
    for label, path in run_jsons.items():
        values = list(json.loads(path.read_text())["test"].values())  # zero-shot, fine-tuned, ceiling
        rows.append((label, *(v["log_loss"] for v in values)))

    fig, ax = plt.subplots(figsize=(8, 1.4 + 1.1 * len(rows)), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    label_kw: dict[str, Any] = {"textcoords": "offset points", "fontsize": 9, "color": INK}
    for i, (_, zero, tuned, ceil) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([ceil, zero], [y, y], color=GRID, linewidth=4, solid_capstyle="round", zorder=1)
        ax.plot([ceil, ceil], [y - 0.2, y + 0.2], color=CEILING, linewidth=2, zorder=2)
        ax.annotate(f"ceiling {ceil:.3f}", (ceil, y), xytext=(0, -24), ha="center", **{**label_kw, "color": MUTED})
        ax.scatter([zero], [y], s=90, color=ZERO_SHOT, edgecolor=SURFACE, linewidth=2, zorder=3)
        if abs(zero - tuned) < 5e-4:
            ax.annotate(f"zero-shot = fine-tuned {zero:.3f}", (zero, y), xytext=(10, 10), ha="left", **label_kw)
        else:
            ax.scatter([tuned], [y], s=90, color=FINE_TUNED, edgecolor=SURFACE, linewidth=2, zorder=4)
            ax.annotate(f"zero-shot {zero:.3f}", (zero, y), xytext=(8, 12), ha="left", **label_kw)
            ax.annotate(f"fine-tuned {tuned:.3f}", (tuned, y), xytext=(-8, -20), ha="right", **label_kw)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], color=INK, fontsize=10)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    lo, hi = min(min(r[1:]) for r in rows), max(max(r[1:]) for r in rows)
    ax.set_xlim(lo - 0.05, hi + 0.12)
    ax.set_xlabel("test log loss (lower is better)", color=MUTED, fontsize=10)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out
