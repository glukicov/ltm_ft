"""Figures for the README and write-up, rendered only from the JSON files under outputs/.

    uv run ltm-ft plot          # -> docs/figures/*.png

Colours are the first three slots of a colour-vision-deficiency-validated categorical palette:
zero-shot orange, fine-tuned (last 4 blocks) blue, fine-tuned with the encoders aqua. Every series
also has its own marker shape and a direct label, so nothing depends on colour alone; the ceiling
is a neutral dashed line or tick.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ZERO_SHOT = "#eb6834"
LAST4 = "#2a78d6"
ENCODERS = "#1baf7a"
CEILING = "#8a8983"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
MARKERS = {ZERO_SHOT: "o", LAST4: "s", ENCODERS: "D"}


def _style(ax: Any, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig: Any, out: Path) -> Path:
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def _load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


def _test_metric(run: dict[str, Any], metric: str) -> tuple[float, float, float]:
    """(zero-shot, fine-tuned, ceiling), in the order `run` writes them."""
    zero, tuned, ceil = (v[metric] for v in run["test"].values())
    return float(zero), float(tuned), float(ceil)


def validation_curves(series: Sequence[tuple[str, Path, str]], out: Path, title: str) -> Path:
    """Validation accuracy against step for several runs that share a split (same step-0 model and ceiling)."""
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200, facecolor=SURFACE)
    _style(ax)
    first = _load(series[0][1])
    zero = 100 * first["history"][0]["val_accuracy"]
    ceiling = 100 * first["data"]["val_ceiling"]["accuracy"]
    ax.axhline(zero, color=ZERO_SHOT, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.axhline(ceiling, color=CEILING, linewidth=1.5, linestyle=(0, (4, 3)))
    last_step = 0
    for label, path, color in series:
        run = _load(path)
        steps = [h["step"] for h in run["history"]]
        acc = [100 * h["val_accuracy"] for h in run["history"]]
        last_step = max(last_step, steps[-1])
        ax.plot(steps, acc, color=color, linewidth=2, marker=MARKERS[color], markersize=6, markeredgecolor=SURFACE)
        ax.annotate(
            f"{label} {acc[-1]:.1f}%",
            (steps[-1], acc[-1]),
            xytext=(0, -17 if ceiling - acc[-1] < 3 else 9),
            textcoords="offset points",
            ha="right",
            color=INK,
            fontsize=10,
        )
    ax.annotate(
        f"ceiling {ceiling:.1f}%",
        (0, ceiling),
        xytext=(0, 6),
        textcoords="offset points",
        ha="left",
        color=INK,
        fontsize=10,
    )
    ax.annotate(
        f"zero-shot {zero:.1f}%",
        (last_step, zero),
        xytext=(0, -17),
        textcoords="offset points",
        ha="right",
        color=INK,
        fontsize=10,
    )
    ax.set_ylim(zero - 5, ceiling + 3)
    ax.set_xlabel("fine-tuning step", color=MUTED, fontsize=10)
    ax.set_ylabel("validation accuracy (%)", color=MUTED, fontsize=10)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    return _save(fig, out)


def before_after(
    panels: dict[str, dict[str, Path]], out: Path, title: str, notes: dict[str, dict[str, str]] | None = None
) -> Path:
    """Test accuracy per task and variant, one panel per machine: zero-shot, fine-tuned and the ceiling.

    `panels` maps a machine label to {row label: results.json}; rows are matched by label across panels.
    `notes` optionally explains a missing row per machine (default "not run").
    """
    row_labels = list(dict.fromkeys(label for rows in panels.values() for label in rows))
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.2 * len(panels), 1.2 + 0.95 * len(row_labels)),
        dpi=200,
        facecolor=SURFACE,
        sharey=True,
        squeeze=False,
    )
    for ax, (machine, rows) in zip(axes[0], panels.items(), strict=True):
        _style(ax, grid_axis="x")
        for i, label in enumerate(row_labels):
            y = len(row_labels) - 1 - i
            if label not in rows:
                note = (notes or {}).get(machine, {}).get(label, "not run")
                ax.annotate(note, (70, y), ha="center", va="center", color=MUTED, fontsize=9)
                continue
            run = _load(rows[label])
            zero, tuned, ceil = (100 * v for v in _test_metric(run, "accuracy"))
            color = ENCODERS if run["config"]["train_encoders"] else LAST4
            ax.plot([min(zero, tuned), ceil], [y, y], color=GRID, linewidth=4, solid_capstyle="round", zorder=1)
            ax.plot([ceil, ceil], [y - 0.22, y + 0.22], color=CEILING, linewidth=2, zorder=2)
            ax.scatter([zero], [y], s=70, color=ZERO_SHOT, marker="o", edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.scatter(
                [tuned], [y], s=70, color=color, marker=MARKERS[color], edgecolor=SURFACE, linewidth=1.5, zorder=4
            )
            text = f"{zero:.1f}% → {tuned:.1f}%"
            ax.annotate(
                text,
                (min(zero, tuned), y),
                xytext=(-10, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                color=INK,
                fontsize=9,
            )
        ax.set_xlim(40, 100)
        ax.set_ylim(-0.7, len(row_labels) - 0.3)
        ax.set_title(machine, loc="left", color=INK, fontsize=11, fontweight="bold")
        ax.set_xlabel("test accuracy (%)", color=MUTED, fontsize=10)
    axes[0][0].set_yticks(range(len(row_labels)), list(reversed(row_labels)), color=INK, fontsize=10)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    return _save(fig, out)


def efficiency(rows: Sequence[tuple[str, float, float, str]], out: Path, title: str) -> Path:
    """Speed-up of the second machine over the first per measurement: (label, first s, second s, unit label)."""
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.7 * len(rows)), dpi=200, facecolor=SURFACE)
    _style(ax, grid_axis="x")
    top = max(a / b for _, a, b, _ in rows)
    for i, (_label, first, second, _unit) in enumerate(rows):
        y = len(rows) - 1 - i
        ratio = first / second
        ax.barh(y, ratio, height=0.55, color=LAST4, zorder=2)
        ax.annotate(
            f"{ratio:.1f}\u00d7   {_seconds(first)} → {_seconds(second)}",
            (ratio, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            color=INK,
            fontsize=9,
        )
    ax.axvline(1, color=CEILING, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], color=INK, fontsize=10)
    ax.set_xlim(0, top * 1.45)
    ax.set_xlabel("times faster (dashed line: same speed)", color=MUTED, fontsize=10)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    return _save(fig, out)


def _seconds(s: float) -> str:
    if s >= 90:
        return f"{s / 60:.1f} min"
    return f"{s:.1f} s" if s >= 1 else f"{s:.2f} s"


def seeds(rows: dict[str, tuple[Path, Path]], out: Path, title: str) -> Path:
    """Per data seed: test accuracy zero-shot, fine-tuned last 4 blocks, fine-tuned with encoders, and the ceiling."""
    fig, ax = plt.subplots(figsize=(8, 1.4 + 0.62 * len(rows)), dpi=200, facecolor=SURFACE)
    _style(ax, grid_axis="x")
    for i, (last4_path, encoders_path) in enumerate(rows.values()):
        y = len(rows) - 1 - i
        zero, last4, ceil = (100 * v for v in _test_metric(_load(last4_path), "accuracy"))
        _, encoders, _ = (100 * v for v in _test_metric(_load(encoders_path), "accuracy"))
        ax.plot([zero, ceil], [y, y], color=GRID, linewidth=3, zorder=1)
        ax.plot([ceil, ceil], [y - 0.25, y + 0.25], color=CEILING, linewidth=2, zorder=2)
        for value, color, name in (
            (zero, ZERO_SHOT, "zero-shot"),
            (last4, LAST4, "last 4 blocks"),
            (encoders, ENCODERS, "+ encoders"),
        ):
            ax.scatter(
                [value],
                [y],
                s=60,
                color=color,
                marker=MARKERS[color],
                edgecolor=SURFACE,
                linewidth=1.2,
                zorder=3,
                label=name if i == 0 else None,
            )
        ax.annotate(
            f"{encoders:.1f}%",
            (encoders, y),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            color=INK,
            fontsize=8,
        )
    ax.set_yticks(range(len(rows)), list(reversed(rows)), color=INK, fontsize=10)
    ax.set_xlim(40, 100)
    ax.set_ylim(-0.7, len(rows) - 0.2)
    ax.set_xlabel("test accuracy (%), grey tick = ceiling", color=MUTED, fontsize=10)
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK, ncols=3, bbox_to_anchor=(0, 1.0))
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=26)
    return _save(fig, out)


def log_loss_curves(series: Sequence[tuple[str, Path]], out: Path, title: str) -> Path:
    """Validation log loss against step for runs on different splits, with ln 2 marked: predicting 0.5 for every row."""
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.axhline(0.6931, color=CEILING, linewidth=1.5, linestyle=(0, (4, 3)))
    styles = ["-", "--", ":"]
    markers = ["D", "s", "o"]
    last_step = 0
    for i, (label, path) in enumerate(series):
        run = _load(path)
        steps = [h["step"] for h in run["history"]]
        loss = [h["val_log_loss"] for h in run["history"]]
        last_step = max(last_step, steps[-1])
        ax.plot(
            steps,
            loss,
            color=ENCODERS,
            linewidth=2,
            linestyle=styles[i % 3],
            marker=markers[i % 3],
            markersize=5,
            markeredgecolor=SURFACE,
        )
        ax.annotate(
            label,
            (steps[-1], loss[-1]),
            xytext=(-4, 10 if loss[-1] < 0.6 else -16 - 13 * i),
            textcoords="offset points",
            ha="right",
            color=INK,
            fontsize=10,
        )
    ax.annotate(
        "predicting 0.5 for every row (log loss ln 2)",
        (0, 0.6931),
        xytext=(0, 7),
        textcoords="offset points",
        ha="left",
        color=MUTED,
        fontsize=9,
    )
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("fine-tuning step (no early stopping)", color=MUTED, fontsize=10)
    ax.set_ylabel("validation log loss (lower is better)", color=MUTED, fontsize=10)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    return _save(fig, out)
