"""`ltm-ft`: zero-shot TabFM vs the same TabFM fine-tuned, on identical synthetic rows.

    uv run ltm-ft data --task checkerboard                     # describe a split, no model needed
    uv run ltm-ft tune --task checkerboard --name lr3e-4       # validation curve only (no test rows)
    uv run ltm-ft run --task checkerboard --name checkerboard  # zero-shot -> fine-tune -> same test rows

`run` writes outputs/<name>/results.json (config, metrics, bootstrap intervals, validation history),
summary.md and test_predictions.csv. Fine-tuned weights are only written with --save-weights and
are never committed: they derive from TabFM's non-commercial checkpoint.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import tabfm
import torch
import typer
from safetensors.torch import save_file

from ltm_ft.data import POSITIVE, TRUE_PROB, Split, Task, labels, make_split
from ltm_ft.evaluate import Metrics, ceiling, paired_bootstrap, predict_positive, score
from ltm_ft.finetune import FinetuneConfig, FinetuneResult, finetune
from ltm_ft.hardware import accelerator_name, synchronize
from ltm_ft.model import load_classifier, pick_device, trainable_state

OUTPUTS_DIR = Path("outputs")

app = typer.Typer(add_completion=False, no_args_is_help=True)

# Options shared by `run` and `tune`.
TaskOpt = Annotated[Task, typer.Option(help="synthetic task")]
TrainOpt = Annotated[int, typer.Option(help="labelled rows: TabFM's context and the fine-tuning data")]
ValOpt = Annotated[int, typer.Option(help="rows for early stopping / tuning")]
SeedOpt = Annotated[int, typer.Option(help="seed for the synthetic data")]
NoiseOpt = Annotated[int, typer.Option(help="extra columns that carry no signal")]
BlocksOpt = Annotated[int, typer.Option(help="last ICL blocks to fine-tune (of 24)")]
EncodersOpt = Annotated[bool, typer.Option(help="also fine-tune the row/column encoders (slow: full backward)")]
LrOpt = Annotated[float, typer.Option(help="AdamW learning rate")]
StepsOpt = Annotated[int, typer.Option(help="gradient steps")]
ContextOpt = Annotated[int, typer.Option(help="context rows per training episode")]
QueryOpt = Annotated[int, typer.Option(help="query rows per training episode")]
EveryOpt = Annotated[int, typer.Option(help="validate every N steps")]
DeviceOpt = Annotated[str, typer.Option(help="torch device (default: mps, then cuda, then cpu)")]


def _config(**kwargs: Any) -> FinetuneConfig:
    return FinetuneConfig(**kwargs)


def _summary(title: str, rows: dict[str, Metrics], deltas: dict[str, dict[str, float]], n_test: int) -> str:
    lines = [
        f"# {title}",
        "",
        f"{n_test:,} held-out rows. Lower log loss is better; higher ROC AUC and accuracy are better.",
        "",
        "| Model | Log loss | ROC AUC | Accuracy |",
        "|---|---|---|---|",
    ]
    for label, m in rows.items():
        lines.append(f"| {label} | {m.log_loss:.4f} | {m.roc_auc:.4f} | {m.accuracy:.1%} |")
    lines += ["", "Fine-tuned minus zero-shot, with 95% paired-bootstrap intervals over test rows:", ""]
    lines += ["| Metric | Δ | 95% interval |", "|---|---|---|"]
    for metric, d in deltas.items():
        scale, unit = (100, " pts") if metric == "accuracy" else (1, "")
        low, high = d["ci_low"] * scale, d["ci_high"] * scale
        lines.append(f"| {metric} | {d['delta'] * scale:+.4f}{unit} | [{low:+.4f}, {high:+.4f}] |")
    return "\n".join(lines) + "\n"


def _environment(device: str) -> dict[str, str]:
    return {
        "device": device,
        "accelerator": accelerator_name(device),
        "torch": torch.__version__,
        "tabfm": tabfm.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
    }


@app.command()
def run(
    name: Annotated[str, typer.Option(help="results go to outputs/<name>/")],
    task: TaskOpt = Task.checkerboard,
    n_train: TrainOpt = 1500,
    n_val: ValOpt = 500,
    n_test: Annotated[int, typer.Option(help="held-out rows to score")] = 3000,
    data_seed: SeedOpt = 7,
    noise_columns: NoiseOpt = 0,
    n_trainable_blocks: BlocksOpt = 4,
    train_encoders: EncodersOpt = False,
    learning_rate: LrOpt = 3e-4,
    max_steps: StepsOpt = 150,
    context_size: ContextOpt = 512,
    query_size: QueryOpt = 256,
    eval_every: EveryOpt = 25,
    patience: Annotated[int, typer.Option(help="stop after N validations without improvement")] = 3,
    n_estimators_test: Annotated[int, typer.Option(help="ensemble members for the test predictions")] = 4,
    seed: Annotated[int, typer.Option()] = 0,
    device: DeviceOpt = "",
    save_weights: Annotated[bool, typer.Option(help="also save the fine-tuned trainable weights")] = False,
) -> None:
    """Score zero-shot TabFM on held-out rows, fine-tune it, and score it again on the same rows."""
    started = time.perf_counter()
    device = pick_device(device)
    out_dir = OUTPUTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    config = _config(
        n_trainable_blocks=n_trainable_blocks,
        train_encoders=train_encoders,
        learning_rate=learning_rate,
        max_steps=max_steps,
        context_size=context_size,
        query_size=query_size,
        eval_every=eval_every,
        patience=patience,
        seed=seed,
    )
    split = make_split(n_train, n_val, n_test, data_seed, noise_columns, task)
    y_test = labels(split.test)
    print(f"{task}: {len(split.train)} train / {len(split.val)} val / {len(split.test)} test rows, device {device}")

    print("loading TabFM classification weights...")
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    model = load_classifier(device)
    synchronize(device)
    timings["model_load_s"] = time.perf_counter() - t0

    def score_test(tag: str) -> np.ndarray:
        synchronize(device)
        t0 = time.perf_counter()
        p = predict_positive(model, split.train, split.test, n_estimators_test, seed)
        synchronize(device)
        timings[f"test_{tag}_s"] = time.perf_counter() - t0
        print(f"[{tag}] test: {score(y_test, p)} ({timings[f'test_{tag}_s']:.0f}s)")
        return p

    # Both scores come from the stock TabFMClassifier on a plain bf16 model: the checkpoint as
    # downloaded, then the same checkpoint with the fine-tuned weights swapped in.
    p_before = score_test("zero-shot")
    t0 = time.perf_counter()
    result = finetune(model, split, config, device)
    timings["finetune_s"] = time.perf_counter() - t0
    p_after = score_test("fine-tuned")

    trained = f"last {n_trainable_blocks} ICL blocks" + (" + encoders" if train_encoders else "")
    rows = {
        "TabFM zero-shot": score(y_test, p_before),
        f"TabFM fine-tuned ({trained}, step {result.best_step})": score(y_test, p_after),
        "Generator's true probabilities (ceiling)": ceiling(split.test),
    }
    deltas = paired_bootstrap(y_test, p_before, p_after, seed=seed)
    summary = _summary(f"{name} ({task})", rows, deltas, len(split.test))
    print("\n" + summary)

    results = {
        "data": _data_info(task, split, data_seed, noise_columns),
        "config": config.as_dict(),
        "n_estimators_test": n_estimators_test,
        "test": {label: m.as_dict() for label, m in rows.items()},
        "delta_finetuned_minus_zero_shot": deltas,
        **_result_info(result),
        "timings_s": {**timings, "total_s": time.perf_counter() - started},
        "efficiency": result.efficiency(),
        "environment": _environment(device),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out_dir / "summary.md").write_text(summary)
    pd.DataFrame(
        {
            TRUE_PROB: split.test[TRUE_PROB],
            "label_is_yes": y_test == POSITIVE,
            "p_zero_shot": p_before.round(5),
            "p_finetuned": p_after.round(5),
        }
    ).to_csv(out_dir / "test_predictions.csv", index_label="row_id")
    if save_weights:
        save_file(trainable_state(model), str(out_dir / "trainable_weights.safetensors"))
    print(f"wrote {out_dir}/results.json, summary.md, test_predictions.csv")


@app.command()
def tune(
    name: Annotated[str, typer.Option(help="results go to outputs/tune/<name>.json")],
    task: TaskOpt = Task.checkerboard,
    n_train: TrainOpt = 1500,
    n_val: ValOpt = 500,
    data_seed: SeedOpt = 7,
    noise_columns: NoiseOpt = 0,
    n_trainable_blocks: BlocksOpt = 4,
    train_encoders: EncodersOpt = False,
    learning_rate: LrOpt = 3e-4,
    max_steps: StepsOpt = 150,
    context_size: ContextOpt = 512,
    query_size: QueryOpt = 256,
    eval_every: EveryOpt = 25,
    seed: Annotated[int, typer.Option()] = 0,
    device: DeviceOpt = "",
) -> None:
    """Fine-tune and record the validation curve only - no test rows are generated or scored.

    Use it to choose hyperparameters; `run` then touches the test set once before and once after.
    """
    started = time.perf_counter()
    device = pick_device(device)
    config = _config(
        n_trainable_blocks=n_trainable_blocks,
        train_encoders=train_encoders,
        learning_rate=learning_rate,
        max_steps=max_steps,
        context_size=context_size,
        query_size=query_size,
        eval_every=eval_every,
        patience=max_steps,  # never stop early: we want the whole curve
        seed=seed,
    )
    split = make_split(n_train, n_val, 0, data_seed, noise_columns, task)
    model = load_classifier(device)
    result = finetune(model, split, config, device)
    out = OUTPUTS_DIR / "tune" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": _data_info(task, split, data_seed, noise_columns),
        "config": config.as_dict(),
        **_result_info(result),
        "total_s": time.perf_counter() - started,
        "efficiency": result.efficiency(),
        "environment": _environment(device),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")


def _data_info(task: Task, split: Split, data_seed: int, noise_columns: int) -> dict[str, object]:
    return {
        "task": str(task),
        "n_train": len(split.train),
        "n_val": len(split.val),
        "n_test": len(split.test),
        "data_seed": data_seed,
        "noise_columns": noise_columns,
        "val_ceiling": ceiling(split.val).as_dict(),
    }


def _result_info(result: FinetuneResult) -> dict[str, object]:
    return {"best_step": result.best_step, "stopped_early": result.stopped_early, "history": result.history}


@app.command()
def data(
    task: TaskOpt = Task.checkerboard,
    n_train: TrainOpt = 1500,
    n_val: ValOpt = 500,
    n_test: Annotated[int, typer.Option()] = 3000,
    data_seed: SeedOpt = 7,
    noise_columns: NoiseOpt = 0,
) -> None:
    """Describe the synthetic split without loading TabFM."""
    split = make_split(n_train, n_val, n_test, data_seed, noise_columns, task)
    for part in ("train", "val", "test"):
        df: pd.DataFrame = getattr(split, part)
        rate = (labels(df) == POSITIVE).mean()
        print(f"{part:>5}: {len(df):>5} rows x {df.shape[1] - 2} features, {rate:.1%} yes, ceiling {ceiling(df)}")


@app.command()
def precision(
    name: Annotated[str, typer.Option(help="results go to outputs/<name>.json")] = "precision/checkerboard",
    task: TaskOpt = Task.checkerboard,
    n_train: TrainOpt = 1500,
    n_val: ValOpt = 500,
    data_seed: SeedOpt = 7,
    n_estimators: Annotated[int, typer.Option()] = 2,
    device: DeviceOpt = "",
) -> None:
    """Zero-shot validation scores of the same checkpoint computed in bfloat16 and in float32."""
    import gc

    device = pick_device(device)
    split = make_split(n_train, n_val, 0, data_seed, 0, task)
    y_val = labels(split.val)
    scores: dict[str, dict[str, float]] = {}
    for precision_name, dtype in (("bfloat16", torch.bfloat16), ("float32", None)):
        model = load_classifier(device, dtype)
        metrics = score(y_val, predict_positive(model, split.train, split.val, n_estimators, 0))
        scores[precision_name] = metrics.as_dict()
        print(f"{precision_name}: {metrics}")
        del model
        gc.collect()
    out = OUTPUTS_DIR / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"data": _data_info(task, split, data_seed, 0), "n_estimators": n_estimators, "val": scores}
    out.write_text(json.dumps({**payload, "environment": _environment(device)}, indent=2) + "\n")
    print(f"wrote {out}")


@app.command()
def plot(
    figures_dir: Annotated[Path, typer.Option(help="where to write the PNGs")] = Path("docs/figures"),
) -> None:
    """Render the README figures from outputs/ (no model needed); figures whose inputs are missing are skipped."""
    from ltm_ft import plots

    def results(*parts: str) -> Path:
        return OUTPUTS_DIR.joinpath(*parts, "results.json")

    machines = {"Apple M4 laptop (MPS)": "m4", "NVIDIA L4 on GKE": "l4"}
    variants = {
        "runners · last 4 blocks": "runners",
        "runners · + encoders": "runners_encoders",
        "checkerboard · last 4 blocks": "checkerboard",
        "checkerboard · + encoders": "checkerboard_encoders",
    }
    panels = {
        machine: {label: results(tag, run) for label, run in variants.items() if results(tag, run).exists()}
        for machine, tag in machines.items()
    }
    if any(panels.values()):
        title = "Zero-shot vs fine-tuned TabFM, same 3,000 test rows per task"
        notes = {"Apple M4 laptop (MPS)": {"checkerboard · + encoders": "stopped: ran out of memory (swapping)"}}
        print(plots.before_after(panels, figures_dir / "before_after.png", title, notes))

    curves = [
        ("last 4 blocks", results("l4", "checkerboard"), plots.LAST4),
        ("+ encoders", results("l4", "checkerboard_encoders"), plots.ENCODERS),
    ]
    if all(path.exists() for _, path, _ in curves):
        title = "Checkerboard on the L4: validation accuracy while fine-tuning"
        print(plots.validation_curves(curves, figures_dir / "checkerboard_validation.png", title))

    timing_inputs = [
        results("m4", "checkerboard"),
        results("l4", "checkerboard"),
        results("l4", "checkerboard_encoders"),
    ]
    m4_encoders_timing = OUTPUTS_DIR / "m4" / "tune" / "checkerboard_encoders_timing.json"
    if all(path.exists() for path in timing_inputs) and m4_encoders_timing.exists():
        m4, l4, l4e = (_load_json(path) for path in timing_inputs)
        m4e = _load_json(m4_encoders_timing)

        def scoring(run: dict[str, Any]) -> float:  # both test passes (zero-shot and fine-tuned) of a run
            return float((run["timings_s"]["test_zero-shot_s"] + run["timings_s"]["test_fine-tuned_s"]) / 2)

        rows = [
            ("gradient step, last 4 blocks", m4["efficiency"]["step_s_median"], l4["efficiency"]["step_s_median"], "s"),
            ("gradient step, + encoders", m4e["efficiency"]["step_s_median"], l4e["efficiency"]["step_s_median"], "s"),
            (
                "validation pass, 500 rows",
                m4["efficiency"]["validation_s_mean"],
                l4["efficiency"]["validation_s_mean"],
                "s",
            ),
            ("scoring 3,000 test rows", scoring(m4), scoring(l4), "s"),
            ("150 fine-tuning steps + validation", m4["timings_s"]["finetune_s"], l4["timings_s"]["finetune_s"], "s"),
            ("whole run, incl. model load", m4["timings_s"]["total_s"], l4["timings_s"]["total_s"], "s"),
        ]
        title = "L4 on GKE vs M4 laptop: same checkerboard run"
        print(plots.efficiency(rows, figures_dir / "efficiency.png", title))

    seed_rows = {"data seed 7": (results("l4", "checkerboard"), results("l4", "checkerboard_encoders"))}
    for d in (17, 27, 37, 47):
        seed_rows[f"data seed {d}"] = (
            results("l4", "seeds", f"last4_data{d}"),
            results("l4", "seeds", f"encoders_data{d}"),
        )
    if all(p.exists() for pair in seed_rows.values() for p in pair):
        title = "Checkerboard on the L4: five data seeds"
        print(plots.seeds(seed_rows, figures_dir / "seeds.png", title))

    long_runs = [
        (f"data seed {d}", OUTPUTS_DIR / "l4" / "tune" / f"encoders_data{d}_300steps.json") for d in (17, 27, 47)
    ]
    if all(path.exists() for _, path in long_runs):
        title = "Training the encoders can collapse to predicting 0.5 (L4, lr 3e-4)"
        print(plots.log_loss_curves(long_runs, figures_dir / "collapse.png", title))


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))
