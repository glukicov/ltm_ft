"""Fine-tune TabFM on one dataset by gradient descent on in-context episodes.

TabFM predicts a query row from labelled context rows in a single forward pass. Fine-tuning keeps
exactly that interface and just trains it: every step draws a random *episode* from the training
rows - `context_size` rows whose labels the model sees, and `query_size` other rows whose labels
it must predict - and takes a gradient step on the query rows' cross-entropy. This is the recipe
from Rubachev et al. 2025 ("On Finetuning Tabular Foundation Models") and Prior Labs' TabPFN
fine-tuning examples; the defaults below follow the latter (AdamW, lr 1e-5, weight decay 0.01,
gradient clipping at 1.0, linear warmup then cosine decay, early stopping on validation).

Each episode is built by the stock `TabFMClassifier` preprocessing, so the model is trained on
exactly the tensors it sees at inference: the same categorical encoding, outlier clipping,
normalisation (`none` or `power`), random feature order and random class-label shift. One
ensemble view is sampled per step, which doubles as data augmentation.

Validation uses the full training set as context - as the test evaluation does - so the step we
keep is the one that is best in the setting we actually deploy.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tabfm import TabFMClassifier
from torch import nn

from ltm_ft.data import Split, features, labels
from ltm_ft.evaluate import predict_positive, score
from ltm_ft.hardware import memory_gb, reset_peak_memory, synchronize
from ltm_ft.model import freeze, load_trainable_state, prepare_for_finetuning, trainable_state

# The classifier builds this many views per fit (one per normalisation method); each step uses one.
_VIEWS_PER_EPISODE = 2


@dataclass(frozen=True)
class FinetuneConfig:
    n_trainable_blocks: int = 4
    train_encoders: bool = False
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_steps: int = 60
    warmup_steps: int = 5
    context_size: int = 768
    query_size: int = 256
    grad_clip: float = 1.0
    eval_every: int = 10
    patience: int = 3
    n_estimators_val: int = 2
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Episode:
    """Model-ready tensors for one (context, query) episode, as `TabFMClassifier` would build them."""

    x: torch.Tensor  # [1, context + query, features] float32
    y: torch.Tensor  # [1, context + query] context labels, padded with -100 over the query rows
    train_size: torch.Tensor  # [1] number of context rows
    cat_mask: torch.Tensor  # [1, features] which columns are categorical
    d: torch.Tensor  # [1] number of real (unpadded) feature columns
    target: torch.Tensor  # [query] query labels, with the view's class shift applied
    n_classes: int
    view: int  # which of the classifier's ensemble views was sampled
    class_shift: int  # that view's class-label offset


def make_episode(
    model: nn.Module,
    context: Any,
    y_context: np.ndarray,
    query: Any,
    y_query: np.ndarray,
    seed: int,
    device: str,
) -> Episode:
    clf = TabFMClassifier(model=model, n_estimators=_VIEWS_PER_EPISODE, random_state=seed)
    clf.fit(context, y_context)  # no gradients: fits encoders and builds the ensemble views
    generator = clf.ensemble_generator_
    views = generator.transform(clf.X_encoder_.transform(query))
    xs, ys, cat_masks, ds, _ = generator.prepare_ensemble_tensors(views)
    offsets = [offset for per_norm in generator.class_shift_offsets_.values() for offset in per_norm]

    view = int(np.random.default_rng(seed).integers(len(offsets)))
    n_rows = xs.shape[1]
    y_padded = np.pad(ys[view : view + 1], ((0, 0), (0, n_rows - ys.shape[1])), constant_values=-100)
    # Views relabel class k as (k + offset) % n_classes, and the model's outputs follow that labelling.
    y_query_encoded = clf.y_encoder_.transform(y_query.reshape(-1, 1)).ravel()
    target = (y_query_encoded + offsets[view]) % clf.n_classes_

    return Episode(
        x=torch.from_numpy(xs[view : view + 1]).to(device, torch.float32),
        y=torch.from_numpy(y_padded).to(device),
        train_size=torch.tensor([ys.shape[1]], device=device),
        cat_mask=torch.from_numpy(cat_masks[view : view + 1]).to(device),
        d=torch.from_numpy(ds[view : view + 1]).to(device),
        target=torch.from_numpy(np.asarray(target, dtype=np.int64)).to(device),
        n_classes=int(clf.n_classes_),
        view=view,
        class_shift=int(offsets[view]),
    )


def episode_loss(model: nn.Module, episode: Episode) -> torch.Tensor:
    """Cross-entropy of the query rows' logits (the model also emits outputs for context rows; they are ignored)."""
    out = model(episode.x, episode.y, episode.train_size, cat_mask=episode.cat_mask, d=episode.d)
    n_context = int(episode.train_size[0])
    logits = out[0, n_context:, : episode.n_classes].float()
    return F.cross_entropy(logits, episode.target)


def lr_multiplier(step: int, warmup_steps: int, max_steps: int) -> float:
    """Linear warmup to the base learning rate, then cosine decay to zero at `max_steps`."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


class Float32Master:
    """Float32 master copies of bf16 parameters: the optimiser updates these, the model runs on rounded copies.

    A 1e-5 step on a weight of 0.02 is a 0.05% change, well below bf16's ~0.8% rounding step, so
    updating bf16 weights directly would mostly round back to the same value.
    """

    def __init__(self, params: list[nn.Parameter]) -> None:
        self.params = params
        self.masters = [p.detach().float().clone() for p in params]

    def gradients_to_masters(self) -> None:
        for p, m in zip(self.params, self.masters, strict=True):
            m.grad = None if p.grad is None else p.grad.float()
            p.grad = None

    @torch.no_grad()
    def masters_to_params(self) -> None:
        for p, m in zip(self.params, self.masters, strict=True):
            p.copy_(m)


@dataclass
class FinetuneResult:
    history: list[dict[str, float]]
    best_step: int
    stopped_early: bool
    step_seconds: list[float]  # wall clock per gradient step, synchronized: episode build + forward + backward + update
    validation_seconds: list[float]  # wall clock per validation pass (full training set as context)
    peak_memory_gb: float | None  # accelerator memory high-water mark (sampled on MPS, exact on CUDA)

    def efficiency(self) -> dict[str, float | int | None]:
        steps = np.asarray(self.step_seconds)
        return {
            "steps": len(steps),
            "step_s_median": float(np.median(steps)) if len(steps) else None,
            "step_s_mean": float(steps.mean()) if len(steps) else None,
            "validation_s_mean": float(np.mean(self.validation_seconds)) if self.validation_seconds else None,
            "peak_memory_gb": self.peak_memory_gb,
        }


def finetune(
    model: nn.Module,
    split: Split,
    config: FinetuneConfig,
    device: str,
    log: Callable[[str], None] = print,
) -> FinetuneResult:
    """Fine-tune `model` in place on `split.train`; the model ends with the best validation step's weights."""
    params = prepare_for_finetuning(model, config.n_trainable_blocks, config.train_encoders)
    master = Float32Master(params)
    encoders = " + row/column encoders" if config.train_encoders else ""
    log(
        f"trainable: {sum(p.numel() for p in params) / 1e6:.0f}M parameters "
        f"(last {config.n_trainable_blocks} ICL blocks + head{encoders})"
    )
    optimizer = torch.optim.AdamW(master.masters, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, config.warmup_steps, config.max_steps)
    )
    rng = np.random.default_rng(config.seed)
    X_train, y_train = features(split.train), labels(split.train)
    episode_rows = config.context_size + config.query_size
    if episode_rows > len(X_train):
        raise ValueError(f"context_size + query_size ({episode_rows}) exceeds the {len(X_train)} training rows")

    step_seconds: list[float] = []
    validation_seconds: list[float] = []
    peak_memory: list[float] = []

    def sample_memory() -> None:
        gb = memory_gb(device)
        if gb is not None:
            peak_memory.append(gb)

    def validate(step: int, train_loss: float) -> float:
        synchronize(device)
        t0 = time.perf_counter()
        p = predict_positive(model, split.train, split.val, config.n_estimators_val, config.seed)
        synchronize(device)
        validation_seconds.append(time.perf_counter() - t0)
        sample_memory()
        metrics = score(labels(split.val), p)
        history.append(
            {"step": step, "train_loss": train_loss, **{f"val_{k}": v for k, v in metrics.as_dict().items()}}
        )
        log(
            f"step {step:>3}  train loss {train_loss:.4f}  val log loss {metrics.log_loss:.4f}  "
            f"val AUC {metrics.roc_auc:.4f}  val acc {metrics.accuracy:.3f}  ({time.perf_counter() - t0:.0f}s)"
        )
        return metrics.log_loss

    history: list[dict[str, float]] = []
    best_loss, best_step, best_state = validate(0, float("nan")), 0, trainable_state(model)
    bad_evals, stopped_early, recent_losses = 0, False, []

    reset_peak_memory(device)
    for step in range(1, config.max_steps + 1):
        synchronize(device)
        t0 = time.perf_counter()
        rows = rng.permutation(len(X_train))[:episode_rows]
        ctx, qry = rows[: config.context_size], rows[config.context_size :]
        episode = make_episode(
            model, X_train.iloc[ctx], y_train[ctx], X_train.iloc[qry], y_train[qry], int(rng.integers(2**31)), device
        )
        loss = episode_loss(model, episode)
        loss.backward()
        master.gradients_to_masters()
        torch.nn.utils.clip_grad_norm_(master.masters, config.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        master.masters_to_params()
        recent_losses.append(loss.item())
        synchronize(device)
        step_seconds.append(time.perf_counter() - t0)
        sample_memory()
        log(f"step {step:>3}  loss {loss.item():.4f}  lr {scheduler.get_last_lr()[0]:.2e}  ({step_seconds[-1]:.1f}s)")

        if step % config.eval_every == 0 or step == config.max_steps:
            val_loss = validate(step, float(np.mean(recent_losses)))
            recent_losses = []
            if val_loss < best_loss:
                best_loss, best_step, best_state, bad_evals = val_loss, step, trainable_state(model), 0
            else:
                bad_evals += 1
                if bad_evals >= config.patience:
                    log(f"no validation improvement in {bad_evals} evaluations: stopping")
                    stopped_early = True
                    break

    load_trainable_state(model, best_state)
    freeze(model)
    log(f"restored the weights from step {best_step} (val log loss {best_loss:.4f})")
    return FinetuneResult(
        history=history,
        best_step=best_step,
        stopped_early=stopped_early,
        step_seconds=step_seconds,
        validation_seconds=validation_seconds,
        peak_memory_gb=max(peak_memory) if peak_memory else None,
    )
