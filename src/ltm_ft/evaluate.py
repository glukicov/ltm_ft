"""Score TabFM's predictions, the generator's ceiling, and the before/after difference.

Accuracy on a few thousand rows moves in steps of a few hundredths of a point, and fine-tuning
effects are small, so the headline metric is **log loss** (it rewards well-calibrated
probabilities on every row, not just the side of 0.5 they land on), with ROC AUC and accuracy
alongside. The before/after difference is measured on the same rows, so its uncertainty comes
from a paired bootstrap over test rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from tabfm import TabFMClassifier
from torch import nn

from ltm_ft.data import POSITIVE, TRUE_PROB, features, labels

EPS = 1e-6


@dataclass(frozen=True)
class Metrics:
    log_loss: float
    roc_auc: float
    accuracy: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def score(y_true: np.ndarray, p_positive: np.ndarray) -> Metrics:
    y = (y_true == POSITIVE).astype(int)
    p = np.clip(p_positive, EPS, 1 - EPS)
    return Metrics(
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        roc_auc=float(roc_auc_score(y, p)),
        accuracy=float(np.mean((p >= 0.5) == y)),
    )


def ceiling(df: pd.DataFrame) -> Metrics:
    """The same metrics for predictions made with the generator's own probabilities."""
    return score(labels(df), df[TRUE_PROB].to_numpy())


def predict_positive(
    model: nn.Module, context: pd.DataFrame, query: pd.DataFrame, n_estimators: int, seed: int
) -> np.ndarray:
    """P(injured) for every `query` row, using the stock TabFM classifier with `context` as its labelled rows."""
    clf = TabFMClassifier(model=model, n_estimators=n_estimators, random_state=seed)
    clf.fit(features(context), labels(context))
    proba = clf.predict_proba(features(query))
    positive = int(np.flatnonzero(clf.classes_ == POSITIVE)[0])
    return np.asarray(proba[:, positive], dtype=float)


def paired_bootstrap(
    y_true: np.ndarray, p_before: np.ndarray, p_after: np.ndarray, n_resamples: int = 2000, seed: int = 0
) -> dict[str, dict[str, float]]:
    """after - before for each metric, with a 95% interval from resampling test rows (same rows for both)."""
    y = (y_true == POSITIVE).astype(int)
    pb, pa = np.clip(p_before, EPS, 1 - EPS), np.clip(p_after, EPS, 1 - EPS)
    nll_b = -(y * np.log(pb) + (1 - y) * np.log(1 - pb))
    nll_a = -(y * np.log(pa) + (1 - y) * np.log(1 - pa))
    acc_b, acc_a = (pb >= 0.5) == y, (pa >= 0.5) == y

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_resamples, len(y)))
    deltas = {
        "log_loss": (nll_a[idx] - nll_b[idx]).mean(axis=1),
        "accuracy": (acc_a[idx].astype(float) - acc_b[idx].astype(float)).mean(axis=1),
    }
    auc_deltas = []
    for row in idx[: min(n_resamples, 500)]:  # AUC is slower to recompute; 500 resamples is plenty
        if y[row].min() != y[row].max():
            auc_deltas.append(roc_auc_score(y[row], pa[row]) - roc_auc_score(y[row], pb[row]))
    deltas["roc_auc"] = np.asarray(auc_deltas)

    point = {
        "log_loss": float(nll_a.mean() - nll_b.mean()),
        "accuracy": float(acc_a.mean() - acc_b.mean()),
        "roc_auc": float(roc_auc_score(y, pa) - roc_auc_score(y, pb)),
    }
    return {
        name: {"delta": point[name], "ci_low": float(np.quantile(d, 0.025)), "ci_high": float(np.quantile(d, 0.975))}
        for name, d in deltas.items()
    }
