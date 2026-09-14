"""Synthetic tables with a known answer: every row records the probability its label was drawn with.

Because the generator's probabilities are known, every table can report the best score any model
could reach (the "ceiling"), and we can draw as many rows as an experiment needs.

- `runners`: will a runner get injured in the next 8 weeks? The generator from the running-injury
  example in https://github.com/glukicov/ltm (all 21 feature columns). Deliberately messy: integers,
  floats, low- and high-cardinality categoricals, an ordinal, yes/no flags, missing values,
  thresholds and interactions - a realistic export rather than a tidy benchmark.

- `checkerboard`: ten uniform features, and the label depends only on which square of a 6x6
  checkerboard the first two fall in. A classic stress test: neither feature is informative on its
  own, and a row's nearest neighbours in all ten dimensions are mostly on other squares. Zero-shot
  TabFM leaves a large gap to the ceiling here, which is what fine-tuning needs to show anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

LABEL = "label"  # runners: injured in the next 8 weeks
TRUE_PROB = "true_prob"  # the generator's ground-truth probability - never a feature
POSITIVE = "yes"


class Task(StrEnum):
    runners = "runners"
    checkerboard = "checkerboard"


def injury_logit(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Log-odds of injury. Loosely inspired by running-injury literature, but invented."""
    ramp = df["weekly_km_change_pct"].to_numpy()
    logit = (
        -1.9
        # Training-load spikes are the classic cause: risk climbs steeply past ~+10%.
        + 0.05 * np.clip(ramp - 10, 0, None)
        - 0.6 * (df["sleep_hours_avg"].to_numpy() - 7.0)
        + 2.0 * (df["previous_injury"] == "yes").to_numpy()
        - 0.8 * df["strength_sessions_per_week"].to_numpy()
        + 2.0 * (df["shoe_age_km"].to_numpy() > 800)
        + 0.06 * np.clip(df["age"].to_numpy() - 40, 0, None)
        + 0.3 * np.clip(df["bmi"].to_numpy() - 25, 0, None)
        # Interactions a linear rule of thumb would miss.
        + 2.2 * ((df["shoe_type"] == "minimalist") & (df["weekly_km"] > 45)).to_numpy()
        + 2.2 * ((df["experience_level"] == "beginner") & df["target_race"].isin(["marathon", "ultra"])).to_numpy()
        - 0.6 * (df["primary_surface"] == "trail").to_numpy()
        + 0.8 * (df["primary_surface"] == "track").to_numpy()
        + 1.2 * (df["runs_per_week"] >= 6).to_numpy()
    )
    return np.asarray(logit + rng.normal(0, 0.3, len(df)), dtype=float)


def generate_runners(rows: int, seed: int, noise_columns: int = 0) -> pd.DataFrame:
    """`rows` independent synthetic runners, with the label and its true probability.

    `noise_columns` appends that many junk telemetry columns (`telemetry_01`, ...) that play no
    part in the risk model - the export of an app that logs everything. The ceiling is unchanged,
    but every row now looks less like its true neighbours.
    """
    rng = np.random.default_rng(seed)
    n = rows

    age = rng.integers(18, 71, n)
    years_running = np.minimum(rng.gamma(2.0, 3.0, n), age - 14).round(1)
    runs_per_week = np.clip(rng.poisson(3.5, n), 1, 7)
    weekly_km = np.clip(runs_per_week * rng.normal(9, 3, n) + years_running * 0.8, 5, 160).round(1)
    pace = np.clip(rng.normal(6.4, 0.9, n) - 0.02 * weekly_km + 0.015 * (age - 40), 3.2, 9.0).round(2)

    experience = np.select(
        [years_running < 2, years_running < 5, (years_running < 9) | (pace > 5.5)],
        ["beginner", "intermediate", "advanced"],
        "elite",
    )
    has_watch = rng.random(n) < 0.75
    vo2max = np.where(has_watch, np.clip(80 - 6.5 * pace - 0.2 * (age - 30) + rng.normal(0, 3, n), 25, 80), np.nan)
    resting_hr = np.where(rng.random(n) < 0.85, np.clip(75 - 0.25 * weekly_km + rng.normal(0, 6, n), 38, 90), np.nan)

    df = pd.DataFrame(
        {
            "age": age,
            "sex": rng.choice(["female", "male"], n),
            "bmi": np.clip(rng.normal(23.5, 3.0, n), 17, 38).round(1),
            "country": rng.choice(
                ["UK", "US", "Germany", "Spain", "Kenya", "Japan", "Australia", "Brazil"],
                n,
                p=[0.22, 0.25, 0.12, 0.1, 0.05, 0.1, 0.08, 0.08],
            ),
            "years_running": years_running,
            "experience_level": experience,
            "runs_per_week": runs_per_week,
            "weekly_km": weekly_km,
            "longest_run_km": np.clip(weekly_km * rng.uniform(0.25, 0.5, n), 3, 60).round(1),
            "avg_pace_min_per_km": pace,
            "resting_hr_bpm": resting_hr.round(0),
            "vo2max_estimate": vo2max.round(1),
            "primary_surface": rng.choice(["road", "trail", "track", "treadmill"], n, p=[0.55, 0.2, 0.1, 0.15]),
            "shoe_type": rng.choice(
                ["neutral", "stability", "minimalist", "max_cushion", "carbon_plated"],
                n,
                p=[0.4, 0.2, 0.1, 0.2, 0.1],
            ),
            "shoe_age_km": np.clip(rng.gamma(2.0, 250, n), 0, 1500).round(0),
            "strength_sessions_per_week": rng.choice([0, 1, 2, 3], n, p=[0.45, 0.3, 0.18, 0.07]),
            "previous_injury": np.where(rng.random(n) < 0.35, "yes", "no"),
            "training_plan": rng.choice(["none", "app", "online_coach", "club"], n, p=[0.4, 0.3, 0.15, 0.15]),
            "target_race": rng.choice(["none", "5k", "10k", "half", "marathon", "ultra"], n),
            "weekly_km_change_pct": np.clip(rng.normal(8, 22, n), -40, 90).round(0),
            "sleep_hours_avg": np.clip(rng.normal(7.0, 0.9, n), 4.5, 9.5).round(1),
        }
    )
    prob = 1 / (1 + np.exp(-injury_logit(df, rng)))
    df[TRUE_PROB] = prob.round(4)
    df[LABEL] = np.where(rng.random(n) < prob, POSITIVE, "no")
    if noise_columns:
        # Drawn from their own stream, so the runners and labels are identical with or without them.
        noise_rng = np.random.default_rng([seed, noise_columns])
        scales = noise_rng.uniform(0.5, 50, noise_columns).round(1)
        noise = noise_rng.normal(0, 1, (n, noise_columns)) * scales
        junk = pd.DataFrame(noise.round(2), columns=[f"telemetry_{i + 1:02d}" for i in range(noise_columns)])
        df = pd.concat([df.drop(columns=[TRUE_PROB, LABEL]), junk, df[[TRUE_PROB, LABEL]]], axis=1)
    return df


def generate_checkerboard(rows: int, seed: int, noise_columns: int = 0) -> pd.DataFrame:
    """Ten U(0, 1) features; P(yes) = sigmoid(+3) on white squares of a 6x6 board over (x0, x1), else sigmoid(-3)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.uniform(0, 1, (rows, 10 + noise_columns)), columns=[f"x{i}" for i in range(10 + noise_columns)]
    )
    square = np.sign(np.sin(2 * np.pi * 3 * df["x0"]) * np.sin(2 * np.pi * 3 * df["x1"])).to_numpy()
    prob = 1 / (1 + np.exp(-3 * square))
    df[TRUE_PROB] = prob
    df[LABEL] = np.where(rng.random(rows) < prob, POSITIVE, "no")
    return df


def generate(task: Task, rows: int, seed: int, noise_columns: int = 0) -> pd.DataFrame:
    if task == Task.checkerboard:
        return generate_checkerboard(rows, seed, noise_columns)
    return generate_runners(rows, seed, noise_columns)


@dataclass(frozen=True)
class Split:
    """Three disjoint sets of runners.

    train: the labelled rows. They are TabFM's in-context rows at prediction time, and the
        rows fine-tuning draws its (context, query) episodes from.
    val:   only used to pick the fine-tuning step to keep (early stopping) and to tune.
    test:  touched twice - once before fine-tuning, once after.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def make_split(
    n_train: int, n_val: int, n_test: int, seed: int, noise_columns: int = 0, task: Task = Task.runners
) -> Split:
    """Each set is generated from its own seed, so changing one set's size never changes the others.

    That lets tuning runs (train + val only) share the exact validation rows of the final run
    without ever seeing its test rows.
    """
    return Split(
        train=generate(task, n_train, seed, noise_columns),
        val=generate(task, n_val, seed + 1, noise_columns).rename(index=lambda i: i + n_train),
        test=generate(task, n_test, seed + 2, noise_columns).rename(index=lambda i: i + n_train + n_val),
    )


def features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[LABEL, TRUE_PROB])


def labels(df: pd.DataFrame) -> np.ndarray:
    return df[LABEL].to_numpy()
