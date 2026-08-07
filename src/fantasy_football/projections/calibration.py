"""Leave-one-season-out backtest of the projection intervals.

A mean projection that ranks players well but claims false precision is
dangerous here, because every risk decision downstream — when to seek variance,
which lineup maximizes P(beating this opponent) — reads the spread, not the
mean. So the spread gets tested as hard as the mean.

The test holds out one season entirely, refits on the rest, and asks two
questions of the held-out year: does the ordering hold up, and does an interval
claiming to cover 80% of outcomes actually cover 80%?

What this does *not* validate: the ESPN blend. ESPN's historical preseason
projections are not retrievable, so the backtest measures the consensus-only
projection. The blend can only inherit that calibration, not prove its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .curve import fit_all

NOMINAL_COVERAGES = (0.50, 0.80, 0.90)


@dataclass
class SeasonResult:
    season: int
    n: int
    mae: float
    baseline_mae: float
    bias: float
    spearman: float
    coverage: dict[float, float] = field(default_factory=dict)

    @property
    def skill(self) -> float:
        """Fraction of the naive baseline's error removed. Zero means useless."""
        return 0.0 if self.baseline_mae == 0 else 1.0 - self.mae / self.baseline_mae


@dataclass
class CalibrationReport:
    seasons: list[SeasonResult]
    pit_values: np.ndarray

    @property
    def overall_coverage(self) -> dict[float, float]:
        return {
            level: float(np.mean([s.coverage[level] for s in self.seasons]))
            for level in NOMINAL_COVERAGES
        }

    @property
    def mean_spearman(self) -> float:
        return float(np.mean([s.spearman for s in self.seasons]))

    @property
    def mean_skill(self) -> float:
        return float(np.mean([s.skill for s in self.seasons]))

    def pit_histogram(self, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Uniform-looking is calibrated; a hump means over- or under-confident."""
        counts, edges = np.histogram(self.pit_values, bins=bins, range=(0.0, 1.0))
        return counts / max(counts.sum(), 1), edges


def _spearman(predicted: np.ndarray, actual: np.ndarray) -> float:
    if len(predicted) < 2:
        return float("nan")
    pred_ranks = np.argsort(np.argsort(predicted))
    act_ranks = np.argsort(np.argsort(actual))
    return float(np.corrcoef(pred_ranks, act_ranks)[0, 1])


def backtest(
    training: pl.DataFrame, positions: tuple[str, ...] = ("QB", "RB", "WR", "TE")
) -> CalibrationReport:
    """Refit without each season in turn and score the projections on it."""
    seasons = sorted(training["season"].unique().to_list())
    results: list[SeasonResult] = []
    pit_all: list[float] = []

    for holdout in seasons:
        fit_data = training.filter(pl.col("season") != holdout)
        test_data = training.filter((pl.col("season") == holdout) & pl.col("pos").is_in(positions))
        if fit_data.is_empty() or test_data.is_empty():
            continue

        curves = fit_all(fit_data)

        predicted, actual, pits = [], [], []
        covered = {level: [] for level in NOMINAL_COVERAGES}

        for row in test_data.iter_rows(named=True):
            curve = curves.get(row["pos"])
            if curve is None or curve.quantiles.size == 0:
                continue

            rank = row["pos_rank"]
            truth = row["actual_points"]

            predicted.append(curve.points_at(rank))
            actual.append(truth)
            pits.append(curve.probability_below(rank, truth))

            for level in NOMINAL_COVERAGES:
                low, high = curve.interval_at(rank, level)
                covered[level].append(low <= truth <= high)

        if not predicted:
            continue

        predicted_arr = np.array(predicted)
        actual_arr = np.array(actual)
        # The honest naive alternative: know nothing about the player beyond his
        # position, and guess that position's average.
        baseline = np.full_like(actual_arr, actual_arr.mean())

        results.append(
            SeasonResult(
                season=holdout,
                n=len(predicted),
                mae=float(np.mean(np.abs(predicted_arr - actual_arr))),
                baseline_mae=float(np.mean(np.abs(baseline - actual_arr))),
                bias=float(np.mean(predicted_arr - actual_arr)),
                spearman=_spearman(predicted_arr, actual_arr),
                coverage={level: float(np.mean(covered[level])) for level in NOMINAL_COVERAGES},
            )
        )
        pit_all.extend(pits)

    return CalibrationReport(seasons=results, pit_values=np.array(pit_all))
