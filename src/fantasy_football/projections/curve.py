"""Preseason positional rank -> expected points, and the spread around it.

Non-parametric on purpose. Fantasy points by rank is roughly a power law, but
only roughly: it flattens through the middle of each position and breaks at the
replacement cliff, and the shape differs by position. A rolling-window fit
follows whatever shape the data has, and the same window yields the dispersion,
which matters as much as the mean here — a projection without an honest spread
cannot support any decision about risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

# Rolling half-width in ranks. Wide enough to smooth season-to-season noise,
# narrow enough to preserve the cliff at the end of each position's starters.
DEFAULT_WINDOW = 6
MIN_SAMPLES = 8

# Outcomes at a given rank are right-skewed with a lump of zeros from players who
# got hurt or lost a job. A mean and standard deviation describe that badly, so
# the curve carries empirical quantiles and intervals are read straight off them.
QUANTILE_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


@dataclass(frozen=True)
class RankCurve:
    """Expected points and residual spread as a function of preseason rank."""

    position: str
    ranks: np.ndarray
    expected: np.ndarray
    spread: np.ndarray
    n_observations: int = 0
    seasons: tuple[int, ...] = field(default_factory=tuple)
    # Shape (len(ranks), len(QUANTILE_LEVELS)); empty when not fitted.
    quantiles: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    def quantile_at(self, rank: float, level: float) -> float:
        """Empirical quantile of outcomes for a player at this preseason rank."""
        if self.quantiles.size == 0:
            raise ValueError("curve carries no quantiles")
        try:
            column = QUANTILE_LEVELS.index(level)
        except ValueError as exc:
            raise ValueError(f"level must be one of {QUANTILE_LEVELS}") from exc
        return float(np.interp(rank, self.ranks, self.quantiles[:, column]))

    def interval_at(self, rank: float, coverage: float = 0.80) -> tuple[float, float]:
        """Central interval with the requested nominal coverage."""
        tail = round((1.0 - coverage) / 2.0, 3)
        upper = round(1.0 - tail, 3)
        return self.quantile_at(rank, tail), self.quantile_at(rank, upper)

    def probability_below(self, rank: float, points: float) -> float:
        """Roughly where an outcome falls in the predicted distribution.

        Interpolates across the stored quantile levels, which is what the
        calibration check needs to build its PIT histogram.
        """
        curve_points = [self.quantile_at(rank, level) for level in QUANTILE_LEVELS]
        return float(np.interp(points, curve_points, QUANTILE_LEVELS))

    def points_at(self, rank: float) -> float:
        """Expected points at a possibly fractional rank.

        Interpolated rather than snapped, because the blended rank of two
        sources is rarely a whole number and snapping would collapse genuinely
        different players onto one value.
        """
        return float(np.interp(rank, self.ranks, self.expected))

    def spread_at(self, rank: float) -> float:
        return float(np.interp(rank, self.ranks, self.spread))

    @property
    def max_rank(self) -> int:
        return int(self.ranks[-1])

    def replacement_points(self, starters_required: int) -> float:
        """Points from the best player nobody is obliged to start.

        Value over replacement is only meaningful against the league's actual
        demand for a position, which is why this takes the requirement rather
        than assuming a conventional cutoff.
        """
        return self.points_at(min(starters_required + 1, self.max_rank))


def fit_rank_curve(
    training: pl.DataFrame,
    position: str,
    window: int = DEFAULT_WINDOW,
    value_col: str = "actual_points",
    monotone: bool = True,
) -> RankCurve | None:
    """Fit one position's curve from the pooled player-seasons.

    `value_col` selects what is being predicted — season points, points per game,
    weekly spread. `monotone` should stay on for anything that genuinely has to
    fall as rank worsens, and off for quantities that do not (games played is
    only weakly related to preseason rank, so forcing it downward invents a
    trend the data does not support).
    """
    subset = (
        training.filter(pl.col("pos") == position)
        .select(["pos_rank", value_col, "season"])
        .drop_nulls(value_col)
    )
    if subset.height < MIN_SAMPLES:
        return None

    ranks_obs = subset["pos_rank"].to_numpy()
    points_obs = subset[value_col].to_numpy().astype(float)

    max_rank = int(ranks_obs.max())
    grid = np.arange(1, max_rank + 1)
    expected = np.zeros(len(grid))
    spread = np.zeros(len(grid))
    quantiles = np.zeros((len(grid), len(QUANTILE_LEVELS)))

    for i, rank in enumerate(grid):
        window_mask = np.abs(ranks_obs - rank) <= window
        # Near the ends the window is one-sided, so widen until it has enough
        # to say anything, rather than reporting a mean over two players.
        extra = window
        while window_mask.sum() < MIN_SAMPLES and extra < max_rank:
            extra += window
            window_mask = np.abs(ranks_obs - rank) <= extra

        sample = points_obs[window_mask]
        expected[i] = sample.mean()
        spread[i] = sample.std(ddof=1) if len(sample) > 1 else 0.0
        quantiles[i] = np.quantile(sample, QUANTILE_LEVELS)

    # Expected points cannot rise with a worse preseason rank. Where the raw
    # window says otherwise it is noise, so enforce the monotonicity — on each
    # quantile level independently, so the whole distribution shifts down
    # together and intervals cannot cross.
    if monotone:
        expected = np.minimum.accumulate(expected)
        quantiles = np.minimum.accumulate(quantiles, axis=0)

    return RankCurve(
        position=position,
        ranks=grid,
        expected=expected,
        spread=spread,
        n_observations=subset.height,
        seasons=tuple(sorted(set(subset["season"].to_list()))),
        quantiles=quantiles,
    )


# Which positions may fill each shared slot. Listed explicitly because a slash in
# the slot name is not a reliable signal — "D/ST" has one and is not a flex.
FLEX_ELIGIBILITY = {
    "RB/WR": ("RB", "WR"),
    "RB/WR/TE": ("RB", "WR", "TE"),
    "WR/TE": ("WR", "TE"),
    "OP": ("QB", "RB", "WR", "TE"),
}

DEDICATED_POSITIONS = ("QB", "RB", "WR", "TE", "K", "D/ST")


def replacement_ranks(settings, curves: dict[str, RankCurve]) -> dict[str, int]:
    """How deep into each position the league's weekly demand actually reaches.

    Dedicated slots are simple multiplication. Flex slots are assigned one at a
    time to whichever eligible position has the most valuable player still
    available at the margin, which is what managers collectively do and what
    determines where each position's replacement level really falls.
    """
    counts = {
        position: settings.starting_slots.get(position, 0) * settings.team_count
        for position in DEDICATED_POSITIONS
    }

    flex_slots = sum(
        n * settings.team_count
        for slot, n in settings.starting_slots.items()
        if slot in FLEX_ELIGIBILITY
    )
    eligible = sorted(
        {
            position
            for slot in settings.starting_slots
            if slot in FLEX_ELIGIBILITY
            for position in FLEX_ELIGIBILITY[slot]
            if position in curves
        }
    )

    for _ in range(flex_slots):
        if not eligible:
            break
        best = max(eligible, key=lambda p: curves[p].points_at(counts[p] + 1))
        counts[best] += 1

    return counts


def replacement_points(settings, curves: dict[str, RankCurve]) -> dict[str, float]:
    ranks = replacement_ranks(settings, curves)
    return {
        position: curve.points_at(min(ranks.get(position, 0) + 1, curve.max_rank))
        for position, curve in curves.items()
    }


def fit_all(
    training: pl.DataFrame,
    window: int = DEFAULT_WINDOW,
    value_col: str = "actual_points",
    monotone: bool = True,
) -> dict[str, RankCurve]:
    curves = {}
    for position in training["pos"].unique().to_list():
        curve = fit_rank_curve(
            training, position, window=window, value_col=value_col, monotone=monotone
        )
        if curve is not None:
            curves[position] = curve
    return curves
