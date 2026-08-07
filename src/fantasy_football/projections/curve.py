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


@dataclass(frozen=True)
class RankCurve:
    """Expected points and residual spread as a function of preseason rank."""

    position: str
    ranks: np.ndarray
    expected: np.ndarray
    spread: np.ndarray
    n_observations: int = 0
    seasons: tuple[int, ...] = field(default_factory=tuple)

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
    training: pl.DataFrame, position: str, window: int = DEFAULT_WINDOW
) -> RankCurve | None:
    """Fit one position's curve from the pooled player-seasons."""
    subset = training.filter(pl.col("pos") == position).select(
        ["pos_rank", "actual_points", "season"]
    )
    if subset.height < MIN_SAMPLES:
        return None

    ranks_obs = subset["pos_rank"].to_numpy()
    points_obs = subset["actual_points"].to_numpy()

    max_rank = int(ranks_obs.max())
    grid = np.arange(1, max_rank + 1)
    expected = np.zeros(len(grid))
    spread = np.zeros(len(grid))

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

    # Expected points cannot rise with a worse preseason rank. Where the raw
    # window says otherwise it is noise, so enforce the monotonicity.
    expected = np.minimum.accumulate(expected)

    return RankCurve(
        position=position,
        ranks=grid,
        expected=expected,
        spread=spread,
        n_observations=subset.height,
        seasons=tuple(sorted(set(subset["season"].to_list()))),
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


def fit_all(training: pl.DataFrame, window: int = DEFAULT_WINDOW) -> dict[str, RankCurve]:
    curves = {}
    for position in training["pos"].unique().to_list():
        curve = fit_rank_curve(training, position, window=window)
        if curve is not None:
            curves[position] = curve
    return curves
