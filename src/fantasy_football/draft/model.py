"""Where a player actually goes, given where the consensus ranks him.

Two facts make this necessary. Drafters do not pick in consensus order — they
reach and they let players slide. And the size of that deviation grows sharply
as the draft wears on: the first three picks are nearly deterministic, while
round 12 is close to random. A single "players go about where they're ranked"
assumption would be confidently wrong at exactly the picks where the decision is
hardest.

So the model is fit as a function of rank: an average drift, and a spread that
is allowed to widen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt

import numpy as np
import polars as pl

WINDOW = 12
MIN_SAMPLES = 10
MIN_SPREAD = 2.0


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


@dataclass(frozen=True)
class PickModel:
    """Distribution of a player's draft slot, conditional on his consensus rank."""

    ranks: np.ndarray
    drift: np.ndarray
    spread: np.ndarray
    n_observations: int = 0
    seasons: tuple[int, ...] = field(default_factory=tuple)

    def expected_pick(self, consensus_rank: float) -> float:
        drift = float(np.interp(consensus_rank, self.ranks, self.drift))
        return consensus_rank + drift

    def pick_spread(self, consensus_rank: float) -> float:
        return max(float(np.interp(consensus_rank, self.ranks, self.spread)), MIN_SPREAD)

    def survival(self, consensus_rank: float, pick_number: int) -> float:
        """Probability the player is still on the board when `pick_number` arrives."""
        mu = self.expected_pick(consensus_rank)
        sigma = self.pick_spread(consensus_rank)
        # P(draft slot >= pick_number), with a half-pick continuity correction.
        return 1.0 - _normal_cdf((pick_number - 0.5 - mu) / sigma)

    def sample_picks(self, consensus_ranks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One plausible draft ordering: a slot drawn for every player at once.

        Sampling every player's slot in a single draw and then sorting gives a
        coherent draft — players compete for the same picks — which repeated
        independent draws would not.
        """
        mus = np.interp(consensus_ranks, self.ranks, self.drift) + consensus_ranks
        sigmas = np.maximum(np.interp(consensus_ranks, self.ranks, self.spread), MIN_SPREAD)
        return rng.normal(mus, sigmas)

    def sample_board_order(
        self, position_on_board: np.ndarray, consensus_ranks: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample a draft ordering over the players still available.

        Two different ranks are in play and conflating them is a real error.
        Who gets taken next is a contest among the players *still on the board*,
        so the ordering is built from each player's position in that shrinking
        list. But how unpredictably he is treated depends on his standing in the
        league as a whole — a consensus top-5 player is taken decisively wherever
        he happens to sit on a depleted board, while a rank-90 player is close to
        a coin flip. So the centre comes from board position and the spread comes
        from absolute consensus rank.
        """
        sigmas = np.maximum(np.interp(consensus_ranks, self.ranks, self.spread), MIN_SPREAD)
        return rng.normal(position_on_board.astype(float), sigmas)


def fit_pick_model(training: pl.DataFrame) -> PickModel | None:
    """Fit drift and spread against consensus rank, pooling the seasons."""
    subset = training.select(["consensus_rank", "overall_pick", "season"]).drop_nulls()
    if subset.height < MIN_SAMPLES:
        return None

    ranks_obs = subset["consensus_rank"].to_numpy().astype(float)
    picks_obs = subset["overall_pick"].to_numpy().astype(float)
    deviation = picks_obs - ranks_obs

    max_rank = int(ranks_obs.max())
    grid = np.arange(1, max_rank + 1)
    drift = np.zeros(len(grid))
    spread = np.zeros(len(grid))

    for i, rank in enumerate(grid):
        mask = np.abs(ranks_obs - rank) <= WINDOW
        extra = WINDOW
        while mask.sum() < MIN_SAMPLES and extra < max_rank:
            extra += WINDOW
            mask = np.abs(ranks_obs - rank) <= extra

        sample = deviation[mask]
        drift[i] = sample.mean()
        spread[i] = sample.std(ddof=1) if len(sample) > 1 else MIN_SPREAD

    return PickModel(
        ranks=grid,
        drift=drift,
        spread=spread,
        n_observations=subset.height,
        seasons=tuple(sorted(set(subset["season"].to_list()))),
    )
