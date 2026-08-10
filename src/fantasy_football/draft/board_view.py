"""What to show on draft night.

The decision and the context come from different places, deliberately. The pick
itself comes from the rule that actually won the rehearsal — highest-ranked
player at a position you still need — because that is the only strategy tested
that beats the humans and nothing beat it.

The survival numbers alongside it come from the simulation. That machinery lost
badly when it was allowed to *choose*, but it is still the best estimate of who
will be gone by your next turn, and knowing that is genuinely useful even when
it is not what decides the pick. Information and authority are separated on
purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..projections.ensemble import Projection
from .recommend import roster_limits
from .state import DraftState

SURVIVAL_TRIALS = 300


@dataclass(frozen=True)
class BoardRow:
    player: str
    position: str
    team: str
    espn_id: int | None
    overall_rank: int
    mean: float
    sd: float
    # Raw projected points are not comparable across positions — a quarterback's
    # 297 is worth less than a running back's 225 because replacement quarterbacks
    # score 250. Value over replacement is the number to read side by side.
    vor: float
    survival: float
    fills_a_need: bool
    recommended: bool = False

    @property
    def likely_gone(self) -> bool:
        return self.survival < 0.5

    def rationale(self) -> str:
        scarcity = (
            f"{self.survival:.0%} chance he lasts to your next pick"
            if self.survival < 0.9
            else "likely to last, but he is the best fit available"
        )
        return (
            f"consensus #{self.overall_rank} ({self.mean:.0f} pts, +/-{self.sd:.0f}); {scarcity}."
        )


def _needed_positions(state: DraftState, projections, settings, by_id) -> set[str]:
    limits = roster_limits(settings)
    counts: dict[str, int] = {}
    for espn_id in state.my_roster:
        projection = by_id.get(espn_id)
        if projection is not None:
            counts[projection.position] = counts.get(projection.position, 0) + 1
    return {p for p, cap in limits.items() if counts.get(p, 0) < cap}


def survival_odds(
    state: DraftState,
    available: list[Projection],
    pick_model,
    trials: int = SURVIVAL_TRIALS,
    seed: int = 0,
) -> dict[int, float]:
    """Chance each available player is still there when you pick again."""
    gap = state.picks_until_my_next()
    if gap is None or gap <= 0 or not available:
        return {i: 1.0 for i in range(len(available))}

    ranks = np.array([p.consensus_overall_rank for p in available], dtype=float)
    board_position = (np.argsort(np.argsort(ranks)) + 1).astype(float)

    rng = np.random.default_rng(seed)
    survived = np.zeros(len(available))
    for _ in range(trials):
        order = np.argsort(pick_model.sample_board_order(board_position, ranks, rng))
        taken = set(order[:gap].tolist())
        for i in range(len(available)):
            if i not in taken:
                survived[i] += 1.0

    return {i: float(survived[i] / trials) for i in range(len(available))}


def board_view(
    state: DraftState,
    projections: list[Projection],
    pick_model,
    settings,
    top: int = 8,
    trials: int = SURVIVAL_TRIALS,
) -> list[BoardRow]:
    """The draft-night board: the pick to make, plus the alternatives."""
    by_id = {p.espn_id: p for p in projections if p.espn_id is not None}
    taken = state.drafted_set
    available = [p for p in projections if p.espn_id not in taken]
    if not available:
        return []

    needed = _needed_positions(state, projections, settings, by_id)
    odds = survival_odds(state, available, pick_model, trials=trials)

    eligible = [(i, p) for i, p in enumerate(available) if p.position in needed] or list(
        enumerate(available)
    )
    eligible.sort(key=lambda pair: pair[1].consensus_overall_rank)

    chosen_index = eligible[0][0]
    rows = [
        BoardRow(
            player=p.player,
            position=p.position,
            team=p.team,
            espn_id=p.espn_id,
            overall_rank=p.consensus_overall_rank,
            mean=p.mean,
            sd=p.sd,
            vor=p.vor,
            survival=odds.get(i, 1.0),
            fills_a_need=p.position in needed,
            recommended=(i == chosen_index),
        )
        for i, p in eligible[:top]
    ]
    return rows
