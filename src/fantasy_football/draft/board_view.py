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
from .strategies import best_available_at_need

SURVIVAL_TRIALS = 300

# Cushion, in picks, before ADP is allowed to say "you can wait". ESPN publishes
# only the mean draft position with no spread, so a player whose ADP sits barely
# past your next turn is a coin flip, not a wait. One round in this league is
# eight picks; requiring a full round of daylight keeps the hint honest.
ADP_WAIT_CUSHION = 8

# Picks looked back over when deciding whether a position is being run on.
RUN_WINDOW = 8
# A position taken this many times more often than its baseline share counts as
# a run worth reacting to.
RUN_THRESHOLD = 1.8

# Roughly how often each position is taken across a whole draft. Used as the
# baseline a run is measured against.
BASELINE_SHARE = {"QB": 0.11, "RB": 0.30, "WR": 0.36, "TE": 0.13, "K": 0.05, "D/ST": 0.05}


def detect_runs(state: DraftState, by_id: dict, window: int = RUN_WINDOW) -> dict[str, float]:
    """How much hotter than normal each position has been running.

    The fitted pick model describes a typical draft. A real one goes through
    phases — six running backs in eight picks, then nobody touches one for two
    rounds — and during a run the static model badly overstates who will survive.
    Returns a multiplier per position: 1.0 is normal, above `RUN_THRESHOLD` means
    that position is disappearing far faster than usual.
    """
    recent = [by_id[i] for i in state.drafted[-window:] if i in by_id]
    if len(recent) < 4:
        return {}

    counts: dict[str, int] = {}
    for projection in recent:
        counts[projection.position] = counts.get(projection.position, 0) + 1

    intensity = {}
    for position, count in counts.items():
        baseline = BASELINE_SHARE.get(position, 0.1) * len(recent)
        if baseline > 0:
            intensity[position] = count / baseline
    return intensity


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
    # Where the wider ESPN population actually takes this player, as an ordinal
    # rank. None when ESPN lists no ADP for him — common for deep bench players
    # and most defences.
    adp_rank: int | None = None
    # Picks of daylight between your next turn and where the room takes him.
    # Positive means ADP expects him to still be there.
    adp_slack: int | None = None

    @property
    def can_wait(self) -> bool:
        """ADP says this player survives your next turn with a round to spare.

        Purely informational. The recommendation is never changed by it — the
        rehearsal was unambiguous that making the selection rule cleverer loses,
        and this is a signal that has never been backtested at all.
        """
        return self.adp_slack is not None and self.adp_slack >= ADP_WAIT_CUSHION

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
    runs: dict[str, float] | None = None,
) -> dict[int, float]:
    """Chance each available player is still there when you pick again."""
    gap = state.picks_until_my_next()
    if gap is None or gap <= 0 or not available:
        return {i: 1.0 for i in range(len(available))}

    ranks = np.array([p.consensus_overall_rank for p in available], dtype=float)
    board_position = (np.argsort(np.argsort(ranks)) + 1).astype(float)

    # During a run, players at the hot position come off the board earlier than
    # their rank implies. Pulling their effective board position forward is what
    # stops the tool from promising a running back will last when five have just
    # gone in six picks.
    if runs:
        for i, projection in enumerate(available):
            intensity = runs.get(projection.position, 1.0)
            if intensity >= RUN_THRESHOLD:
                board_position[i] /= min(intensity, 3.0)

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
    adp: dict[int, int] | None = None,
) -> list[BoardRow]:
    """The draft-night board: the pick to make, plus the alternatives."""
    by_id = {p.espn_id: p for p in projections if p.espn_id is not None}
    taken = state.drafted_set
    available = [p for p in projections if p.espn_id not in taken]
    if not available:
        return []

    needed = _needed_positions(state, projections, settings, by_id)
    runs = detect_runs(state, by_id)
    odds = survival_odds(state, available, pick_model, trials=trials, runs=runs)

    eligible = [(i, p) for i, p in enumerate(available) if p.position in needed] or list(
        enumerate(available)
    )
    eligible.sort(key=lambda pair: pair[1].consensus_overall_rank)

    # The recommendation comes from the strategy itself rather than being
    # reimplemented here. They diverged once — the live board kept taking the
    # highest-ranked player while the strategy had learned to reserve late picks
    # for mandatory positions — and a draft board that disagrees with the
    # strategy it was validated as is worse than either alone.
    chosen_id = best_available_at_need(state, projections, settings, by_id)
    chosen_index = next(
        (i for i, p in enumerate(available) if p.espn_id == chosen_id),
        eligible[0][0],
    )

    # The recommendation must always appear in the rows returned, and it is not
    # always near the top of them: late in the draft the right pick is a kicker
    # ranked 178th, which the display would otherwise truncate away — leaving
    # nothing flagged and any caller falling back to the wrong player.
    eligible = [pair for pair in eligible if pair[0] != chosen_index]
    eligible.insert(0, (chosen_index, available[chosen_index]))
    adp = adp or {}
    next_pick = state.next_pick_for_me()

    def adp_for(projection):
        rank = adp.get(projection.espn_id)
        if rank is None or next_pick is None:
            return rank, None
        return rank, rank - next_pick

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
            adp_rank=adp_for(p)[0],
            adp_slack=adp_for(p)[1],
        )
        for i, p in eligible[:top]
    ]
    return rows
