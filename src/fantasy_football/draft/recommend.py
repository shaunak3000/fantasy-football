"""What to do with the pick in front of you.

Taking the most valuable player available is the obvious move and the wrong one.
The right question is not "who is best?" but "who will not be there next time?"
If the best player on the board and the fourth-best are worth the same — which
happens constantly, since preseason RB3 through RB7 are statistically identical
— and the fourth-best is far more likely to survive eleven more picks, taking
the fourth-best is strictly better.

So every candidate is scored by simulating the picks between now and your next
turn, using the league's own fitted draft behaviour, and asking what your roster
looks like after *both* selections. A player is worth taking early exactly to
the extent that waiting would cost you.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..projections.ensemble import Projection
from .lineup_value import marginal_value, roster_points
from .model import PickModel
from .state import DraftState

DEFAULT_TRIALS = 500
CANDIDATES_CONSIDERED = 40

# How many of each position are worth owning: the starters the league forces,
# plus the flex for anyone eligible, plus a sensible backup allowance. These are
# caps that stop the tool recommending a third quarterback, not targets.
BACKUP_ALLOWANCE = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 0, "D/ST": 0}
FLEX_ELIGIBLE = ("RB", "WR", "TE")


def roster_limits(settings) -> dict[str, int]:
    flex = sum(
        n
        for slot, n in settings.starting_slots.items()
        if slot in ("RB/WR", "RB/WR/TE", "WR/TE", "OP")
    )
    limits = {}
    for position, backup in BACKUP_ALLOWANCE.items():
        starters = settings.starting_slots.get(position, 0)
        if starters == 0:
            continue
        bonus = flex if position in FLEX_ELIGIBLE else 0
        limits[position] = starters + bonus + backup
    return limits


@dataclass(frozen=True)
class Recommendation:
    player: str
    position: str
    team: str
    espn_id: int | None
    consensus_rank: int
    vor: float
    mean: float
    sd: float
    survival: float
    value_now: float
    next_pick_value: float

    def rationale(self) -> str:
        """One line explaining the pick, for reading under a 90-second clock."""
        scarcity = (
            f"{self.survival:.0%} chance he lasts to your next pick"
            if self.survival < 0.9
            else "safe to wait on, but nobody better is here"
        )
        return (
            f"{self.vor:+.0f} over replacement ({self.mean:.0f} pts, +/-{self.sd:.0f}); {scarcity}."
        )

    @property
    def two_pick_value(self) -> float:
        """Value of this pick plus the best you can expect at your next one.

        This is the number the board is sorted on. Comparing candidates on VOR
        alone ignores what taking them costs you later, which is the whole
        reason a high-VOR player who will still be there in ten picks is a worse
        selection than a slightly cheaper one who will not.
        """
        return self.value_now + self.next_pick_value

    @property
    def likely_gone(self) -> bool:
        return self.survival < 0.5

    @property
    def expected_loss_if_passed(self) -> float:
        """What passing on him costs, weighted by how likely he is to vanish."""
        return self.value_now * (1.0 - self.survival)


def _needed_positions(roster_counts: dict[str, int], limits: dict[str, int]) -> set[str]:
    return {p for p, cap in limits.items() if roster_counts.get(p, 0) < cap}


def recommend(
    state: DraftState,
    projections: list[Projection],
    pick_model: PickModel,
    settings,
    trials: int = DEFAULT_TRIALS,
    seed: int = 0,
) -> list[Recommendation]:
    """Rank the available players by what taking them now is actually worth."""
    taken = state.drafted_set

    available = [p for p in projections if p.espn_id is None or p.espn_id not in taken]
    if not available:
        return []

    my_counts: dict[str, int] = {}
    for projection in projections:
        if projection.espn_id in set(state.my_roster):
            my_counts[projection.position] = my_counts.get(projection.position, 0) + 1

    gap = state.picks_until_my_next()
    positions = [p.position for p in available]

    # The decision value is marginal improvement to the lineup you can start,
    # not value over a league-wide replacement. VOR cannot tell a sixth receiver
    # from a first running back; this can, and without it the board drafts one
    # position until the roster is unstartable.
    by_id = {p.espn_id: p for p in projections if p.espn_id is not None}
    my_roster = roster_points(state.my_roster, by_id)
    replacement = {p.position: p.replacement for p in projections}

    vors = np.array(
        [marginal_value(p.position, p.mean, my_roster, settings, replacement) for p in available],
        dtype=float,
    )

    # Value comes from positional rank; draft order comes from overall rank.
    # Board position is where each player sits among those still available, which
    # is what he competes on — a run on running backs really does move everyone
    # else up the board.
    absolute_rank = np.array([p.consensus_overall_rank for p in available], dtype=float)
    board_position = (np.argsort(np.argsort(absolute_rank)) + 1).astype(float)

    order_by_value = np.argsort(-vors)[:CANDIDATES_CONSIDERED]

    if gap is None:
        # Genuinely the last pick of the draft: nothing follows it.
        return [_build(available[i], 1.0, vors[i], 0.0) for i in order_by_value]

    if gap == 0:
        # Back-to-back picks at the turn of a snake. Nobody picks in between, so
        # nothing can be sniped and you are really choosing a *pair*: take the
        # best, then the best of what is left. No simulation is needed because
        # there is no uncertainty to simulate.
        results = []
        for i in order_by_value:
            i = int(i)
            after = {position: list(points) for position, points in my_roster.items()}
            after.setdefault(positions[i], []).append(available[i].mean)
            partner = max(
                (
                    marginal_value(positions[j], available[j].mean, after, settings, replacement)
                    for j in range(len(available))
                    if j != i
                ),
                default=0.0,
            )
            results.append(_build(available[i], 1.0, float(vors[i]), float(partner)))
        results.sort(key=lambda r: -r.two_pick_value)
        return results

    # The best follow-up pick depends on which candidate was taken, because the
    # roster it would join is different. Precompute one preference ordering per
    # candidate — cheap up front, and it keeps the inner trial loop to a scan.
    pool_size = CANDIDATES_CONSIDERED * 4
    follow_up_order: dict[int, list[int]] = {}
    follow_up_value: dict[int, dict[int, float]] = {}

    for candidate in order_by_value:
        candidate = int(candidate)
        after = {position: list(points) for position, points in my_roster.items()}
        after.setdefault(positions[candidate], []).append(available[candidate].mean)

        scored = []
        for i in np.argsort(-vors)[:pool_size]:
            i = int(i)
            if i == candidate:
                continue
            value = marginal_value(positions[i], available[i].mean, after, settings, replacement)
            scored.append((value, i))
        scored.sort(reverse=True)
        follow_up_order[candidate] = [i for _, i in scored]
        follow_up_value[candidate] = {i: value for value, i in scored}

    rng = np.random.default_rng(seed)
    totals = np.zeros(len(order_by_value))
    # Survival is counted from the same simulation that drives the lookahead, so
    # the two can never disagree about who is still on the board.
    survived = np.zeros(len(available))

    for _ in range(trials):
        slots = pick_model.sample_board_order(board_position, absolute_rank, rng)
        draft_order = np.argsort(slots)

        # The players opponents would take if I were not picking at all, plus
        # the one who steps up when I remove somebody they wanted.
        base_taken = set(draft_order[:gap].tolist())
        next_in_line = int(draft_order[gap]) if gap < len(draft_order) else None

        for i in range(len(available)):
            if i not in base_taken:
                survived[i] += 1.0

        for slot_index, candidate in enumerate(order_by_value):
            candidate = int(candidate)
            if candidate in base_taken:
                gone = (base_taken - {candidate}) | (
                    {next_in_line} if next_in_line is not None else set()
                )
            else:
                gone = base_taken
            best_next = 0.0
            for i in follow_up_order[candidate]:
                if i not in gone:
                    best_next = follow_up_value[candidate][i]
                    break

            totals[slot_index] += vors[candidate] + best_next

    totals /= trials
    survived /= trials

    results = []
    for slot_index, candidate in enumerate(order_by_value):
        results.append(
            _build(
                available[candidate],
                survived[candidate],
                vors[candidate],
                totals[slot_index] - vors[candidate],
            )
        )

    results.sort(key=lambda r: -r.two_pick_value)
    return results


def _build(
    projection: Projection, survival: float, vor: float, next_pick_value: float
) -> Recommendation:
    return Recommendation(
        player=projection.player,
        position=projection.position,
        team=projection.team,
        espn_id=projection.espn_id,
        consensus_rank=projection.consensus_rank,
        # `vor` on a recommendation is the marginal gain to *your* lineup, which
        # is what the decision runs on. The league-wide VOR lives on the
        # projection and is only useful for comparing players in the abstract.
        vor=vor,
        mean=projection.mean,
        sd=projection.sd,
        survival=float(survival),
        value_now=vor,
        next_pick_value=float(next_pick_value),
    )
