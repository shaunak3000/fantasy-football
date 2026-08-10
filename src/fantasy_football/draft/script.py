"""A forward plan for your own draft, built before it starts.

The live board answers "what do I take now?". It cannot answer "what will still
be here at 24?", which is the question that actually shapes picks 8 and 9.

That gap matters most at the turn. From slot 8 the picks arrive in pairs
separated by fourteen selections, so committing both halves of a pair to the
same position is a decision you cannot revisit for a long time. Knowing in
advance that the board at 24/25 will almost certainly still hold four startable
running backs — or that it will not — changes what you do at 8.

So the whole draft is simulated repeatedly using the league's fitted behaviour
and the strategy that won the rehearsal, and the results are summarized per
pick: who you would most often end up taking, and how strong that pick is.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..projections.ensemble import Projection
from .state import DraftState, snake_slot_for_pick
from .strategies import best_available_at_need

DEFAULT_TRIALS = 400


@dataclass
class PickForecast:
    pick_number: int
    round_number: int
    likely: list[tuple[str, str, float]] = field(default_factory=list)
    position_mix: dict[str, float] = field(default_factory=dict)
    mean_points: float = 0.0
    mean_vor: float = 0.0

    def headline(self) -> str:
        if not self.likely:
            return "no clear target"
        player, position, share = self.likely[0]
        return f"{player} ({position}) {share:.0%}"

    def alternatives(self, limit: int = 4) -> str:
        return ", ".join(f"{player} {share:.0%}" for player, _, share in self.likely[1 : limit + 1])


def simulate_draft(
    my_slot: int,
    projections: list[Projection],
    pick_model,
    settings,
    rounds: int,
    rng: np.random.Generator,
) -> list[Projection]:
    """One plausible draft; returns the players this slot ends up with, in order."""
    by_id = {p.espn_id: p for p in projections if p.espn_id is not None}
    pool = [p for p in projections if p.espn_id is not None]

    ranks = np.array([p.consensus_overall_rank for p in pool], dtype=float)
    board_position = (np.argsort(np.argsort(ranks)) + 1).astype(float)
    order = np.argsort(pick_model.sample_board_order(board_position, ranks, rng))
    queue = [pool[int(i)] for i in order]

    state = DraftState(team_count=settings.team_count, rounds=rounds, my_slot=my_slot)
    mine: list[Projection] = []
    cursor = 0

    for pick_number in range(1, state.total_picks + 1):
        if snake_slot_for_pick(pick_number, settings.team_count) == my_slot:
            chosen_id = best_available_at_need(state, projections, settings, by_id)
            if chosen_id is None:
                break
            state.record_my_pick(chosen_id)
            mine.append(by_id[chosen_id])
            continue

        while cursor < len(queue) and queue[cursor].espn_id in state.drafted_set:
            cursor += 1
        if cursor >= len(queue):
            break
        state.record(queue[cursor].espn_id)
        cursor += 1

    return mine


def forecast(
    my_slot: int,
    projections: list[Projection],
    pick_model,
    settings,
    rounds: int = 16,
    trials: int = DEFAULT_TRIALS,
    seed: int = 0,
) -> list[PickForecast]:
    """Aggregate many simulated drafts into a per-pick plan."""
    rng = np.random.default_rng(seed)
    state = DraftState(team_count=settings.team_count, rounds=rounds, my_slot=my_slot)
    my_picks = state.my_picks

    tallies: dict[int, Counter] = defaultdict(Counter)
    positions: dict[int, Counter] = defaultdict(Counter)
    points: dict[int, list[float]] = defaultdict(list)
    vors: dict[int, list[float]] = defaultdict(list)

    for _ in range(trials):
        drafted = simulate_draft(my_slot, projections, pick_model, settings, rounds, rng)
        for index, player in enumerate(drafted):
            if index >= len(my_picks):
                break
            pick_number = my_picks[index]
            tallies[pick_number][(player.player, player.position)] += 1
            positions[pick_number][player.position] += 1
            points[pick_number].append(player.mean)
            vors[pick_number].append(player.vor)

    forecasts = []
    for pick_number in my_picks:
        counts = tallies[pick_number]
        total = sum(counts.values())
        if not total:
            continue
        forecasts.append(
            PickForecast(
                pick_number=pick_number,
                round_number=state.round_of(pick_number),
                likely=[
                    (player, position, count / total)
                    for (player, position), count in counts.most_common(6)
                ],
                position_mix={
                    position: count / total
                    for position, count in positions[pick_number].most_common()
                },
                mean_points=float(np.mean(points[pick_number])),
                mean_vor=float(np.mean(vors[pick_number])),
            )
        )
    return forecasts


def target_survival(
    my_slot: int,
    projections: list[Projection],
    pick_model,
    settings,
    rounds: int = 16,
    trials: int = DEFAULT_TRIALS,
    seed: int = 1,
    top: int = 25,
) -> list[tuple[str, str, int, float]]:
    """How often each early-round name is still there at your first pick.

    Answers the only question that matters before the draft starts: of the
    players worth wanting, which ones will you actually get a shot at?
    """
    rng = np.random.default_rng(seed)
    state = DraftState(team_count=settings.team_count, rounds=rounds, my_slot=my_slot)
    first_pick = state.my_picks[0]

    pool = sorted(
        (p for p in projections if p.espn_id is not None),
        key=lambda p: p.consensus_overall_rank,
    )[:top]
    ranks_all = np.array(
        [p.consensus_overall_rank for p in projections if p.espn_id is not None], dtype=float
    )
    board_position = (np.argsort(np.argsort(ranks_all)) + 1).astype(float)
    index_of = {p.espn_id: i for i, p in enumerate(p for p in projections if p.espn_id is not None)}

    survived: Counter = Counter()
    for _ in range(trials):
        order = np.argsort(pick_model.sample_board_order(board_position, ranks_all, rng))
        gone = set(order[: first_pick - 1].tolist())
        for player in pool:
            if index_of[player.espn_id] not in gone:
                survived[player.espn_id] += 1

    return [
        (p.player, p.position, p.consensus_overall_rank, survived[p.espn_id] / trials) for p in pool
    ]
