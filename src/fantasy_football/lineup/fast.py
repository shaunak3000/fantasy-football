"""Lineup points without the solver, for when it has to run a hundred thousand times.

`optimize` solves an assignment problem with PuLP, which is the right way to get
a lineup a human will act on: legality is guaranteed by construction and it
cannot quietly produce an illegal answer at the edges. It is also far too slow to
screen trades. At sixteen give-candidates against seventeen get-candidates across
seven rosters, the search evaluates around 290,000 lineups, which is roughly
forty-five minutes of solver time — the wide-depth search simply never returned.

For *this* league's slots the exact answer needs no solver. There is one flex
slot, over RB/WR/TE. Fill each dedicated slot with the best players at that
position, then give the flex to the best player left who is eligible for it, and
no reshuffle can beat it: taking a worse player for a dedicated slot to free a
better one for the flex trades a point for at most the same point.

That argument holds for a single flex slot and breaks with several overlapping
ones, where the assignment genuinely interacts. `is_greedy_exact` says whether a
league qualifies, and `best_points` falls back to the solver when it does not, so
a league with two flex slots gets a slower screen rather than a wrong one.
"""

from __future__ import annotations

from collections import defaultdict

from ..draft.lineup_value import FLEX_ELIGIBILITY, FLEX_SLOTS


def is_greedy_exact(settings) -> bool:
    """Whether greedy filling provably matches the solver for these slots.

    True when at most one flex slot exists. With two, a player eligible for both
    can be needed in either and the choice interacts — greedy would have to
    guess, so the solver is used instead.
    """
    flex_count = sum(count for slot, count in settings.starting_slots.items() if slot in FLEX_SLOTS)
    return flex_count <= 1


def select(players: list, settings) -> list:
    """The best legal lineup these players can field, as the players themselves.

    Returning the starters rather than just their total is what lets a trade be
    judged week by week — whether the incoming receiver actually plays, and in
    which weeks, falls straight out of which players this picks.

    Takes anything with `.position` and `.mean`, the same shape `optimize` reads.
    A player whose position has no slot in this league simply never starts. Ties
    keep input order, so the choice is deterministic.
    """
    by_position: dict[str, list] = defaultdict(list)
    for player in players:
        by_position[player.position].append(player)
    for pool in by_position.values():
        pool.sort(key=lambda p: -p.mean)

    starters: list = []
    used: dict[str, int] = defaultdict(int)
    flex_pools: list[tuple[str, ...]] = []

    for slot, count in settings.starting_slots.items():
        count = int(count)
        if slot in FLEX_SLOTS:
            flex_pools.extend([FLEX_ELIGIBILITY[slot]] * count)
            continue
        take = by_position.get(slot, [])[:count]
        starters.extend(take)
        used[slot] = len(take)

    for eligible in flex_pools:
        best = None
        best_position = None
        for position in eligible:
            pool = by_position.get(position, ())
            index = used[position]
            if (
                index < len(pool)
                and pool[index].mean > 0
                and (best is None or pool[index].mean > best.mean)
            ):
                best = pool[index]
                best_position = position
        if best is not None:
            starters.append(best)
            used[best_position] += 1

    return starters


def best_points(players: list, settings) -> float:
    """Points of the best legal lineup these players can field."""
    return sum(player.mean for player in select(players, settings))
