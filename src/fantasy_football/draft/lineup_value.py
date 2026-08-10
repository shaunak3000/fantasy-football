"""What a player is actually worth to *your* roster.

Value over replacement ranks players against a league-wide benchmark, which is
the right way to compare a receiver to a running back in the abstract and the
wrong way to make a pick. It has no idea what you already own. Two players with
identical VOR are worth wildly different amounts if one fills an empty starting
slot and the other is your sixth receiver competing for a flex spot.

So a candidate is priced by how much he improves the best lineup you could
start. An unfilled slot is charged at replacement level rather than zero,
because a manager with no tight end does not start nobody — he picks one off
waivers. That keeps the first player at a position valuable without making him
absurdly so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

FLEX_SLOTS = ("RB/WR", "RB/WR/TE", "WR/TE", "OP")
FLEX_ELIGIBILITY = {
    "RB/WR": ("RB", "WR"),
    "RB/WR/TE": ("RB", "WR", "TE"),
    "WR/TE": ("WR", "TE"),
    "OP": ("QB", "RB", "WR", "TE"),
}


def dedicated_slots(settings) -> dict[str, int]:
    return {
        slot: count
        for slot, count in settings.starting_slots.items()
        if slot not in FLEX_SLOTS and count
    }


def flex_slots(settings) -> list[tuple[str, ...]]:
    slots = []
    for slot, count in settings.starting_slots.items():
        if slot in FLEX_SLOTS:
            slots.extend([FLEX_ELIGIBILITY[slot]] * count)
    return slots


def lineup_points(
    roster: Mapping[str, Sequence[float]],
    settings,
    replacement: Mapping[str, float],
) -> float:
    """Best legal starting lineup from the players on hand.

    `roster` maps position to that position's projected point totals. Missing
    starters fall back to replacement level, which is what a manager would
    realistically stream.
    """
    pools = {position: sorted(points, reverse=True) for position, points in roster.items()}
    total = 0.0
    used: dict[str, int] = {}

    for position, count in dedicated_slots(settings).items():
        available = pools.get(position, [])
        floor = replacement.get(position, 0.0)
        for index in range(count):
            if index < len(available):
                # Floored at replacement: owning a player worse than the waiver
                # wire does not force you to start him, so acquiring anyone can
                # never make your lineup worse. Without this, drafting a backup
                # would show up as negative value.
                total += max(available[index], floor)
                used[position] = used.get(position, 0) + 1
            else:
                total += floor

    for eligible in flex_slots(settings):
        best_value = None
        best_position = None
        for position in eligible:
            taken = used.get(position, 0)
            available = pools.get(position, [])
            if taken < len(available):
                value = available[taken]
                if best_value is None or value > best_value:
                    best_value, best_position = value, position
        flex_floor = max((replacement.get(position, 0.0) for position in eligible), default=0.0)
        if best_position is None or best_value <= flex_floor:
            total += flex_floor
        else:
            total += best_value
            used[best_position] = used.get(best_position, 0) + 1

    return total


def marginal_value(
    candidate_position: str,
    candidate_points: float,
    roster: Mapping[str, Sequence[float]],
    settings,
    replacement: Mapping[str, float],
) -> float:
    """How much adding this player improves the lineup you can start."""
    before = lineup_points(roster, settings, replacement)
    updated = {position: list(points) for position, points in roster.items()}
    updated.setdefault(candidate_position, []).append(candidate_points)
    return lineup_points(updated, settings, replacement) - before


def roster_points(espn_ids: Sequence[int], by_id: Mapping[int, object]) -> dict[str, list[float]]:
    """Group a set of drafted players into position -> projected points."""
    roster: dict[str, list[float]] = {}
    for espn_id in espn_ids:
        projection = by_id.get(espn_id)
        if projection is None:
            continue
        roster.setdefault(projection.position, []).append(projection.mean)
    return roster
