"""Who in the room owns what, and what each of them still needs.

The pick model treats every opponent identically: a distribution over where a
player goes, conditional only on his consensus rank. That is deliberately memory-
less, and it is wrong in one specific, correctable way — a manager holding three
quarterbacks does not take a fourth. Roster caps were the entire edge in our own
draft rule, so the same filter should predict opponents better too.

Nothing here models a *reaction* to what we do. It models the constraint each
opponent is already under, which is the part that can be checked against two
seasons of real picks.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

from .state import snake_slot_for_pick

# Positions a flex slot can absorb. A manager at his running-back cap will still
# take a running back if the flex is open, so the cap alone would over-predict
# how quickly a position closes.
FLEX_ELIGIBLE = frozenset({"RB", "WR", "TE"})


def slot_owners(pick_count: int, team_count: int) -> list[int]:
    """The draft slot owning each of the first `pick_count` picks, in order."""
    return [snake_slot_for_pick(pick, team_count) for pick in range(1, pick_count + 1)]


def rosters_from_sequence(
    owners: Sequence[Hashable], picks: Sequence[int]
) -> dict[Hashable, list[int]]:
    """Group an ordered pick sequence by whoever made each pick."""
    rosters: dict[Hashable, list[int]] = {}
    for owner, espn_id in zip(owners, picks, strict=False):
        rosters.setdefault(owner, []).append(espn_id)
    return rosters


def slot_rosters(drafted: Sequence[int], team_count: int) -> dict[int, list[int]]:
    """Every slot's roster so far, derived from the snake order."""
    return rosters_from_sequence(slot_owners(len(drafted), team_count), drafted)


def position_counts(roster: Sequence[int], positions: Mapping[int, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for espn_id in roster:
        position = positions.get(espn_id)
        if position is not None:
            counts[position] = counts.get(position, 0) + 1
    return counts


def open_positions(
    counts: Mapping[str, int], limits: Mapping[str, int], flex_slots: int = 0
) -> set[str]:
    """Positions this roster can still add, honouring the flex.

    A position at its own cap stays open while the flex is unfilled, because the
    flex is where the overflow goes. Treating the cap as hard would have the
    model insisting a manager is done at running back two rounds before he is.
    """
    open_now = {position for position, cap in limits.items() if counts.get(position, 0) < cap}

    flex_used = sum(
        max(0, counts.get(position, 0) - limits.get(position, 0)) for position in FLEX_ELIGIBLE
    )
    if flex_used < flex_slots:
        open_now |= {p for p in FLEX_ELIGIBLE if p in limits}
    return open_now


def slot_needs(
    drafted: Sequence[int],
    team_count: int,
    positions: Mapping[int, str],
    limits: Mapping[str, int],
    flex_slots: int = 0,
) -> dict[int, set[str]]:
    """What every slot in the room can still draft."""
    rosters = slot_rosters(drafted, team_count)
    return {
        slot: open_positions(position_counts(rosters.get(slot, []), positions), limits, flex_slots)
        for slot in range(1, team_count + 1)
    }


def flex_slot_count(settings) -> int:
    """How many flex slots this league starts, from its real slot names.

    Matching on a slash alone is wrong and quietly so: `D/ST` contains one, and
    counting the defence as a flex loosens every roster cap by one.
    """
    total = 0
    for name, count in getattr(settings, "starting_slots", {}).items():
        parts = {part.strip() for part in name.split("/")}
        if len(parts) > 1 and parts <= FLEX_ELIGIBLE:
            total += count or 0
    return total
