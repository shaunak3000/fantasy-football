"""Snake draft bookkeeping: whose turn it is, and when you pick again.

The gap between your picks is the whole reason a draft needs strategy. In an
8-team snake you pick at 3 and then not again until 14 — eleven players vanish
in between — and the size of that gap swings wildly with where you sit. From
slot 1 you wait 15 picks; from slot 8 you pick back to back and then wait 15.
Every recommendation depends on getting this right.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def snake_slot_for_pick(pick_number: int, team_count: int) -> int:
    """Which draft slot (1-indexed) owns a given overall pick."""
    if pick_number < 1:
        raise ValueError("pick numbers start at 1")
    index = pick_number - 1
    round_index = index // team_count
    position = index % team_count
    # Odd-numbered rounds run forward, even-numbered rounds run back.
    return position + 1 if round_index % 2 == 0 else team_count - position


def picks_for_slot(slot: int, team_count: int, rounds: int) -> list[int]:
    """Every overall pick number belonging to one draft slot."""
    return [
        pick
        for pick in range(1, team_count * rounds + 1)
        if snake_slot_for_pick(pick, team_count) == slot
    ]


@dataclass
class DraftState:
    """Live state of a draft in progress."""

    team_count: int
    rounds: int
    my_slot: int
    drafted: list[int] = field(default_factory=list)
    my_roster: list[int] = field(default_factory=list)

    @property
    def total_picks(self) -> int:
        return self.team_count * self.rounds

    @property
    def current_pick(self) -> int:
        return len(self.drafted) + 1

    @property
    def is_complete(self) -> bool:
        return len(self.drafted) >= self.total_picks

    @property
    def my_picks(self) -> list[int]:
        return picks_for_slot(self.my_slot, self.team_count, self.rounds)

    @property
    def is_my_turn(self) -> bool:
        return not self.is_complete and self.current_pick in set(self.my_picks)

    def next_pick_for_me(self, after: int | None = None) -> int | None:
        """My next pick strictly after the given pick (default: the current one)."""
        reference = self.current_pick if after is None else after
        upcoming = [p for p in self.my_picks if p > reference]
        return upcoming[0] if upcoming else None

    def my_picks_remaining(self) -> int:
        """How many selections I still have, including the one on the clock."""
        return len([pick for pick in self.my_picks if pick >= self.current_pick])

    def picks_until_my_next(self) -> int | None:
        """How many players come off the board before I choose again."""
        following = self.next_pick_for_me()
        if following is None:
            return None
        return following - self.current_pick - (1 if self.is_my_turn else 0)

    def record(self, espn_id: int, mine: bool = False) -> None:
        if self.is_complete:
            raise ValueError("draft is already complete")
        self.drafted.append(espn_id)
        if mine:
            self.my_roster.append(espn_id)

    def record_my_pick(self, espn_id: int) -> None:
        self.record(espn_id, mine=True)

    @property
    def drafted_set(self) -> set[int]:
        return set(self.drafted)

    def round_of(self, pick_number: int) -> int:
        return (pick_number - 1) // self.team_count + 1
