"""Roster derivation for the whole room, and the flex rule that softens the caps."""

from fantasy_football.draft.room import (
    FLEX_ELIGIBLE,
    flex_slot_count,
    open_positions,
    position_counts,
    slot_owners,
    slot_rosters,
)


class Settings:
    starting_slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "D/ST": 1, "K": 1, "RB/WR/TE": 1}


class TwoFlex:
    starting_slots = {"QB": 1, "RB": 2, "WR": 2, "RB/WR/TE": 1, "WR/TE": 1, "D/ST": 1}


class TestFlexSlotCount:
    def test_dst_is_not_a_flex(self):
        """`D/ST` contains a slash. Matching on that alone loosens every cap by one."""
        assert flex_slot_count(Settings()) == 1

    def test_counts_every_real_flex(self):
        assert flex_slot_count(TwoFlex()) == 2

    def test_no_flex(self):
        class NoFlex:
            starting_slots = {"QB": 1, "RB": 2, "D/ST": 1}

        assert flex_slot_count(NoFlex()) == 0


class TestSlotOwners:
    def test_snake_turns(self):
        assert slot_owners(10, 4) == [1, 2, 3, 4, 4, 3, 2, 1, 1, 2]

    def test_rosters_follow_the_snake(self):
        rosters = slot_rosters([101, 102, 103, 104, 105, 106], team_count=3)
        assert rosters[1] == [101, 106]
        assert rosters[2] == [102, 105]
        assert rosters[3] == [103, 104]

    def test_partial_round(self):
        rosters = slot_rosters([11, 12], team_count=4)
        assert rosters == {1: [11], 2: [12]}


class TestOpenPositions:
    positions = {1: "RB", 2: "RB", 3: "WR", 4: "QB"}
    limits = {"QB": 2, "RB": 2, "WR": 2, "TE": 1}

    def test_counts(self):
        assert position_counts([1, 2, 3], self.positions) == {"RB": 2, "WR": 1}

    def test_position_at_cap_closes_without_a_flex(self):
        counts = {"RB": 2}
        assert "RB" not in open_positions(counts, self.limits, flex_slots=0)

    def test_flex_keeps_an_overflowing_position_open(self):
        """A manager at his running-back cap still takes one while the flex is empty."""
        counts = {"RB": 2}
        assert "RB" in open_positions(counts, self.limits, flex_slots=1)

    def test_flex_closes_once_it_is_used(self):
        counts = {"RB": 3}  # one past the cap of 2, so the single flex is spent
        assert "RB" not in open_positions(counts, self.limits, flex_slots=1)

    def test_flex_never_opens_a_non_flex_position(self):
        counts = {"QB": 2}
        assert "QB" not in open_positions(counts, self.limits, flex_slots=1)
        assert "QB" not in FLEX_ELIGIBLE
