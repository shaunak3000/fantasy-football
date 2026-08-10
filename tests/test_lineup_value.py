import pytest

from fantasy_football.draft.lineup_value import (
    dedicated_slots,
    flex_slots,
    lineup_points,
    marginal_value,
)


class Settings:
    starting_slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1, "K": 1, "D/ST": 1}
    team_count = 8


REPLACEMENT = {"QB": 250.0, "RB": 170.0, "WR": 175.0, "TE": 145.0, "K": 145.0, "D/ST": 100.0}
SETTINGS = Settings()


def value(roster):
    return lineup_points(roster, SETTINGS, REPLACEMENT)


class TestSlots:
    def test_dedicated_slots_exclude_the_flex(self):
        assert "RB/WR/TE" not in dedicated_slots(SETTINGS)
        assert dedicated_slots(SETTINGS)["RB"] == 2

    def test_flex_slots_list_their_eligibility(self):
        assert flex_slots(SETTINGS) == [("RB", "WR", "TE")]


class TestLineupPoints:
    def test_empty_roster_scores_replacement_everywhere(self):
        # QB + 2RB + 2WR + TE + K + D/ST + flex(best replacement of RB/WR/TE)
        expected = 250 + 170 * 2 + 175 * 2 + 145 + 145 + 100 + 175
        assert value({}) == pytest.approx(expected)

    def test_a_starter_replaces_the_replacement(self):
        base = value({})
        assert value({"QB": [300.0]}) == pytest.approx(base + 50.0)

    def test_extra_players_beyond_the_slots_add_nothing(self):
        two_rbs = value({"RB": [250.0, 240.0]})
        # A third RB can only take the flex; a fourth is worth nothing at all.
        three = value({"RB": [250.0, 240.0, 230.0]})
        four = value({"RB": [250.0, 240.0, 230.0, 225.0]})
        assert three > two_rbs
        assert four == pytest.approx(three)

    def test_flex_takes_the_best_leftover_across_positions(self):
        roster = {"RB": [250.0, 240.0], "WR": [260.0, 255.0, 252.0]}
        # QB/TE/K/DST fall back to replacement; the flex takes the third WR.
        expected = 250.0 + (250 + 240) + (260 + 255) + 145.0 + 145.0 + 100.0 + 252.0
        assert value(roster) == pytest.approx(expected)

    def test_a_player_worse_than_the_waiver_wire_is_not_started(self):
        """You would stream a replacement rather than start a bad backup."""
        assert value({"TE": [90.0]}) == pytest.approx(value({}))


class TestMarginalValue:
    def test_first_player_at_an_empty_position_is_worth_the_gap(self):
        got = marginal_value("QB", 300.0, {}, SETTINGS, REPLACEMENT)
        assert got == pytest.approx(50.0)

    def test_sixth_receiver_is_worth_nothing(self):
        roster = {"WR": [280.0, 270.0, 260.0, 250.0, 240.0]}
        assert marginal_value("WR", 230.0, roster, SETTINGS, REPLACEMENT) == pytest.approx(0.0)

    def test_filling_an_empty_slot_beats_a_better_surplus_player(self):
        """The whole reason VOR alone is not enough.

        With both receiver slots and the flex already filled, a sixth receiver
        adds nothing, while a running back worth fewer raw points fills a slot
        that is otherwise sitting at replacement level.
        """
        roster = {"WR": [280.0, 270.0, 265.0, 260.0, 255.0]}
        rb = marginal_value("RB", 200.0, roster, SETTINGS, REPLACEMENT)
        wr = marginal_value("WR", 250.0, roster, SETTINGS, REPLACEMENT)
        assert rb > wr
        assert wr == pytest.approx(0.0)

    def test_upgrading_a_filled_slot_is_worth_only_the_difference(self):
        roster = {"QB": [280.0]}
        assert marginal_value("QB", 300.0, roster, SETTINGS, REPLACEMENT) == pytest.approx(20.0)

    def test_a_worse_player_at_a_filled_slot_may_still_help_the_flex(self):
        roster = {"RB": [250.0], "WR": [280.0, 270.0]}
        # A second RB fills the empty RB2 slot, so it is worth a lot.
        assert marginal_value("RB", 200.0, roster, SETTINGS, REPLACEMENT) > 25.0

    def test_marginal_value_is_never_negative(self):
        roster = {"RB": [250.0, 240.0], "WR": [280.0, 270.0]}
        for position in ("QB", "RB", "WR", "TE"):
            got = marginal_value(position, 10.0, roster, SETTINGS, REPLACEMENT)
            assert got >= -1e-9
