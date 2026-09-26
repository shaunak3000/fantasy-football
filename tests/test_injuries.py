"""ESPN publishes no return dates, so the repo has to supply them — correctly.

Until this existed a player on injured reserve counted as back the next week. A.J.
Brown, out until November, was valued as a healthy 15.9-point starter from week 4,
which mis-priced every trade touching the receiver room and labelled DK Metcalf
"depth — never starts" when he would have started every week until Brown returned.
"""

import json
from dataclasses import dataclass
from datetime import date

from fantasy_football.data.injuries import (
    IR_MINIMUM_GAMES,
    first_seen_on_ir,
    load_overrides,
    normalize,
    resolve_return_week,
    return_week,
)

# New England's real 2026 dates around the return: a Thursday game in week 7.
NE_GAMES = {
    "NE": [
        (6, date(2026, 10, 18)),
        (7, date(2026, 10, 22)),
        (8, date(2026, 11, 1)),
        (9, date(2026, 11, 8)),
    ],
    "MIA": [(7, date(2026, 10, 25)), (8, date(2026, 11, 2))],
}


class TestResolvingADate:
    def test_back_in_november_is_week_8_for_new_england(self):
        """The case that prompted this. NE's first game on or after 1 November."""
        info = {"team": "NE", "return_date": "2026-11-01"}
        assert resolve_return_week(info, NE_GAMES) == 8

    def test_it_uses_the_players_own_team(self):
        """Miami's week-8 game is on the 2nd; a Miami player back on the 1st
        plays week 8 too, but a date of the 26th lands in week 8 for Miami and
        still week 8 for NE — the team decides, not the calendar."""
        assert resolve_return_week({"team": "MIA", "return_date": "2026-10-26"}, NE_GAMES) == 8
        assert resolve_return_week({"team": "NE", "return_date": "2026-10-20"}, NE_GAMES) == 7

    def test_an_explicit_week_is_taken_as_given(self):
        assert resolve_return_week({"return_week": 11}, None) == 11

    def test_a_date_without_a_schedule_cannot_be_resolved(self):
        assert resolve_return_week({"return_date": "2026-11-01"}, None) is None


class TestWhichSourceWins:
    def test_an_override_beats_the_ir_floor(self):
        overrides = {normalize("A.J. Brown"): {"team": "NE", "return_date": "2026-11-01"}}
        week = return_week("A.J. Brown", 1, "INJURY_RESERVE", 3, overrides, {1: 2}, NE_GAMES)
        assert week == 8

    def test_an_override_applies_even_when_espn_says_healthy(self):
        """The news runs ahead of the status."""
        overrides = {normalize("X"): {"return_week": 9}}
        assert return_week("X", 1, "ACTIVE", 3, overrides, {}, None) == 9

    def test_an_override_in_the_past_means_available(self):
        overrides = {normalize("X"): {"return_week": 2}}
        assert return_week("X", 1, "INJURY_RESERVE", 3, overrides, {}, None) is None

    def test_ir_without_an_override_uses_the_four_game_minimum(self):
        """First seen on IR in week 3 -> misses 3, 4, 5, 6 -> back week 7."""
        week = return_week("Brooks", 7, "INJURY_RESERVE", 3, {}, {7: 3}, None)
        assert week == 3 + IR_MINIMUM_GAMES

    def test_ir_never_returns_sooner_than_next_week(self):
        """An old placement whose minimum has run out is still out this week."""
        assert return_week("Old", 5, "INJURY_RESERVE", 10, {}, {5: 1}, None) == 11

    def test_out_and_doubtful_are_this_week_only(self):
        for status in ("OUT", "DOUBTFUL", "QUESTIONABLE", "ACTIVE", None):
            assert return_week("P", 1, status, 3, {}, {}, None) is None

    def test_names_match_despite_punctuation(self):
        assert normalize("A.J. Brown") == normalize("AJ Brown") == normalize("a.j.  brown")


class TestTheFile:
    def test_it_loads_the_season_asked_for(self, tmp_path):
        path = tmp_path / "returns.json"
        path.write_text(json.dumps({"2026": {"A.J. Brown": {"return_week": 8}}, "2025": {}}))
        assert load_overrides(2026, path) == {"aj brown": {"return_week": 8}}
        assert load_overrides(2025, path) == {}

    def test_a_missing_file_means_no_overrides(self, tmp_path):
        assert load_overrides(2026, tmp_path / "absent.json") == {}

    def test_the_real_file_parses(self):
        """Empty by default: ESPN's report is the source, the file only corrects it."""
        assert isinstance(load_overrides(2026), dict)


@dataclass
class Snap:
    week: int
    teams: dict


@dataclass
class Seen:
    espn_id: int
    injury_status: str | None


class TestFirstSeenOnIr:
    def test_it_is_the_earliest_captured_week(self):
        snaps = [
            Snap(3, {1: [Seen(7, "INJURY_RESERVE")]}),
            Snap(2, {1: [Seen(7, "INJURY_RESERVE"), Seen(8, "OUT")]}),
            Snap(1, {1: [Seen(7, "QUESTIONABLE")]}),
        ]
        assert first_seen_on_ir(snaps) == {7: 2}

    def test_out_is_not_ir(self):
        assert first_seen_on_ir([Snap(1, {1: [Seen(8, "OUT")]})]) == {}


def test_the_minimum_is_the_nfl_rule():
    assert IR_MINIMUM_GAMES == 4
