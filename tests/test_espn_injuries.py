"""ESPN's injury page is the only public source of return dates, and it is not an
API — the data is a JSON blob the page embeds. These tests pin the parse against
the real structure, the year inference ESPN leaves out, and the fallbacks."""

import json
from datetime import date

import pytest

from fantasy_football.data import espn_injuries
from fantasy_football.data.espn_injuries import (
    ReportEntry,
    infer_date,
    load_report,
    parse_report,
)
from fantasy_football.data.injuries import return_info


def page(groups):
    """A page in ESPN's real shape: `window['__espnfitt__']=` followed by JSON."""
    blob = {"page": {"content": {"injuries": groups}}}
    return f"<html><script>window['__espnfitt__']={json.dumps(blob)};</script></html>"


def item(pid, name, status, when):
    return {
        "athlete": {"name": name, "href": f"https://www.espn.com/nfl/player/_/id/{pid}/x"},
        "statusDesc": status,
        "date": when,
        "description": "news",
    }


REAL_SHAPE = page(
    [
        {
            "displayName": "New England Patriots",
            "logo": "https://a.espncdn.com/i/teamlogos/nfl/500/ne.png",
            "items": [item(4047646, "A.J. Brown", "Injured Reserve", "Nov 1")],
        },
        {
            "displayName": "Los Angeles Rams",
            "logo": "https://a.espncdn.com/i/teamlogos/nfl/500/lar.png",
            "items": [item(1, "Rams Guy", "Out", "Feb 10")],
        },
        {
            "displayName": "Washington Commanders",
            "logo": "https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png",
            "items": [item(2, "Commander", "Questionable", "Sep 28")],
        },
    ]
)


class TestParsing:
    def test_it_reads_the_embedded_blob(self):
        entries = {e.espn_id: e for e in parse_report(REAL_SHAPE, 2026)}
        brown = entries[4047646]
        assert brown.name == "A.J. Brown" and brown.team == "NE"
        assert brown.status == "Injured Reserve" and brown.return_date == date(2026, 11, 1)

    def test_espn_team_codes_become_schedule_codes(self):
        entries = {e.espn_id: e for e in parse_report(REAL_SHAPE, 2026)}
        assert entries[1].team == "LA" and entries[2].team == "WAS"

    def test_a_changed_page_fails_loudly(self):
        with pytest.raises(ValueError):
            parse_report("<html>no blob here</html>", 2026)
        with pytest.raises(ValueError):
            parse_report("window['__espnfitt__']={\"page\": {}}", 2026)


class TestTheMissingYear:
    def test_autumn_is_this_season(self):
        assert infer_date("Nov 1", 2026) == date(2026, 11, 1)
        assert infer_date("Sep 28", 2026) == date(2026, 9, 28)

    def test_winter_is_the_next_calendar_year(self):
        """ESPN writes "Feb" for season-ending injuries."""
        assert infer_date("Feb 10", 2026) == date(2027, 2, 10)
        assert infer_date("Jan 3", 2026) == date(2027, 1, 3)

    def test_nonsense_is_none(self):
        for text in (None, "", "TBD", "Novem 1", "Feb 30"):
            assert infer_date(text, 2026) is None


GAMES = {
    "NE": [(7, date(2026, 10, 22)), (8, date(2026, 11, 1)), (14, date(2026, 12, 10))],
    "LA": [(13, date(2026, 12, 6)), (14, date(2026, 12, 13))],
}


def reported(*entries):
    return {e.espn_id: e for e in entries}


class TestPrecedence:
    brown = ReportEntry(4047646, "A.J. Brown", "NE", "Injured Reserve", date(2026, 11, 1), "")

    def test_espn_dates_an_ir_player(self):
        info = return_info(
            "A.J. Brown", 4047646, "INJURY_RESERVE", 3, {}, {}, GAMES, reported(self.brown)
        )
        assert info.week == 8 and info.source == "ESPN est. Nov 1"

    def test_espn_beats_the_ir_floor(self):
        """The floor said week 7 for Jonathon Brooks; ESPN said 8 November."""
        info = return_info(
            "A.J. Brown",
            4047646,
            "INJURY_RESERVE",
            3,
            {},
            {4047646: 3},
            GAMES,
            reported(self.brown),
        )
        assert info.week == 8

    def test_an_override_beats_espn(self):
        overrides = {"aj brown": {"return_week": 10}}
        info = return_info(
            "A.J. Brown", 4047646, "INJURY_RESERVE", 3, overrides, {}, GAMES, reported(self.brown)
        )
        assert info.week == 10 and info.source == "injury_returns.json"

    def test_out_can_be_longer_than_a_week(self):
        """Zach Charbonnet was OUT until 11 October — not this week only."""
        entry = ReportEntry(7, "X", "NE", "Out", date(2026, 11, 1), "")
        assert return_info("X", 7, "OUT", 3, {}, {}, GAMES, reported(entry)).week == 8

    def test_questionable_players_are_assumed_to_play(self):
        entry = ReportEntry(7, "X", "NE", "Questionable", date(2026, 11, 1), "")
        assert return_info("X", 7, "QUESTIONABLE", 3, {}, {}, GAMES, reported(entry)) is None

    def test_a_date_past_the_last_game_is_out_for_the_season(self):
        entry = ReportEntry(1, "Rams Guy", "LA", "Injured Reserve", date(2027, 2, 10), "")
        info = return_info("Rams Guy", 1, "INJURY_RESERVE", 3, {}, {}, GAMES, reported(entry))
        assert info.week == 15 and info.source == "ESPN est. out for season"

    def test_espn_saying_back_already_means_available(self):
        entry = ReportEntry(7, "X", "NE", "Injured Reserve", date(2026, 10, 20), "")
        assert return_info("X", 7, "INJURY_RESERVE", 8, {}, {7: 1}, GAMES, reported(entry)) is None

    def test_without_espn_ir_falls_back_to_the_floor(self):
        info = return_info("X", 7, "INJURY_RESERVE", 3, {}, {7: 3}, GAMES, None)
        assert info.week == 7 and info.source == "IR 4-game floor"


class TestCaching:
    @pytest.fixture(autouse=True)
    def isolated_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(espn_injuries, "cache_path", lambda name: tmp_path / name)

    def test_a_fresh_cache_is_used_without_downloading(self):
        clock = [1_000_000.0]
        load_report(2026, download=lambda: REAL_SHAPE, now=lambda: clock[0])

        def must_not_download():
            raise AssertionError("should have used the cache")

        clock[0] += 3600
        report = load_report(2026, download=must_not_download, now=lambda: clock[0])
        assert report.from_cache and 4047646 in report.entries
        assert report.entries[4047646].return_date == date(2026, 11, 1)

    def test_a_stale_cache_is_refreshed(self):
        clock = [1_000_000.0]
        load_report(2026, download=lambda: REAL_SHAPE, now=lambda: clock[0])
        clock[0] += 7 * 3600
        calls = []
        load_report(2026, download=lambda: calls.append(1) or REAL_SHAPE, now=lambda: clock[0])
        assert calls == [1]

    def test_a_failed_download_raises_for_the_caller_to_handle(self):
        def offline():
            raise ConnectionError("no network")

        with pytest.raises(ConnectionError):
            load_report(2026, download=offline)
