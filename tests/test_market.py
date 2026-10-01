"""The live market blend: exactly the version `check_kalshi` measured, applied to
this week, and nothing that could quietly move the wrong player."""

import pytest

from fantasy_football.data.espn import WeeklyProjection, projected_yards
from fantasy_football.projections.market import (
    load_offsets,
    market_adjustments,
    open_medians,
    save_offsets,
    week_window,
)

WEEK4 = (1_000_000, 1_000_000 + 5 * 86400)
INSIDE = "1970-01-12T22:00:00Z"  # 1,000,800 — inside WEEK4
LATER = "1970-01-25T00:00:00Z"  # a future week's game


def market(title, bid, ask, occurs=INSIDE):
    return {
        "title": title,
        "ticker": title,
        "occurrence_datetime": occurs,
        "yes_bid_dollars": f"{bid:.4f}",
        "yes_ask_dollars": f"{ask:.4f}",
    }


class FakeClient:
    def __init__(self, by_series):
        self.by_series = by_series

    def markets(self, series, status):
        assert status == "open"
        return self.by_series.get(series, [])


class TestReadingTheOpenMarket:
    def test_it_reads_the_crossing_from_live_quotes(self):
        client = FakeClient(
            {
                "KXNFLRSHYDS": [
                    market("Kyren Williams: 50+ rushing yards", 0.59, 0.61),
                    market("Kyren Williams: 60+ rushing yards", 0.46, 0.48),
                ]
            }
        )
        medians = open_medians(client, WEEK4)
        assert medians[("kyren williams", "rushing")] == pytest.approx(57.69, 0.01)

    def test_only_this_weeks_games_count(self):
        """Open markets include games weeks away; those are not this week's."""
        client = FakeClient(
            {
                "KXNFLRSHYDS": [
                    market("X Y: 50+ rushing yards", 0.6, 0.62, occurs=LATER),
                    market("X Y: 60+ rushing yards", 0.4, 0.42, occurs=LATER),
                ]
            }
        )
        assert open_medians(client, WEEK4) == {}

    def test_placeholder_quotes_are_ignored(self):
        """A 15c/94c rung is a market maker's placeholder, not a price."""
        client = FakeClient(
            {
                "KXNFLRECYDS": [
                    market("A B: 50+ receiving yards", 0.15, 0.94),
                    market("A B: 60+ receiving yards", 0.40, 0.42),
                ]
            }
        )
        assert open_medians(client, WEEK4) == {}


def proj(espn_id, name, points, **yards):
    return WeeklyProjection(espn_id, name, "RB", points, "ACTIVE", tuple(yards.items()))


OFFSETS = {"rushing": -2.0, "receiving": -3.0, "passing": -5.0}


class TestTheAdjustment:
    def test_market_yards_replace_espns_after_the_offset(self):
        """ESPN 60 rushing; market 68; the market typically reads 2 under, so
        +10 yards of genuine disagreement = +1.0 point."""
        adj = market_adjustments(
            {1: proj(1, "Kyren Williams", 13.0, rushing=60.0)},
            {("kyren williams", "rushing"): 68.0},
            OFFSETS,
        )
        assert adj == {1: pytest.approx(1.0)}

    def test_both_yardage_stats_of_a_back_count(self):
        adj = market_adjustments(
            {1: proj(1, "Back", 14.0, rushing=60.0, receiving=20.0)},
            {("back", "rushing"): 58.0, ("back", "receiving"): 27.0},
            OFFSETS,
        )
        # rushing 0.1*(58-60+2)=0, receiving 0.1*(27-20+3)=1.0
        assert adj == {1: pytest.approx(1.0)}

    def test_no_offsets_means_no_blend(self):
        """Without the measured offset the market reads low by construction."""
        assert (
            market_adjustments({1: proj(1, "A", 10.0, rushing=50.0)}, {("a", "rushing"): 40.0}, {})
            == {}
        )

    def test_a_name_shared_by_two_players_is_never_guessed(self):
        published = {
            1: proj(1, "Mike Williams", 10.0, receiving=50.0),
            2: proj(2, "Mike Williams", 8.0, receiving=40.0),
        }
        assert market_adjustments(published, {("mike williams", "receiving"): 90.0}, OFFSETS) == {}

    def test_a_player_espn_rules_out_is_left_alone(self):
        assert (
            market_adjustments(
                {1: proj(1, "A", 0.0, rushing=0.0)}, {("a", "rushing"): 50.0}, OFFSETS
            )
            == {}
        )


class TestPlumbing:
    def test_espn_yards_are_read_from_the_scored_id_or_its_twin(self):
        assert projected_yards({"42": 71.5, "24": 3.0}) == (("rushing", 3.0), ("receiving", 71.5))
        assert projected_yards({"61": 40.0}) == (("receiving", 40.0),)

    def test_offsets_round_trip(self, tmp_path):
        path = tmp_path / "offsets.json"
        save_offsets({"rushing": -2.345}, note="t", path=path)
        assert load_offsets(path) == {"rushing": -2.35}
        assert load_offsets(tmp_path / "missing.json") == {}

    def test_the_week_window_spans_the_slate(self):
        games = {"NE": [(4, 100), (5, 900)], "DET": [(4, 300)]}
        assert week_window(games, 4) == (100, 300 + 86400)
        assert week_window(games, 9) is None
