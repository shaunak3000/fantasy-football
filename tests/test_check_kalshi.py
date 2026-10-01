"""The measurement's own logic, offline. The one property that must never break:
nothing from the week being scored reaches the prediction for it."""

import pytest

from fantasy_football.check_kalshi import _blend, _pairwise, price_ladder
from fantasy_football.data.kalshi import LadderRung, Quote


class FakeClient:
    """Prices a ladder from a dict, and counts what it was asked for."""

    def __init__(self, probs, spread=0.04):
        self.probs, self.spread, self.asked = probs, spread, []

    def quote_before(self, series, ticker, ts):
        self.asked.append(int(ticker))
        p = self.probs[int(ticker)]
        return Quote(p - self.spread / 2, p + self.spread / 2)


def ladder(*thresholds):
    return [LadderRung("X", t, "receiving", str(t), "S", "") for t in thresholds]


class TestWalkingTheLadder:
    probs = {40: 0.82, 50: 0.74, 60: 0.64, 70: 0.55, 80: 0.46, 90: 0.38, 100: 0.30}

    def test_it_finds_the_crossing(self):
        median, spread = price_ladder(FakeClient(self.probs), ladder(*self.probs), 0, 72)
        assert median == pytest.approx(75.555, 0.01) and spread == pytest.approx(0.04)

    def test_it_prices_only_the_rungs_it_needs(self):
        """Seven rungs, starting near the guess: two requests, not seven."""
        client = FakeClient(self.probs)
        price_ladder(client, ladder(*self.probs), 0, 72)
        assert sorted(client.asked) == [70, 80]

    def test_a_bad_guess_still_walks_to_the_crossing(self):
        client = FakeClient(self.probs)
        median, _ = price_ladder(client, ladder(*self.probs), 0, 5)
        assert median == pytest.approx(75.555, 0.01)

    def test_a_ladder_that_never_crosses_has_no_median(self):
        assert price_ladder(FakeClient({40: 0.9, 50: 0.8}), ladder(40, 50), 0, 45) == (None, None)


def row(week, name, stat, espn, kalshi, actual, proj_pts=10.0, act_pts=10.0):
    return dict(
        week=week,
        name=name,
        stat=stat,
        espn=espn,
        kalshi=kalshi,
        actual=actual,
        proj_pts=proj_pts,
        act_pts=act_pts,
        spread=0.02,
    )


class TestTheRankingTest:
    def test_it_scores_pairs_within_a_week_and_stat(self):
        rows = [row(1, "a", "receiving", 50, 80, 90), row(1, "b", "receiving", 60, 40, 30)]
        assert _pairwise(rows, "kalshi") == (1.0, 1)
        assert _pairwise(rows, "espn") == (0.0, 1)

    def test_different_weeks_are_never_compared(self):
        rows = [row(1, "a", "receiving", 50, 80, 90), row(2, "b", "receiving", 60, 40, 30)]
        assert _pairwise(rows, "kalshi")[1] == 0


class TestTheBlendNeverSeesItsOwnWeek:
    def test_the_offset_comes_from_other_weeks_only(self):
        """Week 1's market runs 10 yards under ESPN, week 2's 30 under. Scoring
        week 2 must subtract week 1's offset (-10), not its own (-30)."""
        rows = [
            row(1, "a", "receiving", 70, 60, 60),
            row(2, "b", "receiving", 70, 40, 40, proj_pts=10.0, act_pts=8.0),
        ]
        blended = {round(a, 3): b for _, b, a in _blend(rows, [1, 2])}
        # week 2: shift = 0.1 * (40 - 70 - (-10)) = -2.0 -> 8.0
        assert blended[8.0] == pytest.approx(8.0)

    def test_a_player_with_two_yardage_stats_gets_both(self):
        rows = [
            row(1, "rb", "rushing", 60, 70, 70),
            row(1, "rb", "receiving", 20, 30, 30),
            row(2, "z", "rushing", 50, 50, 50),
            row(2, "z", "receiving", 20, 20, 20),
        ]
        blended = _blend(rows, [1, 2])
        espn, mine, _ = next(b for b in blended if b[1] != b[0])
        # week 1: offsets from week 2 are 0, so shift = 0.1*10 + 0.1*10 = 2.0
        assert mine - espn == pytest.approx(2.0)


class TestTheRefresh:
    def rows(self):
        return [
            row(1, "a", "receiving", 70, 62, 60, proj_pts=10.0, act_pts=9.0),
            row(1, "b", "rushing", 60, 50, 55, proj_pts=12.0, act_pts=11.0),
            row(2, "c", "receiving", 50, 46, 48, proj_pts=8.0, act_pts=8.0),
            row(2, "d", "rushing", 40, 36, 30, proj_pts=9.0, act_pts=7.0),
        ]

    def test_it_writes_offsets_coverage_and_summary(self, tmp_path):
        from fantasy_football.check_kalshi import refresh_offsets

        path = tmp_path / "o.json"
        payload = refresh_offsets(2026, 2, rows=self.rows(), path=path)
        assert payload["through_week"] == 2 and payload["season"] == 2026
        assert payload["offsets"]["receiving"] == pytest.approx(-6.0)  # (-8 + -4) / 2
        assert payload["offsets"]["rushing"] == pytest.approx(-7.0)  # (-10 + -4) / 2
        assert payload["summary"]["n"] == 4 and "verdict" in payload["summary"]

    def test_an_empty_measurement_never_overwrites_a_good_file(self, tmp_path):
        from fantasy_football.check_kalshi import refresh_offsets

        path = tmp_path / "o.json"
        path.write_text('{"keep": true}')
        with pytest.raises(RuntimeError):
            refresh_offsets(2026, 3, rows=[], path=path)
        assert path.read_text() == '{"keep": true}'
