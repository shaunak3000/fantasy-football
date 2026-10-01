"""Kalshi is read for one number per player-week: the crowd's median yards, priced
before kickoff. Each step that produces it is pinned here, offline."""

import io
import json
import urllib.error

import pytest

from fantasy_football.data import kalshi
from fantasy_football.data.kalshi import (
    KalshiClient,
    Quote,
    ladder_median,
    parse_rung,
    quote_from_candles,
)


class TestTheMedian:
    def test_it_interpolates_where_the_ladder_crosses_half(self):
        # 60+: 64%, 70+: 55%, 80+: 46% -> crosses 50% just past 75
        assert ladder_median([(60, 0.64), (70, 0.55), (80, 0.46)]) == pytest.approx(75.555, 0.01)

    def test_rung_order_does_not_matter(self):
        assert ladder_median([(80, 0.46), (60, 0.64), (70, 0.55)]) == pytest.approx(75.555, 0.01)

    def test_an_inverted_thin_quote_is_smoothed_not_trusted(self):
        """70+ quoted above 60+ is impossible; the ladder is forced downhill."""
        assert ladder_median([(60, 0.60), (70, 0.70), (80, 0.40)]) == pytest.approx(75.0)

    def test_a_ladder_that_never_brackets_half_has_no_median(self):
        assert ladder_median([(60, 0.80), (70, 0.70)]) is None
        assert ladder_median([(60, 0.40), (70, 0.30)]) is None
        assert ladder_median([(60, 0.5)]) is None


class TestPricingBeforeKickoff:
    def candle(self, end, bid, ask):
        return {
            "end_period_ts": end,
            "yes_bid": {"close_dollars": f"{bid:.4f}"},
            "yes_ask": {"close_dollars": f"{ask:.4f}"},
        }

    def test_it_takes_the_last_close_at_or_before_the_cutoff(self):
        candles = [
            self.candle(100, 0.40, 0.44),
            self.candle(200, 0.50, 0.52),
            self.candle(300, 0.9, 0.95),
        ]
        assert quote_from_candles(candles, 250) == Quote(0.50, 0.52)

    def test_nothing_after_the_cutoff_leaks_in(self):
        """The candle at 300 has the game in it. It must never be read."""
        assert quote_from_candles([self.candle(300, 0.9, 0.95)], 250) is None

    def test_spread_and_mid(self):
        q = Quote(0.15, 0.94)
        assert q.mid == pytest.approx(0.545) and q.spread == pytest.approx(0.79)

    def test_a_malformed_candle_is_skipped(self):
        assert quote_from_candles([{"end_period_ts": 100, "yes_bid": {}}], 200) is None


class TestTitles:
    def test_a_ladder_title_parses(self):
        rung = parse_rung(
            {
                "title": "CeeDee Lamb: 70+ receiving yards",
                "ticker": "T",
                "occurrence_datetime": "2026-09-29T03:15:00Z",
            },
            "KXNFLRECYDS",
        )
        assert (rung.player, rung.threshold, rung.stat) == ("CeeDee Lamb", 70, "receiving")

    def test_names_with_punctuation_survive(self):
        rung = parse_rung({"title": "A.J. Brown: 100+ receiving yards", "ticker": "T"}, "S")
        assert rung.player == "A.J. Brown"

    def test_other_titles_are_ignored(self):
        assert parse_rung({"title": "Los Angeles R wins", "ticker": "T"}, "S") is None


class TestTheClient:
    def test_it_backs_off_on_rate_limits_and_retries(self, tmp_path, monkeypatch):
        monkeypatch.setattr(kalshi, "cache_path", lambda *p: tmp_path.joinpath(*p))
        calls, slept = [], []

        def opener(url):
            calls.append(url)
            if len(calls) < 3:
                raise urllib.error.HTTPError(url, 429, "slow down", {}, None)
            return io.BytesIO(json.dumps({"ok": True}).encode())

        client = KalshiClient(opener=opener, sleep=slept.append)
        assert client.get("/x") == {"ok": True}
        assert len(calls) == 3 and slept[:2] == [2.0, 4.0]

    def test_cached_history_is_never_refetched(self, tmp_path, monkeypatch):
        (tmp_path / "kalshi").mkdir()
        monkeypatch.setattr(kalshi, "cache_path", lambda *p: tmp_path.joinpath(*p))
        (tmp_path / "kalshi" / "k.json").write_text(json.dumps({"cached": 1}))

        def no_network(url):
            raise AssertionError("should have used the cache")

        assert KalshiClient(opener=no_network).get("/x", cache_key="k") == {"cached": 1}

    def test_a_dropped_connection_is_retried_not_fatal(self, tmp_path, monkeypatch):
        """The first backfill died 500 ladders in on a connection reset."""
        monkeypatch.setattr(kalshi, "cache_path", lambda *p: tmp_path.joinpath(*p))
        calls = []

        def opener(url):
            calls.append(url)
            if len(calls) == 1:
                raise urllib.error.URLError(ConnectionResetError(54, "reset by peer"))
            if len(calls) == 2:
                raise TimeoutError("timed out")
            return io.BytesIO(json.dumps({"ok": True}).encode())

        assert KalshiClient(opener=opener, sleep=lambda s: None).get("/x") == {"ok": True}
        assert len(calls) == 3

    def test_a_real_client_error_is_not_retried(self, tmp_path, monkeypatch):
        monkeypatch.setattr(kalshi, "cache_path", lambda *p: tmp_path.joinpath(*p))

        def opener(url):
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)

        with pytest.raises(urllib.error.HTTPError):
            KalshiClient(opener=opener, sleep=lambda s: None).get("/x")
