"""Kalshi prediction-market prices for NFL player props — read-only, public data.

ESPN's weekly projection is the only input to start/sit, and a few memorable
misses (Drake Maye, Zay Flowers, Harold Fannin) raised the question of whether a
market does better. Kalshi lists ladders of binary markets per player — "Lamb:
60+ receiving yards", "70+", "80+" — so each ladder is a distribution, and the
yardage where it crosses 50% is the crowd's median.

Nothing here trades or needs an account. Everything is the public market-data
API, cached to disk because the history endpoint is one request per market and
the exchange rate-limits (429) quickly.

**No look-ahead.** A market's `occurrence_datetime` is roughly when the game
*ends*; pricing there leaks the result. Prices are read from the hourly bid/ask
history at a time the caller chooses — one hour before kickoff, from the NFL
schedule. **No false precision.** An untraded market can sit at 15c bid / 94c
ask; every price carries its spread so the caller can drop the ones that say
nothing.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from ..config import cache_path

API = "https://api.elections.kalshi.com/trade-api/v2"

#: Yardage ladders and the stat each measures.
LADDER_SERIES = {
    "KXNFLRECYDS": "receiving",
    "KXNFLRSHYDS": "rushing",
    "KXNFLPASSYDS": "passing",
}

_TITLE = re.compile(
    r"^(?P<player>.+?):\s*(?P<threshold>\d+)\+\s*(?P<stat>receiving|rushing|passing) yards$", re.I
)


class KalshiClient:
    """GET with backoff on 429, and a disk cache for anything immutable."""

    def __init__(self, pause: float = 0.12, opener=None, sleep=time.sleep):
        self.pause = pause
        self._open = opener or (lambda url: urllib.request.urlopen(url, timeout=30))
        self._sleep = sleep

    def get(self, path: str, cache_key: str | None = None, tries: int = 8) -> dict:
        if cache_key:
            hit = cache_path("kalshi", f"{cache_key}.json")
            if hit.exists():
                return json.loads(hit.read_text(encoding="utf-8"))
        for attempt in range(tries):
            try:
                with self._open(API + path) as response:
                    payload = json.load(response)
                self._sleep(self.pause)
                if cache_key:
                    cache_path("kalshi", f"{cache_key}.json").write_text(
                        json.dumps(payload), encoding="utf-8"
                    )
                return payload
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504):
                    self._sleep(min(2.0 * (attempt + 1), 15.0))
                    continue
                raise
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                # A dropped connection is as transient as a rate limit. The first
                # full backfill died 500 ladders in on "connection reset by peer"
                # because only 429 was retried.
                self._sleep(min(2.0 * (attempt + 1), 15.0))
        raise RuntimeError(f"Kalshi kept failing {path}")

    def markets(self, series: str, status: str) -> list[dict]:
        """Every market in a series with the given status, following the cursor."""
        out, cursor = [], ""
        while True:
            suffix = f"&cursor={cursor}" if cursor else ""
            page = self.get(f"/markets?series_ticker={series}&status={status}&limit=1000{suffix}")
            out.extend(page.get("markets", []))
            cursor = page.get("cursor") or ""
            if not cursor or not page.get("markets"):
                return out

    def quote_before(self, series: str, ticker: str, ts: int) -> Quote | None:
        """Bid/ask at the last hourly close at or before `ts`. Settled history is
        immutable, so it is cached forever."""
        payload = self.get(
            f"/series/{series}/markets/{ticker}/candlesticks"
            f"?start_ts={ts - 6 * 3600}&end_ts={ts}&period_interval=60",
            cache_key=f"{ticker}@{ts}",
        )
        return quote_from_candles(payload.get("candlesticks", []), ts)


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


def quote_from_candles(candles: list[dict], ts: int) -> Quote | None:
    """The closing bid/ask of the last candle ending at or before `ts`."""
    usable = [c for c in candles if int(c.get("end_period_ts", 0)) <= ts]
    if not usable:
        return None
    last = max(usable, key=lambda c: int(c["end_period_ts"]))
    try:
        bid = float(last["yes_bid"]["close_dollars"])
        ask = float(last["yes_ask"]["close_dollars"])
    except (KeyError, TypeError, ValueError):
        return None
    if not 0 <= bid <= ask <= 1:
        return None
    return Quote(bid, ask)


def quote_from_market(market: dict) -> Quote | None:
    """The current bid/ask carried in an open-market listing — no history needed."""
    try:
        bid = float(market.get("yes_bid_dollars") or 0)
        ask = float(market.get("yes_ask_dollars") or 0)
    except (TypeError, ValueError):
        return None
    if not 0 <= bid <= ask <= 1 or ask == 0:
        return None
    return Quote(bid, ask)


@dataclass(frozen=True)
class LadderRung:
    player: str
    threshold: int
    stat: str
    ticker: str
    series: str
    occurs: str


def parse_rung(market: dict, series: str) -> LadderRung | None:
    match = _TITLE.match((market.get("title") or "").strip())
    if not match:
        return None
    return LadderRung(
        player=match["player"].strip(),
        threshold=int(match["threshold"]),
        stat=match["stat"].lower(),
        ticker=market["ticker"],
        series=series,
        occurs=market.get("occurrence_datetime") or "",
    )


def ladder_median(points: list[tuple[int, float]]) -> float | None:
    """Where P(yards >= t) crosses 50%, by linear interpolation between rungs.

    Probabilities are forced non-increasing first, since thin quotes can invert
    two neighbouring rungs. None when the ladder never brackets 50% — a median
    off the end of the ladder is a guess, not a reading.
    """
    rows = sorted(points)
    if len(rows) < 2:
        return None
    fixed, ceiling = [], 1.0
    for threshold, p in rows:
        ceiling = min(ceiling, p)
        fixed.append((threshold, ceiling))
    for (t0, p0), (t1, p1) in zip(fixed, fixed[1:], strict=False):
        if p0 >= 0.5 >= p1:
            if p0 == p1:
                return float(t0)
            return t0 + (t1 - t0) * (p0 - 0.5) / (p0 - p1)
    return None
