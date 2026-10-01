"""The market's second opinion on this week's projections.

`check_kalshi` measured one specific blend on weeks 1-3 of 2026 and found it
slightly better than ESPN alone: keep ESPN's projected points, but replace its
yardage with the yardage where Kalshi's ladder for that player crosses 50%.
Fantasy-point error fell from 5.61 to 5.54 per player-week, 95% interval
[-0.12, -0.02] — small, real, and mostly from running backs. This module applies
exactly that blend to the live week and nothing more ambitious.

ESPN projects a mean; a ladder's crossing is a median, and yardage is
right-skewed, so the market reads lower by construction. The measured offset for
each stat is stored by `check_kalshi` in `data/kalshi_offsets.json` and taken off
before the swap. Without that file there is no fair blend, so none is made.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

from ..config import REPO_ROOT
from ..data.injuries import normalize
from ..data.kalshi import LADDER_SERIES, KalshiClient, ladder_median, parse_rung, quote_from_market

POINTS_PER_YARD = {"passing": 0.04, "rushing": 0.1, "receiving": 0.1}
#: A rung wider than this is a market maker's placeholder, not a price.
TIGHT_SPREAD = 0.10
OFFSETS_FILE = REPO_ROOT / "data" / "kalshi_offsets.json"
ET = ZoneInfo("America/New_York")


def kickoffs(season: int) -> dict[str, list[tuple[int, int]]]:
    """Schedule team code -> [(week, kickoff unix ts)]."""
    import nflreadpy as nfl
    import polars as pl

    games: dict[str, list[tuple[int, int]]] = defaultdict(list)
    sched = nfl.load_schedules([season]).filter(pl.col("game_type") == "REG")
    for r in sched.iter_rows(named=True):
        when = datetime.fromisoformat(f"{r['gameday']}T{r['gametime']}").replace(tzinfo=ET)
        for team in (r["home_team"], r["away_team"]):
            games[team].append((int(r["week"]), int(when.timestamp())))
    return games


def week_window(games: dict[str, list[tuple[int, int]]], week: int) -> tuple[int, int] | None:
    """From the week's first kickoff to a day after its last."""
    times = [ts for slate in games.values() for w, ts in slate if w == week]
    if not times:
        return None
    return min(times), max(times) + 86400


def load_offsets_file(path=OFFSETS_FILE) -> dict:
    """The whole offsets file: the offsets, the weeks they cover, and the
    measurement summary from the last refresh."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_offsets(path=OFFSETS_FILE) -> dict[str, float]:
    return load_offsets_file(path).get("offsets", {})


def save_offsets(
    offsets: dict[str, float],
    note: str,
    path=OFFSETS_FILE,
    season: int | None = None,
    through_week: int | None = None,
    summary: dict | None = None,
) -> dict:
    payload = {
        "_note": note,
        "season": season,
        #: The last finished week measured. `weekly.py` refreshes when the league
        #: has finished a week past this one; a free-text note could not be checked.
        "through_week": through_week,
        "offsets": {k: round(v, 2) for k, v in offsets.items()},
        "summary": summary or {},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def week_to_refresh(meta: dict, season: int, live_week: int) -> int | None:
    """The finished week to measure through, or None if the file is current.

    A week is finished once the league has moved past it, so the last finished
    week is the one before the live scoring period. A file for another season,
    or one written before coverage was recorded, counts as covering nothing.
    """
    finished = live_week - 1
    if finished < 1:
        return None
    covered = meta.get("through_week") if meta.get("season") == season else None
    return finished if (covered or 0) < finished else None


def describe_offsets(meta: dict) -> str:
    """One line on what the blend's correction rests on, for every report."""
    through, summary = meta.get("through_week"), meta.get("summary") or {}
    if not through or not summary:
        return "offsets coverage unknown"
    return (
        f"measured through wk {through} (n={summary['n']}): blend "
        f"{summary['diff']:+.2f} pts/player-week vs ESPN "
        f"[{summary['lo']:+.2f}, {summary['hi']:+.2f}]"
    )


def open_medians(client: KalshiClient, window: tuple[int, int]) -> dict[tuple[str, str], float]:
    """(normalized player, stat) -> the crowd's median yards, from open markets.

    Only rungs that settle inside this week's window, and only tight quotes:
    an untraded rung can sit at 15c/94c and says nothing.
    """
    start, end = window
    ladders: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for series in LADDER_SERIES:
        for market in client.markets(series, "open"):
            rung = parse_rung(market, series)
            if rung is None or not rung.occurs:
                continue
            settles = datetime.fromisoformat(rung.occurs.replace("Z", "+00:00")).timestamp()
            if not start <= settles <= end:
                continue
            quote = quote_from_market(market)
            if quote is None or quote.spread > TIGHT_SPREAD:
                continue
            ladders[(normalize(rung.player), rung.stat)].append((rung.threshold, quote.mid))
    out = {}
    for key, points in ladders.items():
        median = ladder_median(points)
        if median is not None:
            out[key] = median
    return out


def market_adjustments(
    published: dict, medians: dict[tuple[str, str], float], offsets: dict[str, float]
) -> dict[int, float]:
    """ESPN id -> points to add to ESPN's projection, from swapping in the market's yards.

    Players whose names collide after normalisation are skipped rather than
    guessed — a wrong match moves the wrong player's projection. Players ESPN
    projects at zero (byes, ruled out) are left alone.
    """
    if not offsets:
        return {}
    counts: dict[str, int] = defaultdict(int)
    for projection in published.values():
        counts[normalize(projection.name)] += 1
    out = {}
    for espn_id, projection in published.items():
        name = normalize(projection.name)
        if projection.points <= 0 or counts[name] > 1:
            continue
        delta, used = 0.0, False
        for stat, espn_yards in projection.yards:
            median = medians.get((name, stat))
            if median is None or stat not in offsets:
                continue
            delta += POINTS_PER_YARD[stat] * (median - espn_yards - offsets[stat])
            used = True
        if used:
            out[espn_id] = round(delta, 2)
    return out
