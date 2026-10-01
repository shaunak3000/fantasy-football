"""Does the Kalshi crowd project players better than ESPN?

    uv run python -m fantasy_football.check_kalshi [season]

ESPN's weekly projection is the only input to start/sit. Kalshi lists ladders of
yardage markets per player, and the yardage where a ladder crosses 50% is the
crowd's median. This replays every finished week and asks three questions:

1. **Yards against yards.** ESPN publishes projected yards as well as points, so
   the fairest test is whose yardage estimate was closer to what happened.
2. **Order.** Within a stat and week, how often does each put two players in the
   right order? Scale-free, so the mean-versus-median difference cannot help.
3. **The blend that would be used.** Keep ESPN's points but swap its yardage for
   the market's, and see whether fantasy points get more predictable.

Fairness. ESPN projects a mean; a ladder's crossing is a median, and yardage is
right-skewed, so a median wins on absolute error by construction and loses on
squared error. Both are reported. The blend corrects for the gap with an offset
learned only from the *other* weeks. Every market is priced an hour before
kickoff; Kalshi's `occurrence_datetime` is roughly the end of the game and would
leak the result.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime

from espn_api.football import League

from .config import load_credentials
from .data.espn import PLAYER_POSITION_BY_ID, YARD_IDS
from .data.injuries import normalize
from .data.kalshi import LADDER_SERIES, KalshiClient, ladder_median, parse_rung
from .projections.market import (
    OFFSETS_FILE,
    POINTS_PER_YARD,
    TIGHT_SPREAD,
    kickoffs,
    save_offsets,
)


def _stat(stats: dict, ids: tuple[int, ...]) -> float | None:
    for key in ids:
        if str(key) in stats:
            return float(stats[str(key)])
    return None


def espn_week(league: League, week: int) -> dict[str, dict]:
    """Normalized name -> projected and actual yards and points, for one week."""
    out = {}
    for status in (["ONTEAM"], ["FREEAGENT", "WAIVERS"]):
        body = {
            "players": {
                "filterStatus": {"value": status},
                "limit": 700,
                "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
            }
        }
        data = league.espn_request.league_get(
            params={"view": "kona_player_info", "scoringPeriodId": week},
            headers={"x-fantasy-filter": json.dumps(body)},
        )
        for entry in data.get("players", []):
            p = entry.get("player") or {}
            blocks = {
                s.get("statSourceId"): s
                for s in p.get("stats", [])
                if s.get("statSplitTypeId") == 1 and s.get("scoringPeriodId") == week
            }
            proj, act = blocks.get(1), blocks.get(0)
            if not proj or not act or not act.get("stats"):
                continue
            row = {
                "name": p.get("fullName", ""),
                "position": PLAYER_POSITION_BY_ID.get(p.get("defaultPositionId", -1)),
                "team": p.get("proTeamId"),
                "proj_pts": float(proj.get("appliedTotal") or 0.0),
                "act_pts": float(act.get("appliedTotal") or 0.0),
            }
            for stat, ids in YARD_IDS.items():
                row[f"proj_{stat}"] = _stat(proj.get("stats") or {}, ids)
                row[f"act_{stat}"] = _stat(act.get("stats") or {}, ids)
            out[normalize(row["name"])] = row
    return out


def price_ladder(client: KalshiClient, rungs: list, cutoff: int, guess: float | None):
    """Price just enough rungs to find where the ladder crosses 50%.

    Starts at the rung nearest `guess` and walks toward the crossing, so a
    ten-rung ladder usually costs two or three history requests, not ten.
    Returns (median, widest spread among the rungs used) or (None, None).
    """
    rungs = sorted(rungs, key=lambda r: r.threshold)
    if len(rungs) < 2:
        return None, None
    start = min(range(len(rungs)), key=lambda i: abs(rungs[i].threshold - (guess or 0)))
    priced: dict[int, tuple[float, float]] = {}

    def price(i):
        if i not in priced:
            q = client.quote_before(rungs[i].series, rungs[i].ticker, cutoff)
            priced[i] = (q.mid, q.spread) if q else (None, None)
        return priced[i][0]

    i = start
    p = price(i)
    if p is None:
        return None, None
    step = 1 if p >= 0.5 else -1
    while 0 <= i + step < len(rungs):
        nxt = price(i + step)
        if nxt is None:
            return None, None
        if (step == 1 and nxt < 0.5) or (step == -1 and nxt >= 0.5):
            break
        i += step
    points = [(rungs[k].threshold, v[0]) for k, v in priced.items() if v[0] is not None]
    median = ladder_median(points)
    if median is None:
        return None, None
    used = [v[1] for v in priced.values() if v[1] is not None]
    return median, max(used) if used else None


def collect(season: int, weeks: list[int]) -> list[dict]:
    creds = load_credentials()
    league = League(league_id=creds.league_id, year=season, espn_s2=creds.espn_s2, swid=creds.swid)
    from espn_api.football.constant import PRO_TEAM_MAP

    alias = {"WSH": "WAS", "LAR": "LA"}
    games = kickoffs(season)
    client = KalshiClient()
    espn = {w: espn_week(league, w) for w in weeks}

    ladders = defaultdict(list)  # (name, stat, occurs-date) -> rungs
    for series in LADDER_SERIES:
        for market in client.markets(series, "settled"):
            rung = parse_rung(market, series)
            if rung:
                ladders[(normalize(rung.player), rung.stat, rung.occurs[:10])].append(rung)

    rows = []
    total = len(ladders)
    for n, ((name, stat, _occurs), rungs) in enumerate(ladders.items(), 1):
        if n % 100 == 0:
            print(f"  ...priced {n}/{total} ladders", file=sys.stderr)
        for week in weeks:
            player = espn[week].get(name)
            if not player or player.get(f"act_{stat}") is None:
                continue
            team = PRO_TEAM_MAP.get(player["team"], "")
            team = alias.get(team, team)
            game = next((ts for w, ts in games.get(team, []) if w == week), None)
            if game is None:
                continue
            # The market belongs to this game if it settles within a day of kickoff.
            ends = datetime.fromisoformat(rungs[0].occurs.replace("Z", "+00:00")).timestamp()
            if not (game <= ends <= game + 86400):
                continue
            median, spread = price_ladder(client, rungs, game - 3600, player.get(f"proj_{stat}"))
            if median is None:
                continue
            rows.append(
                {
                    "week": week,
                    "name": player["name"],
                    "position": player["position"],
                    "stat": stat,
                    "kalshi": median,
                    "spread": spread,
                    "espn": player.get(f"proj_{stat}"),
                    "actual": player[f"act_{stat}"],
                    "proj_pts": player["proj_pts"],
                    "act_pts": player["act_pts"],
                }
            )
    return rows


def _err(rows, key, kind):
    errs = [r[key] - r["actual"] for r in rows]
    if kind == "mae":
        return sum(abs(e) for e in errs) / len(errs)
    return math.sqrt(sum(e * e for e in errs) / len(errs))


def _pairwise(rows, key):
    right = total = 0
    groups = defaultdict(list)
    for r in rows:
        groups[(r["week"], r["stat"])].append(r)
    for group in groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a["actual"] == b["actual"] or a[key] == b[key]:
                    continue
                total += 1
                right += (a[key] > b[key]) == (a["actual"] > b["actual"])
    return right / total if total else float("nan"), total


def _blend(rows, weeks):
    """Fantasy points with ESPN's yardage swapped for the market's.

    The median-below-mean offset for each week is learned from the other weeks
    only, so the correction never sees the week it is scoring.
    """
    out = []
    for week in weeks:
        train = [r for r in rows if r["week"] != week]
        offset = {
            stat: (
                sum(r["kalshi"] - r["espn"] for r in train if r["stat"] == stat)
                / max(1, sum(1 for r in train if r["stat"] == stat))
            )
            for stat in POINTS_PER_YARD
        }
        by_player = defaultdict(list)
        for r in rows:
            if r["week"] == week:
                by_player[r["name"]].append(r)
        for parts in by_player.values():
            shift = sum(
                POINTS_PER_YARD[p["stat"]] * (p["kalshi"] - p["espn"] - offset[p["stat"]])
                for p in parts
            )
            first = parts[0]
            out.append((first["proj_pts"], first["proj_pts"] + shift, first["act_pts"]))
    return out


def report(rows: list[dict], weeks: list[int]) -> None:
    rows = [r for r in rows if r["espn"] is not None]
    tight = [r for r in rows if r["spread"] is not None and r["spread"] <= TIGHT_SPREAD]
    print(f"\nKalshi vs ESPN — weeks {weeks[0]}-{weeks[-1]}, priced an hour before kickoff\n")
    print(
        f"  player-weeks matched: {len(rows)}"
        f"   (tight quotes, spread <= {TIGHT_SPREAD:.0%}: {len(tight)})\n"
    )
    print("  1. YARDS — whose estimate was closer?        ESPN     Kalshi")
    for label, subset in (("all", rows), ("tight quotes only", tight)):
        for stat in ("receiving", "rushing", "passing"):
            s = [r for r in subset if r["stat"] == stat]
            if len(s) < 10:
                continue
            print(
                f"     {label:<18}{stat:<10} n={len(s):<4}"
                f" MAE  {_err(s, 'espn', 'mae'):6.1f}  {_err(s, 'kalshi', 'mae'):7.1f}"
                f"    RMSE {_err(s, 'espn', 'rmse'):6.1f}  {_err(s, 'kalshi', 'rmse'):7.1f}"
            )
    print("\n  2. ORDER — share of player pairs put in the right order")
    for label, subset in (("all", rows), ("tight quotes only", tight)):
        e, n = _pairwise(subset, "espn")
        k, _ = _pairwise(subset, "kalshi")
        print(f"     {label:<18} ESPN {e:.1%}   Kalshi {k:.1%}   ({n:,} pairs)")
    summary = blend_summary(rows, weeks)
    if summary:
        print("\n  3. FANTASY POINTS — ESPN's points with the market's yardage swapped in")
        print(
            f"     n={summary['n']} player-weeks   MAE  ESPN {summary['espn_mae']:.2f}"
            f"   blend {summary['blend_mae']:.2f}"
        )
        print(
            f"     blend minus ESPN: {summary['diff']:+.2f} pts per player-week,"
            f" 95% interval [{summary['lo']:+.2f}, {summary['hi']:+.2f}]"
        )
        print(f"     -> {summary['verdict']}")


def blend_summary(rows: list[dict], weeks: list[int]) -> dict | None:
    """The fantasy-point test as numbers: error of each, their gap, and its interval."""
    blend = _blend([r for r in rows if r["espn"] is not None], weeks)
    if not blend:
        return None
    d = [abs(b - a) - abs(e - a) for e, b, a in blend]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    lo, hi = boots[50], boots[1949]
    return {
        "n": len(blend),
        "espn_mae": round(sum(abs(e - a) for e, _, a in blend) / len(blend), 3),
        "blend_mae": round(sum(abs(b - a) for _, b, a in blend) / len(blend), 3),
        "diff": round(sum(d) / len(d), 3),
        "lo": round(lo, 3),
        "hi": round(hi, 3),
        "verdict": (
            "the blend is better"
            if hi < 0
            else "ESPN is better"
            if lo > 0
            else "no measurable difference yet"
        ),
    }


def offsets_from(rows: list[dict]) -> dict[str, float]:
    """The median-below-mean gap per stat, from every measured week."""
    usable = [r for r in rows if r["espn"] is not None]
    return {
        stat: sum(r["kalshi"] - r["espn"] for r in usable if r["stat"] == stat)
        / sum(1 for r in usable if r["stat"] == stat)
        for stat in POINTS_PER_YARD
        if any(r["stat"] == stat for r in usable)
    }


def refresh_offsets(
    season: int, through_week: int, rows: list[dict] | None = None, path=None
) -> dict:
    """Measure weeks 1..through_week and rewrite the offsets file.

    Past prices are cached, so a weekly refresh fetches only the newest week.
    Raises if nothing could be matched, so the caller keeps the old file rather
    than overwriting a good correction with an empty one.
    """
    weeks = list(range(1, through_week + 1))
    rows = rows if rows is not None else collect(season, weeks)
    if not rows:
        raise RuntimeError("no player-weeks matched to priced Kalshi ladders")
    return save_offsets(
        offsets_from(rows),
        note=f"check_kalshi {season}, weeks 1-{through_week}, n={len(rows)}",
        season=season,
        through_week=through_week,
        summary=blend_summary(rows, weeks),
        path=path or OFFSETS_FILE,
    )


def main(argv: list[str]) -> int:
    season = int(argv[0]) if argv else 2026
    last = int(argv[1]) if len(argv) > 1 else 3
    weeks = list(range(1, last + 1))
    rows = collect(season, weeks)
    if not rows:
        print("No player-weeks could be matched to priced Kalshi ladders.")
        return 1
    report(rows, weeks)
    payload = refresh_offsets(season, last, rows=rows)
    print(
        "\n  offsets saved (market median minus ESPN mean, yards): "
        + ", ".join(f"{k} {v:+.1f}" for k, v in payload["offsets"].items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
