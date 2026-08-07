"""Validate the scoring engine against ESPN's own 2025 results.

    uv run python -m fantasy_football.check_scoring [season] [max_week]

The gate for the whole repo: recompute every player-week from raw stats using
only the league's settings, and match ESPN's published total. Until this passes,
no projection built on top of the engine means anything.
"""

from __future__ import annotations

import sys
from collections import Counter

from espn_api.football import League
from espn_api.football.constant import SETTINGS_SCORING_FORMAT_MAP

from .config import load_credentials
from .data.espn import fetch_player_weeks, fetch_raw_settings, parse_settings
from .projections.scoring import ScoringEngine, find_double_counted_stats

TOLERANCE = 0.01


def stat_name(stat_id: int) -> str:
    meta = SETTINGS_SCORING_FORMAT_MAP.get(stat_id)
    return f"{meta['abbr']} ({meta['label']})" if meta else f"stat {stat_id}"


def main(argv: list[str]) -> int:
    season = int(argv[0]) if argv else 2025
    max_week = int(argv[1]) if len(argv) > 1 else 17

    creds = load_credentials()
    league = League(league_id=creds.league_id, year=season, espn_s2=creds.espn_s2, swid=creds.swid)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, season)
    engine = ScoringEngine.from_settings(settings)

    print(f"{settings.name} {season} — {settings.scoring_label}, {len(engine.rules)} scoring rules")

    doubles = find_double_counted_stats(engine)
    if doubles:
        print(f"WARNING: league scores both ids of a duplicated stat: {doubles}")

    checked = mismatched = 0
    worst: list[tuple[float, str, int, float, float]] = []
    culprits: Counter[int] = Counter()

    for week in range(1, max_week + 1):
        try:
            player_weeks = fetch_player_weeks(league, week)
        except Exception as exc:
            print(f"  week {week}: fetch failed ({type(exc).__name__}), skipping")
            continue

        for pw in player_weeks:
            # Players on a bye or inactive produce an empty stat line; ESPN
            # reports 0.0 and there is nothing to verify.
            if not pw.stats:
                continue

            expected = pw.applied_total
            actual = engine.score(pw.stats, pw.position_id)
            checked += 1

            if abs(actual - expected) > TOLERANCE:
                mismatched += 1
                worst.append((abs(actual - expected), pw.name, week, expected, actual))
                mine = engine.explain(pw.stats, pw.position_id)
                theirs = pw.applied_stats
                for sid in set(mine) | set(theirs):
                    if abs(mine.get(sid, 0.0) - theirs.get(sid, 0.0)) > TOLERANCE:
                        culprits[sid] += 1

        print(f"  week {week:>2}: {checked:>5} checked, {mismatched} mismatched")

    rate = 100 * (checked - mismatched) / checked if checked else 0.0
    print(f"\n{checked - mismatched}/{checked} player-weeks reproduced exactly ({rate:.2f}%)")

    if mismatched:
        print("\nStats responsible for the disagreements:")
        for sid, n in culprits.most_common(10):
            print(f"  {n:>5}x  {stat_name(sid)}")
        print("\nLargest discrepancies:")
        for diff, name, week, expected, actual in sorted(worst, reverse=True)[:10]:
            print(
                f"  {name:<26} wk{week:<3} espn={expected:>8.2f} mine={actual:>8.2f} ({diff:+.2f})"
            )
        return 1

    print("Gate passed — the engine reproduces ESPN exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
