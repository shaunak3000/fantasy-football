"""How much did managers leave on the bench in 2025?

    uv run python -m fantasy_football.check_lineups [season]

The question that decides whether a lineup optimizer is worth having. If every
manager already starts near their best possible lineup, the optimizer is a
rounding error and the effort belongs elsewhere. If they routinely bench their
best players, that is a standing edge available every single week.

This measures the ceiling, not our skill: it compares what each manager started
against the best lineup their roster could have produced *with hindsight*. No
tool achieves that. But it bounds the prize, and a prize worth less than the
noise is not worth chasing.
"""

from __future__ import annotations

import sys
from collections import defaultdict

from espn_api.football import League

from .config import load_credentials
from .data.espn import ACTUAL_STAT_SOURCE, WEEKLY_SPLIT, fetch_raw_settings, parse_settings
from .draft.lineup_value import FLEX_ELIGIBILITY, FLEX_SLOTS
from .projections.scoring import ScoringEngine

BENCH_SLOTS = {20, 21, 24}  # bench, IR, and the "everything else" slot
POSITION_BY_ID = {0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "D/ST", 17: "K", 23: "RB/WR/TE"}


def _starting_requirements(settings) -> list[tuple[str, tuple[str, ...]]]:
    slots = []
    for name, count in settings.starting_slots.items():
        eligible = FLEX_ELIGIBILITY.get(name, (name,)) if name in FLEX_SLOTS else (name,)
        slots.extend([(name, tuple(eligible))] * int(count))
    return slots


def _best_possible(players: list[tuple[str, float]], requirements) -> float:
    """Highest-scoring legal lineup, chosen with hindsight."""
    remaining = sorted(players, key=lambda row: -row[1])
    total = 0.0
    used = set()
    # Fill the most restrictive slots first so a flex cannot steal a player a
    # dedicated slot needed.
    for _, eligible in sorted(requirements, key=lambda r: len(r[1])):
        for index, (position, points) in enumerate(remaining):
            if index in used or position not in eligible:
                continue
            total += points
            used.add(index)
            break
    return total


def main(argv: list[str]) -> int:
    season = int(argv[0]) if argv else 2025
    creds = load_credentials()
    league = League(league_id=creds.league_id, year=season, espn_s2=creds.espn_s2, swid=creds.swid)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, season)
    engine = ScoringEngine.from_settings(settings)
    requirements = _starting_requirements(settings)

    names = {team.team_id: team.team_name for team in league.teams}
    started = defaultdict(float)
    optimal = defaultdict(float)
    weeks_counted = 0

    for week in range(1, settings.regular_season_weeks + 1):
        try:
            payload = league.espn_request.league_get(
                params={"view": "mRoster", "scoringPeriodId": week}
            )
        except Exception as exc:
            print(f"  week {week}: fetch failed ({type(exc).__name__})")
            continue

        any_data = False
        for team in payload.get("teams", []):
            team_id = team.get("id")
            roster_rows: list[tuple[str, float]] = []
            week_started = 0.0

            for entry in team.get("roster", {}).get("entries", []):
                slot_id = entry.get("lineupSlotId")
                player = entry.get("playerPoolEntry", {}).get("player", {})
                actual = next(
                    (
                        s
                        for s in player.get("stats", [])
                        if s.get("statSourceId") == ACTUAL_STAT_SOURCE
                        and s.get("statSplitTypeId") == WEEKLY_SPLIT
                        and s.get("scoringPeriodId") == week
                    ),
                    None,
                )
                if actual is None:
                    continue
                stats = {int(k): float(v) for k, v in (actual.get("stats") or {}).items()}
                if not stats:
                    continue

                points = engine.score(stats, player.get("defaultPositionId", -1))
                position = POSITION_BY_ID.get(player.get("defaultPositionId", -1))
                if position is None:
                    continue

                any_data = True
                roster_rows.append((position, points))
                if slot_id not in BENCH_SLOTS:
                    week_started += points

            if roster_rows:
                started[team_id] += week_started
                optimal[team_id] += _best_possible(roster_rows, requirements)

        if any_data:
            weeks_counted += 1

    if not weeks_counted:
        print("No weekly roster data available.")
        return 1

    print(f"\n{settings.name} {season} - points left on the bench, {weeks_counted} weeks\n")
    print(f"  {'team':<30}{'started':>10}{'best':>10}{'left':>9}{'/week':>8}")

    losses = []
    for team_id in sorted(started, key=lambda t: optimal[t] - started[t], reverse=True):
        lost = optimal[team_id] - started[team_id]
        losses.append(lost)
        print(
            f"  {names.get(team_id, team_id):<30}{started[team_id]:>10.0f}"
            f"{optimal[team_id]:>10.0f}{lost:>9.0f}{lost / weeks_counted:>8.1f}"
        )

    average = sum(losses) / len(losses)
    print(
        f"\n  League average: {average:.0f} points left on the bench "
        f"({average / weeks_counted:.1f} per week)"
    )
    print("\n  This is the hindsight ceiling, not an achievable target - nobody knows")
    print("  in advance which bench player will go off. But it bounds the prize:")
    print("  an optimizer that captures even a quarter of it is worth having.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
