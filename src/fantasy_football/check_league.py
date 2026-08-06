"""Verify ESPN credentials and print the league's configuration.

    uv run python -m fantasy_football.check_league

This is the first thing to run after filling in .env — it confirms the cookies
work and shows exactly what settings the rest of the repo will build against.
"""

from __future__ import annotations

import sys

from .config import load_credentials
from .data.espn import connect, fetch_raw_settings, parse_settings, snapshot_settings


def main() -> int:
    try:
        creds = load_credentials()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Connecting to league {creds.league_id}, season {creds.season}...")
    try:
        league = connect(creds)
    except Exception as exc:
        print(f"\nConnection failed: {exc}", file=sys.stderr)
        print(
            "\nIf this is a 401, the cookies are wrong or expired — pull fresh "
            "values for espn_s2 and SWID from devtools (see README). Check that "
            "SWID kept its curly braces.",
            file=sys.stderr,
        )
        return 1

    settings = parse_settings(fetch_raw_settings(league), creds.league_id, creds.season)

    print(f"\n{settings.describe()}\n")

    print("Starting lineup:")
    for slot, count in settings.starting_slots.items():
        print(f"  {slot:<10} {count}")
    print(f"  {'BENCH':<10} {settings.bench_size}")
    if settings.ir_slots:
        print(f"  {'IR':<10} {settings.ir_slots}")

    acquisitions = (
        f"FAAB ${settings.acquisition_budget}" if settings.uses_faab else "waiver priority"
    )
    print(
        f"\nKeepers: {settings.keeper_count or 'none'}"
        f" | Playoff teams: {settings.playoff_team_count}"
        f" | Acquisitions: {acquisitions}"
    )

    print(f"\nTeams ({len(league.teams)}):")
    for team in league.teams:
        print(f"  {team.team_name}")

    nonzero = [r for r in settings.scoring_rules if r.points]
    print(f"\nScoring rules in effect ({len(nonzero)}):")
    for rule in nonzero:
        print(f"  {rule.abbr:<8} {rule.points:>6}  {rule.label}")

    path = snapshot_settings(league, creds)
    print(f"\nSnapshot written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
