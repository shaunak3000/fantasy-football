"""What weekly lineup data do we actually have?

    uv run python -m fantasy_football.check_snapshots [season]

ESPN serves only the current week's roster, so the lineups managers set can be
recorded as the season runs or not at all — see `data/snapshots.py`. This says
which weeks are in the bag, which are missing and gone, and which still have
results to collect.

A gap in this table is permanent. It is the difference between being able to
settle whether the lineup optimizer beats the league and never being able to.
"""

from __future__ import annotations

import sys

from .config import load_credentials
from .data.snapshots import load_season

# The season runs to 14 matchup periods in 2026 and 13 in 2025; asking the API
# would need credentials, and this command is meant to work offline.
LIKELY_LAST_WEEK = 14


def main(argv: list[str]) -> int:
    season = int(argv[0]) if argv else 2026
    try:
        league_name = str(load_credentials().league_id)
    except Exception:  # noqa: BLE001 - coverage should be readable without credentials
        league_name = "?"

    weeks = {snapshot.week: snapshot for snapshot in load_season(season)}
    print(f"\nWeekly lineup snapshots — league {league_name}, {season}\n")

    if not weeks:
        print("  NONE CAPTURED. Every week that passes without a snapshot is")
        print("  permanently unrecoverable. Run `weekly.py` to start collecting.")
        return 1

    print(f"  {'week':>6}{'rosters':>9}{'scored':>8}{'lineups':>10}  captured")
    for week in sorted(weeks):
        snapshot = weeks[week]
        scored = sum(
            1 for players in snapshot.teams.values() for p in players if p.actual is not None
        )
        quality = "real" if snapshot.trustworthy else "STALE"
        print(
            f"  {week:>6}{len(snapshot.teams):>9}{scored:>8}{quality:>10}  {snapshot.captured_at}"
        )

    captured = sorted(weeks)
    missing = [w for w in range(1, max(captured) + 1) if w not in weeks]
    print(f"\n  {len(captured)} week(s) captured, through week {max(captured)}.")
    if missing:
        print(f"  MISSING AND UNRECOVERABLE: {', '.join(str(w) for w in missing)}")
    # `has_results` is true once anybody has scored, which a Thursday capture
    # satisfies while missing 100+ stat lines. What matters is whether the
    # scores were pulled after the week ended — see `results_final`.
    pending = sorted(w for w, s in weeks.items() if not s.results_final)
    if pending:
        print(f"  Still awaiting results (rerun weekly.py to fill in): {pending}")
    if max(captured) < LIKELY_LAST_WEEK:
        print(
            f"  Weeks {max(captured) + 1}-{LIKELY_LAST_WEEK} still to come "
            f"— keep running weekly.py."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
