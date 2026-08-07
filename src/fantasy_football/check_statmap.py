"""Verify that nflverse stats, scored by our engine, agree with ESPN's own points.

    uv run python -m fantasy_football.check_statmap [season]

Step 5 proved the engine reproduces ESPN given ESPN's stats. This proves the
nflverse translation feeds it the same numbers, which is what makes it safe to
fit projections on nflverse's complete NFL history instead of the handful of
players who happened to be rostered in this league.
"""

from __future__ import annotations

import sys

import polars as pl
from espn_api.football import League

from .config import load_credentials
from .data.espn import fetch_player_weeks, fetch_raw_settings, parse_settings
from .data.ids import attach_espn_ids
from .data.nflverse import load_weekly_stats
from .data.statmap import UNMAPPED_STAT_IDS, score_frame
from .projections.scoring import ScoringEngine

TOLERANCE = 0.05
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")


def main(argv: list[str]) -> int:
    season = int(argv[0]) if argv else 2025
    creds = load_credentials()
    league = League(league_id=creds.league_id, year=season, espn_s2=creds.espn_s2, swid=creds.swid)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, season)
    engine = ScoringEngine.from_settings(settings)

    print(f"{settings.name} {season} — scoring nflverse stats with the league's own rules\n")

    weekly = load_weekly_stats([season]).filter(
        (pl.col("season_type") == "REG") & pl.col("position").is_in(SKILL_POSITIONS)
    )
    scored = score_frame(weekly, engine).select(
        ["player_display_name", "position", "team", "week", "fantasy_points"]
    )
    print(f"nflverse: {scored.height} skill-position player-weeks scored")

    report = attach_espn_ids(
        scored, name_col="player_display_name", pos_col="position", team_col="team"
    )
    mine = report.matched.filter(pl.col("espn_id").is_not_null()).select(
        ["espn_id", "week", "fantasy_points", "player_display_name", "position"]
    )

    espn_rows = []
    for week in range(1, 18):
        try:
            for pw in fetch_player_weeks(league, week):
                if pw.stats:
                    espn_rows.append((pw.espn_id, week, pw.applied_total))
        except Exception as exc:
            print(f"  week {week}: ESPN fetch failed ({type(exc).__name__})")

    espn = pl.DataFrame(espn_rows, schema=["espn_id", "week", "espn_points"], orient="row")
    print(f"ESPN:     {espn.height} rostered player-weeks to compare against\n")

    joined = mine.join(espn, on=["espn_id", "week"], how="inner").with_columns(
        (pl.col("fantasy_points") - pl.col("espn_points")).alias("diff")
    )
    if joined.is_empty():
        print("No overlapping player-weeks — cannot verify.")
        return 1

    agree = joined.filter(pl.col("diff").abs() <= TOLERANCE).height
    rate = 100 * agree / joined.height
    print(f"{agree}/{joined.height} player-weeks agree within {TOLERANCE} ({rate:.2f}%)")

    # The direction of the error is what matters more than its frequency. Every
    # unmapped rule is a *bonus*, so a correct mapping can only ever fall short.
    # A single point scored above ESPN would mean a stat is being double-counted,
    # which is a real defect rather than a known gap.
    over = joined.filter(pl.col("diff") > TOLERANCE)
    if over.is_empty():
        print("No player-week scores above ESPN — nothing is double-counted.")
    else:
        print(f"\nDEFECT: {over.height} player-weeks scored ABOVE ESPN (should be impossible):")
        print(
            over.sort("diff", descending=True)
            .head(10)
            .select(
                ["player_display_name", "position", "week", "fantasy_points", "espn_points", "diff"]
            )
            .to_pandas()
            .to_string(index=False)
        )

    shortfall = joined.filter(pl.col("diff") < -TOLERANCE)["diff"].sum()
    n_players = joined["espn_id"].n_unique()
    print(
        f"Total shortfall {shortfall:.1f} pts across {n_players} players "
        f"= {abs(shortfall) / max(n_players, 1):.2f} pts per player-season"
    )

    disagree = joined.filter(pl.col("diff").abs() > TOLERANCE)
    if not disagree.is_empty():
        print("\nBy position:")
        by_pos = (
            disagree.group_by("position")
            .agg(pl.len().alias("n"), pl.col("diff").abs().mean().round(2).alias("mean_abs_diff"))
            .sort("n", descending=True)
        )
        print(by_pos.to_pandas().to_string(index=False))

        print("\nLargest disagreements:")
        worst = disagree.sort(pl.col("diff").abs(), descending=True).head(12)
        print(
            worst.select(
                ["player_display_name", "position", "week", "fantasy_points", "espn_points", "diff"]
            )
            .to_pandas()
            .to_string(index=False)
        )

    print(f"\nKnown unmapped rules ({len(UNMAPPED_STAT_IDS)}), each worth 1-6 points and rare:")
    for stat_id, why in list(UNMAPPED_STAT_IDS.items())[:4]:
        print(f"  {stat_id}: {why}")

    passed = over.is_empty() and rate >= 97.0
    print("\nGate passed." if passed else "\nGate FAILED.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
