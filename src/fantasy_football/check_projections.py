"""Build and inspect this season's projections.

uv run python -m fantasy_football.check_projections [season]
"""

from __future__ import annotations

import sys

import polars as pl

from .projections.build import build
from .projections.ensemble import to_frame


def main(argv: list[str]) -> int:
    season = int(argv[0]) if argv else 2026
    result = build(season)

    print(
        f"{result.settings.name} {season} — {result.settings.scoring_label}, "
        f"{result.settings.team_count} teams"
    )
    print(
        f"Fit on {result.training_rows} historical player-seasons; "
        f"{len(result.projections)} players projected, "
        f"{result.espn_covered} with an ESPN projection to blend\n"
    )

    print("Replacement level by position (where league demand runs out):")
    for position, points in sorted(result.replacement.items(), key=lambda kv: -kv[1]):
        depth = result.replacement_rank.get(position, 0)
        print(f"  {position:<5} rank {depth:>3} -> {points:>6.1f} pts")

    frame = to_frame(result.projections)

    print("\n=== Top 30 by value over replacement ===")
    print(frame.head(30).to_pandas().to_string(index=False))

    print("\n=== Best available at each position ===")
    for position in ("QB", "RB", "WR", "TE", "K"):
        sub = frame.filter(pl.col("pos") == position).head(3)
        if not sub.is_empty():
            print(f"\n{position}:")
            print(sub.to_pandas().to_string(index=False))

    print("\n=== Where the two sources disagree most ===")
    disagree = (
        frame.filter(pl.col("espn_rank").is_not_null())
        .with_columns((pl.col("espn_rank") - pl.col("consensus_rank")).alias("gap"))
        .filter(pl.col("consensus_rank") <= 40)
    )
    print("\nESPN much higher on:")
    print(
        disagree.sort("gap")
        .head(8)
        .select(["player", "pos", "consensus_rank", "espn_rank", "gap", "mean", "sd"])
        .to_pandas()
        .to_string(index=False)
    )
    print("\nConsensus much higher on:")
    print(
        disagree.sort("gap", descending=True)
        .head(8)
        .select(["player", "pos", "consensus_rank", "espn_rank", "gap", "mean", "sd"])
        .to_pandas()
        .to_string(index=False)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
