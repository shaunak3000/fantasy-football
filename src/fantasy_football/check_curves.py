"""Fit and inspect the preseason-rank -> points curves.

    uv run python -m fantasy_football.check_curves

Shows what the consensus rank has historically been worth at each position, how
wide the outcomes are around it, and where replacement level sits given this
league's actual starting requirements.
"""

from __future__ import annotations

import polars as pl
from espn_api.football import League

from .config import load_credentials
from .data.espn import fetch_raw_settings, parse_settings
from .projections.curve import fit_all, replacement_points, replacement_ranks
from .projections.history import training_table
from .projections.scoring import ScoringEngine

TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]


def main() -> int:
    creds = load_credentials()
    league = League(league_id=creds.league_id, year=2026, espn_s2=creds.espn_s2, swid=creds.swid)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, 2026)
    engine = ScoringEngine.from_settings(settings)

    print(f"Training on {TRAIN_SEASONS} under {settings.name} scoring ({settings.scoring_label})\n")
    training, coverage = training_table(TRAIN_SEASONS, engine)
    for cov in coverage:
        print(cov.line())

    if training.is_empty():
        print("\nNo training data assembled.")
        return 1

    print(f"\nTraining rows: {training.height}")
    zeros = training.filter(pl.col("games_played") == 0).height
    print(
        f"Ranked but never played: {zeros} ({100 * zeros / training.height:.1f}%) — kept as zeros"
    )

    curves = fit_all(training)
    ranks = replacement_ranks(settings, curves)
    repl_points = replacement_points(settings, curves)

    for position in ("QB", "RB", "WR", "TE", "K"):
        curve = curves.get(position)
        if curve is None:
            continue
        need = ranks.get(position, 0)
        repl = repl_points[position]
        print(
            f"\n=== {position} ({curve.n_observations} player-seasons, "
            f"{len(curve.seasons)} seasons) ==="
        )
        print(f"  league starts {need} {position}s weekly -> replacement ~{repl:.0f} pts")
        print(f"  {'rank':>5} {'expected':>9} {'spread':>8} {'VOR':>7}")
        for rank in (1, 3, 6, 9, 12, 18, 24, 30, 36, 48):
            if rank > curve.max_rank:
                break
            exp = curve.points_at(rank)
            print(f"  {rank:>5} {exp:>9.1f} {curve.spread_at(rank):>8.1f} {exp - repl:>7.1f}")

    print("\nValue of the top player at each position over that position's replacement:")
    for position, curve in sorted(
        curves.items(), key=lambda kv: kv[1].points_at(1) - repl_points[kv[0]], reverse=True
    ):
        vor = curve.points_at(1) - repl_points[position]
        print(f"  {position:<5} {vor:>7.1f} pts   ({vor / 17:>4.1f} per week)")

    print("\nSpread is the standard deviation of actual outcomes at that rank —")
    print("it includes players who got hurt, which is why it is large.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
