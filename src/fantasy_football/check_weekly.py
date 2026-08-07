"""Inspect the weekly projection model.

uv run python -m fantasy_football.check_weekly
"""

from __future__ import annotations

from espn_api.football import League

from .config import load_credentials
from .data.espn import fetch_raw_settings, parse_settings
from .data.ids import attach_espn_ids
from .data.nflverse import load_consensus_board
from .projections.build import build
from .projections.history import training_table
from .projections.scoring import ScoringEngine
from .projections.weekly import WeeklyModel, bye_weeks_from_board

TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]


def main() -> int:
    creds = load_credentials()
    league = League(league_id=creds.league_id, year=2026, espn_s2=creds.espn_s2, swid=creds.swid)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, 2026)
    engine = ScoringEngine.from_settings(settings)

    training, _ = training_table(TRAIN_SEASONS, engine)
    model = WeeklyModel.fit(training)

    print("Per-game production and week-to-week spread by preseason rank\n")
    for position in ("QB", "RB", "WR", "TE"):
        per_game = model.per_game.get(position)
        spread = model.spread.get(position)
        if per_game is None or spread is None:
            continue
        print(f"=== {position} ===")
        print(f"  {'rank':>5} {'pts/gm':>8} {'wk sd':>8} {'cv':>6} {'exp gms':>8}")
        for rank in (1, 3, 6, 12, 24, 36):
            if rank > per_game.max_rank:
                break
            mean = per_game.points_at(rank)
            sd = spread.points_at(rank)
            cv = sd / mean if mean else 0.0
            print(
                f"  {rank:>5} {mean:>8.1f} {sd:>8.1f} {cv:>6.2f} "
                f"{model.expected_games(position, rank):>8.1f}"
            )
        print()

    print("Coefficient of variation is spread relative to production. A higher")
    print("number means a player swings more for the same average — useful when")
    print("chasing a ceiling, costly when protecting a lead.\n")

    result = build(2026)
    board = attach_espn_ids(load_consensus_board()).matched
    byes = bye_weeks_from_board(board)
    print(f"Bye weeks known for {len(byes)} players\n")

    print("=== Week 1 outlook, top 12 by season value ===")
    print(f"  {'player':<22} {'pos':<4} {'mean':>7} {'sd':>6} {'P(>15)':>8}")
    shown = 0
    for projection in result.projections:
        outlook = model.outlook(projection, week=1, bye_week=byes.get(projection.player))
        if not outlook.is_startable:
            continue
        print(
            f"  {projection.player:<22} {projection.position:<4} "
            f"{outlook.mean:>7.1f} {outlook.sd:>6.1f} {outlook.probability_over(15.0):>7.0%}"
        )
        shown += 1
        if shown >= 12:
            break

    on_bye = [
        p.player
        for p in result.projections[:60]
        if model.outlook(p, week=5, bye_week=byes.get(p.player)).on_bye
    ]
    print(f"\nTop-60 players on bye in week 5: {', '.join(on_bye) if on_bye else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
