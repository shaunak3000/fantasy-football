"""The in-season command: what to start, who to claim, what to offer.

    uv run python -m fantasy_football.weekly [week]

One report a week. The lineup section is the part with a measured edge — 74% of
a 272-point-a-season prize, beating all eight managers in the 2025 backtest.
The waiver and trade sections are advisory: they are priced in championship
probability, which is the right unit, but neither has been validated against
what managers actually did, and that is stated rather than hidden.
"""

from __future__ import annotations

import sys

from espn_api.football import League

from .config import load_credentials
from .data.espn import fetch_raw_settings, parse_settings
from .draft.cache import load_bundle
from .draft.live import my_team_id
from .lineup.optimizer import best_lineup_against, optimize
from .projections.history import training_table
from .projections.scoring import ScoringEngine
from .projections.weekly import WeeklyModel
from .season.league_state import build_state, bye_weeks_by_espn_id
from .season.simulator import TeamSeason, simulate
from .transactions.evaluate import find_trades, rank_waiver_targets

SEASON = 2026
TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]
SIM_TRIALS = 2000


def _team_seasons(state) -> list[TeamSeason]:
    teams = []
    for team_id, roster in state.rosters.items():
        lineup = optimize(roster, state.settings)
        wins, losses, points = state.banked.get(team_id, (0, 0, 0.0))
        teams.append(
            TeamSeason(
                team_id=team_id,
                name=state.names.get(team_id, str(team_id)),
                weekly_mean=lineup.mean,
                weekly_sd=lineup.sd,
                wins=wins,
                losses=losses,
                points_for=points,
            )
        )
    return teams


def main(argv: list[str]) -> int:
    week = int(argv[0]) if argv else None
    # A past season can be replayed to exercise this end to end against real
    # rosters, schedules and standings, rather than waiting for a season to
    # start. Bye weeks come from the current consensus board, so they are only
    # meaningful for the live season — a replay is a smoke test, not a backtest.
    season = int(argv[1]) if len(argv) > 1 else SEASON

    creds = load_credentials()
    league = League(league_id=creds.league_id, year=season, espn_s2=creds.espn_s2, swid=creds.swid)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, season)
    engine = ScoringEngine.from_settings(settings)
    week = week or max(getattr(league, "nfl_week", 1), 1)

    bundle = load_bundle(SEASON)
    if bundle is None:
        print("No prepared bundle. Run: live_draft prepare", file=sys.stderr)
        return 1

    training, _ = training_table(TRAIN_SEASONS, engine)
    weekly_model = WeeklyModel.fit(training)
    state = build_state(
        league,
        settings,
        bundle.projections.projections,
        weekly_model,
        current_week=week,
        byes=bye_weeks_by_espn_id() if season == SEASON else {},
    )

    my_id = my_team_id(league, creds.swid)
    if my_id is None:
        my_id = next((tid for tid in state.rosters), None)

    if my_id is None or my_id not in state.rosters:
        print("Could not identify your team.", file=sys.stderr)
        return 1

    print(f"\n# {settings.name} {season} — week {week}\n")

    teams = _team_seasons(state)
    outcome = simulate(teams, state.schedule, settings.playoff_team_count, trials=SIM_TRIALS)
    print("## Where the league stands\n")
    print(outcome.summary(state.names))

    opponent_id = (state.schedule.get(my_id) or [None])[0]
    opponent = next((t for t in teams if t.team_id == opponent_id), None)

    print("\n## Start these\n")
    roster = state.rosters[my_id]
    if opponent is not None:
        lineup, win_probability = best_lineup_against(
            roster, settings, opponent.weekly_mean, opponent.weekly_sd
        )
        print(
            f"Facing {state.names.get(opponent_id, '?')} — win probability {win_probability:.0%}\n"
        )
    else:
        lineup = optimize(roster, settings)
        print("No opponent found for this week; maximizing expected points.\n")

    for slot, players in lineup.starters.items():
        for player in players:
            print(f"  {slot:<10} {player.player:<24} {player.mean:>6.1f} +/-{player.sd:>5.1f}")

    # The solver leaves a slot empty rather than starting someone worth zero, so
    # an absent slot is a real signal: nobody on the roster can fill it.
    unfilled = [
        slot
        for slot in settings.starting_slots
        if not lineup.starters.get(slot)
        and settings.starting_slots[slot] > len(lineup.starters.get(slot, []))
    ]
    if unfilled:
        print(f"\n  NOBODY TO START at: {', '.join(unfilled)} — make a claim")

    on_bye = [p for p in roster if p.on_bye]
    if on_bye:
        print(f"\n  ON BYE this week (worth 0): {', '.join(p.player for p in on_bye)}")

    # A bye collision two weeks out is fixable now and unfixable then, so the
    # report looks ahead rather than only at the week in front of you.
    upcoming: dict[int, list[str]] = {}
    for player in roster:
        if player.bye_week and week < player.bye_week <= week + 3:
            upcoming.setdefault(player.bye_week, []).append(f"{player.player} ({player.position})")
    for bye_week in sorted(upcoming):
        names_hit = upcoming[bye_week]
        warning = "  <- thin, plan a claim" if len(names_hit) >= 3 else ""
        print(f"  Week {bye_week} byes: {', '.join(names_hit)}{warning}")

    bench = [p for p in roster if p not in lineup.players]
    changes = [p for p in lineup.players if not p.started]
    if not lineup.players:
        # An empty roster produces an empty lineup and no changes, which must
        # not be reported as everything being fine.
        print("\n  No rostered players to set a lineup from.")
    elif changes:
        print("\n  CHANGES from your current lineup:")
        for player in changes:
            print(f"    start {player.player} ({player.position})")
    else:
        print("\n  Your current lineup is already optimal.")

    print("\n## Waiver targets\n")
    print("_Advisory — priced in title probability, not yet validated against history._\n")
    waivers = rank_waiver_targets(
        my_id,
        state.rosters,
        state.names,
        settings,
        state.schedule,
        settings.playoff_team_count,
        state.free_agents,
        bench,
        banked=state.banked,
        top=5,
        trials=300,
    )
    for move in waivers[:4]:
        marker = "  <- worth doing" if move.worth_doing else ""
        print(f"  {move.summary()}{marker}")

    print("\n## Trades worth proposing\n")
    print("_Advisory. Only trades that help both sides are listed; a proposal they_")
    print("_refuse is worthless, so their title delta is shown too._\n")
    trades = find_trades(
        my_id,
        state.rosters,
        state.names,
        settings,
        state.schedule,
        settings.playoff_team_count,
        banked=state.banked,
        trials=200,
    )
    if not trades:
        print("  Nothing mutually beneficial found this week.")
    for move in trades[:4]:
        print(
            f"  {move.summary()}  |  {state.names.get(move.counterparty_id, '?')}: "
            f"{move.counterparty_delta:+.2%}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
