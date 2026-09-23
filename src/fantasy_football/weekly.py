"""The in-season command: what to start, who to claim, what to offer.

    uv run python -m fantasy_football.weekly [week]

One report a week, and one baseline underneath all of it. Every section quotes
the same starting title probability, computed once, with the Monte Carlo error
on it stated — and any recommendation whose effect is smaller than that error is
withheld rather than dressed up as a finding.

The lineup section runs on ESPN's published weekly projections, which is the
input `check_optimizer` replays finished seasons on. The season simulator
underneath the probabilities runs on the fitted rank curves instead, because it
plays out thirteen weeks and must not assume they all look like this one.
"""

from __future__ import annotations

import sys

from espn_api.football import League

from .config import load_credentials
from .data.espn import fetch_raw_settings, fetch_weekly_projections, parse_settings
from .data.snapshots import capture_and_save, weeks_awaiting_results
from .draft.cache import load_bundle
from .draft.live import my_team_id
from .lineup.optimizer import best_lineup_against, optimize
from .projections.history import training_table
from .projections.scoring import ScoringEngine
from .projections.weekly import WeeklyModel
from .season.league_state import build_state, bye_weeks_by_espn_id, this_week
from .transactions.evaluate import (
    DEFAULT_TRIALS,
    SimulationContext,
    find_trades,
    rank_waiver_targets,
)
from .transactions.streaming import find_stream

SEASON = 2026
TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]
SIM_TRIALS = DEFAULT_TRIALS

#: How far a team's projected weekly mean sits from what it really averages.
#: Measured on 2025: the projection-maximizing lineup's realized points per week
#: differed from each team's actual scoring average with a spread of 3.7 points
#: across the eight teams. Treating a projection as exact makes the simulator
#: overconfident, which `check_simulator` catches directly.
#:
#: This is a *lower* bound. It was measured against end-of-season rosters, the
#: only snapshot ESPN keeps, so it cannot see a team that was weaker in October
#: than the roster it finished with; and the replay finds that larger values
#: calibrate better still. Revisit once a season of weekly snapshots exists.
PROJECTION_MEAN_UNCERTAINTY = 3.7


def _lineup_section(state, my_id, settings, week) -> list:
    """Print the start/sit advice and return the bench, for the waiver screen."""
    roster = state.rosters[my_id]
    week_roster = this_week(roster)

    opponent_id = (state.schedule.get(my_id) or [None])[0]
    opponent_roster = state.rosters.get(opponent_id)

    if opponent_roster:
        opponent_lineup = optimize(this_week(opponent_roster), settings)
        lineup, win_probability = best_lineup_against(
            week_roster, settings, opponent_lineup.mean, opponent_lineup.sd
        )
        print(
            f"Facing {state.names.get(opponent_id, '?')} "
            f"(projected {opponent_lineup.mean:.1f}) — win probability {win_probability:.0%}\n"
        )
    else:
        lineup = optimize(week_roster, settings)
        print("No opponent found for this week; maximizing expected points.\n")

    for slot, players in lineup.starters.items():
        for player in players:
            print(f"  {slot:<10} {player.player:<24} {player.mean:>6.1f} +/-{player.sd:>5.1f}")

    # The solver leaves a slot empty rather than starting someone worth zero, so
    # an absent slot is a real signal: nobody on the roster can fill it.
    unfilled = [
        slot
        for slot in settings.starting_slots
        if settings.starting_slots[slot] > len(lineup.starters.get(slot, []))
    ]
    if unfilled:
        print(f"\n  NOBODY TO START at: {', '.join(unfilled)} — make a claim")

    sidelined = [p for p in week_roster if p.unavailable]
    if sidelined:
        print(
            "\n  OUT this week (worth 0): "
            + ", ".join(f"{p.player} [{p.injury_status}]" for p in sidelined)
        )
    on_bye = [p for p in week_roster if p.on_bye]
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

    chosen = {p.espn_id for p in lineup.players}
    current = [p for p in week_roster if p.started]
    bench = [p for p in roster if p.espn_id not in chosen]

    if not lineup.players:
        # An empty roster produces an empty lineup and no changes, which must
        # not be reported as everything being fine.
        print("\n  No rostered players to set a lineup from.")
        return bench

    if not current:
        print("\n  ESPN reports no starting lineup set for this week.")
        return bench

    # Valued on what they will actually score: a player who is out is worth
    # zero in the lineup you currently have, however good his projection looks.
    current_points = sum(p.mean for p in current)
    gain = lineup.mean - current_points
    benched = [p for p in current if p.espn_id not in chosen]
    promoted = [p for p in lineup.players if not p.started]

    if not benched and not promoted:
        print(f"\n  Your current lineup is already optimal ({current_points:.1f} projected).")
        return bench

    print(f"\n  CHANGES worth {gain:+.1f} points this week")
    print(f"    current {current_points:.1f} -> best legal {lineup.mean:.1f}")
    for out, into in zip(benched, promoted, strict=False):
        reason = f" [{out.injury_status}]" if out.unavailable else ""
        print(f"    start {into.player} ({into.position})  for  {out.player}{reason}")
    for extra in promoted[len(benched) :]:
        print(f"    start {extra.player} ({extra.position})")
    for extra in benched[len(promoted) :]:
        print(f"    bench {extra.player}")

    return bench


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
    # Hoisted rather than called inline: the streaming check below needs the same
    # numbers, and this is a 900-player request not worth making twice.
    published = fetch_weekly_projections(league, week)
    state = build_state(
        league,
        settings,
        bundle.projections.projections,
        weekly_model,
        current_week=week,
        byes=bye_weeks_by_espn_id() if season == SEASON else {},
        weekly_projections=published,
    )

    my_id = my_team_id(league, creds.swid)
    if my_id is None:
        my_id = next((tid for tid in state.rosters), None)

    if my_id is None or my_id not in state.rosters:
        print("Could not identify your team.", file=sys.stderr)
        return 1

    print(f"\n# {settings.name} {season} — week {week}\n")

    # Record this week's lineups before anything else. ESPN serves only the
    # current week's roster, so a week that goes uncaptured is gone for good —
    # this is the one thing here that cannot be rerun later. It always snapshots
    # the *live* scoring period, not the week being reported on, and it never
    # raises: a failed snapshot must not take the report down with it.
    live_week = max(int(getattr(league, "nfl_week", week) or week), 1)
    captures = [capture_and_save(league, live_week, season)]

    # The live capture has the lineups but no scores — the games have not been
    # played. Nothing used to come back for them, so the record was half of what
    # a lineup backtest needs. Finished weeks missing stat lines are re-fetched
    # here; `save` merges only `actual` and keeps the slots recorded at the time.
    for finished_week in weeks_awaiting_results(season, live_week):
        captures.append(capture_and_save(league, finished_week, season))

    print(f"_Snapshot: {'; '.join(captures)}._\n")

    # One baseline, computed once, shared by every section below. The standings
    # table, the waiver deltas and the trade deltas are all differences against
    # this single simulation.
    context = SimulationContext(
        names=state.names,
        settings=settings,
        schedule=state.schedule,
        playoff_teams=settings.playoff_team_count,
        banked=state.banked,
        trials=SIM_TRIALS,
        mean_uncertainty=PROJECTION_MEAN_UNCERTAINTY,
    )
    baseline = context.outcome(state.rosters)
    stderr = baseline.standard_error(my_id)

    print("## Where the league stands\n")
    print(baseline.summary(state.names))
    print(
        f"\n  {state.names.get(my_id, '?')} baseline title odds "
        f"{baseline.championship.get(my_id, 0.0):.1%} +/-{stderr:.2%} "
        f"({SIM_TRIALS:,} trials). Moves worth less than that are not reported."
    )

    print("\n## Start these\n")
    bench = _lineup_section(state, my_id, settings, week)

    print("\n## Waiver targets\n")
    print("_Advisory — priced in title probability, not yet validated against history._")
    print("_This league uses waiver priority, not FAAB: a claim spends a position._\n")
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
        top=len(state.free_agents),
        context=context,
        baseline=baseline,
    )
    # Listing four bench adds at +0.00% apiece is not a ranking, it is padding.
    # If nothing on the wire cracks the lineup, that is the whole finding, and
    # the right advice is to keep the priority position rather than spend it.
    useful = [a for a in waivers if a.points_gain > 0]
    if not useful:
        checked = len(state.free_agents)
        best = waivers[0].move.description if waivers else "nothing"
        print(
            f"  None of the top {checked} free agents would start for us "
            f"(best was {best}).\n  Hold waiver priority."
        )
    for advice in useful:
        print(f"  {advice.move.summary()}")
        print(f"      {advice.points_gain:+.1f} pts/week to the lineup — {advice.verdict}")

    # Defence is the one position where the weekly matchup is worth chasing —
    # about 8% of variation against 2-3% for skill players — and the one the
    # optimizer above cannot help with, because it only ranks the defence
    # already on the roster and never looks at the wire for a better one.
    stream = find_stream(
        published,
        my_ids={p.espn_id for p in state.rosters[my_id]},
        rostered_ids={p.espn_id for roster in state.rosters.values() for p in roster},
    )
    if stream is not None:
        print("\n## Stream a defence\n")
        print(f"  {stream.summary()}")
        if stream.runners_up:
            others = ", ".join(f"{name} {points:.1f}" for name, points in stream.runners_up)
            print(f"      also free: {others}")

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
        context=context,
        baseline=baseline,
    )
    if not trades:
        print("  Nothing mutually beneficial found this week.")
    for move in trades[:4]:
        print(
            f"  {move.summary()}  |  {state.names.get(move.counterparty_id, '?')}: "
            f"{move.counterparty_delta:+.2%} +/-{move.counterparty_stderr:.2%}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
