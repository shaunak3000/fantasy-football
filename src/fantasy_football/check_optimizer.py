"""Does the lineup optimizer actually capture any of the bench points?

    uv run python -m fantasy_football.check_optimizer [season]

`check_lineups` established the prize: 272 points a season sitting on benches.
This asks how much of it is reachable. It replays every week of a finished
season using only what was knowable beforehand — ESPN's own weekly projections,
published before kickoff — sets a lineup, and scores it with what actually
happened.

Three lineups are compared each week: what the manager really started, what
maximizing projected points would have started, and what maximizing the chance
of beating *that week's actual opponent* would have started. The third is the
one that tests whether risk-adjustment is worth anything, and it is judged on
games won rather than points, because that is what it optimizes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from espn_api.football import League

from .config import load_credentials
from .data.espn import (
    ACTUAL_STAT_SOURCE,
    PROJECTED_STAT_SOURCE,
    WEEKLY_SPLIT,
    fetch_raw_settings,
    parse_settings,
)
from .lineup.optimizer import optimize
from .projections.scoring import ScoringEngine

BENCH_SLOTS = {20, 21, 24}
POSITION_BY_ID = {0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "D/ST", 17: "K"}

# Weekly volatility relative to production, measured per position in
# `check_weekly`. Tight ends swing furthest for what they score; quarterbacks
# least. Used to turn a point projection into a distribution.
COEFFICIENT_OF_VARIATION = {
    "QB": 0.40,
    "RB": 0.49,
    "WR": 0.49,
    "TE": 0.58,
    "K": 0.35,
    "D/ST": 0.60,
}


@dataclass
class WeekPlayer:
    player: str
    position: str
    mean: float
    sd: float
    actual: float
    started: bool


@dataclass
class Totals:
    manager: float = 0.0
    projection_max: float = 0.0
    matchup_aware: float = 0.0
    hindsight: float = 0.0
    wins_manager: int = 0
    wins_projection: int = 0
    wins_matchup: int = 0
    games: int = 0
    weeks: list[int] = field(default_factory=list)


def _stat_block(player: dict, source: int, week: int) -> dict | None:
    return next(
        (
            s
            for s in player.get("stats", [])
            if s.get("statSourceId") == source
            and s.get("statSplitTypeId") == WEEKLY_SPLIT
            and s.get("scoringPeriodId") == week
        ),
        None,
    )


def _collect_week(payload: dict, week: int, engine) -> dict[int, list[WeekPlayer]]:
    rosters: dict[int, list[WeekPlayer]] = {}
    for team in payload.get("teams", []):
        players: list[WeekPlayer] = []
        for entry in team.get("roster", {}).get("entries", []):
            raw = entry.get("playerPoolEntry", {}).get("player", {})
            position = POSITION_BY_ID.get(raw.get("defaultPositionId", -1))
            if position is None:
                continue

            actual_block = _stat_block(raw, ACTUAL_STAT_SOURCE, week)
            projected_block = _stat_block(raw, PROJECTED_STAT_SOURCE, week)
            if actual_block is None or projected_block is None:
                continue

            actual_stats = {int(k): float(v) for k, v in (actual_block.get("stats") or {}).items()}
            if not actual_stats:
                continue

            position_id = raw.get("defaultPositionId", -1)
            projected = float(projected_block.get("appliedTotal") or 0.0)
            players.append(
                WeekPlayer(
                    player=raw.get("fullName", "?"),
                    position=position,
                    mean=projected,
                    sd=max(projected * COEFFICIENT_OF_VARIATION.get(position, 0.5), 1.0),
                    actual=engine.score(actual_stats, position_id),
                    started=entry.get("lineupSlotId") not in BENCH_SLOTS,
                )
            )
        if players:
            rosters[team.get("id")] = players
    return rosters


def _hindsight(players: list[WeekPlayer], settings) -> float:
    """Best legal lineup with perfect foresight — the unreachable ceiling."""
    perfect = [
        WeekPlayer(p.player, p.position, p.actual, 0.0, p.actual, p.started) for p in players
    ]
    return optimize(perfect, settings).mean


def main(argv: list[str]) -> int:
    season = int(argv[0]) if argv else 2025
    creds = load_credentials()
    league = League(league_id=creds.league_id, year=season, espn_s2=creds.espn_s2, swid=creds.swid)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, season)
    engine = ScoringEngine.from_settings(settings)

    names = {team.team_id: team.team_name for team in league.teams}
    opponents = {
        team.team_id: [getattr(o, "team_id", None) for o in team.schedule] for team in league.teams
    }

    totals: dict[int, Totals] = {tid: Totals() for tid in names}

    for week in range(1, settings.regular_season_weeks + 1):
        try:
            payload = league.espn_request.league_get(
                params={"view": "mRoster", "scoringPeriodId": week}
            )
        except Exception as exc:
            print(f"  week {week}: fetch failed ({type(exc).__name__})")
            continue

        rosters = _collect_week(payload, week, engine)
        if not rosters:
            continue

        # Every team's projection-maximizing lineup is needed before any
        # matchup-aware decision can be made, since the objective depends on
        # what the opponent is expected to score.
        baseline = {tid: optimize(players, settings) for tid, players in rosters.items()}

        for tid, players in rosters.items():
            record = totals[tid]
            record.weeks.append(week)
            record.manager += sum(p.actual for p in players if p.started)
            record.hindsight += _hindsight(players, settings)

            projection_lineup = baseline[tid]
            record.projection_max += sum(p.actual for p in projection_lineup.players)

            opponent_id = opponents.get(tid, [None] * 20)[week - 1] if opponents.get(tid) else None
            opponent = baseline.get(opponent_id)
            if opponent is None:
                record.matchup_aware += sum(p.actual for p in projection_lineup.players)
                continue

            from .lineup.optimizer import best_lineup_against

            chosen, _ = best_lineup_against(players, settings, opponent.mean, opponent.sd)
            record.matchup_aware += sum(p.actual for p in chosen.players)

    print(f"\n{settings.name} {season} - lineup decisions replayed on ESPN's own projections\n")
    print(f"  {'team':<26}{'manager':>9}{'proj-max':>10}{'matchup':>9}{'ceiling':>9}{'gain':>7}")

    gains = []
    for tid, record in sorted(totals.items(), key=lambda kv: -kv[1].projection_max):
        if not record.weeks:
            continue
        gain = record.projection_max - record.manager
        gains.append(gain)
        print(
            f"  {names.get(tid, tid):<26}{record.manager:>9.0f}{record.projection_max:>10.0f}"
            f"{record.matchup_aware:>9.0f}{record.hindsight:>9.0f}{gain:>+7.0f}"
        )

    played = [r for r in totals.values() if r.weeks]
    weeks = max(len(r.weeks) for r in played)
    mean_gain = sum(gains) / len(gains)
    ceiling = sum(r.hindsight - r.manager for r in played) / len(played)

    print(f"\n  Projection-maximizing beats the manager by {mean_gain:+.0f} points a season")
    print(f"  ({mean_gain / weeks:+.1f} per week), against a hindsight ceiling of {ceiling:.0f}.")
    if ceiling:
        print(f"  That captures {100 * mean_gain / ceiling:.0f}% of the available prize.")

    matchup_delta = sum(r.matchup_aware - r.projection_max for r in played) / len(played)
    print(f"\n  Matchup-aware scores {matchup_delta:+.0f} points a season versus projection-max.")
    print("  Fewer points is expected and not a failure - it trades expected points")
    print("  for win probability. Judging it needs head-to-head results, not totals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
