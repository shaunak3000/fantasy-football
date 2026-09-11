"""Is the season simulator telling the truth?

    uv run python -m fantasy_football.check_simulator [season]

Every other claim in this repo is backed by a replay of a real season. The
simulator was the exception: it produced the headline number — P(finishing
first) — and nothing checked it against anything that actually happened.

This replays a finished season week by week. Standing at the start of each week,
using only scores from weeks already played, it simulates the rest of the season
and records what it predicted. Then it scores those predictions against what
really happened.

**It deliberately uses real weekly team scores, not rosters.** ESPN does not
keep historical weekly lineups — a request for week 3 hands back the roster and
the lineup slots as they stand today, which is why `box_scores()` fails on a past
season. Anything built on replayed rosters is therefore measuring the team you
finished with, not the team you fielded. Actual weekly scores have no such
problem: they are what the teams really scored, in the order they really scored
them. That tests the part this module is responsible for — turning scoring
distributions and a schedule into probabilities — and leaves the accuracy of the
roster projections to `check_optimizer`.

Two things are measured:

* **Calibration.** When it says 60%, does it happen about 60% of the time?
* **Skill.** Does it beat naive baselines that need no simulation at all? A
  forecast can be perfectly calibrated and still useless if a simpler rule is
  just as good.
"""

from __future__ import annotations

import math
import statistics
import sys

import numpy as np
from espn_api.football import League

from .config import load_credentials
from .data.espn import fetch_raw_settings, parse_settings
from .season.simulator import TeamSeason, simulate

TRIALS = 20000
#: Weeks of a team's own scoring it takes before we stop leaning on the league
#: average. Small samples of weekly fantasy scores are extremely noisy, and an
#: unshrunk three-week mean would have the simulator chasing hot starts.
SHRINKAGE_WEEKS = 4.0
PROBABILITY_FLOOR = 0.01


def _weekly_scores_and_schedule(
    league: League, regular_season_weeks: int
) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, int]]]:
    """Real per-week scores and opponents, straight from the scoreboard."""
    payload = league.espn_request.league_get(
        params={"view": ["mMatchupScore"], "scoringPeriodId": 1}
    )
    scores: dict[int, dict[int, float]] = {}
    opponents: dict[int, dict[int, int]] = {}

    for matchup in payload.get("schedule", []):
        period = matchup.get("matchupPeriodId")
        if period is None or period > regular_season_weeks:
            continue
        home = (matchup.get("home") or {}).get("teamId")
        away = (matchup.get("away") or {}).get("teamId")
        if home is None or away is None:
            continue
        opponents.setdefault(home, {})[period] = away
        opponents.setdefault(away, {})[period] = home
        for side in ("home", "away"):
            entry = matchup.get(side) or {}
            team_id = entry.get("teamId")
            for week, points in (entry.get("pointsByScoringPeriod") or {}).items():
                if int(week) <= regular_season_weeks:
                    scores.setdefault(team_id, {})[int(week)] = float(points)
    return scores, opponents


def _record_through(
    scores: dict[int, dict[int, float]],
    opponents: dict[int, dict[int, int]],
    through: int,
) -> dict[int, tuple[int, int, float]]:
    """Wins, losses and points for each team over weeks 1..through."""
    record: dict[int, tuple[int, int, float]] = {}
    for team_id, by_week in scores.items():
        wins = losses = 0
        points = 0.0
        for week in range(1, through + 1):
            mine = by_week.get(week)
            if mine is None:
                continue
            points += mine
            opponent = opponents.get(team_id, {}).get(week)
            theirs = scores.get(opponent, {}).get(week) if opponent else None
            if theirs is None:
                continue
            if mine > theirs:
                wins += 1
            elif mine < theirs:
                losses += 1
        record[team_id] = (wins, losses, points)
    return record


def _qualifiers(record: dict[int, tuple[int, int, float]], playoff_teams: int) -> set[int]:
    ranked = sorted(record, key=lambda tid: (-record[tid][0], -record[tid][2]))
    return set(ranked[:playoff_teams])


def _estimate(
    scores: dict[int, dict[int, float]], before_week: int
) -> tuple[dict[int, float], float, float]:
    """Each team's expected weekly score, the weekly spread, and how wrong the
    mean itself is likely to be — all from history only."""
    history = {
        team_id: [by_week[w] for w in range(1, before_week) if w in by_week]
        for team_id, by_week in scores.items()
    }
    pooled = [value for values in history.values() for value in values]
    league_mean = statistics.mean(pooled) if pooled else 100.0

    means = {}
    for team_id, values in history.items():
        n = len(values)
        team_mean = statistics.mean(values) if values else league_mean
        means[team_id] = (n * team_mean + SHRINKAGE_WEEKS * league_mean) / (n + SHRINKAGE_WEEKS)

    # One spread for everyone: with at most a dozen observations per team, a
    # per-team variance estimate is mostly noise, and the teams that matter are
    # separated by their means rather than their volatility.
    deviations = [
        value - statistics.mean(values)
        for values in history.values()
        if len(values) >= 2
        for value in values
    ]
    spread = max(statistics.pstdev(deviations) if len(deviations) >= 4 else 25.0, 1.0)

    # How far the estimated mean is likely to be from the truth: the ordinary
    # standard error of a mean, sd/sqrt(n). This is derived rather than fitted,
    # which matters — a constant tuned on this same season would score better
    # here and mean nothing. Before any games it is the spread between teams,
    # since that is the whole range a team could turn out to occupy.
    played = max(len(values) for values in history.values()) if history else 0
    if played >= 1:
        uncertainty = spread / math.sqrt(played)
    else:
        team_means = [statistics.mean(v.values()) for v in scores.values() if v]
        uncertainty = statistics.pstdev(team_means) if len(team_means) >= 2 else spread
    return means, spread, uncertainty


def _brier(predictions: list[float], outcomes: list[int]) -> float:
    return float(np.mean([(p - y) ** 2 for p, y in zip(predictions, outcomes, strict=True)]))


def _scaled(weights: dict[int, float], total: float) -> dict[int, float]:
    """Turn positive weights into probabilities summing to `total`."""
    mass = sum(weights.values())
    if mass <= 0:
        return {tid: total / len(weights) for tid in weights}
    return {
        tid: min(max(total * weight / mass, PROBABILITY_FLOOR), 1.0 - PROBABILITY_FLOOR)
        for tid, weight in weights.items()
    }


def main(argv: list[str]) -> int:
    season = int(argv[0]) if argv else 2025
    creds = load_credentials()
    league = League(league_id=creds.league_id, year=season, espn_s2=creds.espn_s2, swid=creds.swid)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, season)
    weeks = settings.regular_season_weeks
    playoff_teams = settings.playoff_team_count

    names = {team.team_id: team.team_name for team in league.teams}
    scores, opponents = _weekly_scores_and_schedule(league, weeks)
    if not scores:
        print("No completed weekly scores for that season.")
        return 1

    final = _record_through(scores, opponents, weeks)
    truth = _qualifiers(final, playoff_teams)

    print(f"\n{settings.name} {season} — season simulator replayed week by week\n")
    print(f"  Actually made the top {playoff_teams}: {', '.join(sorted(names[t] for t in truth))}")
    print(f"  {TRIALS:,} trials per week, predictions use only earlier weeks.\n")

    sim_p: list[float] = []
    record_p: list[float] = []
    strength_p: list[float] = []
    uniform_p: list[float] = []
    outcomes: list[int] = []
    win_error: list[float] = []

    print(f"  {'from week':>10}{'sim':>9}{'record':>9}{'strength':>10}{'uniform':>9}")
    for start in range(1, weeks + 1):
        banked = _record_through(scores, opponents, start - 1)
        means, spread, uncertainty = _estimate(scores, start)

        remaining = {
            team_id: [opponents.get(team_id, {}).get(w) for w in range(start, weeks + 1)]
            for team_id in scores
        }
        teams = [
            TeamSeason(
                team_id=team_id,
                name=names.get(team_id, str(team_id)),
                weekly_mean=means[team_id],
                weekly_sd=spread,
                wins=banked[team_id][0],
                losses=banked[team_id][1],
                points_for=banked[team_id][2],
                mean_uncertainty=uncertainty,
            )
            for team_id in scores
        ]
        outcome = simulate(teams, remaining, playoff_teams, trials=TRIALS, seed=start)

        games = max(start - 1, 0)
        record_weights = {t: (banked[t][0] + 0.5) / (games + 1) for t in scores}
        record_scaled = _scaled(record_weights, playoff_teams)
        strength_scaled = _scaled(dict(means), playoff_teams)

        week_sim, week_rec, week_str, week_uni, week_y = [], [], [], [], []
        for team_id in scores:
            y = 1 if team_id in truth else 0
            week_sim.append(outcome.playoffs[team_id])
            week_rec.append(record_scaled[team_id])
            week_str.append(strength_scaled[team_id])
            week_uni.append(playoff_teams / len(scores))
            week_y.append(y)
            win_error.append(outcome.mean_wins[team_id] - final[team_id][0])

        sim_p += week_sim
        record_p += week_rec
        strength_p += week_str
        uniform_p += week_uni
        outcomes += week_y
        print(
            f"  {start:>10}{_brier(week_sim, week_y):>9.3f}{_brier(week_rec, week_y):>9.3f}"
            f"{_brier(week_str, week_y):>10.3f}{_brier(week_uni, week_y):>9.3f}"
        )

    print("\n  Brier score over every team-week (lower is better):")
    results = {
        "simulator": _brier(sim_p, outcomes),
        "record so far": _brier(record_p, outcomes),
        "roster strength": _brier(strength_p, outcomes),
        "uniform 50%": _brier(uniform_p, outcomes),
    }
    for label, value in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"    {label:<18}{value:.4f}")

    best_naive = min(results["record so far"], results["roster strength"], results["uniform 50%"])
    edge = best_naive - results["simulator"]
    print(
        f"\n  The simulator beats the best naive baseline by {edge:+.4f} Brier"
        if edge > 0
        else f"\n  The simulator LOSES to the best naive baseline by {edge:+.4f} Brier"
    )

    print("\n  Calibration — when it says X%, how often did it happen?\n")
    print(f"  {'predicted':>18}{'n':>6}{'mean predicted':>16}{'actually happened':>19}")
    edges = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
    for low, high in zip(edges, edges[1:], strict=False):
        bucket = [(p, y) for p, y in zip(sim_p, outcomes, strict=True) if low <= p < high]
        if not bucket:
            continue
        predicted = statistics.mean(p for p, _ in bucket)
        realized = statistics.mean(y for _, y in bucket)
        print(
            f"  {f'{low:.0%}-{min(high, 1.0):.0%}':>18}{len(bucket):>6}"
            f"{predicted:>16.1%}{realized:>19.1%}"
        )

    print(f"\n  Mean error in predicted final wins: {statistics.mean(win_error):+.2f}")
    print(f"  Mean absolute error: {statistics.mean(abs(e) for e in win_error):.2f}")
    print("\n  What this does not test: whether the roster projections feeding the")
    print("  live simulator are any good. This replay takes each team's scoring")
    print("  distribution from its own past results, so it checks the schedule,")
    print("  seeding and bracket machinery, not the inputs. One season of one")
    print("  league is also 8 teams and 13 weeks — treat a small Brier edge as")
    print("  suggestive, not settled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
