"""Monte Carlo the rest of the season to get P(finishing first).

This is the objective the whole project is nominally about, and in this league
it diverges sharply from "score the most points". Four of eight teams make the
playoffs, so qualifying is close to free; the title is then two single-game coin
flips. A team can lead the league all year and win nothing.

That has a concrete consequence the points-maximizing view misses entirely: once
a playoff berth is secure, additional regular-season points are worth almost
nothing, while anything that raises your ceiling in two specific weeks is worth a
great deal. Only a simulation that plays the bracket out can price that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TeamSeason:
    """A team's weekly scoring distribution, plus results already banked."""

    team_id: int
    name: str
    weekly_mean: float
    weekly_sd: float
    wins: int = 0
    losses: int = 0
    points_for: float = 0.0

    def draw(self, weeks: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.weekly_mean, max(self.weekly_sd, 1e-6), size=weeks)


@dataclass
class SeasonOutcome:
    championship: dict[int, float] = field(default_factory=dict)
    playoffs: dict[int, float] = field(default_factory=dict)
    mean_wins: dict[int, float] = field(default_factory=dict)

    def summary(self, teams: dict[int, str], top: int = 8) -> str:
        rows = sorted(self.championship.items(), key=lambda kv: -kv[1])[:top]
        return "\n".join(
            f"  {teams.get(tid, tid):<28} title {p:>6.1%}   playoffs "
            f"{self.playoffs.get(tid, 0):>6.1%}   wins {self.mean_wins.get(tid, 0):>4.1f}"
            for tid, p in rows
        )


def simulate(
    teams: list[TeamSeason],
    remaining_schedule: dict[int, list[int | None]],
    playoff_teams: int,
    trials: int = 2000,
    seed: int = 0,
) -> SeasonOutcome:
    """Play the rest of the season repeatedly and count titles.

    `remaining_schedule` maps team id to its opponents, one per remaining week,
    with None for a bye. Every team's weeks are drawn together so that a single
    trial is one coherent season rather than independent per-matchup draws.
    """
    rng = np.random.default_rng(seed)
    weeks = max((len(games) for games in remaining_schedule.values()), default=0)
    by_id = {team.team_id: team for team in teams}

    titles = dict.fromkeys(by_id, 0)
    berths = dict.fromkeys(by_id, 0)
    total_wins = dict.fromkeys(by_id, 0.0)

    for _ in range(trials):
        scores = {team.team_id: team.draw(weeks, rng) for team in teams}
        wins = {tid: by_id[tid].wins for tid in by_id}
        points = {tid: by_id[tid].points_for for tid in by_id}

        for week in range(weeks):
            for tid, opponents in remaining_schedule.items():
                if week >= len(opponents):
                    continue
                opponent = opponents[week]
                points[tid] += scores[tid][week]
                if opponent is None or opponent not in scores:
                    continue
                # Each matchup is seen from both sides; count the win once.
                if tid < opponent and scores[tid][week] > scores[opponent][week]:
                    wins[tid] += 1
                elif tid < opponent:
                    wins[opponent] += 1

        # Seeding: wins first, total points as the tiebreak, which is this
        # league's actual rule.
        standings = sorted(by_id, key=lambda tid: (-wins[tid], -points[tid]))
        qualifiers = standings[:playoff_teams]
        for tid in qualifiers:
            berths[tid] += 1
        for tid in by_id:
            total_wins[tid] += wins[tid]

        champion = _play_bracket(qualifiers, by_id, rng)
        titles[champion] += 1

    return SeasonOutcome(
        championship={tid: titles[tid] / trials for tid in by_id},
        playoffs={tid: berths[tid] / trials for tid in by_id},
        mean_wins={tid: total_wins[tid] / trials for tid in by_id},
    )


def _play_bracket(seeds: list[int], by_id: dict[int, TeamSeason], rng) -> int:
    """Single-elimination from the given seeding, best seed plays worst."""
    remaining = list(seeds)
    while len(remaining) > 1:
        next_round = []
        for high, low in zip(remaining, reversed(remaining), strict=False):
            if high == low or len(next_round) * 2 >= len(remaining):
                break
            high_score = rng.normal(by_id[high].weekly_mean, max(by_id[high].weekly_sd, 1e-6))
            low_score = rng.normal(by_id[low].weekly_mean, max(by_id[low].weekly_sd, 1e-6))
            next_round.append(high if high_score >= low_score else low)
        if not next_round:
            break
        remaining = next_round
    return remaining[0] if remaining else seeds[0]
