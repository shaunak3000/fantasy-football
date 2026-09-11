"""Monte Carlo the rest of the season to get P(finishing first).

This is the objective the whole project is nominally about, and in this league
it diverges sharply from "score the most points". Four of eight teams make the
playoffs, so qualifying is close to free; the title is then two single-game coin
flips. A team can lead the league all year and win nothing.

That has a concrete consequence the points-maximizing view misses entirely: once
a playoff berth is secure, additional regular-season points are worth almost
nothing, while anything that raises your ceiling in two specific weeks is worth a
great deal. Only a simulation that plays the bracket out can price that.

Every trial is kept, not just its average. A recommendation is always a
*difference* between two simulations, and the difference of two noisy numbers is
noisier than either — so the per-trial record is what makes it possible to say
whether a reported edge is real. See `paired_delta`.
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
    #: Uncertainty in `weekly_mean` itself, as opposed to week-to-week variation
    #: around it. See `simulate` for why leaving this at zero makes the
    #: simulator overconfident.
    mean_uncertainty: float = 0.0

    def draw(self, weeks: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.weekly_mean, max(self.weekly_sd, 1e-6), size=weeks)


@dataclass
class SeasonOutcome:
    championship: dict[int, float] = field(default_factory=dict)
    playoffs: dict[int, float] = field(default_factory=dict)
    mean_wins: dict[int, float] = field(default_factory=dict)
    #: Per-trial championship indicator per team, kept so that the difference
    #: between two runs can be given an honest error bar.
    title_draws: dict[int, np.ndarray] = field(default_factory=dict)
    trials: int = 0

    def standard_error(self, team_id: int) -> float:
        """Monte Carlo error on this team's title probability."""
        probability = self.championship.get(team_id, 0.0)
        if self.trials <= 1:
            return 0.0
        return float(np.sqrt(probability * (1.0 - probability) / self.trials))

    def summary(self, teams: dict[int, str], top: int = 8) -> str:
        rows = sorted(self.championship.items(), key=lambda kv: -kv[1])[:top]
        return "\n".join(
            f"  {teams.get(tid, tid):<28} title {p:>6.1%}   playoffs "
            f"{self.playoffs.get(tid, 0):>6.1%}   wins {self.mean_wins.get(tid, 0):>4.1f}"
            for tid, p in rows
        )


def paired_delta(before: SeasonOutcome, after: SeasonOutcome, team_id: int) -> tuple[float, float]:
    """Change in title probability, with the standard error of *that change*.

    Both runs share a seed, so trial `i` sees the same luck in each and the
    difference cancels most of it. That pairing is what makes a half-percent
    move measurable at all — but it has to be measured, not assumed. Comparing
    the two probabilities and quoting the error of a single run would overstate
    the precision; differencing trial by trial gives the real figure.

    Returns `(delta, standard_error)`. A delta smaller than its own standard
    error is not a finding, it is the random number generator.
    """
    left = before.title_draws.get(team_id)
    right = after.title_draws.get(team_id)
    if left is None or right is None or len(left) != len(right) or len(left) < 2:
        delta = after.championship.get(team_id, 0.0) - before.championship.get(team_id, 0.0)
        return delta, float("inf")

    difference = right.astype(np.float64) - left.astype(np.float64)
    return float(difference.mean()), float(difference.std(ddof=1) / np.sqrt(len(difference)))


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

    All trials run at once as array operations. That is not only faster: the
    trade search evaluates hundreds of candidate rosters, and a simulation cheap
    enough to run at high trial counts is the difference between a readable
    signal and a coin flip dressed up as a recommendation.

    **Two sources of uncertainty, not one.** A team's weekly score varies around
    its mean, and that is what `weekly_sd` describes. But the mean itself is an
    estimate — a projection of a roster, or an average of a handful of games —
    and it can simply be wrong. Simulating only the first treats a noisy guess
    at a team's strength as a known fact, and the resulting probabilities come
    out far too confident: replaying 2025, the simulator said 58% for outcomes
    that happened 36% of the time, and lost to "rank them by their record".
    `mean_uncertainty` draws each team's true strength once per trial, so a
    thirteen-week run no longer assumes it knew the answer in week 3.
    """
    if not teams or trials <= 0:
        return SeasonOutcome(trials=max(trials, 0))

    rng = np.random.default_rng(seed)
    ids = [team.team_id for team in teams]
    index_of = {team_id: i for i, team_id in enumerate(ids)}
    n = len(ids)
    weeks = max((len(games) for games in remaining_schedule.values()), default=0)

    means = np.array([t.weekly_mean for t in teams], dtype=np.float64)
    sds = np.array([max(t.weekly_sd, 1e-6) for t in teams], dtype=np.float64)
    uncertainty = np.array([max(t.mean_uncertainty, 0.0) for t in teams], dtype=np.float64)

    # One draw of "how good is this team really", held fixed across the whole
    # trial — a team that is better than projected is better in week 3 and in
    # the final alike, which is exactly the correlation that makes a season
    # less predictable than independent weeks would suggest.
    true_means = means[:, None] + rng.normal(0.0, 1.0, size=(n, trials)) * uncertainty[:, None]

    # opponents[i, w] is the row of team i's week-w opponent, or -1 for a bye.
    opponents = np.full((n, weeks), -1, dtype=np.int64)
    for team_id, games in remaining_schedule.items():
        i = index_of.get(team_id)
        if i is None:
            continue
        for w, opponent in enumerate(games[:weeks]):
            j = index_of.get(opponent) if opponent is not None else None
            if j is not None:
                opponents[i, w] = j

    wins = np.array([t.wins for t in teams], dtype=np.float64)[:, None].repeat(trials, axis=1)
    points = np.array([t.points_for for t in teams], dtype=np.float64)[:, None].repeat(
        trials, axis=1
    )

    if weeks:
        scores = (
            true_means[:, None, :]
            + rng.normal(0.0, 1.0, size=(n, weeks, trials)) * sds[:, None, None]
        )
        points += scores.sum(axis=1)
        for w in range(weeks):
            opponent_row = opponents[:, w]
            played = opponent_row >= 0
            if not played.any():
                continue
            mine = scores[:, w, :]
            theirs = scores[opponent_row, w, :]
            # A tie gives nobody a win, which for continuous scores never
            # happens; stating it keeps the two sides of a matchup symmetric.
            wins += np.where(played[:, None] & (mine > theirs), 1.0, 0.0)

    # Seeding: wins first, total points as the tiebreak, which is this league's
    # actual rule. lexsort is stable, so a dead heat falls back to team order.
    order = np.lexsort((-points, -wins), axis=0)
    qualifiers = order[:playoff_teams]

    berths = np.zeros(n, dtype=np.float64)
    np.add.at(berths, qualifiers.ravel(), 1.0)

    champions = _bracket_winners(qualifiers, true_means, sds, rng)
    title_draws = {ids[i]: (champions == i) for i in range(n)}

    return SeasonOutcome(
        championship={ids[i]: float(title_draws[ids[i]].mean()) for i in range(n)},
        playoffs={ids[i]: float(berths[i] / trials) for i in range(n)},
        mean_wins={ids[i]: float(wins[i].mean()) for i in range(n)},
        title_draws=title_draws,
        trials=trials,
    )


def _bracket_winners(
    qualifiers: np.ndarray, true_means: np.ndarray, sds: np.ndarray, rng
) -> np.ndarray:
    """Single elimination from the given seeding: best seed plays worst.

    `qualifiers` is (seeds, trials) of team rows, seeded best first, and
    `true_means` is (teams, trials) — each team's strength in that trial, so a
    team that was better than projected all season is still better than
    projected in the final. Returns the champion's row for each trial.

    With an odd number left the top seed takes the bye, which is the only
    sensible reading of "best seed plays worst" and the case the previous
    implementation silently dropped a team on.
    """
    trial_index = np.arange(qualifiers.shape[1])

    def play(rows: np.ndarray) -> np.ndarray:
        centre = true_means[rows, trial_index]
        return centre + rng.normal(0.0, 1.0, size=rows.shape) * sds[rows]

    remaining = qualifiers
    while remaining.shape[0] > 1:
        count = remaining.shape[0]
        byes = remaining[:1] if count % 2 else remaining[:0]
        playing = remaining[1:] if count % 2 else remaining
        half = playing.shape[0] // 2
        high, low = playing[:half], playing[half:][::-1]

        winners = np.where(play(high) >= play(low), high, low)
        remaining = np.concatenate([byes, winners], axis=0)

    return remaining[0]
