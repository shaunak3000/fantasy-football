"""Pick a starting lineup.

Two things make this more than sorting by projected points.

The first is legality: slots have eligibility rules, a player fills exactly one,
and a lineup that breaks those rules is worthless however good it looks. That is
an assignment problem, so it is solved as one rather than with greedy heuristics
that quietly produce illegal answers at the edges.

The second is that maximizing expected points is often the wrong objective. You
do not win a week by scoring a lot; you win by scoring more than one specific
opponent. Against a projected blowout in either direction the expected-points
lineup is close to irrelevant — a heavy underdog needs variance and should start
the boom-bust tight end, a heavy favourite should start the boring one. Since the
chance of winning is not linear in the selection, the optimizer walks a
risk-appetite frontier with the solver and then scores each candidate lineup
exactly, which gets the right answer without pretending the objective is linear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt

import pulp

from ..draft.lineup_value import FLEX_ELIGIBILITY, FLEX_SLOTS

# Risk appetites swept when hunting for the best chance of winning. Negative
# values deliberately seek a *lower* ceiling: when heavily favoured, the safest
# lineup beats the highest-scoring one.
RISK_GRID = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5)


@dataclass(frozen=True)
class LineupSlot:
    name: str
    eligible: tuple[str, ...]


@dataclass
class Lineup:
    starters: dict[str, list] = field(default_factory=dict)
    mean: float = 0.0
    sd: float = 0.0

    @property
    def players(self) -> list:
        return [player for group in self.starters.values() for player in group]

    def win_probability(self, opponent_mean: float, opponent_sd: float) -> float:
        spread = sqrt(self.sd**2 + opponent_sd**2)
        if spread <= 0:
            return 1.0 if self.mean > opponent_mean else 0.0
        z = (self.mean - opponent_mean) / spread
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    def describe(self) -> str:
        parts = []
        for slot, players in self.starters.items():
            for player in players:
                parts.append(f"{slot}:{player.player}")
        return ", ".join(parts)


def slots_for(settings) -> list[LineupSlot]:
    """Expand the league's starting requirements into individual slots."""
    slots: list[LineupSlot] = []
    for name, count in settings.starting_slots.items():
        eligible = FLEX_ELIGIBILITY.get(name, (name,)) if name in FLEX_SLOTS else (name,)
        slots.extend([LineupSlot(name=name, eligible=tuple(eligible))] * int(count))
    return slots


def optimize(candidates: list, settings, risk: float = 0.0) -> Lineup:
    """Best legal lineup at a given risk appetite.

    `risk` above zero prefers volatility, below zero prefers a floor. Zero is
    plain expected points. Candidates need `.player`, `.position`, `.mean`, `.sd`.
    """
    slots = slots_for(settings)
    startable = [c for c in candidates if any(c.position in s.eligible for s in slots)]
    if not startable or not slots:
        return Lineup()

    problem = pulp.LpProblem("lineup", pulp.LpMaximize)
    assign = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
        for i, player in enumerate(startable)
        for j, slot in enumerate(slots)
        if player.position in slot.eligible
    }
    if not assign:
        return Lineup()

    problem += pulp.lpSum(
        variable * (startable[i].mean + risk * startable[i].sd)
        for (i, j), variable in assign.items()
    )

    for j in range(len(slots)):
        in_slot = [v for (i, jj), v in assign.items() if jj == j]
        if in_slot:
            problem += pulp.lpSum(in_slot) <= 1

    for i in range(len(startable)):
        of_player = [v for (ii, j), v in assign.items() if ii == i]
        if of_player:
            problem += pulp.lpSum(of_player) <= 1

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    lineup = Lineup()
    variance = 0.0
    for (i, j), variable in assign.items():
        if variable.value() and variable.value() > 0.5:
            player = startable[i]
            lineup.starters.setdefault(slots[j].name, []).append(player)
            lineup.mean += player.mean
            variance += player.sd**2
    lineup.sd = sqrt(variance)
    return lineup


def best_lineup_against(
    candidates: list,
    settings,
    opponent_mean: float,
    opponent_sd: float,
    risk_grid: tuple[float, ...] = RISK_GRID,
) -> tuple[Lineup, float]:
    """The lineup with the best chance of beating this specific opponent.

    Sweeps risk appetites, then scores each resulting lineup by its exact win
    probability. The winner is frequently *not* the highest-scoring lineup, which
    is the entire point.
    """
    best: Lineup | None = None
    best_probability = -1.0

    for risk in risk_grid:
        lineup = optimize(candidates, settings, risk=risk)
        if not lineup.players:
            continue
        probability = lineup.win_probability(opponent_mean, opponent_sd)
        if probability > best_probability:
            best, best_probability = lineup, probability

    return (best or Lineup()), max(best_probability, 0.0)
