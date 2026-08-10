"""Price a roster move in the only currency that matters: title probability.

A waiver claim that adds twelve points a week is worth a great deal in week 3
and almost nothing in week 13 to a team already locked into a playoff seed. A
trade that raises your floor helps a favourite and actively hurts an underdog
who needs variance. Points cannot express either of those; championship
probability can, which is why every move here is quoted as a delta in P(first)
rather than a delta in points.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..lineup.optimizer import optimize
from ..season.simulator import TeamSeason, simulate


@dataclass(frozen=True)
class MoveEvaluation:
    description: str
    title_before: float
    title_after: float
    weekly_points_change: float
    counterparty_id: int | None = None
    counterparty_before: float = 0.0
    counterparty_after: float = 0.0

    @property
    def counterparty_delta(self) -> float:
        """What the trade does to the other side's title odds.

        A trade they will refuse is worth nothing, so the number that decides
        whether a proposal is worth sending is theirs, not yours.
        """
        return self.counterparty_after - self.counterparty_before

    @property
    def mutually_acceptable(self) -> bool:
        return self.title_delta > 0 and self.counterparty_delta > 0

    @property
    def title_delta(self) -> float:
        return self.title_after - self.title_before

    @property
    def worth_doing(self) -> bool:
        return self.title_delta > 0

    def summary(self) -> str:
        direction = "+" if self.title_delta >= 0 else ""
        return (
            f"{self.description}: title {self.title_before:.1%} -> {self.title_after:.1%} "
            f"({direction}{self.title_delta:.2%}), "
            f"{self.weekly_points_change:+.1f} pts/week"
        )


def roster_strength(roster: list, settings) -> tuple[float, float]:
    """Weekly mean and spread of the best lineup this roster can start."""
    lineup = optimize(roster, settings)
    return lineup.mean, lineup.sd


def _title_odds(
    my_team_id: int,
    rosters: dict[int, list],
    names: dict[int, str],
    settings,
    schedule: dict[int, list[int | None]],
    playoff_teams: int,
    banked: dict[int, tuple[int, int, float]],
    trials: int,
    seed: int,
) -> float:
    teams = []
    for team_id, roster in rosters.items():
        mean, sd = roster_strength(roster, settings)
        wins, losses, points = banked.get(team_id, (0, 0, 0.0))
        teams.append(
            TeamSeason(
                team_id=team_id,
                name=names.get(team_id, str(team_id)),
                weekly_mean=mean,
                weekly_sd=sd,
                wins=wins,
                losses=losses,
                points_for=points,
            )
        )
    outcome = simulate(teams, schedule, playoff_teams, trials=trials, seed=seed)
    return outcome.championship.get(my_team_id, 0.0)


def evaluate_move(
    my_team_id: int,
    rosters: dict[int, list],
    names: dict[int, str],
    settings,
    schedule: dict[int, list[int | None]],
    playoff_teams: int,
    add: list | None = None,
    drop: list | None = None,
    banked: dict[int, tuple[int, int, float]] | None = None,
    trials: int = 600,
    seed: int = 0,
    counterparty_id: int | None = None,
) -> MoveEvaluation:
    """Simulate the season with and without a proposed move.

    Leave `counterparty_id` unset for a waiver claim, where the incoming player
    genuinely comes from outside the league. Set it for a trade, so the other
    side's roster changes too — see the note below on why that matters.

    The same random seed is used for both simulations so the comparison isolates
    the move rather than the luck of the draw — without that, a 1% title change
    would be indistinguishable from Monte Carlo noise.
    """
    banked = banked or {}
    add = add or []
    drop = drop or []

    def odds_for(team_id, state):
        return _title_odds(
            team_id, state, names, settings, schedule, playoff_teams, banked, trials, seed
        )

    before = odds_for(my_team_id, rosters)
    before_mean, _ = roster_strength(rosters[my_team_id], settings)
    counterparty_before = odds_for(counterparty_id, rosters) if counterparty_id is not None else 0.0

    dropped = {getattr(p, "player", None) for p in drop}
    acquired = {getattr(p, "player", None) for p in add}

    updated = {tid: list(roster) for tid, roster in rosters.items()}
    updated[my_team_id] = [
        p for p in rosters[my_team_id] if getattr(p, "player", None) not in dropped
    ] + list(add)

    # A trade is two-sided: whoever owned the incoming players must lose them,
    # and must gain whatever went the other way. Without this the acquired
    # player scores for both teams at once, which silently overstates every
    # trade — and misses the real prize in an eight-team league, that taking a
    # player off a rival weakens a direct competitor for one of four berths.
    if counterparty_id is not None and counterparty_id in updated:
        updated[counterparty_id] = [
            p for p in rosters[counterparty_id] if getattr(p, "player", None) not in acquired
        ] + list(drop)

    after = odds_for(my_team_id, updated)
    after_mean, _ = roster_strength(updated[my_team_id], settings)
    counterparty_after = odds_for(counterparty_id, updated) if counterparty_id is not None else 0.0

    labels = []
    if add:
        labels.append("add " + ", ".join(getattr(p, "player", "?") for p in add))
    if drop:
        labels.append(
            ("send " if counterparty_id is not None else "drop ")
            + ", ".join(getattr(p, "player", "?") for p in drop)
        )

    return MoveEvaluation(
        description=" / ".join(labels) or "no change",
        title_before=before,
        title_after=after,
        weekly_points_change=after_mean - before_mean,
        counterparty_id=counterparty_id,
        counterparty_before=counterparty_before,
        counterparty_after=counterparty_after,
    )


def find_trades(
    my_team_id: int,
    rosters: dict[int, list],
    names: dict[int, str],
    settings,
    schedule: dict[int, list[int | None]],
    playoff_teams: int,
    banked: dict[int, tuple[int, int, float]] | None = None,
    give_depth: int = 3,
    get_depth: int = 3,
    trials: int = 300,
) -> list[MoveEvaluation]:
    """Search every opponent roster for one-for-one swaps that help both sides.

    Only mutually beneficial trades are returned. A proposal that raises your
    title odds and lowers theirs is not a trade, it is a wish — the search
    filters those out rather than making you sift them.
    """
    my_roster = rosters.get(my_team_id, [])
    if not my_roster:
        return []

    # Trade your surplus, not your best: the players most likely to be spare are
    # the ones your lineup already cannot start.
    give_candidates = sorted(my_roster, key=lambda p: p.mean, reverse=True)[len(my_roster) // 2 :][
        :give_depth
    ]

    proposals = []
    for team_id, roster in rosters.items():
        if team_id == my_team_id or not roster:
            continue
        get_candidates = sorted(roster, key=lambda p: p.mean, reverse=True)[:get_depth]
        for give in give_candidates:
            for get in get_candidates:
                if give.position == get.position and give.mean >= get.mean:
                    continue
                proposals.append(
                    evaluate_move(
                        my_team_id,
                        rosters,
                        names,
                        settings,
                        schedule,
                        playoff_teams,
                        add=[get],
                        drop=[give],
                        banked=banked,
                        trials=trials,
                        counterparty_id=team_id,
                    )
                )

    accepted = [p for p in proposals if p.mutually_acceptable]
    accepted.sort(key=lambda p: -p.title_delta)
    return accepted


def rank_waiver_targets(
    my_team_id: int,
    rosters: dict[int, list],
    names: dict[int, str],
    settings,
    schedule: dict[int, list[int | None]],
    playoff_teams: int,
    free_agents: list,
    droppable: list,
    banked: dict[int, tuple[int, int, float]] | None = None,
    top: int = 8,
    trials: int = 400,
) -> list[MoveEvaluation]:
    """Best available add/drop pairs, ranked by championship impact."""
    evaluations = []
    for candidate in free_agents[:top]:
        worst = min(droppable, key=lambda p: p.mean, default=None)
        if worst is None:
            continue
        evaluations.append(
            evaluate_move(
                my_team_id,
                rosters,
                names,
                settings,
                schedule,
                playoff_teams,
                add=[candidate],
                drop=[worst],
                banked=banked,
                trials=trials,
            )
        )
    evaluations.sort(key=lambda e: -e.title_delta)
    return evaluations
