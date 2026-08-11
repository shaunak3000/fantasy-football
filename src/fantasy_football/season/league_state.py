"""Turn the live ESPN league into the inputs the simulator and trade engine want.

Those modules deliberately take plain data — rosters, a schedule, banked
results — so they can be tested without a network. This is the one place that
knows how to get that data out of ESPN, which keeps the awkwardness contained.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from espn_api.football import League

from ..data.ids import attach_espn_ids
from ..data.nflverse import load_consensus_board
from ..projections.weekly import WeeklyModel


def bye_weeks_by_espn_id() -> dict[int, int]:
    """Bye week per ESPN player id, taken from the consensus board."""
    matched = attach_espn_ids(load_consensus_board()).matched
    return {
        int(row["espn_id"]): int(row["bye"])
        for row in matched.iter_rows(named=True)
        if row.get("espn_id") is not None and row.get("bye") is not None
    }


@dataclass
class RosterPlayer:
    """A rostered player expressed the way the optimizer expects."""

    player: str
    position: str
    espn_id: int
    mean: float
    sd: float
    slot_id: int = -1
    bye_week: int | None = None
    on_bye: bool = False

    @property
    def started(self) -> bool:
        return self.slot_id not in {20, 21, 24}


@dataclass
class LeagueState:
    settings: object
    rosters: dict[int, list[RosterPlayer]] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)
    schedule: dict[int, list[int | None]] = field(default_factory=dict)
    banked: dict[int, tuple[int, int, float]] = field(default_factory=dict)
    free_agents: list[RosterPlayer] = field(default_factory=list)
    current_week: int = 1

    @property
    def remaining_weeks(self) -> int:
        return max((len(games) for games in self.schedule.values()), default=0)


POSITION_BY_ID = {0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "D/ST", 17: "K"}


def _weekly_estimate(projection, model: WeeklyModel) -> tuple[float, float]:
    """Per-game mean and spread for a player, from the rank curves."""
    per_game = model.per_game.get(projection.position)
    spread = model.spread.get(projection.position)
    rank = getattr(projection, "blended_rank", getattr(projection, "consensus_rank", 50))
    mean = per_game.points_at(rank) if per_game else 0.0
    sd = spread.points_at(rank) if spread else max(mean * 0.5, 1.0)
    return mean, sd


def build_state(
    league: League,
    settings,
    projections: list,
    weekly_model: WeeklyModel,
    current_week: int,
    free_agent_pool: int = 60,
    byes: dict[int, int] | None = None,
) -> LeagueState:
    """Assemble rosters, remaining schedule, and standings from the live league."""
    by_id = {p.espn_id: p for p in projections if p.espn_id is not None}
    byes = byes or {}

    def to_roster_player(espn_id, name, position, slot_id=-1) -> RosterPlayer | None:
        projection = by_id.get(espn_id)
        if projection is None:
            return None
        mean, sd = _weekly_estimate(projection, weekly_model)

        # A player on bye scores exactly zero, so he must be worth zero to the
        # optimizer. Without this the solver cheerfully starts him — the single
        # most damaging routine mistake there is, and one no amount of
        # projection accuracy elsewhere can make up for.
        bye_week = byes.get(espn_id)
        on_bye = bye_week is not None and bye_week == current_week
        if on_bye:
            mean, sd = 0.0, 0.0

        return RosterPlayer(
            player=name,
            position=position,
            espn_id=espn_id,
            mean=mean,
            sd=sd,
            slot_id=slot_id,
            bye_week=bye_week,
            on_bye=on_bye,
        )

    state = LeagueState(settings=settings, current_week=current_week)

    for team in league.teams:
        state.names[team.team_id] = team.team_name
        players = []
        for player in team.roster:
            position = getattr(player, "position", None)
            entry = to_roster_player(player.playerId, player.name, position)
            if entry is not None:
                players.append(entry)
        state.rosters[team.team_id] = players
        state.banked[team.team_id] = (
            getattr(team, "wins", 0),
            getattr(team, "losses", 0),
            float(getattr(team, "points_for", 0.0)),
        )

        upcoming = []
        for week_index, opponent in enumerate(getattr(team, "schedule", []), start=1):
            if week_index < current_week:
                continue
            if week_index > settings.regular_season_weeks:
                break
            upcoming.append(getattr(opponent, "team_id", None))
        state.schedule[team.team_id] = upcoming

    for player in league.free_agents(size=free_agent_pool):
        position = POSITION_BY_ID.get(getattr(player, "position", None)) or getattr(
            player, "position", None
        )
        entry = to_roster_player(player.playerId, player.name, position)
        if entry is not None:
            state.free_agents.append(entry)

    return state
