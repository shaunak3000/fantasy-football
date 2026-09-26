"""Turn the live ESPN league into the inputs the simulator and trade engine want.

Those modules deliberately take plain data — rosters, a schedule, banked
results — so they can be tested without a network. This is the one place that
knows how to get that data out of ESPN, which keeps the awkwardness contained.

Every player carries **two** valuations, because two different questions get
asked of him and they have different right answers:

* `week_mean`/`week_sd` — what he will score *this* Sunday, from ESPN's own
  weekly projection, with byes and injuries zeroed. This drives start/sit.
* `mean`/`sd` — a typical week for the rest of the season, from the fitted rank
  curves. This drives the season simulator, which plays out thirteen weeks and
  must not assume every one of them looks like this one.

Collapsing the two was the original mistake: a season-long rank curve cannot see
that a tight end is facing the league's worst defence, and a single week's
projection should not be extrapolated across a whole season.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from espn_api.football import League
from espn_api.football.constant import POSITION_MAP

from ..data.espn import WeeklyProjection
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


# Lineup-slot ids that mean "not in the starting lineup". espn-api hands us the
# slot as a *name*, so the name is mapped back through its own table rather than
# hardcoding numbers a future ESPN change could move.
SLOT_ID_BY_NAME = {
    name: slot_id for slot_id, name in POSITION_MAP.items() if isinstance(slot_id, int)
}
BENCH_SLOT_IDS = frozenset({20, 21, 24})  # BE, IR, ER
UNKNOWN_SLOT_ID = -1


@dataclass
class RosterPlayer:
    """A rostered player expressed the way the optimizer expects."""

    player: str
    position: str
    espn_id: int
    mean: float
    sd: float
    slot_id: int = UNKNOWN_SLOT_ID
    bye_week: int | None = None
    on_bye: bool = False
    week_mean: float = 0.0
    week_sd: float = 0.0
    injury_status: str | None = None
    unavailable: bool = False
    #: First week this player can score, when he is out beyond this week — from
    #: `data/injury_returns.json` or the IR four-game floor. None means available.
    return_week: int | None = None

    @property
    def started(self) -> bool:
        """Whether ESPN currently has this player in a starting slot.

        An unknown slot is *not* treated as started. The previous default did
        the opposite, which quietly made every player on the roster a "current
        starter" and turned the current-versus-optimal comparison into a
        comparison of the best nine against all sixteen — a test nothing could
        fail, reported every week as "your lineup is already optimal".
        """
        return self.slot_id != UNKNOWN_SLOT_ID and self.slot_id not in BENCH_SLOT_IDS

    @property
    def playable(self) -> bool:
        return not self.on_bye and not self.unavailable

    def this_week(self) -> RosterPlayer:
        """A copy whose `mean`/`sd` are this week's numbers.

        The optimizer reads `.mean` and `.sd`, so swapping them here lets the
        same solver answer the weekly question without knowing there are two
        valuations in play.
        """
        return replace(self, mean=self.week_mean, sd=self.week_sd)


def this_week(roster: list[RosterPlayer]) -> list[RosterPlayer]:
    return [player.this_week() for player in roster]


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


# Games a healthy player is available for, used to convert a season projection
# into a per-week one when no fitted weekly curve exists.
TYPICAL_GAMES_PLAYED = 16.0
# Weekly volatility relative to production for positions with no fitted spread.
UNFITTED_WEEKLY_CV = 0.55

# Weekly volatility relative to production, measured per position in
# `check_weekly`. Tight ends swing furthest for what they score; quarterbacks
# least. Used to turn ESPN's single weekly point projection into a distribution,
# which it does not publish.
WEEKLY_CV = {
    "QB": 0.40,
    "RB": 0.49,
    "WR": 0.49,
    "TE": 0.58,
    "K": 0.35,
    "D/ST": 0.60,
}


def _weekly_estimate(projection, model: WeeklyModel) -> tuple[float, float]:
    """Per-game mean and spread for a player, over the rest of the season.

    Uses the fitted per-game curves where they exist. Positions without one —
    team defenses, whose scoring inputs were never mapped — fall back to their
    season projection divided across a season. Without that fallback every
    defense is worth zero every week, so the optimizer cannot tell them apart
    and quietly understates the lineup it recommends.
    """
    rank = getattr(projection, "blended_rank", getattr(projection, "consensus_rank", 50))
    per_game = model.per_game.get(projection.position)
    spread = model.spread.get(projection.position)

    if per_game is None:
        mean = float(getattr(projection, "mean", 0.0)) / TYPICAL_GAMES_PLAYED
        return mean, max(mean * UNFITTED_WEEKLY_CV, 1.0)

    mean = per_game.points_at(rank)
    sd = spread.points_at(rank) if spread else max(mean * UNFITTED_WEEKLY_CV, 1.0)
    return mean, sd


def build_state(
    league: League,
    settings,
    projections: list,
    weekly_model: WeeklyModel,
    current_week: int,
    free_agent_pool: int = 60,
    byes: dict[int, int] | None = None,
    weekly_projections: dict[int, WeeklyProjection] | None = None,
    return_weeks: dict[int, int] | None = None,
) -> LeagueState:
    """Assemble rosters, remaining schedule, and standings from the live league.

    `return_weeks` maps ESPN id to the first week a long-term injured player can
    score (see `data/injuries.py`). It is computed by the caller so that this
    function stays free of file and network access.
    """
    by_id = {p.espn_id: p for p in projections if p.espn_id is not None}
    byes = byes or {}
    return_weeks = return_weeks or {}
    weekly_projections = weekly_projections or {}

    def to_roster_player(espn_id, name, position, slot_id=UNKNOWN_SLOT_ID) -> RosterPlayer | None:
        projection = by_id.get(espn_id)
        if projection is None:
            return None
        mean, sd = _weekly_estimate(projection, weekly_model)

        published = weekly_projections.get(espn_id)
        if published is not None and published.points > 0:
            week_mean = published.points
            week_sd = max(week_mean * WEEKLY_CV.get(position, UNFITTED_WEEKLY_CV), 1.0)
        else:
            week_mean, week_sd = mean, sd

        # A player on bye scores exactly zero, so he must be worth zero to the
        # optimizer. Without this the solver cheerfully starts him — the single
        # most damaging routine mistake there is, and one no amount of
        # projection accuracy elsewhere can make up for.
        bye_week = byes.get(espn_id)
        on_bye = bye_week is not None and bye_week == current_week

        # An injured player is the same mistake wearing different clothes, and
        # ESPN publishes a projection for him regardless: A.J. Brown sat on
        # injured reserve with a 13.8-point week-2 projection attached, and the
        # solver duly started him. Only *this week* is zeroed — a player on IR
        # in September is usually back well before the season ends, so the
        # rest-of-season valuation is left alone.
        injury_status = published.injury_status if published is not None else None
        unavailable = published is not None and published.unavailable
        if on_bye or unavailable:
            week_mean, week_sd = 0.0, 0.0

        return RosterPlayer(
            player=name,
            position=position,
            espn_id=espn_id,
            mean=mean,
            sd=sd,
            slot_id=slot_id,
            bye_week=bye_week,
            on_bye=on_bye,
            week_mean=week_mean,
            week_sd=week_sd,
            injury_status=injury_status,
            unavailable=unavailable,
            return_week=return_weeks.get(espn_id),
        )

    state = LeagueState(settings=settings, current_week=current_week)

    for team in league.teams:
        state.names[team.team_id] = team.team_name
        players = []
        for player in team.roster:
            position = getattr(player, "position", None)
            # espn-api reports the lineup slot by name; the id is what the rest
            # of the repo speaks, and what distinguishes a bench spot from a
            # flex one.
            slot_id = SLOT_ID_BY_NAME.get(getattr(player, "lineupSlot", None), UNKNOWN_SLOT_ID)
            entry = to_roster_player(player.playerId, player.name, position, slot_id)
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
