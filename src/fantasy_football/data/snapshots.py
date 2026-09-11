"""Capture what each manager actually started, week by week, while it still exists.

**This is the only irreplaceable data in the repo.** ESPN serves exactly one
roster per league: the current one. A request for a past week's `mRoster` comes
back with today's players in today's lineup slots — week 1 and week 13 of a
finished season are byte-identical — so there is no way to ask, later, what
somebody started in week 3. `box_scores()` fails on historical seasons for the
same reason.

Everything else here can be refetched. Rankings, projections, stat lines and
scoreboard totals are all still there next month. A week's lineups are not, and
once the scoring period rolls over they are gone permanently.

That gap is why this repo's headline claim had to be withdrawn: `check_optimizer`
appeared to show the lineup optimizer beating all eight managers by 200 points a
season, but the "manager" baseline it compared against was the lineup each
manager *finished* with, replayed against every week, because that is the only
thing ESPN would tell it. Against real scoreboard totals the edge measured +0.5
points a week — indistinguishable from zero.

Settling that question needs lineups recorded as they happen. A season of these
files is what makes `check_optimizer` honest, and nothing captured after the
fact can substitute for them.

Snapshots are written to `data/snapshots/`, which is tracked in git rather than
gitignored like `data/cache/`, so they travel between machines with the code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from espn_api.football import League

from ..config import snapshot_path
from .espn import (
    ACTUAL_STAT_SOURCE,
    PLAYER_POSITION_BY_ID,
    PROJECTED_STAT_SOURCE,
    WEEKLY_SPLIT,
)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PlayerSnapshot:
    """One player's situation in one week, as it stood when captured."""

    espn_id: int
    name: str
    position: str | None
    #: The lineup slot ESPN had him in. This is the field that cannot be
    #: recovered later, and the reason this module exists.
    slot_id: int | None
    injury_status: str | None
    projected: float | None
    actual: float | None

    @property
    def started(self) -> bool:
        return self.slot_id is not None and self.slot_id not in BENCH_SLOT_IDS


BENCH_SLOT_IDS = frozenset({20, 21, 24})  # BE, IR, ER


@dataclass(frozen=True)
class WeekSnapshot:
    season: int
    week: int
    captured_at: str
    #: The league's live scoring period when this was taken. If it does not
    #: equal `week`, the lineups in here are not that week's lineups — they are
    #: whatever the rosters looked like at capture time, and are not evidence.
    nfl_week_at_capture: int
    schema: int
    teams: dict[int, list[PlayerSnapshot]]
    records: dict[int, tuple[int, int, float]]
    names: dict[int, str]

    @property
    def trustworthy(self) -> bool:
        """Whether the lineups here really are this week's lineups."""
        return self.nfl_week_at_capture == self.week

    @property
    def has_results(self) -> bool:
        return any(
            player.actual is not None for players in self.teams.values() for player in players
        )


def _stat(player: dict, source: int, week: int) -> dict | None:
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


def capture(league: League, week: int, season: int | None = None) -> WeekSnapshot:
    """Record every roster in the league exactly as it stands right now."""
    payload = league.espn_request.league_get(params={"view": "mRoster", "scoringPeriodId": week})
    live_week = max(int(getattr(league, "nfl_week", week) or week), 1)

    teams: dict[int, list[PlayerSnapshot]] = {}
    for team in payload.get("teams", []):
        team_id = team.get("id")
        players = []
        for entry in team.get("roster", {}).get("entries", []):
            raw = entry.get("playerPoolEntry", {}).get("player", {})
            actual_block = _stat(raw, ACTUAL_STAT_SOURCE, week)
            projected_block = _stat(raw, PROJECTED_STAT_SOURCE, week)
            # An empty stat block means the game has not been played, which is
            # different from a player who played and scored nothing.
            actual = None
            if actual_block and (actual_block.get("stats") or {}):
                actual = float(actual_block.get("appliedTotal") or 0.0)
            players.append(
                PlayerSnapshot(
                    espn_id=int(raw.get("id", 0)),
                    name=raw.get("fullName", "?"),
                    position=PLAYER_POSITION_BY_ID.get(raw.get("defaultPositionId", -1)),
                    slot_id=entry.get("lineupSlotId"),
                    injury_status=raw.get("injuryStatus"),
                    projected=(
                        float(projected_block.get("appliedTotal") or 0.0)
                        if projected_block
                        else None
                    ),
                    actual=actual,
                )
            )
        if players:
            teams[team_id] = players

    return WeekSnapshot(
        season=season or league.year,
        week=week,
        captured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        nfl_week_at_capture=live_week,
        schema=SCHEMA_VERSION,
        teams=teams,
        records={
            t.team_id: (
                int(getattr(t, "wins", 0)),
                int(getattr(t, "losses", 0)),
                float(getattr(t, "points_for", 0.0)),
            )
            for t in league.teams
        },
        names={t.team_id: t.team_name for t in league.teams},
    )


def path_for(season: int, week: int) -> Path:
    return snapshot_path(str(season), f"week{week:02d}.json")


def save(snapshot: WeekSnapshot, force: bool = False) -> tuple[Path | None, str]:
    """Write a snapshot, refusing to destroy a better one.

    Returns `(path, reason)`; `path` is None when nothing was written.

    The refusals matter more than the writes. Running `weekly.py 3` in week 10
    would otherwise overwrite the real week-3 lineups with the week-10 roster,
    because that is what ESPN hands back — quietly replacing the only copy of
    the data with a worthless one. So a stale capture never lands, and a capture
    without results never replaces one that has them.
    """
    destination = path_for(snapshot.season, snapshot.week)
    existing = load(snapshot.season, snapshot.week)

    if not snapshot.trustworthy and not force:
        # The lineups in a stale capture are worthless, but the *stat lines*
        # are not: ESPN keeps those per week forever, and a week captured live
        # on Sunday has only half its scores in. So a late capture is allowed to
        # fill in results, as long as it does not touch the recorded slots.
        if existing is not None and existing.trustworthy and snapshot.has_results:
            snapshot = _with_results_from(existing, snapshot)
        else:
            return None, (
                f"refused: asked for week {snapshot.week} while the league is on "
                f"week {snapshot.nfl_week_at_capture}, so these are not that week's lineups"
            )

    if existing is not None and not force and existing.has_results and not snapshot.has_results:
        return None, "refused: would replace a snapshot that has results with one that has none"

    payload = {
        "season": snapshot.season,
        "week": snapshot.week,
        "captured_at": snapshot.captured_at,
        "nfl_week_at_capture": snapshot.nfl_week_at_capture,
        "schema": snapshot.schema,
        "names": {str(k): v for k, v in snapshot.names.items()},
        "records": {str(k): list(v) for k, v in snapshot.records.items()},
        "teams": {
            str(team_id): [asdict(p) for p in players]
            for team_id, players in snapshot.teams.items()
        },
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    scored = sum(1 for players in snapshot.teams.values() for p in players if p.actual is not None)
    verb = "updated" if existing is not None else "captured"
    return destination, (
        f"{verb} week {snapshot.week} ({len(snapshot.teams)} rosters, {scored} players scored)"
    )


def _with_results_from(kept: WeekSnapshot, fresh: WeekSnapshot) -> WeekSnapshot:
    """The lineups we already recorded, with stat lines brought up to date.

    Slots come from `kept` because those are the ones taken while the week was
    live and are the only record of them. Only `actual` is taken from `fresh`.
    """
    scores = {
        player.espn_id: player.actual
        for players in fresh.teams.values()
        for player in players
        if player.actual is not None
    }
    teams = {
        team_id: [
            replace(player, actual=scores.get(player.espn_id, player.actual)) for player in players
        ]
        for team_id, players in kept.teams.items()
    }
    return replace(
        kept,
        teams=teams,
        captured_at=kept.captured_at,
        records=fresh.records or kept.records,
    )


def load(season: int, week: int) -> WeekSnapshot | None:
    path = path_for(season, week)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return WeekSnapshot(
        season=raw["season"],
        week=raw["week"],
        captured_at=raw["captured_at"],
        nfl_week_at_capture=raw.get("nfl_week_at_capture", raw["week"]),
        schema=raw.get("schema", 1),
        teams={
            int(team_id): [PlayerSnapshot(**p) for p in players]
            for team_id, players in raw["teams"].items()
        },
        records={int(k): tuple(v) for k, v in raw.get("records", {}).items()},
        names={int(k): v for k, v in raw.get("names", {}).items()},
    )


def load_season(season: int) -> list[WeekSnapshot]:
    """Every captured week, in order. The input a real lineup backtest needs."""
    directory = snapshot_path(str(season))
    if not directory.exists():
        return []
    weeks = []
    for path in sorted(directory.glob("week*.json")):
        snapshot = load(season, int(path.stem.removeprefix("week")))
        if snapshot is not None:
            weeks.append(snapshot)
    return weeks


def capture_and_save(league: League, week: int, season: int | None = None) -> str:
    """Capture the given week and report what happened, never raising.

    This runs as a side effect of the weekly report, where a network hiccup or
    an ESPN change must not take the report down with it.
    """
    try:
        _, reason = save(capture(league, week, season))
    except Exception as exc:  # noqa: BLE001 - a failed snapshot must not break the report
        return f"snapshot failed: {type(exc).__name__}: {exc}"
    return reason
