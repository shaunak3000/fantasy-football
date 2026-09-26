"""When an injured player comes back — which ESPN will not tell us.

The league API publishes a status (OUT, INJURY_RESERVE) and nothing else: no
return date, no expected week, on any view. Until this existed the whole repo
assumed a player on injured reserve was out for the current week only and back
the next, so A.J. Brown — out until November — counted as a healthy 15.9-point
starter from week 4. That made every trade touching the receiver room wrong, and
labelled DK Metcalf "depth — never starts" when he would in fact have started
every week until Brown returned.

Three sources, in order:

1. `data/injury_returns.json`, maintained by hand. It exists to *correct* the
   next source when the news is ahead of it, and it always wins — so an entry
   left in after the news moves on silently overrides a fresher date. Delete
   entries once they stop being corrections.
2. ESPN's public injury report (`espn_injuries.py`): an estimated return date for
   every injured player, keyed by the same id as the fantasy API. Used for IR,
   OUT, DOUBTFUL and suspensions; QUESTIONABLE players are assumed to play.
3. Otherwise, for injured reserve only, the NFL's four-game minimum counted from
   the week the player was first seen on IR in the weekly snapshots — a floor,
   not a forecast. It ran two weeks short for Jonathon Brooks (floor week 7,
   ESPN 8 November), which is why it is the last resort.

Every date resolves to the player's own team's first game on or after it, so
"back in November" is week 8 for New England, and a date past the team's last
regular-season game means out for the season — ESPN writes "Feb" for those.
Anything with no source keeps the old behaviour: out this week only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ..config import REPO_ROOT
from .espn_injuries import DATED_STATUSES

RETURNS_FILE = REPO_ROOT / "data" / "injury_returns.json"

#: NFL rules: a player placed on injured reserve must miss at least four games.
IR_MINIMUM_GAMES = 4
IR_STATUSES = frozenset({"INJURY_RESERVE"})


def normalize(name: str) -> str:
    return " ".join(name.lower().replace(".", "").replace("'", "").split())


def load_overrides(season: int, path: Path = RETURNS_FILE) -> dict[str, dict]:
    """Hand-entered return information for `season`, keyed by normalized name."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get(str(season), {}) or {}
    return {normalize(name): info for name, info in entries.items()}


def team_game_weeks(season: int) -> dict[str, list[tuple[int, date]]]:
    """Each team's regular-season (week, game date), in order. Byes are absent."""
    import nflreadpy as nfl
    import polars as pl

    schedule = nfl.load_schedules([season]).filter(pl.col("game_type") == "REG")
    games: dict[str, list[tuple[int, date]]] = {}
    for row in schedule.iter_rows(named=True):
        when = date.fromisoformat(str(row["gameday"]))
        for team in (row["home_team"], row["away_team"]):
            games.setdefault(team, []).append((int(row["week"]), when))
    for team in games:
        games[team].sort()
    return games


def resolve_return_week(
    info: dict, games: dict[str, list[tuple[int, date]]] | None = None
) -> int | None:
    """The first week the player can play, from an override entry.

    `return_week` is taken as given. `return_date` becomes the first week in
    which the player's own team plays on or after that date; without a team it
    falls back to the first week with any game on or after it.
    """
    if info.get("return_week") is not None:
        return int(info["return_week"])
    if not info.get("return_date") or not games:
        return None
    target = date.fromisoformat(info["return_date"])
    team = info.get("team")
    candidates = games.get(team) if team else None
    if candidates is None:
        candidates = sorted({g for team_games in games.values() for g in team_games})
    for week, when in candidates:
        if when >= target:
            return week
    # After the team's last regular-season game: out for the season. A week past
    # the schedule keeps him out of every remaining week and out of the bracket.
    return candidates[-1][0] + 1 if candidates else None


def first_seen_on_ir(snapshots) -> dict[int, int]:
    """ESPN id -> earliest captured week in which the player was on IR."""
    seen: dict[int, int] = {}
    for snapshot in sorted(snapshots, key=lambda s: s.week):
        for players in snapshot.teams.values():
            for player in players:
                if (player.injury_status or "").upper() in IR_STATUSES:
                    seen.setdefault(player.espn_id, snapshot.week)
    return seen


@dataclass(frozen=True)
class ReturnInfo:
    """When a player is back, and where that came from — printed in the report."""

    week: int
    source: str


def _describe(when: date | None, week: int, games: dict | None, team: str | None) -> str:
    last = max((w for w, _ in (games or {}).get(team or "", [])), default=None)
    if last is not None and week > last:
        return "out for season"
    return f"{when:%b} {when.day}" if when else f"wk{week}"


def return_info(
    name: str,
    espn_id: int | None,
    injury_status: str | None,
    current_week: int,
    overrides: dict[str, dict],
    first_seen: dict[int, int],
    games: dict[str, list[tuple[int, date]]] | None = None,
    reported: dict | None = None,
) -> ReturnInfo | None:
    """The first week this player can score and its source, or None if available.

    Each source that can answer, answers — a date resolving to this week or
    earlier means available now, and the weaker sources are not consulted.
    """
    info = overrides.get(normalize(name))
    if info is not None:
        week = resolve_return_week(info, games)
        if week is not None:
            if week <= current_week:
                return None
            return ReturnInfo(week, "injury_returns.json")

    entry = (reported or {}).get(espn_id) if espn_id is not None else None
    if entry is not None and entry.status in DATED_STATUSES and entry.return_date:
        week = resolve_return_week(
            {"team": entry.team, "return_date": entry.return_date.isoformat()}, games
        )
        if week is not None:
            if week <= current_week:
                return None
            label = _describe(entry.return_date, week, games, entry.team)
            return ReturnInfo(week, f"ESPN est. {label}")

    if (injury_status or "").upper() in IR_STATUSES:
        placed = first_seen.get(espn_id, current_week) if espn_id is not None else current_week
        return ReturnInfo(max(current_week + 1, placed + IR_MINIMUM_GAMES), "IR 4-game floor")
    return None


def return_week(*args, **kwargs) -> int | None:
    """`return_info` without the source."""
    info = return_info(*args, **kwargs)
    return info.week if info else None


def league_return_weeks(
    published: dict, current_week: int, season: int, snapshots, reported: dict | None = None
) -> dict[int, ReturnInfo]:
    """ESPN id -> return info, for every player ESPN published this week.

    `published` is `fetch_weekly_projections` output, covering every rostered
    player and the top of the wire, so both sides of any trade are valued the
    same way. The schedule is fetched only when a date needs resolving; if that
    fails, dates cannot be placed and IR falls back to the four-game floor.
    """
    overrides = load_overrides(season)
    first_seen = first_seen_on_ir(snapshots)
    games = None
    if reported or any("return_date" in info for info in overrides.values()):
        try:
            games = team_game_weeks(season)
        except Exception:  # noqa: BLE001 - never let a schedule fetch break the report
            games = None
    out: dict[int, ReturnInfo] = {}
    for espn_id, projection in published.items():
        info = return_info(
            projection.name,
            espn_id,
            projection.injury_status,
            current_week,
            overrides,
            first_seen,
            games,
            reported,
        )
        if info is not None:
            out[espn_id] = info
    return out
