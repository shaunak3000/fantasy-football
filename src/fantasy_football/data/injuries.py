"""When an injured player comes back — which ESPN will not tell us.

The league API publishes a status (OUT, INJURY_RESERVE) and nothing else: no
return date, no expected week, on any view. Until this existed the whole repo
assumed a player on injured reserve was out for the current week only and back
the next, so A.J. Brown — out until November — counted as a healthy 15.9-point
starter from week 4. That made every trade touching the receiver room wrong, and
labelled DK Metcalf "depth — never starts" when he would in fact have started
every week until Brown returned.

Two sources, in order:

1. `data/injury_returns.json`, maintained by hand from the news. Tracked in git.
   A `return_date` is resolved to the player's own team's first game on or after
   it, so "back in November" lands on the right week even when his team plays on
   a Thursday.
2. Otherwise, for injured reserve only, the NFL's four-game minimum counted from
   the week the player was first seen on IR in the weekly snapshots. That is a
   floor, not a forecast — a real placement can be earlier than our first
   capture, and many injuries run well past four games — but it is far closer
   than "back next week", and it applies to every roster, so the other side of a
   trade is valued the same way.

OUT and DOUBTFUL keep the old behaviour: out this week only.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ..config import REPO_ROOT

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
    return None


def first_seen_on_ir(snapshots) -> dict[int, int]:
    """ESPN id -> earliest captured week in which the player was on IR."""
    seen: dict[int, int] = {}
    for snapshot in sorted(snapshots, key=lambda s: s.week):
        for players in snapshot.teams.values():
            for player in players:
                if (player.injury_status or "").upper() in IR_STATUSES:
                    seen.setdefault(player.espn_id, snapshot.week)
    return seen


def return_week(
    name: str,
    espn_id: int | None,
    injury_status: str | None,
    current_week: int,
    overrides: dict[str, dict],
    first_seen: dict[int, int],
    games: dict[str, list[tuple[int, date]]] | None = None,
) -> int | None:
    """The first week this player can score, or None if he is available now.

    An override always wins, even for a player ESPN now lists as healthy — the
    news runs ahead of the status. IR without an override falls back to the
    four-game minimum. Anything else is out at most for the current week, which
    is already handled where availability is computed.
    """
    info = overrides.get(normalize(name))
    if info is not None:
        week = resolve_return_week(info, games)
        if week is not None:
            return week if week > current_week else None
    if (injury_status or "").upper() in IR_STATUSES:
        placed = first_seen.get(espn_id, current_week) if espn_id is not None else current_week
        return max(current_week + 1, placed + IR_MINIMUM_GAMES)
    return None


def league_return_weeks(
    published: dict, current_week: int, season: int, snapshots
) -> dict[int, int]:
    """ESPN id -> first week back, for every player ESPN published this week.

    `published` is `fetch_weekly_projections` output, which covers every rostered
    player and the top of the wire, so both sides of any trade are covered. The
    schedule is only fetched when an override needs a date resolved, and a failure
    there degrades to the IR floor rather than taking the report down.
    """
    overrides = load_overrides(season)
    first_seen = first_seen_on_ir(snapshots)
    games = None
    if any("return_date" in info for info in overrides.values()):
        try:
            games = team_game_weeks(season)
        except Exception:  # noqa: BLE001 - never let a schedule fetch break the report
            games = None
    out: dict[int, int] = {}
    for espn_id, projection in published.items():
        week = return_week(
            projection.name,
            espn_id,
            projection.injury_status,
            current_week,
            overrides,
            first_seen,
            games,
        )
        if week is not None:
            out[espn_id] = week
    return out


def source_of(name: str, season: int) -> str:
    """Where a return week came from, for the report."""
    return "injury_returns.json" if normalize(name) in load_overrides(season) else "IR 4-game floor"
