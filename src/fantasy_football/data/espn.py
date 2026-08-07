"""ESPN league client.

Everything downstream reads the league's real configuration from here — scoring
rules, roster slots, league size, keeper and FAAB settings. Nothing about the
league is hardcoded anywhere else in the repo.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from espn_api.football import League
from espn_api.football.constant import POSITION_MAP, SETTINGS_SCORING_FORMAT_MAP

from ..config import EspnCredentials, cache_path, load_credentials

RECEPTION_STAT_ID = 53

STARTABLE_SLOTS = frozenset(
    {"QB", "RB", "WR", "TE", "K", "D/ST", "FLEX", "RB/WR", "RB/WR/TE", "WR/TE", "OP", "DP"}
)
NON_STARTABLE_SLOTS = frozenset({"BE", "IR", "ER", "Rookie", ""})


# ESPN's player-position enumeration, which is *not* the lineup-slot enumeration
# in POSITION_MAP. Team defenses happen to be 16 in both.
DST_POSITION_ID = 16


@dataclass(frozen=True)
class ScoringRule:
    """A single scoring rule, with the per-position exceptions ESPN allows.

    A league can score the same stat differently by position — this one gives a
    team defense -2 for allowing 400-449 yards while scoring it 0 for everyone
    else. Collapsing the override onto the base value (which is what espn-api
    does) would apply defensive scoring to skill players.
    """

    stat_id: int
    abbr: str
    label: str
    points: float
    position_overrides: Mapping[int, float] = field(default_factory=dict)

    def points_for(self, position_id: int | None = None) -> float:
        if position_id is not None and position_id in self.position_overrides:
            return self.position_overrides[position_id]
        return self.points

    @property
    def is_active(self) -> bool:
        return bool(self.points) or any(self.position_overrides.values())


@dataclass(frozen=True)
class LeagueSettings:
    """The league's configuration, flattened into the shape the rest of the repo wants."""

    league_id: int
    season: int
    name: str
    team_count: int
    scoring_type: str | None
    points_per_reception: float
    roster_slots: dict[str, int]
    starting_slots: dict[str, int]
    bench_size: int
    ir_slots: int
    keeper_count: int
    playoff_team_count: int
    regular_season_weeks: int
    playoff_matchup_length: int
    uses_faab: bool
    acquisition_budget: int
    scoring_rules: list[ScoringRule] = field(default_factory=list)

    @property
    def starters_per_team(self) -> int:
        return sum(self.starting_slots.values())

    @property
    def scoring_label(self) -> str:
        ppr = self.points_per_reception
        if ppr == 0:
            return "standard"
        if ppr == 1:
            return "full PPR"
        return f"{ppr} PPR"

    def describe(self) -> str:
        slots = ", ".join(f"{n}{slot}" for slot, n in self.starting_slots.items())
        return (
            f"{self.name} — {self.team_count} teams, {self.scoring_label}, "
            f"starters: {slots}, bench {self.bench_size}, "
            f"{self.regular_season_weeks}-week regular season, "
            f"{self.playoff_team_count} playoff teams"
        )


def connect(creds: EspnCredentials | None = None) -> League:
    creds = creds or load_credentials()
    return League(
        league_id=creds.league_id,
        year=creds.season,
        espn_s2=creds.espn_s2,
        swid=creds.swid,
    )


def fetch_raw_settings(league: League) -> dict:
    """Pull the unparsed mSettings payload.

    espn-api's own `position_slot_counts` maps slot ids positionally, which only
    happens to line up when ESPN returns all 26 slots in order. Parsing the raw
    payload keyed by slot id removes that assumption, and the raw blob is what
    gets snapshotted for offline tests.
    """
    data = league.espn_request.league_get(params={"view": "mSettings"})
    return data.get("settings", data)


def parse_settings(raw: dict, league_id: int, season: int) -> LeagueSettings:
    roster = raw.get("rosterSettings", {})
    schedule = raw.get("scheduleSettings", {})
    scoring = raw.get("scoringSettings", {})
    draft = raw.get("draftSettings", {})
    acquisition = raw.get("acquisitionSettings", {})

    slot_counts = {
        POSITION_MAP.get(int(slot_id), f"SLOT_{slot_id}"): count
        for slot_id, count in roster.get("lineupSlotCounts", {}).items()
        if count
    }

    rules = []
    ppr = 0.0
    for item in scoring.get("scoringItems", []):
        stat_id = item["statId"]
        meta = SETTINGS_SCORING_FORMAT_MAP.get(stat_id, {"abbr": "UNK", "label": "Unknown"})
        base = float(item.get("points", 0) or 0)
        overrides = {int(k): float(v) for k, v in (item.get("pointsOverrides") or {}).items()}
        rules.append(
            ScoringRule(
                stat_id=stat_id,
                abbr=meta["abbr"],
                label=meta["label"],
                points=base,
                position_overrides=overrides,
            )
        )
        if stat_id == RECEPTION_STAT_ID:
            ppr = base

    return LeagueSettings(
        league_id=league_id,
        season=season,
        name=raw.get("name", "Unknown league"),
        team_count=raw.get("size", 0),
        scoring_type=scoring.get("scoringType"),
        points_per_reception=ppr,
        roster_slots=slot_counts,
        starting_slots={k: v for k, v in slot_counts.items() if k in STARTABLE_SLOTS},
        bench_size=slot_counts.get("BE", 0),
        ir_slots=slot_counts.get("IR", 0),
        keeper_count=draft.get("keeperCount", 0),
        playoff_team_count=schedule.get("playoffTeamCount", 0),
        regular_season_weeks=schedule.get("matchupPeriodCount", 0),
        playoff_matchup_length=schedule.get("playoffMatchupPeriodLength", 0),
        uses_faab=bool(acquisition.get("isUsingAcquisitionBudget", False)),
        acquisition_budget=acquisition.get("acquisitionBudget", 0),
        scoring_rules=rules,
    )


def load_settings(
    league: League | None = None, creds: EspnCredentials | None = None
) -> LeagueSettings:
    creds = creds or load_credentials()
    league = league or connect(creds)
    return parse_settings(fetch_raw_settings(league), creds.league_id, creds.season)


def snapshot_settings(league: League, creds: EspnCredentials) -> Path:
    """Write the raw settings payload to cache so the test suite runs offline."""
    raw = fetch_raw_settings(league)
    path = cache_path(f"settings_{creds.league_id}_{creds.season}.json")
    path.write_text(
        json.dumps(
            {
                "fetched_at": datetime.now(UTC).isoformat(),
                "league_id": creds.league_id,
                "season": creds.season,
                "settings": raw,
                "parsed": asdict(parse_settings(raw, creds.league_id, creds.season)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


ACTUAL_STAT_SOURCE = 0
PROJECTED_STAT_SOURCE = 1
WEEKLY_SPLIT = 1


@dataclass(frozen=True)
class PlayerWeek:
    """One player's actual production in one week, in ESPN's own stat-id space.

    `stats` is raw production keyed by stat id; `applied_total` is the points ESPN
    awarded. Keeping both is what makes the scoring engine checkable: the engine
    reads `stats` and must land on `applied_total`.
    """

    espn_id: int
    name: str
    position_id: int
    week: int
    stats: Mapping[int, float]
    applied_stats: Mapping[int, float]
    applied_total: float


def fetch_player_weeks(league: League, week: int) -> list[PlayerWeek]:
    """Every rostered player's actual stat line for one week.

    Reads the raw roster payload rather than `box_scores()`, which raises on
    historical seasons because ESPN omits `rosterForCurrentScoringPeriod` there.
    """
    data = league.espn_request.league_get(params={"view": "mRoster", "scoringPeriodId": week})

    out: list[PlayerWeek] = []
    for team in data.get("teams", []):
        for entry in team.get("roster", {}).get("entries", []):
            player = entry.get("playerPoolEntry", {}).get("player", {})
            actual = next(
                (
                    s
                    for s in player.get("stats", [])
                    if s.get("statSourceId") == ACTUAL_STAT_SOURCE
                    and s.get("statSplitTypeId") == WEEKLY_SPLIT
                    and s.get("scoringPeriodId") == week
                ),
                None,
            )
            if actual is None:
                continue

            out.append(
                PlayerWeek(
                    espn_id=player.get("id"),
                    name=player.get("fullName", ""),
                    position_id=player.get("defaultPositionId", -1),
                    week=week,
                    stats={int(k): float(v) for k, v in (actual.get("stats") or {}).items()},
                    applied_stats={
                        int(k): float(v) for k, v in (actual.get("appliedStats") or {}).items()
                    },
                    applied_total=float(actual.get("appliedTotal") or 0.0),
                )
            )
    return out


def load_settings_snapshot(league_id: int, season: int) -> LeagueSettings:
    path = cache_path(f"settings_{league_id}_{season}.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return parse_settings(payload["settings"], league_id, season)
