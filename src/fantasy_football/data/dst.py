"""Team-defense stat lines, in ESPN's stat-id space.

Defenses score from a different world than skill players: points allowed, yards
allowed, and takeaways, none of which appear in a player stats table. That is
why the position had no historical fit and was, for a while, silently missing
from the board entirely.

Everything needed is derivable from nflverse: a team's own defensive counters
come straight from team stats, while points and yards allowed come from the
other side of the same game. Once assembled, the ordinary scoring engine can
price a defense exactly as it prices anyone else.
"""

from __future__ import annotations

import polars as pl

from .nflverse import _cached

# ESPN awards points-allowed and yards-allowed in bands rather than per unit, so
# a defense's stat line carries a 1 in exactly one bucket from each ladder.
# Gaps are intentional: 18-27 points and 300-349 yards score nothing.
POINTS_ALLOWED_TIERS = (
    (0, 0, 89),
    (1, 6, 90),
    (7, 13, 91),
    (14, 17, 92),
    (28, 34, 123),
    (35, 45, 124),
    (46, 999, 125),
)
YARDS_ALLOWED_TIERS = (
    (0, 99, 128),
    (100, 199, 129),
    (200, 299, 130),
    (350, 399, 132),
    (400, 449, 133),
    (450, 499, 134),
    (500, 549, 135),
    (550, 9999, 136),
)

# Direct counters, mapped to the stat id the league scores them under.
DEFENSIVE_COUNTERS = {
    "def_sacks": 99,
    "def_interceptions": 95,
    "fumble_recovery_opp": 96,
    "def_fumbles_forced": 106,
    "def_safeties": 98,
    # nflverse reports defensive touchdowns as one number, without saying whether
    # they came from an interception or a fumble. Both score 6 here, so the
    # distinction does not change the total.
    "def_tds": 103,
}


def _tier_id(value: float, tiers) -> int | None:
    for low, high, stat_id in tiers:
        if low <= value <= high:
            return stat_id
    return None


def load_team_stats(seasons: list[int], refresh: bool = False) -> pl.DataFrame:
    import nflreadpy as nfl

    key = "_".join(str(s) for s in seasons)
    return _cached(
        f"team_stats_{key}",
        lambda: nfl.load_team_stats(seasons=seasons, summary_level="week"),
        refresh,
    )


def dst_stat_lines(seasons: list[int], refresh: bool = False) -> list[dict]:
    """One row per defense per week, with ESPN stat ids as columns.

    Points and yards allowed are read off the opponent's row in the same game,
    which is what makes this possible at all — a defense's most valuable stats
    are really facts about the offense it faced.
    """
    stats = load_team_stats(seasons, refresh=refresh).filter(pl.col("season_type") == "REG")

    offense = stats.select(
        pl.col("season"),
        pl.col("week"),
        pl.col("team").alias("opponent_team"),
        (pl.col("passing_yards").fill_null(0) + pl.col("rushing_yards").fill_null(0)).alias(
            "yards_allowed"
        ),
    )

    joined = stats.join(offense, on=["season", "week", "opponent_team"], how="left")

    # Points allowed come from the scoreboard rather than being reconstructed
    # from touchdowns and kicks, which would miss two-point plays and safeties.
    import nflreadpy as nfl

    schedules = nfl.load_schedules(seasons=seasons)
    home = schedules.select(
        pl.col("season"),
        pl.col("week"),
        pl.col("home_team").alias("team"),
        pl.col("away_score").alias("points_allowed"),
    )
    away = schedules.select(
        pl.col("season"),
        pl.col("week"),
        pl.col("away_team").alias("team"),
        pl.col("home_score").alias("points_allowed"),
    )
    allowed = pl.concat([home, away], how="vertical")

    joined = joined.join(allowed, on=["season", "week", "team"], how="left")

    rows = []
    for row in joined.iter_rows(named=True):
        line: dict[int, float] = {}
        for column, stat_id in DEFENSIVE_COUNTERS.items():
            value = row.get(column)
            if value:
                line[stat_id] = line.get(stat_id, 0.0) + float(value)

        points_allowed = row.get("points_allowed")
        if points_allowed is not None:
            tier = _tier_id(float(points_allowed), POINTS_ALLOWED_TIERS)
            if tier:
                line[tier] = 1.0

        yards_allowed = row.get("yards_allowed")
        if yards_allowed is not None:
            tier = _tier_id(float(yards_allowed), YARDS_ALLOWED_TIERS)
            if tier:
                line[tier] = 1.0

        rows.append(
            {
                "season": row["season"],
                "week": row["week"],
                "team": row["team"],
                "stats": line,
            }
        )

    # Returned as plain records rather than a frame: the stat line is a dict
    # keyed by integer stat id, which polars cannot hold in a column.
    return rows


def score_dst_seasons(seasons: list[int], engine, dst_position_id: int = 16) -> pl.DataFrame:
    """Season fantasy totals per defense, under this league's own scoring."""
    scored = [
        {
            "season": row["season"],
            "team": row["team"],
            "points": engine.score(row["stats"], dst_position_id),
        }
        for row in dst_stat_lines(seasons)
    ]
    frame = pl.DataFrame(scored)
    return (
        frame.group_by(["season", "team"])
        .agg(
            pl.col("points").sum().alias("actual_points"),
            pl.len().alias("games_played"),
            pl.col("points").std().alias("weekly_sd"),
        )
        .sort(["season", "actual_points"], descending=[False, True])
    )
