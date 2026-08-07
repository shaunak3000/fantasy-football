"""Translate nflverse stat columns into ESPN's stat-id space.

The scoring engine speaks ESPN stat ids. ESPN only publishes stat lines for
players on a roster in this league, which is a small and biased slice of the
NFL — useless for fitting replacement level or a positional rank curve, both of
which need every player who took a snap. nflverse has the complete history, so
this module is the adapter that lets the engine score it.

Known unmapped rules are declared rather than ignored: see `UNMAPPED_STAT_IDS`.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

# nflverse column -> ESPN stat id, where the two mean exactly the same thing.
DIRECT_COLUMNS: Mapping[str, int] = {
    "passing_yards": 3,
    "passing_tds": 4,
    "passing_2pt_conversions": 19,
    "passing_interceptions": 20,
    "rushing_yards": 24,
    "rushing_tds": 25,
    "rushing_2pt_conversions": 26,
    "receiving_yards": 42,
    "receiving_tds": 43,
    "receiving_2pt_conversions": 44,
    "receptions": 53,
    "fumbles_lost_total": 72,
    "fg_made_40_49": 77,
    "fg_made": 83,
    "fg_missed": 85,
    "pat_made": 86,
    "fg_made_50_59": 198,
    "fg_made_60_": 201,
}

# ESPN stat id <- sum of several nflverse columns.
SUMMED_COLUMNS: Mapping[int, tuple[str, ...]] = {
    # ESPN's "FG Made (0-39 yards)" spans three nflverse distance buckets.
    80: ("fg_made_0_19", "fg_made_20_29", "fg_made_30_39"),
}

# Single-game bonuses, which exist only at weekly granularity.
GAME_BONUSES: Mapping[int, tuple[str, float]] = {
    18: ("passing_yards", 400.0),  # 400+ yard passing game
    38: ("rushing_yards", 200.0),  # 200+ yard rushing game
}

# Scoring rules this league uses that nflverse player_stats cannot reconstruct.
# All are worth 1 point and require play-level detail (the length of the specific
# scoring play) or are team-defense stats that live in a different feed entirely.
# Declared so the residual in `check_statmap` is explained rather than mysterious.
UNMAPPED_STAT_IDS: Mapping[int, str] = {
    36: "50+ yard TD rush bonus — needs play-level TD distance",
    46: "50+ yard TD reception bonus — needs play-level TD distance",
    63: "Fumble recovered for TD — not attributed in player_stats",
    93: "Blocked punt/FG return TD — team defense feed",
    101: "Kickoff return TD — not separable from special_teams_tds",
    102: "Punt return TD — not separable from special_teams_tds",
    103: "Interception return TD — team defense feed",
    104: "Fumble return TD — team defense feed",
    206: "2pt return — team defense feed",
    209: "1pt safety — vanishingly rare",
}

# Team defenses score from an entirely different set of inputs (points allowed,
# yards allowed, sacks, takeaways) that player_stats does not carry.
DST_ONLY_STAT_IDS = frozenset(
    {
        89,
        90,
        91,
        92,
        95,
        96,
        97,
        98,
        99,
        106,
        117,
        119,
        123,
        124,
        125,
        128,
        129,
        130,
        132,
        133,
        134,
        135,
        136,
    }
)


def stat_expressions() -> list[pl.Expr]:
    """Polars expressions producing one column per mapped ESPN stat id."""
    exprs: list[pl.Expr] = []

    for column, stat_id in DIRECT_COLUMNS.items():
        exprs.append(pl.col(column).fill_null(0).cast(pl.Float64).alias(f"stat_{stat_id}"))

    for stat_id, columns in SUMMED_COLUMNS.items():
        total = pl.lit(0.0)
        for column in columns:
            total = total + pl.col(column).fill_null(0).cast(pl.Float64)
        exprs.append(total.alias(f"stat_{stat_id}"))

    for stat_id, (column, threshold) in GAME_BONUSES.items():
        exprs.append(
            (pl.col(column).fill_null(0) >= threshold).cast(pl.Float64).alias(f"stat_{stat_id}")
        )

    return exprs


def mapped_stat_ids() -> list[int]:
    return sorted(set(DIRECT_COLUMNS.values()) | set(SUMMED_COLUMNS) | set(GAME_BONUSES))


def to_espn_stats(frame: pl.DataFrame) -> pl.DataFrame:
    """Add a `stat_<id>` column for every ESPN stat id nflverse can supply."""
    return frame.with_columns(stat_expressions())


def score_frame(frame: pl.DataFrame, engine, position_id: int | None = None) -> pl.DataFrame:
    """Attach `fantasy_points` computed under this league's scoring rules.

    Builds a single weighted-sum expression rather than scoring row by row, so a
    full season of every NFL player stays fast.
    """
    with_stats = to_espn_stats(frame)

    total = pl.lit(0.0)
    for stat_id in mapped_stat_ids():
        points = engine.points_for_stat(stat_id, position_id)
        if points:
            total = total + pl.col(f"stat_{stat_id}") * points

    return with_stats.with_columns(total.alias("fantasy_points"))
