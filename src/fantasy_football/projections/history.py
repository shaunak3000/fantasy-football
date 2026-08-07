"""Historical preseason expectations paired with what actually happened.

This is the training set behind every projection in the repo: for each of the
last several seasons, what the consensus thought of a player in August and what
he actually scored under this league's rules.

Fitting on *preseason* rank rather than final rank is the point. A curve fit on
final rank answers "what did the player who finished RB5 score?", which is an
order statistic — it is guaranteed to look extreme at the top and cannot be
achieved in expectation by anyone you draft. The question a draft board actually
needs is "what does the player ranked RB5 in August go on to score?", and that
is a different, much flatter, and much noisier curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..data.ids import attach_espn_ids
from ..data.nflverse import load_rankings_history, load_weekly_stats
from ..data.statmap import score_frame

SCORABLE_POSITIONS = ("QB", "RB", "WR", "TE", "K")
REDRAFT_BOARD = "redraft-overall"

# Boards published from August through the first days of September represent the
# information a drafter actually had. Anything later is contaminated by results.
PRESEASON_MONTHS = ("08",)
EARLY_SEPT_CUTOFF = "10"


@dataclass(frozen=True)
class Coverage:
    season: int
    ranked: int
    matched: int
    played: int

    @property
    def match_rate(self) -> float:
        return self.matched / self.ranked if self.ranked else 0.0

    def line(self) -> str:
        return (
            f"  {self.season}: {self.ranked:>4} ranked, {self.matched:>4} id-matched "
            f"({100 * self.match_rate:.0f}%), {self.played:>4} played a snap"
        )


def _preseason_snapshot(rankings: pl.DataFrame, season: int) -> pl.DataFrame:
    """The last consensus board published before the season began."""
    dated = rankings.with_columns(pl.col("scrape_date").cast(pl.String))
    month = pl.col("scrape_date").str.slice(5, 2)
    day = pl.col("scrape_date").str.slice(8, 2)

    window = dated.filter(
        (pl.col("page_type") == REDRAFT_BOARD)
        & (pl.col("scrape_date").str.slice(0, 4) == str(season))
        & (month.is_in(PRESEASON_MONTHS) | ((month == "09") & (day <= EARLY_SEPT_CUTOFF)))
    )
    if window.is_empty():
        return window

    latest = window["scrape_date"].max()
    return window.filter(pl.col("scrape_date") == latest)


def preseason_board(season: int, refresh: bool = False) -> pl.DataFrame:
    """Consensus board as of the last preseason snapshot, with positional rank."""
    board = _preseason_snapshot(load_rankings_history(refresh=refresh), season)
    if board.is_empty():
        return board

    return (
        board.filter(pl.col("pos").is_in(SCORABLE_POSITIONS))
        .sort("ecr")
        .with_columns(pl.lit(season).alias("season"))
        .with_columns(pl.col("ecr").rank("ordinal").over("pos").cast(pl.Int32).alias("pos_rank"))
        .select(["season", "player", "pos", "team", "ecr", "sd", "best", "worst", "pos_rank"])
    )


def season_actuals(season: int, engine, refresh: bool = False) -> pl.DataFrame:
    """Actual regular-season fantasy points per player, under this league's rules."""
    weekly = load_weekly_stats([season], refresh=refresh).filter(
        (pl.col("season_type") == "REG") & pl.col("position").is_in(SCORABLE_POSITIONS)
    )
    scored = score_frame(weekly, engine)

    return (
        scored.group_by("player_id")
        .agg(
            pl.col("fantasy_points").sum().alias("actual_points"),
            pl.len().alias("games_played"),
            pl.col("fantasy_points").std().alias("weekly_sd"),
            pl.col("player_display_name").first().alias("nfl_name"),
        )
        .with_columns(pl.lit(season).alias("season"))
    )


def training_table(
    seasons: list[int], engine, refresh: bool = False
) -> tuple[pl.DataFrame, list[Coverage]]:
    """Preseason rank joined to realized outcome, one row per ranked player-season.

    Players who were ranked, matched to an id, and then never played are kept
    with zero points. They are not noise — a first-round pick who tears an ACL in
    week 1 is precisely the downside the variance estimate exists to capture, and
    dropping them would make every projection look far safer than it is.
    """
    frames = []
    coverage = []

    for season in seasons:
        board = preseason_board(season, refresh=refresh)
        if board.is_empty():
            continue

        report = attach_espn_ids(board, name_col="player", pos_col="pos", team_col="team")
        ranked = report.matched.filter(pl.col("gsis_id").is_not_null())

        actuals = season_actuals(season, engine, refresh=refresh)
        joined = ranked.join(
            actuals, left_on="gsis_id", right_on="player_id", how="left", suffix="_act"
        ).with_columns(
            pl.col("actual_points").fill_null(0.0),
            pl.col("games_played").fill_null(0),
            pl.col("weekly_sd").fill_null(0.0),
        )

        # Points per game *played*, which is a different quantity from the
        # season total and the one a weekly lineup decision needs: given the
        # player suits up, what does he do? Availability is modelled separately,
        # so leave it null rather than zero for players who never played —
        # a zero here would be a statement about production, not absence.
        joined = joined.with_columns(
            pl.when(pl.col("games_played") > 0)
            .then(pl.col("actual_points") / pl.col("games_played"))
            .otherwise(None)
            .alias("points_per_game")
        )

        coverage.append(
            Coverage(
                season=season,
                ranked=board.height,
                matched=ranked.height,
                played=joined.filter(pl.col("games_played") > 0).height,
            )
        )
        frames.append(
            joined.select(
                [
                    "season",
                    "player",
                    "pos",
                    "team",
                    "ecr",
                    "sd",
                    "best",
                    "worst",
                    "pos_rank",
                    "actual_points",
                    "games_played",
                    "weekly_sd",
                    "points_per_game",
                ]
            )
        )

    if not frames:
        return pl.DataFrame(), coverage

    return pl.concat(frames, how="vertical"), coverage
