"""Per-week projections, which are what the lineup optimizer actually consumes.

A season total cannot drive a weekly start/sit decision. Three separate things
have to be pulled apart, because they behave differently and get used
differently:

- **Production when playing** — points per game *played*. Not the season total
  divided by seventeen, which silently blends in the weeks a player missed and
  understates what he does when he suits up.
- **Week-to-week spread** — how much a player swings around his own average.
  This is the quantity that decides whether to chase a ceiling or protect a
  floor, and it is unrelated to how good the player is: a boom-bust WR2 and a
  steady WR2 have the same mean and very different usefulness depending on
  whether you are favored.
- **Availability** — bye weeks, and the background rate of missing time. Kept
  separate because a manager responds to it by starting someone else, not by
  accepting a lower score.

All three are fit with the same rank-curve machinery as the season projection,
so they inherit its calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .curve import RankCurve, fit_all
from .ensemble import Projection

# A player needs enough games for his own weekly spread to mean anything.
MIN_GAMES_FOR_SPREAD = 6

NFL_REGULAR_SEASON_GAMES = 17


@dataclass(frozen=True)
class WeeklyOutlook:
    player: str
    position: str
    week: int
    mean: float
    sd: float
    on_bye: bool

    @property
    def is_startable(self) -> bool:
        return not self.on_bye and self.mean > 0

    def probability_over(self, threshold: float) -> float:
        """Rough chance of clearing a threshold this week.

        A normal approximation, which is defensible for a single week — weekly
        scoring is far less skewed than a season total, where a lost season
        piles mass at zero. Treat it as a guide, not a calibrated number; the
        season-level intervals are the ones that were backtested.
        """
        if self.on_bye or self.sd <= 0:
            return 0.0 if self.mean < threshold else 1.0
        from math import erf, sqrt

        z = (threshold - self.mean) / self.sd
        return float(1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))


@dataclass
class WeeklyModel:
    """Rank curves for per-game production, weekly spread, and availability."""

    per_game: dict[str, RankCurve]
    spread: dict[str, RankCurve]
    games: dict[str, RankCurve]

    @classmethod
    def fit(cls, training: pl.DataFrame) -> WeeklyModel:
        played = training.filter(pl.col("games_played") >= MIN_GAMES_FOR_SPREAD)
        return cls(
            per_game=fit_all(played, value_col="points_per_game"),
            # Spread is allowed to fall with rank — better players score more and
            # so swing more in absolute terms — but availability is not, because
            # preseason rank barely predicts who stays healthy.
            spread=fit_all(played, value_col="weekly_sd"),
            games=fit_all(training, value_col="games_played", monotone=False),
        )

    def expected_games(self, position: str, rank: float) -> float:
        curve = self.games.get(position)
        if curve is None:
            return float(NFL_REGULAR_SEASON_GAMES)
        return min(curve.points_at(rank), float(NFL_REGULAR_SEASON_GAMES))

    def outlook(
        self, projection: Projection, week: int, bye_week: int | None = None
    ) -> WeeklyOutlook:
        position = projection.position
        rank = projection.blended_rank
        on_bye = bye_week is not None and week == bye_week

        per_game = self.per_game.get(position)
        spread = self.spread.get(position)

        mean = 0.0 if on_bye or per_game is None else per_game.points_at(rank)
        sd = 0.0 if on_bye or spread is None else spread.points_at(rank)

        return WeeklyOutlook(
            player=projection.player,
            position=position,
            week=week,
            mean=mean,
            sd=sd,
            on_bye=on_bye,
        )

    def season_outlooks(
        self,
        projections: list[Projection],
        byes: dict[str, int],
        weeks: range,
    ) -> list[WeeklyOutlook]:
        return [
            self.outlook(projection, week, byes.get(projection.player))
            for projection in projections
            for week in weeks
        ]


def bye_weeks_from_board(board: pl.DataFrame) -> dict[str, int]:
    """Bye week per player, straight off the consensus board."""
    if "bye" not in board.columns:
        return {}
    return {
        row["player"]: int(row["bye"])
        for row in board.iter_rows(named=True)
        if row.get("bye") is not None
    }
