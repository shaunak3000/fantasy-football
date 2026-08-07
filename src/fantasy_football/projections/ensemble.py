"""Blend the available opinions into one calibrated projection per player.

Two independent sources have a view on 2026: FantasyPros consensus rank, and
ESPN's own point projections. They are not on the same scale and cannot simply
be averaged. ESPN projects what a player does if he plays a full healthy season,
so its point totals run systematically high — averaging them against a
historically calibrated number would import that optimism wholesale.

So both are reduced to the thing they genuinely measure, which is an *ordering*
of players within a position, blended there, and mapped through the rank curve
once at the end. The curve was fit on what players ranked at that spot actually
went on to score, injuries and all, so calibration happens in exactly one place
and neither source can smuggle its own scale in.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .curve import RankCurve

# How much weight the expert consensus carries against ESPN's projections.
# Consensus aggregates many analysts and is the better single source; ESPN is
# kept because it is independent and reacts faster to depth-chart news.
DEFAULT_CONSENSUS_WEIGHT = 0.65

# A player both sources rank very differently is genuinely less certain than one
# they agree on. This scales the historical spread up when they disagree.
DISAGREEMENT_SPREAD_FACTOR = 0.35


@dataclass(frozen=True)
class Projection:
    player: str
    position: str
    team: str
    espn_id: int | None
    consensus_rank: int
    espn_rank: int | None
    blended_rank: float
    mean: float
    sd: float
    replacement: float

    @property
    def vor(self) -> float:
        """Points above what the league could get for free at this position."""
        return self.mean - self.replacement

    @property
    def rank_disagreement(self) -> int | None:
        if self.espn_rank is None:
            return None
        return self.espn_rank - self.consensus_rank


def _positional_rank(frame: pl.DataFrame, value_col: str, descending: bool) -> pl.DataFrame:
    return frame.with_columns(
        pl.col(value_col)
        .rank("ordinal", descending=descending)
        .over("pos")
        .cast(pl.Int32)
        .alias(f"{value_col}_pos_rank")
    )


def build_projections(
    board: pl.DataFrame,
    espn_points: dict[int, float],
    curves: dict[str, RankCurve],
    replacement: dict[str, float],
    consensus_weight: float = DEFAULT_CONSENSUS_WEIGHT,
) -> list[Projection]:
    """Combine the consensus board and ESPN's projections into one ranked list.

    `board` must already carry `espn_id` from the identity bridge. Players ESPN
    has no projection for still get a projection from consensus alone rather
    than being dropped.
    """
    ranked = _positional_rank(board.sort("ecr"), "ecr", descending=False)

    espn_col = pl.col("espn_id").map_elements(
        lambda pid: espn_points.get(pid), return_dtype=pl.Float64
    )
    ranked = ranked.with_columns(espn_col.alias("espn_points"))

    # Rank by ESPN projection within position, over the players ESPN covers.
    covered = ranked.filter(pl.col("espn_points").is_not_null())
    covered = _positional_rank(covered, "espn_points", descending=True)
    espn_ranks = dict(
        zip(
            covered["player"].to_list(),
            covered["espn_points_pos_rank"].to_list(),
            strict=True,
        )
    )

    projections: list[Projection] = []
    for row in ranked.iter_rows(named=True):
        position = row["pos"]
        curve = curves.get(position)
        if curve is None:
            continue

        consensus_rank = int(row["ecr_pos_rank"])
        espn_rank = espn_ranks.get(row["player"])

        if espn_rank is None:
            blended = float(consensus_rank)
            disagreement = 0
        else:
            blended = consensus_weight * consensus_rank + (1.0 - consensus_weight) * espn_rank
            disagreement = abs(espn_rank - consensus_rank)

        mean = curve.points_at(blended)
        base_spread = curve.spread_at(blended)
        # Disagreement is measured relative to how deep the position runs, so a
        # 10-rank gap among 40 tight ends counts for more than among 120 wideouts.
        relative = disagreement / max(curve.max_rank, 1)
        sd = base_spread * (1.0 + DISAGREEMENT_SPREAD_FACTOR * min(relative * 4.0, 1.0))

        projections.append(
            Projection(
                player=row["player"],
                position=position,
                team=row["team"],
                espn_id=row.get("espn_id"),
                consensus_rank=consensus_rank,
                espn_rank=espn_rank,
                blended_rank=blended,
                mean=mean,
                sd=sd,
                replacement=replacement.get(position, 0.0),
            )
        )

    # Where the curve is flat, several players legitimately share an expected
    # value — five seasons cannot separate preseason RB3 from RB7. Order those
    # ties by blended rank rather than leaving them arbitrary, but leave the
    # means equal, because inventing a gap would be inventing information.
    projections.sort(key=lambda p: (-p.vor, p.blended_rank))
    return projections


def to_frame(projections: list[Projection]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "player": p.player,
                "pos": p.position,
                "team": p.team,
                "espn_id": p.espn_id,
                "consensus_rank": p.consensus_rank,
                "espn_rank": p.espn_rank,
                "blended_rank": round(p.blended_rank, 1),
                "mean": round(p.mean, 1),
                "sd": round(p.sd, 1),
                "vor": round(p.vor, 1),
            }
            for p in projections
        ]
    )
