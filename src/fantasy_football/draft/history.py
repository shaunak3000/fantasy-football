"""Prior drafts, joined to what the consensus said at the time.

The point of this module is to learn how *this* league drafts, which is not how
the internet drafts. Public ADP describes millions of anonymous drafters; your
family reaches for its own favourites, forgets kickers exist until round 15, and
panics on quarterbacks at a particular moment. Two years of their actual picks
describe that better than any national average.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from espn_api.football import League

from ..data.ids import attach_espn_ids
from ..projections.history import preseason_board


@dataclass(frozen=True)
class DraftPick:
    season: int
    overall_pick: int
    round_num: int
    round_pick: int
    team_name: str
    espn_id: int
    player_name: str


def load_draft(league: League, season: int) -> list[DraftPick]:
    team_count = len(league.teams)
    picks = []
    for pick in league.draft:
        round_num = getattr(pick, "round_num", 0)
        round_pick = getattr(pick, "round_pick", 0)
        if not round_num or not round_pick:
            continue
        picks.append(
            DraftPick(
                season=season,
                overall_pick=(round_num - 1) * team_count + round_pick,
                round_num=round_num,
                round_pick=round_pick,
                team_name=getattr(getattr(pick, "team", None), "team_name", "?"),
                espn_id=getattr(pick, "playerId", -1),
                player_name=getattr(pick, "playerName", ""),
            )
        )
    return sorted(picks, key=lambda p: p.overall_pick)


def draft_frame(picks: list[DraftPick]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "season": p.season,
                "overall_pick": p.overall_pick,
                "round_num": p.round_num,
                "team_name": p.team_name,
                "espn_id": p.espn_id,
                "player_name": p.player_name,
            }
            for p in picks
        ]
    )


def board_with_ids(season: int) -> pl.DataFrame:
    """That season's preseason consensus board, carrying ESPN ids."""
    board = preseason_board(season)
    if board.is_empty():
        return board

    matched = attach_espn_ids(board, name_col="player", pos_col="pos", team_col="team")
    return (
        matched.matched.filter(pl.col("espn_id").is_not_null())
        .with_columns(pl.col("ecr").rank("ordinal").cast(pl.Int32).alias("consensus_rank"))
        .select(["season", "player", "pos", "espn_id", "ecr", "consensus_rank", "pos_rank"])
    )


def pick_training(
    seasons: list[int], creds, undrafted_penalty: int = 1
) -> tuple[pl.DataFrame, dict[int, int]]:
    """Join every historical pick to the consensus rank the drafter could see.

    Players who were ranked but went undrafted are kept, recorded just past the
    end of that draft. Dropping them would make the league look far more
    disciplined than it is: the model needs to know that a chunk of ranked
    players go entirely unwanted.
    """
    frames = []
    draft_sizes: dict[int, int] = {}

    for season in seasons:
        league = League(
            league_id=creds.league_id, year=season, espn_s2=creds.espn_s2, swid=creds.swid
        )
        picks = load_draft(league, season)
        if not picks:
            continue

        total_picks = max(p.overall_pick for p in picks)
        draft_sizes[season] = total_picks

        board = board_with_ids(season)
        if board.is_empty():
            continue

        joined = board.join(
            draft_frame(picks).select(["espn_id", "overall_pick"]), on="espn_id", how="left"
        ).with_columns(
            pl.col("overall_pick").fill_null(total_picks + undrafted_penalty).alias("overall_pick"),
            pl.col("overall_pick").is_null().alias("undrafted"),
        )
        frames.append(joined.with_columns(pl.lit(total_picks).alias("draft_size")))

    if not frames:
        return pl.DataFrame(), draft_sizes

    return pl.concat(frames, how="vertical"), draft_sizes
