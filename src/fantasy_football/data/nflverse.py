"""nflverse ingest, with local parquet caching.

Everything here is free public data. Caching keeps the draft-day board fast and
keeps the test suite off the network.

A note on what `load_ff_rankings` actually is: it returns FantasyPros *expert
consensus rankings* (ECR) with dispersion (`sd`, `best`, `worst`) — not projected
points. Point projections come from ESPN and from the opportunity model. The
dispersion columns are valuable in their own right: they are the only public
measure of how much the experts disagree about a player, which feeds the
variance estimates the risk-aware side of this repo runs on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import nflreadpy as nfl
import polars as pl

from ..config import cache_path

# FantasyPros publishes a board per format/position. This league is a redraft
# league, so the overall redraft board is the relevant one.
DEFAULT_BOARD = "redraft-overall"

FANTASY_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")


def _cached(name: str, loader: Callable[[], pl.DataFrame], refresh: bool = False) -> pl.DataFrame:
    path = cache_path(f"{name}.parquet")
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    frame = loader()
    frame.write_parquet(path)
    return frame


def load_player_ids(refresh: bool = False) -> pl.DataFrame:
    """Cross-platform player identity table — the ESPN <-> nflverse join source."""
    return _cached("ff_playerids", nfl.load_ff_playerids, refresh)


def load_rankings(refresh: bool = False) -> pl.DataFrame:
    """Current consensus ranking boards, every format and position."""
    return _cached("ff_rankings_draft", lambda: nfl.load_ff_rankings("draft"), refresh)


def load_rankings_history(refresh: bool = False) -> pl.DataFrame:
    """Every consensus board ever scraped, back to 2019.

    Large (~1.8M rows) but it is the only way to see what the consensus believed
    *before* a season, which is what any honest backtest has to be fit on.
    """
    return _cached("ff_rankings_all", lambda: nfl.load_ff_rankings("all"), refresh)


def load_consensus_board(board: str = DEFAULT_BOARD, refresh: bool = False) -> pl.DataFrame:
    """One ranking board, sorted best-to-worst, with only the columns worth keeping.

    `ecr` is the consensus rank; `sd` is disagreement between experts; `best` and
    `worst` bound the range of individual opinions.
    """
    frame = load_rankings(refresh=refresh)
    available = set(frame["page_type"].unique().to_list())
    if board not in available:
        raise ValueError(
            f"Unknown ranking board {board!r}. Available: {sorted(x for x in available if x)}"
        )

    return (
        frame.filter(pl.col("page_type") == board)
        .select(["player", "pos", "team", "ecr", "sd", "best", "worst", "bye", "scrape_date"])
        .sort("ecr")
    )


def available_boards(refresh: bool = False) -> list[str]:
    boards = load_rankings(refresh=refresh)["page_type"].unique().to_list()
    return sorted(b for b in boards if b)


def load_opportunity(
    seasons: Sequence[int] | int | None = None, refresh: bool = False
) -> pl.DataFrame:
    """Expected fantasy points from the ffopportunity model — usage-based, not name-based."""
    key = "all" if seasons is None else "_".join(str(s) for s in _as_seasons(seasons))
    return _cached(
        f"ff_opportunity_{key}",
        lambda: nfl.load_ff_opportunity(seasons=seasons, stat_type="weekly"),
        refresh,
    )


def load_weekly_stats(
    seasons: Sequence[int] | int | None = None, refresh: bool = False
) -> pl.DataFrame:
    """Actual weekly player stats — the ground truth the scoring engine reconciles against."""
    key = "all" if seasons is None else "_".join(str(s) for s in _as_seasons(seasons))
    return _cached(
        f"player_stats_{key}",
        lambda: nfl.load_player_stats(seasons=seasons, summary_level="week"),
        refresh,
    )


def _as_seasons(seasons: Sequence[int] | int) -> list[int]:
    return [seasons] if isinstance(seasons, int) else list(seasons)
