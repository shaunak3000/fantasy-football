"""Report how well the ESPN <-> nflverse identity bridge covers draftable players.

    uv run python -m fantasy_football.check_bridge

Match rate on the top of the board is the number that matters — an unmatched
player deep in the rankings costs nothing, an unmatched first-rounder breaks the
draft board.
"""

from __future__ import annotations

import polars as pl

from .data.ids import attach_espn_ids
from .data.nflverse import load_consensus_board


def main() -> int:
    board = load_consensus_board()
    print(f"Board: {board.height} players, scraped {board['scrape_date'][0]}\n")

    report = attach_espn_ids(board)

    print("ESPN id coverage by board depth:")
    for depth in (25, 50, 100, 150, 200, 300, board.height):
        head = report.matched.head(depth)
        skill = head.filter(pl.col("norm_pos") != "DST")
        hit = skill["espn_id"].is_not_null().sum()
        n = skill.height
        rate = 100 * hit / n if n else 0.0
        flag = "" if rate >= 99 else "  <-- gap"
        print(f"  top {depth:>4}: {hit:>4}/{n:<4} ({rate:5.1f}%){flag}")

    print(f"\nOverall (excluding team defenses): {report.summary()}")

    if report.unmatched.height:
        print(f"\nUnmatched players ({report.unmatched.height}), worst-ranked first:")
        cols = ["player", "pos", "team", "ecr"]
        print(report.unmatched.select(cols).sort("ecr").head(20).to_pandas().to_string(index=False))
    else:
        print("\nNo unmatched players.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
