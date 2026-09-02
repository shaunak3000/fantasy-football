"""Regenerate the board table inside docs/index.html from the prepared bundle.

The page argues that the draft rule is just consensus rank plus roster discipline.
Printing the resulting order is the honest way to show it — but a hand-pasted table
on a page about measurement would rot silently, so it is generated here and rewritten
in place between markers.

    uv run python docs/render_board.py

Re-run after `live_draft prepare`; the stamped date on the page comes from the bundle.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

from fantasy_football.data.nflverse import load_consensus_board
from fantasy_football.draft.cache import load_bundle

SEASON = 2026
ROWS = 120
# Byes falling in the fantasy semifinal. Everything else is a regular-season
# collision, which this league's bench depth absorbs for 1-5% of a lineup.
SEMIFINAL_WEEK = 14
# ADP gap, in picks, past which the room is meaningfully later than our board.
SLIDE = 15

START = "<!-- BOARD:START -->"
END = "<!-- BOARD:END -->"


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_rows() -> tuple[str, str, int]:
    bundle = load_bundle(SEASON)
    if bundle is None:
        sys.exit("No prepared bundle. Run: uv run python -m fantasy_football.live_draft prepare")

    board = load_consensus_board()
    scraped = str(board["scrape_date"][0]) if board.height else "unknown"
    byes = {r["player"]: r["bye"] for r in board.iter_rows(named=True) if r.get("bye")}

    ordered = sorted(bundle.projections.projections, key=lambda p: p.consensus_overall_rank)[:ROWS]

    out = []
    for p in ordered:
        bye = byes.get(p.player)
        adp = bundle.adp.get(p.espn_id)
        gap = adp - p.consensus_overall_rank if adp else None

        classes = []
        if bye == SEMIFINAL_WEEK:
            classes.append("semi")
        if gap is not None and gap >= SLIDE:
            classes.append("slide")
        cls = f' class="{" ".join(classes)}"' if classes else ""

        bye_cell = f"{bye}" if bye else "&mdash;"
        if bye == SEMIFINAL_WEEK:
            bye_cell = f'<span class="warn">{bye}</span>'

        if gap is None:
            adp_cell, gap_cell = "&mdash;", "&mdash;"
        else:
            adp_cell = str(adp)
            gap_cell = f'<span class="{"slid" if gap >= SLIDE else ""}">{gap:+d}</span>'

        out.append(
            f"<tr{cls}><td>{p.consensus_overall_rank}</td><td>{esc(p.player)}</td>"
            f"<td>{p.position}</td><td>{esc(p.team)}</td><td>{bye_cell}</td>"
            f"<td>{adp_cell}</td><td>{gap_cell}</td></tr>"
        )

    prepared = datetime.fromisoformat(bundle.prepared_at).strftime("%d %B %Y")
    stamp = (
        f"Consensus board scraped {scraped}; ADP pulled {prepared}. "
        f"{len(bundle.adp)} players carry an ADP."
    )
    return "\n".join(out), stamp, len(ordered)


def main() -> int:
    rows, stamp, count = build_rows()
    page = Path(__file__).with_name("index.html")
    html = page.read_text()

    if START not in html or END not in html:
        sys.exit(f"Markers {START} / {END} not found in {page}")

    block = f"{START}\n{rows}\n{END}"
    html = re.sub(re.escape(START) + r".*?" + re.escape(END), block, html, flags=re.S)
    html = re.sub(
        r'(<p class="stamp">).*?(</p>)',
        lambda m: m.group(1) + stamp + m.group(2),
        html,
        flags=re.S,
    )
    page.write_text(html)
    print(f"Wrote {count} rows to {page}\n  {stamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
