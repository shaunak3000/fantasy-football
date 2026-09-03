"""Does modelling an opponent's roster predict his picks better than rank alone?

The live board answers one question about every player: will he still be there
when I choose again. That answer currently comes from a memoryless model — where
a player goes, given only his consensus rank. It cannot know that the manager on
the clock already has three receivers.

Roster discipline was the entire edge in our own draft rule (`check_rehearsal`:
371 points, all of it the cap filter). This asks whether the same filter also
describes the other seven managers, replayed against every pick they actually
made in 2024 and 2025.

    uv run python -m fantasy_football.check_opponents

Reported as the rank of the pick that was actually made inside each model's
predicted ordering. Lower is better; 1.0 would be a model that called every pick.
"""

from __future__ import annotations

import sys

from espn_api.football import League

from .config import load_credentials
from .data.espn import fetch_raw_settings, parse_settings
from .draft.history import board_with_ids, load_draft
from .draft.recommend import roster_limits
from .draft.room import flex_slot_count, open_positions, position_counts

SEASONS = (2024, 2025)
# Round 1 is close to deterministic and every model calls it, so including it
# flatters both equally and hides the difference where decisions are actually made.
FIRST_SCORED_ROUND = 2
TOP_K = 5
# How far back a player at a capped position is pushed, in board places. Managers
# do occasionally take a sixth receiver, so "at cap" is a strong signal and not a
# prohibition — excluding those players outright sends them to rank 380 and one
# such pick swamps a hundred correct ones. Two rounds of an eight-team draft is
# the demotion: enough to matter, bounded enough to survive being wrong.
# Chosen for that reason, not tuned against the seasons it is scored on.
CAP_DEMOTION = 16


def _season_league(creds, season: int) -> League:
    return League(league_id=creds.league_id, year=season, espn_s2=creds.espn_s2, swid=creds.swid)


def replay(creds, season: int) -> dict | None:
    league = _season_league(creds, season)
    picks = load_draft(league, season)
    if not picks:
        return None

    # That season's own settings. 2024 ran nine teams and a different bench, so
    # judging those picks against 2026's caps would invent discipline nobody had.
    season_settings = parse_settings(fetch_raw_settings(league), creds.league_id, season)
    limits = roster_limits(season_settings)
    flex_slots = flex_slot_count(season_settings)

    board = board_with_ids(season)
    if board.is_empty():
        return None

    position = {int(r["espn_id"]): r["pos"] for r in board.iter_rows(named=True)}
    rank = {int(r["espn_id"]): int(r["consensus_rank"]) for r in board.iter_rows(named=True)}
    # The board is the universe both models predict from. A pick outside it is
    # unpredictable by either and would only add noise.
    ordered_board = sorted(rank, key=lambda espn_id: rank[espn_id])

    taken: set[int] = set()
    rosters: dict[str, list[int]] = {}

    static_ranks, reactive_ranks = [], []
    static_hits = reactive_hits = 0

    for pick in picks:
        owner = pick.team_name
        actual = pick.espn_id
        scored = pick.round_num >= FIRST_SCORED_ROUND and actual in rank

        if scored:
            available = [espn_id for espn_id in ordered_board if espn_id not in taken]

            # Static: the memoryless model. Ordering by consensus rank is exactly
            # what its expected-pick curve produces, since the drift is monotone.
            static_order = available

            # Reactive: the same players, reordered so positions this manager can
            # still roster come first. Both models must rank the WHOLE board, or
            # the comparison is not one — an earlier version dropped capped
            # positions entirely and charged a max penalty when someone drafted
            # one anyway, and those seven picks across two seasons produced the
            # entire apparent loss.
            counts = position_counts(rosters.get(owner, []), position)
            needs = open_positions(counts, limits, flex_slots)
            reactive_order = sorted(
                range(len(available)),
                key=lambda i: i + (0 if position.get(available[i]) in needs else CAP_DEMOTION),
            )
            reactive_order = [available[i] for i in reactive_order]

            static_ranks.append(static_order.index(actual) + 1)
            reactive_ranks.append(reactive_order.index(actual) + 1)
            static_hits += static_ranks[-1] <= TOP_K
            reactive_hits += reactive_ranks[-1] <= TOP_K

        taken.add(actual)
        rosters.setdefault(owner, []).append(actual)

    if not static_ranks:
        return None

    n = len(static_ranks)
    return {
        "season": season,
        "limits": limits,
        "flex": flex_slots,
        "picks": n,
        "static_mean": sum(static_ranks) / n,
        "reactive_mean": sum(reactive_ranks) / n,
        "static_top": static_hits / n,
        "reactive_top": reactive_hits / n,
        "static_median": sorted(static_ranks)[n // 2],
        "reactive_median": sorted(reactive_ranks)[n // 2],
        "better": sum(1 for a, b in zip(static_ranks, reactive_ranks, strict=True) if b < a),
        "worse": sum(1 for a, b in zip(static_ranks, reactive_ranks, strict=True) if b > a),
    }


def main() -> int:
    creds = load_credentials()
    print("Predicting the other managers' picks, replayed against what they did.\n")

    rows = [r for r in (replay(creds, s) for s in SEASONS) if r]
    if not rows:
        print("No draft history available.", file=sys.stderr)
        return 1

    head = f"{'season':>8}{'picks':>8}{'static':>10}{'reactive':>10}{'better by':>12}"
    print(head)
    print("-" * len(head))
    for r in rows:
        gain = r["static_mean"] - r["reactive_mean"]
        print(f"  {r['season']} caps {r['limits']} flex {r['flex']}")
        print(
            f"{r['season']:>8}{r['picks']:>8}{r['static_mean']:>10.2f}"
            f"{r['reactive_mean']:>10.2f}{gain:>+12.2f}"
        )

    total = sum(r["picks"] for r in rows)
    static = sum(r["static_mean"] * r["picks"] for r in rows) / total
    reactive = sum(r["reactive_mean"] * r["picks"] for r in rows) / total
    s_top = sum(r["static_top"] * r["picks"] for r in rows) / total
    r_top = sum(r["reactive_top"] * r["picks"] for r in rows) / total

    print("-" * len(head))
    print(f"{'all':>8}{total:>8}{static:>10.2f}{reactive:>10.2f}{static - reactive:>+12.2f}")
    better = sum(r["better"] for r in rows)
    worse = sum(r["worse"] for r in rows)
    print(f"\n  mean rank of the actual pick   static {static:.2f}  ->  reactive {reactive:.2f}")
    print(f"  actual pick inside top {TOP_K}       static {s_top:.1%}  ->  reactive {r_top:.1%}")
    same = total - better - worse
    print(f"  picks ranked better by reactive  {better}   worse {worse}   unchanged {same}")

    if reactive < static:
        print("\n  Reactive wins. Roster state describes these managers better than rank alone.")
    else:
        print("\n  No gain. Keep the memoryless model; it is simpler and it is not losing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
