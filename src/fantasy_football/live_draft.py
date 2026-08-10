"""Draft-night command.

    uv run python -m fantasy_football.live_draft prepare    # do this the day before
    uv run python -m fantasy_football.live_draft watch      # do this during the draft
    uv run python -m fantasy_football.live_draft manual     # if the feed dies

`watch` needs nothing from you. Draft in ESPN exactly as you normally would;
this notices each pick, re-ranks, and tells you what to take when your turn
comes around. `manual` is the same board driven by typing picks, for an offline
draft or a dead feed.
"""

from __future__ import annotations

import sys
import time

from espn_api.football import League

from .config import load_credentials
from .data.espn import fetch_raw_settings, parse_settings
from .draft.cache import load_bundle, save_bundle
from .draft.history import pick_training
from .draft.live import DraftFeed, draft_order, my_team_id, slot_for_team, sync_state
from .draft.model import fit_pick_model
from .draft.recommend import recommend, roster_limits
from .draft.state import DraftState
from .projections.build import build

SEASON = 2026
ROUNDS = 16
DRAFT_SEASONS = [2024, 2025]
POLL_SECONDS = 3.0
LIVE_TRIALS = 250


def _league(creds, season=SEASON) -> League:
    return League(league_id=creds.league_id, year=season, espn_s2=creds.espn_s2, swid=creds.swid)


def prepare() -> int:
    creds = load_credentials()
    print("Fitting projections and draft behaviour - this takes a minute.\n")

    projections = build(SEASON)
    print(f"  projections: {len(projections.projections)} players")

    training, sizes = pick_training(DRAFT_SEASONS, creds)
    model = fit_pick_model(training)
    if model is None:
        print("  could not fit a draft model - not enough history")
        return 1
    print(f"  draft model:  {model.n_observations} picks from {sizes}")

    path = save_bundle(SEASON, projections, model)
    print(f"\nSaved to {path}")
    print("Re-run this the morning of the draft so the rankings are fresh.")
    return 0


def _load_or_complain():
    bundle = load_bundle(SEASON)
    if bundle is None:
        print("No prepared bundle. Run:  live_draft prepare", file=sys.stderr)
        return None
    if bundle.is_stale:
        print(f"WARNING: bundle is {bundle.age_hours:.0f}h old - rankings may have moved.")
        print("         Run 'live_draft prepare' again for current numbers.\n")
    return bundle


def _render(state: DraftState, bundle, settings, by_id, quiet: bool = False) -> None:
    """Show the board. Kept short on purpose — there is a clock running."""
    roster = [by_id[i].position for i in state.my_roster if i in by_id]
    limits = roster_limits(settings)
    counts = {p: roster.count(p) for p in set(roster)}
    still_need = [
        f"{p}({limits[p] - counts.get(p, 0)})" for p in limits if counts.get(p, 0) < limits[p]
    ]

    print("\n" + "=" * 72)
    print(
        f"PICK {state.current_pick}  |  round {state.round_of(state.current_pick)}  |  "
        f"your roster: {', '.join(roster) if roster else 'empty'}"
    )
    if not state.is_my_turn:
        following = state.next_pick_for_me()
        if following:
            print(
                f"Not your turn - you pick at {following} ({following - state.current_pick} away)"
            )
        else:
            print("Your draft is done.")
        if quiet:
            print("=" * 72)
            return

    options = recommend(
        state, bundle.projections.projections, bundle.pick_model, settings, trials=LIVE_TRIALS
    )
    if not options:
        print("No players left to recommend.")
        print("=" * 72)
        return

    if state.is_my_turn:
        best = options[0]
        print(f"\n  >>> TAKE: {best.player} ({best.position}, {best.team})")
        print(f"      {best.rationale()}")

    print(f"\n  {'player':<22} {'pos':<4} {'VOR':>6} {'survive':>8} {'2-pick':>8}")
    for option in options[:6]:
        flag = " <-gone" if option.likely_gone else ""
        print(
            f"  {option.player:<22} {option.position:<4} {option.vor:>6.1f} "
            f"{option.survival:>7.0%} {option.two_pick_value:>8.1f}{flag}"
        )
    print(f"\n  still need: {', '.join(still_need) if still_need else 'nothing'}")
    print("=" * 72)


def watch() -> int:
    bundle = _load_or_complain()
    if bundle is None:
        return 1

    creds = load_credentials()
    league = _league(creds)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, SEASON)

    order = draft_order(league)
    team_id = my_team_id(league, creds.swid)
    slot = slot_for_team(order, team_id)
    if slot is None:
        print("Could not identify your team from the draft order.", file=sys.stderr)
        print(f"Draft order (team ids): {order}", file=sys.stderr)
        return 1

    print(f"You are team {team_id}, draft slot {slot} of {len(order)}")
    state = DraftState(team_count=settings.team_count, rounds=ROUNDS, my_slot=slot)
    print(f"Your picks: {state.my_picks}\n")

    feed = DraftFeed(league=league)
    by_id = {p.espn_id: p for p in bundle.projections.projections if p.espn_id is not None}

    print("Watching for picks. Draft in ESPN as normal; Ctrl-C to stop.\n")
    last_shown = -1
    try:
        while True:
            feed.poll()
            if feed.last_error:
                print(f"  (feed error: {feed.last_error} - retrying)")

            sync_state(state, feed, team_id)
            if feed.complete:
                print("\nDraft complete.")
                _render(state, bundle, settings, by_id, quiet=True)
                return 0

            if feed.pick_count != last_shown:
                last_shown = feed.pick_count
                _render(state, bundle, settings, by_id, quiet=not state.is_my_turn)

            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


def manual() -> int:
    bundle = _load_or_complain()
    if bundle is None:
        return 1

    creds = load_credentials()
    league = _league(creds)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, SEASON)

    raw_slot = input(f"Your draft slot (1-{settings.team_count}): ").strip()
    slot = int(raw_slot) if raw_slot.isdigit() else 1

    state = DraftState(team_count=settings.team_count, rounds=ROUNDS, my_slot=slot)
    projections = bundle.projections.projections
    by_id = {p.espn_id: p for p in projections if p.espn_id is not None}
    by_name = {p.player.lower(): p for p in projections}

    print("\nType a player name after each pick. 'undo' reverts, 'q' quits.")
    print("Your own picks are recorded automatically when it is your turn.\n")

    while not state.is_complete:
        _render(state, bundle, settings, by_id, quiet=not state.is_my_turn)
        entry = input("pick> ").strip()
        if entry.lower() in {"q", "quit", "exit"}:
            return 0
        if entry.lower() == "undo":
            if state.drafted:
                removed = state.drafted.pop()
                if removed in state.my_roster:
                    state.my_roster.remove(removed)
                print(f"  removed {by_id[removed].player if removed in by_id else removed}")
            continue
        if not entry:
            continue

        matches = [p for name, p in by_name.items() if entry.lower() in name]
        available = [p for p in matches if p.espn_id not in state.drafted_set]
        if not available:
            print(f"  no available player matching {entry!r}")
            continue
        if len(available) > 1:
            print(f"  ambiguous: {', '.join(p.player for p in available[:6])}")
            continue

        chosen = available[0]
        state.record(chosen.espn_id, mine=state.is_my_turn)
        print(f"  recorded {chosen.player} ({chosen.position})")

    print("\nDraft complete.")
    return 0


def main(argv: list[str]) -> int:
    command = argv[0] if argv else "watch"
    if command == "prepare":
        return prepare()
    if command == "watch":
        return watch()
    if command == "manual":
        return manual()
    print(f"Unknown command {command!r}. Use: prepare | watch | manual", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
