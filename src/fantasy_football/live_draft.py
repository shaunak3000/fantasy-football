"""Draft-night command.

    uv run python -m fantasy_football.live_draft prepare    # do this the day before
    uv run python -m fantasy_football.live_draft watch      # do this during the draft
    uv run python -m fantasy_football.live_draft manual     # if the feed dies

`watch` needs nothing from you. Draft in ESPN exactly as you normally would;
this notices each pick, re-ranks, and tells you what to take when your turn
comes around. `manual` is the same board driven by typing picks, for an offline
draft or a dead feed.

The pick it recommends comes from the only strategy that survived the rehearsal
against the 2025 draft: the highest-ranked player at a position you still need.
Several more elaborate rules were tested and none beat it; one lost badly. The
simulation still runs, but only to tell you who is likely to be gone by your
next turn — it advises, it does not decide. See `check_rehearsal` for the bench.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from espn_api.football import League

from .config import load_credentials
from .data.espn import fetch_raw_settings, parse_settings
from .draft.board_view import board_view
from .draft.cache import load_bundle, save_bundle
from .draft.history import pick_training
from .draft.live import DraftFeed, draft_order, my_team_id, slot_for_team, sync_state
from .draft.model import fit_pick_model
from .draft.recommend import roster_limits
from .draft.script import DEFAULT_TRIALS, forecast, target_survival
from .draft.state import DraftState, snake_slot_for_pick
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

    rows = board_view(
        state, bundle.projections.projections, bundle.pick_model, settings, trials=LIVE_TRIALS
    )
    if not rows:
        print("No players left to recommend.")
        print("=" * 72)
        return

    if state.is_my_turn:
        best = next((row for row in rows if row.recommended), rows[0])
        print(f"\n  >>> TAKE: {best.player} ({best.position}, {best.team})")
        print(f"      {best.rationale()}")

    print(f"\n  {'player':<22} {'pos':<4} {'rank':>5} {'VOR':>6} {'survive':>8}")
    for row in rows[:6]:
        flag = " <-gone" if row.likely_gone else ""
        print(
            f"  {row.player:<22} {row.position:<4} {row.overall_rank:>5} "
            f"{row.vor:>6.0f} {row.survival:>7.0%}{flag}"
        )
    print(f"\n  still need: {', '.join(still_need) if still_need else 'nothing'}")
    print("=" * 72)


def plan() -> int:
    """Forward plan for the whole draft, produced before it starts."""
    bundle = _load_or_complain()
    if bundle is None:
        return 1

    creds = load_credentials()
    league = _league(creds)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, SEASON)

    order = draft_order(league)
    team_id = my_team_id(league, creds.swid)
    slot = slot_for_team(order, team_id) or 1

    projections = bundle.projections.projections
    print(f"Draft slot {slot} of {len(order)}. Simulating {DEFAULT_TRIALS} drafts...\n")

    print("Who is still on the board at your first pick:")
    print(f"  {'player':<24}{'pos':<5}{'rank':>5}{'available':>11}")
    for player, position, rank, survival in target_survival(
        slot, projections, bundle.pick_model, settings, rounds=ROUNDS
    ):
        marker = "" if survival > 0.5 else "   unlikely"
        print(f"  {player:<24}{position:<5}{rank:>5}{survival:>10.0%}{marker}")

    print("\nWhat you end up with, pick by pick:")
    print(f"  {'pick':>5} {'rnd':>4}  {'most likely':<28}{'pos mix':<22}{'proj':>7}{'VOR':>7}")
    for item in forecast(slot, projections, bundle.pick_model, settings, rounds=ROUNDS):
        mix = ", ".join(f"{p} {s:.0%}" for p, s in list(item.position_mix.items())[:3])
        print(
            f"  {item.pick_number:>5} {item.round_number:>4}  {item.headline():<28}"
            f"{mix:<22}{item.mean_points:>7.0f}{item.mean_vor:>7.0f}"
        )

    print("\n  'most likely' is the single most frequent outcome across simulations;")
    print("  the position mix shows what the pick usually turns into. Treat this as")
    print("  the shape of your draft, not a script to follow blindly.")
    return 0


def simulate() -> int:
    """Play one plausible draft out in full, every team, every pick.

    The per-pick plan shows what you end up with; this shows the room. Seeing
    the snake actually turn — who is picking while you wait, how far a run
    travels before it reaches you — is what makes the fourteen-pick gap between
    your pairs concrete rather than a number.
    """
    bundle = _load_or_complain()
    if bundle is None:
        return 1

    creds = load_credentials()
    league = _league(creds)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, SEASON)

    order = draft_order(league)
    slot = slot_for_team(order, my_team_id(league, creds.swid)) or 1
    names = {t.team_id: t.team_name for t in league.teams}
    slot_names = {i + 1: names.get(tid, f"slot {i + 1}") for i, tid in enumerate(order)}

    projections = bundle.projections.projections
    by_id = {p.espn_id: p for p in projections if p.espn_id is not None}
    state = DraftState(team_count=settings.team_count, rounds=ROUNDS, my_slot=slot)

    rng = np.random.default_rng(0)
    ranks = np.array([p.consensus_overall_rank for p in projections], dtype=float)
    board_position = (np.argsort(np.argsort(ranks)) + 1).astype(float)
    queue = [
        projections[int(i)]
        for i in np.argsort(bundle.pick_model.sample_board_order(board_position, ranks, rng))
    ]

    print(f"One simulated draft. You are slot {slot} ({slot_names.get(slot)}).\n")

    cursor = 0
    current_round = 0
    while not state.is_complete:
        pick_number = state.current_pick
        pick_round = state.round_of(pick_number)
        if pick_round != current_round:
            current_round = pick_round
            direction = "-->" if pick_round % 2 else "<--"
            print(f"\n  ---- Round {pick_round} {direction} ----")

        picking_slot = snake_slot_for_pick(pick_number, settings.team_count)
        mine = picking_slot == slot

        if mine:
            rows = board_view(state, projections, bundle.pick_model, settings, trials=40)
            if not rows:
                break
            choice = next((r for r in rows if r.recommended), rows[0])
            player = by_id[choice.espn_id]
            state.record_my_pick(choice.espn_id)
        else:
            while cursor < len(queue) and queue[cursor].espn_id in state.drafted_set:
                cursor += 1
            if cursor >= len(queue):
                break
            player = queue[cursor]
            state.record(player.espn_id)
            cursor += 1

        marker = "  <<< YOU" if mine else ""
        print(
            f"  {pick_number:>3}  {slot_names.get(picking_slot, picking_slot)[:20]:<20} "
            f"{player.player:<24}{player.position:<6}{marker}"
        )

    print("\n  Your picks are the marked rows. Note how the order reverses each")
    print("  round: at the turn you pick twice in a row, then wait the longest.")
    return 0


def dryrun() -> int:
    """Drive a whole draft through the live code path, with no live draft.

    The draft happens once. Anything that only breaks in round 12 — an empty
    position, a roster cap, a crash on an exhausted board — has to be found
    now, because there is no second attempt and no time to debug at the time.
    Opponents pick from the fitted model; the board picks exactly as it would
    on the night.
    """
    bundle = _load_or_complain()
    if bundle is None:
        return 1

    creds = load_credentials()
    league = _league(creds)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, SEASON)

    order = draft_order(league)
    slot = slot_for_team(order, my_team_id(league, creds.swid)) or 1

    projections = bundle.projections.projections
    state = DraftState(team_count=settings.team_count, rounds=ROUNDS, my_slot=slot)

    rng = np.random.default_rng(0)
    ranks = np.array([p.consensus_overall_rank for p in projections], dtype=float)
    board_position = (np.argsort(np.argsort(ranks)) + 1).astype(float)
    queue = [
        projections[int(i)]
        for i in np.argsort(bundle.pick_model.sample_board_order(board_position, ranks, rng))
    ]

    print(
        f"Dry run: slot {slot} of {len(order)}, {ROUNDS} rounds, "
        f"{len(projections)} players on the board.\n"
    )

    cursor = 0
    my_picks = []
    while not state.is_complete:
        if state.is_my_turn:
            rows = board_view(state, projections, bundle.pick_model, settings, trials=60)
            if not rows:
                print(f"  pick {state.current_pick}: BOARD EXHAUSTED")
                break
            choice = next((r for r in rows if r.recommended), rows[0])
            my_picks.append(choice)
            print(
                f"  pick {state.current_pick:>3} (rd {state.round_of(state.current_pick):>2})  "
                f"{choice.player:<24}{choice.position:<6}"
                f"rank {choice.overall_rank:>3}  survive {choice.survival:>4.0%}"
            )
            state.record_my_pick(choice.espn_id)
            continue

        while cursor < len(queue) and queue[cursor].espn_id in state.drafted_set:
            cursor += 1
        if cursor >= len(queue):
            print(f"  pick {state.current_pick}: opponents exhausted the board")
            break
        state.record(queue[cursor].espn_id)
        cursor += 1

    counts: dict[str, int] = {}
    for pick in my_picks:
        counts[pick.position] = counts.get(pick.position, 0) + 1

    print(f"\n  Roster: {counts}")
    limits = roster_limits(settings)
    problems = [
        f"{position} {count} exceeds cap {limits[position]}"
        for position, count in counts.items()
        if position in limits and count > limits[position]
    ]
    missing = [
        position
        for position in settings.starting_slots
        if position not in ("RB/WR/TE",) and counts.get(position, 0) == 0
    ]
    if missing:
        problems.append(f"no player at required position(s): {', '.join(missing)}")

    if problems:
        print("\n  PROBLEMS:")
        for problem in problems:
            print(f"    {problem}")
        return 1

    print(f"  {len(my_picks)} picks made, all caps respected, every slot fillable.")
    print("  Dry run passed.")
    return 0


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
    if command == "plan":
        return plan()
    if command == "dryrun":
        return dryrun()
    if command == "simulate":
        return simulate()
    if command == "watch":
        return watch()
    if command == "manual":
        return manual()
    print(
        f"Unknown command {command!r}. Use: prepare | plan | watch | manual",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
