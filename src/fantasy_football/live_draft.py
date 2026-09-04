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
from espn_api.requests.espn_requests import ESPNAccessDenied, ESPNInvalidLeague

from .config import env_file, load_credentials
from .data.espn import fetch_adp, fetch_raw_settings, parse_settings
from .data.nflverse import load_consensus_board
from .draft.board_view import board_view
from .draft.cache import load_bundle, save_bundle
from .draft.disrupt import DEFAULT_FIRST_ROUND, disruptive_pick
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
# Consecutive failed polls before the board is declared untrustworthy. A few
# seconds of transient error is normal; half a minute of it is not.
FEED_FAILURE_LIMIT = 10

# Polls to allow with the draft reported in progress and not one pick visible
# before declaring the feed silent. The clock is 90 seconds a pick, so a slow
# opening pick is normal and must not trip this; two minutes of an active draft
# with a completely empty board is not normal, it is the REST view never being
# written. Measured against an ESPN mock, where that is exactly what happens.
FEED_SILENT_POLLS = 40

# Size of the numbered shortlist in manual mode. Measured against the 250 real
# picks in this league's 2024 and 2025 drafts, the player taken was inside the
# top 20 available 80.8% of the time (top 10: 64.8%, top 30: 86.8%). Twenty is
# where the curve flattens against the screen space it costs — four picks in five
# become one keystroke instead of a typed name, under a 90-second clock.
SHORTLIST = 20


def _league(creds, season=SEASON) -> League:
    """Open the league, turning an auth failure into instructions rather than a stack trace.

    Every command starts here, so this is the first thing that runs at 7:35 PM on
    draft night — and expired cookies are the single most likely way it fails.
    A traceback at that moment is the worst possible output: it buries the one
    fact that matters (go refresh two cookies) under twenty lines of espn_api
    internals. The mid-draft feed banner already handles the same failure once
    polling is underway; this covers the startup path it cannot reach.
    """
    try:
        return League(
            league_id=creds.league_id, year=season, espn_s2=creds.espn_s2, swid=creds.swid
        )
    except ESPNInvalidLeague as exc:
        # Distinct from an auth failure and must not be reported as one. Since the
        # league id can be overridden (ESPN_LEAGUE_ID=... to point at a mock draft),
        # a typo lands here — and sending someone to re-copy two cookies over a
        # wrong id would waste the one resource draft night does not have.
        print("\n" + "!" * 72, file=sys.stderr)
        print(f"  NO SUCH LEAGUE: {creds.league_id} (season {season}).", file=sys.stderr)
        print("  Your cookies are fine — this is the league id.", file=sys.stderr)
        print(
            f"\n  Reading ESPN_LEAGUE_ID from: {env_file() or _CREDENTIAL_SOURCE_UNKNOWN}",
            file=sys.stderr,
        )
        print(
            "  If you meant to point at a mock draft, check the leagueId in its URL.",
            file=sys.stderr,
        )
        print(f"\n  ESPN said: {exc}", file=sys.stderr)
        print("!" * 72 + "\n", file=sys.stderr)
        raise SystemExit(1) from None
    except ESPNAccessDenied as exc:
        source = env_file()
        print("\n" + "!" * 72, file=sys.stderr)
        print("  ESPN REJECTED YOUR CREDENTIALS. Nothing has started.", file=sys.stderr)
        print("  Almost always this means espn_s2 or SWID has expired.", file=sys.stderr)
        print(
            f"\n  Fix: refresh ESPN_S2 + ESPN_SWID in {source or _CREDENTIAL_SOURCE_UNKNOWN}",
            file=sys.stderr,
        )
        print("       (Firefox: F12 -> Storage -> Cookies -> fantasy.espn.com),", file=sys.stderr)
        print("       then re-run this exact command. Nothing is lost.", file=sys.stderr)
        print(f"\n  ESPN said: {exc}", file=sys.stderr)
        print("!" * 72 + "\n", file=sys.stderr)
        raise SystemExit(1) from None


# Shown when no env file was found at all, which load_credentials would normally
# have caught first — belt and braces so the message is never a bare "None".
_CREDENTIAL_SOURCE_UNKNOWN = "your credentials file (see README)"


def prepare() -> int:
    creds = load_credentials()
    print("Fitting projections and draft behaviour - this takes a minute.\n")

    projections = build(SEASON)
    board = load_consensus_board()
    scraped = board["scrape_date"][0] if board.height else "unknown"
    print(f"  projections: {len(projections.projections)} players")
    print(f"  rankings:    consensus board scraped {scraped}")

    training, sizes = pick_training(DRAFT_SEASONS, creds)
    model = fit_pick_model(training)
    if model is None:
        print("  could not fit a draft model - not enough history")
        return 1
    print(f"  draft model:  {model.n_observations} picks from {sizes}")

    # ADP is the freshest input in the bundle — ESPN restamps it daily, while the
    # consensus board is a weekly scrape. It is also the only one that is pure
    # context, so a failure here must not cost us a prepared draft.
    try:
        adp = fetch_adp(_league(creds))
        print(f"  ADP:          {len(adp)} players from live ESPN drafts")
    except Exception as exc:  # noqa: BLE001 - context only, never worth failing prepare
        adp = {}
        print(f"  ADP:          unavailable ({exc}) - board will run without it")

    path = save_bundle(SEASON, projections, model, adp=adp)
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

    adp = getattr(bundle, "adp", None)
    rows = board_view(
        state,
        bundle.projections.projections,
        bundle.pick_model,
        settings,
        trials=LIVE_TRIALS,
        adp=adp,
    )
    if not rows:
        print("No players left to recommend.")
        print("=" * 72)
        return

    covered = False
    if state.is_my_turn:
        best = next((row for row in rows if row.recommended), rows[0])
        print(f"\n  >>> TAKE: {best.player} ({best.position}, {best.team})")
        print(f"      {best.rationale()}")
        covered = _print_alternative(state, bundle, settings, by_id, adp, best)

    print(f"\n  {'player':<22} {'pos':<4} {'rank':>5} {'VOR':>6} {'survive':>8} {'ADP':>5}")
    for row in rows[:6]:
        flag = " <-gone" if row.likely_gone else ""
        if row.can_wait:
            flag = f" <-can wait (+{row.adp_slack})"
        adp = str(row.adp_rank) if row.adp_rank else "-"
        print(
            f"  {row.player:<22} {row.position:<4} {row.overall_rank:>5} "
            f"{row.vor:>6.0f} {row.survival:>7.0%} {adp:>5}{flag}"
        )

    if not covered:
        _print_wait_hint(rows, state)
    print(f"\n  still need: {', '.join(still_need) if still_need else 'nothing'}")
    print("=" * 72)


def _print_alternative(state, bundle, settings, by_id, adp, best) -> bool:
    """Offer the adversarial play as an alternative, never as the pick.

    The recommendation above always comes from the rule that won the rehearsal.
    These plays have never been scored against it, so presenting one as the answer
    would give an untested idea the same standing as a measured one — and under a
    90-second clock a silent substitution is the one thing you cannot audit.
    Naming it, with its reason, leaves the choice where it belongs.
    """
    if best.espn_id is None:
        return False
    play = disruptive_pick(
        state,
        bundle.projections.projections,
        settings,
        by_id,
        adp,
        current_id=best.espn_id,
    )
    if play is None or play.espn_id == best.espn_id:
        return False
    alt = by_id.get(play.espn_id)
    if alt is None:
        return False
    print(f"      ALT [{play.label}]: {alt.player} ({alt.position}, {alt.team})")
    print(f"           {play.why}")
    # The deferral says exactly what the wait hint says, only concretely — it
    # names the swap rather than describing the gap. Printing both spends the
    # clock twice on one idea.
    return play.label == "ADP deferral"


def _print_wait_hint(rows, state) -> None:
    """Say so when the room would let the recommended player slide.

    Our board ranks by consensus and has no idea when anyone actually goes, so
    it will happily spend an early pick on someone the room ignores for another
    fifty. This does not change the recommendation — it names the cost, and a
    human with the roster in front of him decides.
    """
    best = next((row for row in rows if row.recommended), None)
    if best is None or not best.can_wait:
        return
    next_pick = state.next_pick_for_me()
    # Only worth raising if there is something to do instead: a player who fills
    # a need and, by ADP, will NOT survive that long.
    alternatives = [
        row
        for row in rows
        if row is not best and row.fills_a_need and row.adp_slack is not None and row.adp_slack < 0
    ]
    print(
        f"\n  NOTE: ADP puts {best.player} around pick {best.adp_rank}; "
        f"you pick again at {next_pick}."
    )
    if alternatives:
        names = ", ".join(f"{r.player} (ADP {r.adp_rank})" for r in alternatives[:2])
        print(f"        Likely gone by then: {names}")
    print("        Untested signal - the recommendation above is unchanged.")


def plan(argv: list[str] | None = None) -> int:
    """Forward plan for the whole draft, produced before it starts.

    Takes an optional slot number. This league randomizes the draft order an
    hour before kickoff, so the slot read from `pickOrder` beforehand is not the
    one you will draft from — plan for a specific slot only after the room opens,
    and use the all-slots table before that.
    """
    bundle = _load_or_complain()
    if bundle is None:
        return 1

    creds = load_credentials()
    league = _league(creds)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, SEASON)

    order = draft_order(league)
    projections = bundle.projections.projections

    requested = int(argv[0]) if argv and argv[0].isdigit() else None
    if requested is None:
        print("Draft order is randomized an hour before kickoff, so no slot is")
        print("fixed yet. Here is the shape of every possible draw:\n")
        _all_slots_summary(projections, bundle.pick_model, settings, len(order))
        print("\nOnce the room opens and the order is set, re-run with your slot:")
        print("    live_draft plan <slot>")
        slot = slot_for_team(order, my_team_id(league, creds.swid)) or 1
        print(f"\nBelow is provisional detail for slot {slot}, from the CURRENT")
        print("pre-shuffle order. Treat it as illustrative, not as your plan.")
    else:
        slot = requested

    print(f"\nDraft slot {slot} of {len(order)}. Simulating {DEFAULT_TRIALS} drafts...\n")

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


def _all_slots_summary(projections, pick_model, settings, team_count: int) -> None:
    """First picks, longest wait, and a realistic target for every slot."""
    print(f"  {'slot':>5}  {'first four picks':<24}{'longest wait':>13}{'likely at #1':>22}")
    for slot in range(1, team_count + 1):
        state = DraftState(team_count=team_count, rounds=ROUNDS, my_slot=slot)
        picks = state.my_picks
        gaps = [b - a for a, b in zip(picks, picks[1:], strict=False)]
        survivors = target_survival(
            slot, projections, pick_model, settings, rounds=ROUNDS, trials=120, top=14
        )
        likely = next((name for name, _, _, odds in survivors if odds >= 0.5), "-")
        print(f"  {slot:>5}  {str(picks[:4]):<24}{max(gaps) if gaps else 0:>13}{likely[:21]:>22}")
    print("\n  The rule you follow does not change with the slot - only how long you")
    print("  wait between turns. Early slots pick once then wait; the last slot picks")
    print("  twice back to back then waits longest.")


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


def watch(argv: list[str] | None = None) -> int:
    # --disrupt used to gate these; they are advisory and now always on. Still
    # accepted so muscle memory does not produce an error against a draft clock.
    if "--disrupt" in (argv or []):
        print("  (--disrupt is now the default; the flag is no longer needed.)")
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

    print(
        f"  From round {DEFAULT_FIRST_ROUND} an ALT line offers an adversarial alternative. "
        "It never changes the recommendation - you choose."
    )
    print("Watching for picks. Draft in ESPN as normal; Ctrl-C to stop.\n")
    last_shown = -1
    consecutive_errors = 0
    silent_polls = 0
    warned_silent = False
    try:
        while True:
            feed.poll()
            if feed.last_error:
                consecutive_errors += 1
                print(f"  (feed error #{consecutive_errors}: {feed.last_error} - retrying)")
                # A couple of dropped polls are normal. A sustained run of them
                # means the feed is not coming back on its own — usually expired
                # cookies — and silently showing a frozen board during a draft
                # is the worst possible failure. Say so, loudly, with the way out.
                if consecutive_errors == FEED_FAILURE_LIMIT:
                    print("\n" + "!" * 72)
                    print("  FEED HAS FAILED REPEATEDLY. The board below is STALE.")
                    print("  Most likely your ESPN cookies expired mid-draft.")
                    print(f"  Fix: refresh ESPN_S2 + ESPN_SWID in {env_file() or 'your env file'},")
                    print("       then restart this command —")
                    print("       it rebuilds from scratch and loses nothing.")
                    print("  Or:  run 'live_draft manual' — it recovers the picks so far")
                    print("       from ESPN and lets you continue by typing.")
                    print("!" * 72 + "\n")
            else:
                consecutive_errors = 0

            # The draft is running and the feed says nobody has picked. Early on
            # that is just a slow first pick; sustained, it means ESPN is not
            # writing picks where we can read them, and the board will sit on
            # pick 1 all night looking exactly like a quiet draft.
            if feed.in_progress and not feed.picks:
                silent_polls += 1
                if silent_polls >= FEED_SILENT_POLLS and not warned_silent:
                    warned_silent = True
                    print("\n" + "!" * 72)
                    print("  DRAFT IS RUNNING BUT THE FEED SHOWS NO PICKS.")
                    print("  ESPN is not publishing this draft to the API - the board")
                    print("  below will never advance. This is how mock drafts behave.")
                    print("\n  Switch now:  Ctrl-C, then")
                    print("      uv run python -m fantasy_football.live_draft manual")
                    print("  and type each pick as it happens. Nothing is lost.")
                    print("!" * 72 + "\n")
            else:
                silent_polls = 0

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


def _shortlist(state, projections) -> list:
    """The most likely next picks, highest consensus rank first."""
    taken = state.drafted_set
    available = [p for p in projections if p.espn_id not in taken]
    available.sort(key=lambda p: p.consensus_overall_rank)
    return available[:SHORTLIST]


def _print_shortlist(shortlist: list) -> None:
    """Two columns, so twenty names cost ten lines rather than twenty."""
    if not shortlist:
        return
    half = (len(shortlist) + 1) // 2
    print()
    for i in range(half):
        cells = []
        for index in (i, i + half):
            if index < len(shortlist):
                p = shortlist[index]
                cells.append(f"{index + 1:>3} {p.player[:20]:<20} {p.position:<4}")
        print("  " + "  ".join(cells))


def _resolve(token: str, shortlist: list, by_name: dict, state):
    """A shortlist number, or a name fragment. Numbers are checked first."""
    if token.isdigit():
        index = int(token) - 1
        if 0 <= index < len(shortlist):
            return shortlist[index]
        print(f"  {token} is not on the shortlist")
        return None

    matches = [p for name, p in by_name.items() if token.lower() in name]
    available = [p for p in matches if p.espn_id not in state.drafted_set]
    if not available:
        print(f"  no available player matching {token!r}")
        return None
    if len(available) > 1:
        print(f"  ambiguous: {', '.join(p.player for p in available[:6])}")
        return None
    return available[0]


def manual() -> int:
    bundle = _load_or_complain()
    if bundle is None:
        return 1

    creds = load_credentials()
    league = _league(creds)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, SEASON)

    # Seed from the live feed if it is reachable at all. Manual mode exists for
    # when `watch` has failed, and a draft that has failed at pick 60 must not
    # require retyping sixty picks against a running clock — take whatever ESPN
    # will still give, then carry on by hand from there.
    order = draft_order(league)
    detected = slot_for_team(order, my_team_id(league, creds.swid))

    state = DraftState(team_count=settings.team_count, rounds=ROUNDS, my_slot=detected or 1)
    seed_feed = DraftFeed(league=league)
    if seed_feed.poll() or seed_feed.picks:
        sync_state(state, seed_feed, my_team_id(league, creds.swid))
        print(
            f"Recovered {len(seed_feed.picks)} picks from ESPN; resuming at "
            f"pick {state.current_pick}."
        )
    else:
        print(f"No live feed ({seed_feed.last_error or 'nothing drafted yet'}); starting fresh.")

    if detected is None:
        raw_slot = input(f"Your draft slot (1-{settings.team_count}): ").strip()
        state.my_slot = int(raw_slot) if raw_slot.isdigit() else 1
    else:
        print(f"Draft slot {detected} of {len(order)}.")

    projections = bundle.projections.projections
    by_id = {p.espn_id: p for p in projections if p.espn_id is not None}
    by_name = {p.player.lower(): p for p in projections}

    print("\nEnter each pick by NUMBER from the shortlist, or by name.")
    print("Several at once is fine: '3 7 12' or '3, Lamb'. 'undo' reverts, 'q' quits.")
    print("Your own picks are recorded automatically when it is your turn.\n")

    while not state.is_complete:
        _render(state, bundle, settings, by_id, quiet=not state.is_my_turn)
        shortlist = _shortlist(state, projections)
        _print_shortlist(shortlist)
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

        # One line may carry a burst of picks — catching up after looking away is
        # the whole reason manual mode is bearable, and retyping them one prompt
        # at a time against a running clock is not.
        for token in (t.strip() for t in entry.replace(",", " ").split()):
            if not token or state.is_complete:
                continue
            chosen = _resolve(token, shortlist, by_name, state)
            if chosen is None:
                continue
            state.record(chosen.espn_id, mine=state.is_my_turn)
            print(f"  recorded {chosen.player} ({chosen.position})")
            # The shortlist is stale the moment anything is taken from it.
            shortlist = _shortlist(state, projections)

    print("\nDraft complete.")
    return 0


def main(argv: list[str]) -> int:
    command = argv[0] if argv else "watch"
    if command == "prepare":
        return prepare()
    if command == "plan":
        return plan(argv[1:])
    if command == "dryrun":
        return dryrun()
    if command == "simulate":
        return simulate()
    if command == "watch":
        return watch(argv[1:])
    if command == "manual":
        return manual()
    print(
        f"Unknown command {command!r}. Use: prepare | plan | watch | manual",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
