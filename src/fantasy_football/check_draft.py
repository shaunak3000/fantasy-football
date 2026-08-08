"""Draft board and rehearsal.

    uv run python -m fantasy_football.check_draft [slot]

Fits the league's own drafting behaviour from 2024-25, then shows what the board
recommends at each of your picks in an empty 2026 draft.
"""

from __future__ import annotations

import sys

from espn_api.football import League

from .config import load_credentials
from .data.espn import fetch_raw_settings, parse_settings
from .draft.history import pick_training
from .draft.model import fit_pick_model
from .draft.recommend import recommend, roster_limits
from .draft.state import DraftState
from .projections.build import build

DRAFT_SEASONS = [2024, 2025]
ROUNDS = 16


def main(argv: list[str]) -> int:
    my_slot = int(argv[0]) if argv else 1

    creds = load_credentials()
    league = League(league_id=creds.league_id, year=2026, espn_s2=creds.espn_s2, swid=creds.swid)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, 2026)

    print(f"{settings.name} — fitting draft behaviour from {DRAFT_SEASONS}\n")
    training, sizes = pick_training(DRAFT_SEASONS, creds)
    model = fit_pick_model(training)
    if model is None:
        print("Not enough draft history to fit a model.")
        return 1

    print(f"Fit on {model.n_observations} ranked players across seasons {model.seasons}")
    print(f"Draft sizes: {sizes}\n")

    print("How this league drafts, versus consensus:")
    print(f"  {'consensus rank':>15} {'goes at pick':>13} {'spread':>8}")
    for rank in (1, 5, 10, 20, 40, 60, 90, 120):
        if rank > int(model.ranks[-1]):
            break
        print(f"  {rank:>15} {model.expected_pick(rank):>13.1f} {model.pick_spread(rank):>8.1f}")
    print("\n  Spread widens as the draft wears on — early picks are near-certain,")
    print("  late picks are close to random.\n")

    result = build(2026)
    limits = roster_limits(settings)
    print(f"Roster caps: {limits}\n")

    state = DraftState(team_count=settings.team_count, rounds=ROUNDS, my_slot=my_slot)
    print(f"=== Draft slot {my_slot}: your picks are {state.my_picks[:8]}... ===\n")

    by_id = {p.espn_id: p for p in result.projections if p.espn_id is not None}

    for round_number in range(1, 6):
        if state.is_complete:
            break
        # Advance the board to my next pick by letting the model take players.
        while not state.is_my_turn and not state.is_complete:
            picks = recommend(
                state, result.projections, model, settings, trials=1, seed=state.current_pick
            )
            if not picks:
                break
            state.record(picks[0].espn_id or -1)

        if state.is_complete:
            break

        gap = state.picks_until_my_next()
        print(
            f"--- Round {round_number}, overall pick {state.current_pick} "
            f"({gap} players go before you pick again) ---"
        )
        options = recommend(state, result.projections, model, settings, trials=300)
        print(
            f"  {'player':<22} {'pos':<4} {'VOR':>7} {'survive':>8} "
            f"{'next pick':>10} {'2-pick':>8} {'loss':>7}"
        )
        for option in options[:8]:
            flag = "  <- likely gone" if option.likely_gone else ""
            print(
                f"  {option.player:<22} {option.position:<4} {option.vor:>7.1f} "
                f"{option.survival:>7.0%} {option.next_pick_value:>10.1f} "
                f"{option.two_pick_value:>8.1f} {option.expected_loss_if_passed:>7.1f}{flag}"
            )

        pick = options[0]
        state.record_my_pick(pick.espn_id or -1)
        roster = [by_id[i].position for i in state.my_roster if i in by_id]
        print(f"  => take {pick.player} ({pick.position}).  roster so far: {roster}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
