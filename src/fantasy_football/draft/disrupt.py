"""Optional adversarial plays for the back half of the draft.

**Advisory, always.** Six cleverer strategies lost the rehearsal and nothing here
has been scored against it, so these never replace the recommendation — `watch`
prints the play as an ALT line beside the rule's pick, with its reason, and you
choose. An untested idea does not get the same standing as a measured one, and
under a 90-second clock a silent substitution is the one thing you cannot audit.

Shown by default precisely because it cannot decide anything. A flag gated this
while it could still override; once it could only inform, the flag was one more
thing to remember at 7:35 PM for no protection.

What makes it defensible at all is a measurement rather than a theory. Replaying
all eight slots, **7.1 of our 16 picks land on a player at or below replacement**,
starting around round six. By our own model those picks are worth nothing: the
waiver wire will offer as much all season. A pick that buys nothing can be spent
on something other than value without giving anything up, which is the only
honest basis for adversarial play.

The objective helps too. We optimise P(first), not points, so relative damage
counts — a move costing us 2 and costing a rival 12 is a gain even though it is
negative in expected points. That reasoning also forbids the greedy version: a
move costing us 20 to cost them 5 is simply bad, which is what most disruption
ideas turn out to be.

Three plays, in the order they are worth anything:

1. **Handcuff denial.** Take the direct backup to a rival's starting running back.
   RB value is workload-driven and the successor is knowable in advance, so this
   is the one position where denial has teeth in a league whose waiver wire is
   this deep. Free, because the alternative was a sub-replacement lottery ticket.
2. **Take the pair at the turn.** Denial is only free when you wanted the player
   anyway. Picking back to back is the one moment that is true.
3. **ADP deferral.** Prefer the player the room takes soonest among options our
   own board rates equally. Cannot cost value; it only breaks ties.
"""

from __future__ import annotations

from dataclasses import dataclass

from .recommend import roster_limits
from .room import flex_slot_count, open_positions, position_counts, slot_rosters

# Nothing here runs before this round. Everything above it is where our value
# comes from, and every play below is only free because the pick is not.
DEFAULT_FIRST_ROUND = 7

# A candidate must be at or below replacement before we spend his pick on
# disruption. This is the same threshold the live rule uses to stop optimising.
FREE_PICK_VOR = 0.0

# ADP places of daylight before deferral is allowed to break a tie.
TIE_SLIDE = 12
# How close two players must be on our board to count as a tie worth breaking.
TIE_RANK_WINDOW = 6


@dataclass(frozen=True)
class Play:
    espn_id: int
    label: str
    why: str


def rival_starters(state, by_id, settings, position: str = "RB", top_n: int = 2) -> dict[int, list]:
    """Each rival's best players at a position, by our board's ordering.

    Only the top couple matter. Denying the backup to somebody's fourth running
    back protects nothing, because that starter was never going to play.
    """
    rosters = slot_rosters(state.drafted, settings.team_count)
    out: dict[int, list] = {}
    for slot, roster in rosters.items():
        if slot == state.my_slot:
            continue
        held = [by_id[i] for i in roster if i in by_id and by_id[i].position == position]
        held.sort(key=lambda p: p.consensus_overall_rank)
        if held:
            out[slot] = held[:top_n]
    return out


def handcuff(state, available, by_id, settings) -> Play | None:
    """The best available backup to a rival's starting running back.

    Matched on NFL team rather than a depth chart: the highest-ranked running back
    still on the board who plays for the same club as somebody's starter is that
    starter's replacement in all but name. This avoids adding a depth-chart feed
    two days before a draft, and the failure mode is mild — at worst it takes a
    committee back, which is what the pick would have been anyway.
    """
    starters = rival_starters(state, by_id, settings)
    if not starters:
        return None

    owned_teams: dict[str, list] = {}
    for slot, players in starters.items():
        for player in players:
            owned_teams.setdefault(player.team, []).append((slot, player))

    candidates = [
        p
        for p in available
        if p.position == "RB" and p.team in owned_teams and p.espn_id is not None
    ]
    if not candidates:
        return None

    best = min(candidates, key=lambda p: p.consensus_overall_rank)
    slot, blocked = min(owned_teams[best.team], key=lambda pair: pair[1].consensus_overall_rank)
    return Play(
        espn_id=best.espn_id,
        label="handcuff denial",
        why=(
            f"backs up {blocked.player} ({best.team}), held by slot {slot}. "
            "If that back goes down, the replacement is ours, not theirs."
        ),
    )


def pair_at_the_turn(state, available, needed) -> Play | None:
    """Back-to-back picks are the one place denial costs nothing.

    With no opponent picking in between, taking two of the same position is both
    the value-maximising move and the denying one, so there is no trade-off to
    weigh.
    """
    if state.picks_until_my_next() != 0:
        return None
    eligible = [p for p in available if p.position in needed and p.espn_id is not None]
    if len(eligible) < 2:
        return None
    first, second = sorted(eligible, key=lambda p: p.consensus_overall_rank)[:2]
    if first.position != second.position:
        return None
    return Play(
        espn_id=first.espn_id,
        label="pair at the turn",
        why=(
            f"you pick again immediately; {first.player} and {second.player} are the "
            f"top two {first.position}s left, so you can take both."
        ),
    )


def defer_for_adp(available, needed, adp, next_pick, current) -> Play | None:
    """Break a tie towards the player the room takes soonest.

    Only fires when our own board rates two players within a few places of each
    other, so it cannot trade value away. The one it defers is the one ADP says
    survives; the one it takes is the one that will not.

    `current` must be the pick the live rule would actually make. Deriving a
    "top choice" here instead looked equivalent and was not — the rule reserves
    late picks for mandatory positions and breaks lottery ties on ceiling, so the
    two disagree exactly when this play fires, and it would defer a player the
    rule was never going to take.
    """
    if not adp or next_pick is None or current is None:
        return None
    if current.espn_id not in adp or current.position not in needed:
        return None
    eligible = [p for p in available if p.position in needed and p.espn_id in adp]
    if len(eligible) < 2:
        return None

    eligible.sort(key=lambda p: p.consensus_overall_rank)
    top = current
    tied = [
        p
        for p in eligible
        if p.espn_id != top.espn_id
        and abs(p.consensus_overall_rank - top.consensus_overall_rank) <= TIE_RANK_WINDOW
    ]
    if not tied:
        return None

    top_slack = adp[top.espn_id] - next_pick
    # Only worth acting on when our first choice is the one that will still be
    # there and somebody equivalent will not.
    if top_slack < TIE_SLIDE:
        return None
    urgent = [p for p in tied if adp[p.espn_id] - next_pick < 0]
    if not urgent:
        return None

    take = min(urgent, key=lambda p: p.consensus_overall_rank)
    return Play(
        espn_id=take.espn_id,
        label="ADP deferral",
        why=(
            f"{top.player} is our #{top.consensus_overall_rank} but ADP {adp[top.espn_id]} "
            f"says he lasts past pick {next_pick}; {take.player} will not."
        ),
    )


def disruptive_pick(
    state,
    projections,
    settings,
    by_id,
    adp: dict[int, int] | None = None,
    first_round: int = DEFAULT_FIRST_ROUND,
    current_id: int | None = None,
) -> Play | None:
    """The adversarial pick for this turn, or None to leave the rule alone.

    Returns None before the gate round, and None whenever the ordinary
    recommendation is still worth real points — disruption is funded entirely by
    picks that were going to be wasted.
    """
    if state.round_of(state.current_pick) < first_round:
        return None

    taken = state.drafted_set
    available = [p for p in projections if p.espn_id not in taken]
    if not available:
        return None

    limits = roster_limits(settings)
    counts = position_counts(state.my_roster, {i: p.position for i, p in by_id.items()})
    needed = open_positions(counts, limits, flex_slot_count(settings))
    if not needed:
        return None

    next_pick = state.next_pick_for_me()

    # Deferral is free at any point past the gate, because it only breaks ties.
    play = defer_for_adp(available, needed, adp, next_pick, by_id.get(current_id))
    if play is not None:
        return play

    play = pair_at_the_turn(state, available, needed)
    if play is not None:
        return play

    # The rest is only affordable once the board stops offering real value.
    best = max(
        (p for p in available if p.position in needed),
        key=lambda p: p.vor,
        default=None,
    )
    if best is None or best.vor > FREE_PICK_VOR:
        return None

    return handcuff(state, available, by_id, settings)
