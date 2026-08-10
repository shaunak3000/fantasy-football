"""Competing draft strategies, so they can be measured against each other.

Every strategy has the same signature and is judged the same way: replay a real
draft with it in one slot and score the roster it produces against what those
players actually did. No strategy earns a place in the tool by being clever.

The four here differ in what they think a pick is worth:

- **best available at need** — the incumbent, and hard to beat. Takes the
  highest-ranked player at any position not yet full.
- **tiers** — the same, but jumps a position whose tier is about to be emptied
  before your next turn. Encodes scarcity without pretending to precision.
- **rollout** — simulates the whole remaining draft and keeps the candidate whose
  *finished* roster is best. Fixes the myopia of judging a pick against the
  roster you hold today rather than the one you will end with.
- **championship** — the same rollout, but scored by the chance of finishing with
  the best roster rather than the highest expected points, so variance counts.
"""

from __future__ import annotations

import numpy as np

from .lineup_value import lineup_points, marginal_value, roster_points
from .recommend import roster_limits
from .state import DraftState, snake_slot_for_pick

DEFAULT_ROLLOUT_TRIALS = 20
DEFAULT_ROLLOUT_CANDIDATES = 10

# A gap this large between consecutive players at a position marks a tier break.
TIER_GAP_POINTS = 12.0


def _counts(roster_ids, by_id) -> dict[str, int]:
    counts: dict[str, int] = {}
    for espn_id in roster_ids:
        projection = by_id.get(espn_id)
        if projection is not None:
            counts[projection.position] = counts.get(projection.position, 0) + 1
    return counts


def _needed(counts, limits) -> set[str]:
    return {p for p, cap in limits.items() if counts.get(p, 0) < cap}


def best_available_at_need(state, projections, settings, by_id):
    """Highest-ranked player at a position still open. The baseline to beat."""
    taken = state.drafted_set
    available = [p for p in projections if p.espn_id not in taken]
    if not available:
        return None

    limits = roster_limits(settings)
    needed = _needed(_counts(state.my_roster, by_id), limits)
    eligible = [p for p in available if p.position in needed] or available
    return min(eligible, key=lambda p: p.consensus_overall_rank).espn_id


def tiered(state, projections, settings, by_id):
    """Best available at need, unless a tier is about to be wiped out.

    Tiers capture the only thing that really justifies reaching: when the drop
    to the next group at a position is a cliff rather than a step, and the cliff
    will be reached before your next turn, taking that player now is worth more
    than his rank suggests.
    """
    taken = state.drafted_set
    available = [p for p in projections if p.espn_id not in taken]
    if not available:
        return None

    limits = roster_limits(settings)
    needed = _needed(_counts(state.my_roster, by_id), limits)
    eligible = [p for p in available if p.position in needed] or available

    gap = state.picks_until_my_next()
    if gap is None or gap <= 0:
        return min(eligible, key=lambda p: p.consensus_overall_rank).espn_id

    best_score = None
    best_id = None
    for position in {p.position for p in eligible}:
        pool = sorted((p for p in available if p.position == position), key=lambda p: -p.mean)
        if not pool:
            continue
        # How many players sit in the same tier as the best one left here.
        tier_size = 1
        for earlier, later in zip(pool, pool[1:], strict=False):
            if earlier.mean - later.mean >= TIER_GAP_POINTS:
                break
            tier_size += 1

        leader = pool[0]
        if leader.position not in {p.position for p in eligible}:
            continue
        # Urgency rises as the tier gets thinner relative to the wait.
        urgency = 1.0 + max(0.0, (gap - tier_size)) / max(gap, 1)
        score = leader.mean * urgency
        if best_score is None or score > best_score:
            best_score, best_id = score, leader.espn_id

    return best_id or min(eligible, key=lambda p: p.consensus_overall_rank).espn_id


def _simulate_remainder(state, first_pick_id, available, pick_model, settings, by_id, rng):
    """Play the rest of the draft out and return every team's final roster."""
    limits = roster_limits(settings)
    remaining = [p for p in available if p.espn_id != first_pick_id]

    ranks = np.array([p.consensus_overall_rank for p in remaining], dtype=float)
    board_position = (np.argsort(np.argsort(ranks)) + 1).astype(float)
    order = np.argsort(pick_model.sample_board_order(board_position, ranks, rng))
    queue = [remaining[int(i)] for i in order]

    rosters: dict[int, list] = {slot: [] for slot in range(1, settings.team_count + 1)}
    for espn_id in state.drafted:
        projection = by_id.get(espn_id)
        if projection is not None:
            # Prior picks are attributed to their owner only for my own slot;
            # opponents' exact rosters before now do not affect what is left.
            pass
    rosters[state.my_slot] = [by_id[i] for i in state.my_roster if i in by_id]
    rosters[state.my_slot].append(by_id[first_pick_id])

    cursor = 0
    for pick_number in range(state.current_pick + 1, state.total_picks + 1):
        if cursor >= len(queue):
            break
        slot = snake_slot_for_pick(pick_number, settings.team_count)
        if slot == state.my_slot:
            counts = {}
            for projection in rosters[slot]:
                counts[projection.position] = counts.get(projection.position, 0) + 1
            needed = _needed(counts, limits)
            choice_index = next(
                (i for i in range(cursor, len(queue)) if queue[i].position in needed), cursor
            )
            rosters[slot].append(queue.pop(choice_index))
        else:
            rosters[slot].append(queue.pop(cursor))

    return rosters


def _roster_stats(players, settings, replacement) -> tuple[float, float]:
    grouped: dict[str, list[float]] = {}
    spreads: dict[str, list[float]] = {}
    for projection in players:
        grouped.setdefault(projection.position, []).append(projection.mean)
        spreads.setdefault(projection.position, []).append(projection.sd)

    mean = lineup_points(grouped, settings, replacement)
    # Starters dominate the variance; approximating with the best few is enough
    # to rank strategies, which is all this number is used for.
    top_sds = sorted((sd for values in spreads.values() for sd in values), reverse=True)[:7]
    sd = float(np.sqrt(sum(value**2 for value in top_sds))) if top_sds else 0.0
    return mean, sd


def rollout(
    pick_model,
    trials=DEFAULT_ROLLOUT_TRIALS,
    candidates=DEFAULT_ROLLOUT_CANDIDATES,
    objective="points",
    seed=0,
):
    """Judge a pick by the roster it leads to, not the roster it joins."""

    def choose(state, projections, settings, by_id):
        taken = state.drafted_set
        available = [p for p in projections if p.espn_id not in taken]
        if not available:
            return None

        limits = roster_limits(settings)
        replacement = {p.position: p.replacement for p in projections}
        my_roster = roster_points(state.my_roster, by_id)
        needed = _needed(_counts(state.my_roster, by_id), limits)

        # Consider both the best raw talent and the best slot-fillers, because
        # the whole question is which of those two instincts is right here.
        pool = [p for p in available if p.position in needed] or available
        by_talent = sorted(pool, key=lambda p: p.consensus_overall_rank)[:candidates]
        by_fit = sorted(
            pool,
            key=lambda p: -marginal_value(p.position, p.mean, my_roster, settings, replacement),
        )[:candidates]
        shortlist = list({p.espn_id: p for p in by_talent + by_fit}.values())

        rng = np.random.default_rng(seed + state.current_pick)
        best_score, best_id = None, None

        for candidate in shortlist:
            total = 0.0
            for _ in range(trials):
                rosters = _simulate_remainder(
                    state, candidate.espn_id, available, pick_model, settings, by_id, rng
                )
                stats = {
                    slot: _roster_stats(players, settings, replacement)
                    for slot, players in rosters.items()
                    if players
                }
                mine = stats.get(state.my_slot)
                if mine is None:
                    continue
                if objective == "points":
                    total += mine[0]
                else:
                    # Championship proxy: draw a season for every team and ask
                    # how often mine finishes on top. Variance counts here,
                    # which is the entire reason this objective exists.
                    draws = {
                        slot: rng.normal(mean, sd if sd > 0 else 1.0)
                        for slot, (mean, sd) in stats.items()
                    }
                    total += 1.0 if draws[state.my_slot] >= max(draws.values()) else 0.0

            score = total / trials
            if best_score is None or score > best_score:
                best_score, best_id = score, candidate.espn_id

        return best_id

    return choose


def snake_state_for(settings, rounds, slot) -> DraftState:
    return DraftState(team_count=settings.team_count, rounds=rounds, my_slot=slot)
