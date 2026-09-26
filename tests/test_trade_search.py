"""The trade search has to see swaps, weeks, and both sides.

Week 3 of 2026 exposed three blind spots at once. The best deal on the board —
DK Metcalf and Chase Brown for Nico Collins and Kyren Williams — was never found,
because the candidate pool was cut by each player's value *on his own*, and
Metcalf on his own added nothing to a roster already starting better receivers.
He only mattered as Collins' replacement, which is a property of the package.
Found by hand, it was then +3.93% on a flat valuation and +3.00% once byes were
counted, while the other side's gain fell from +1.22% to +0.19% — a -12.5 week
for them that nothing could previously show.
"""

from dataclasses import dataclass

import pytest

from fantasy_football.season.weekly_profile import COVER, DEPTH, STARTER, PlayerUsage
from fantasy_football.transactions.evaluate import (
    MoveEvaluation,
    SimulationContext,
    _distinct,
    evaluate_move,
    find_trades,
    marginal_gain,
)


class Settings:
    starting_slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1}


SETTINGS = Settings()


@dataclass
class P:
    player: str
    position: str
    mean: float
    sd: float = 5.0
    bye_week: int | None = None
    return_week: int | None = None


def filler(prefix, q=1.0):
    return [
        P(f"{prefix}-QB", "QB", 18 * q),
        P(f"{prefix}-RB1", "RB", 15 * q),
        P(f"{prefix}-RB2", "RB", 12 * q),
        P(f"{prefix}-WR1", "WR", 15 * q),
        P(f"{prefix}-WR2", "WR", 12 * q),
        P(f"{prefix}-TE", "TE", 9 * q),
        P(f"{prefix}-FX", "WR", 9 * q),
    ]


def swap_league():
    """Our receiver surplus against their running-back surplus, with a twist:
    the receiver we get back is worse than ours and adds nothing on his own."""
    mine = [
        P("QB", "QB", 18.0),
        P("RB1", "RB", 16.0),
        P("RB2", "RB", 8.0),
        P("WR1", "WR", 20.0),
        P("WR2", "WR", 19.0),
        P("WR3", "WR", 15.0),  # our flex; the Nico Collins of this league
        P("TE", "TE", 9.0),
    ]
    theirs = [
        P("QB2", "QB", 18.0),
        P("RB-A", "RB", 18.0),
        P("RB-B", "RB", 17.0),
        P("RB-C", "RB", 14.0),
        P("RB-D", "RB", 13.0),
        P("WR-x", "WR", 12.0),  # the DK Metcalf: zero standalone value to us
        P("WR-y", "WR", 3.0),
        P("TE2", "TE", 9.0),
    ]
    rosters = {1: mine, 2: theirs, 3: filler("T3"), 4: filler("T4")}
    names = {i: f"Team {i}" for i in rosters}
    weeks = 6
    schedule = {tid: [] for tid in rosters}
    order = list(rosters)
    for w in range(weeks):
        for i, tid in enumerate(order):
            schedule[tid].append(order[(i + w + 1) % len(order)])
    return rosters, names, schedule


def found_deal(moves, get, give):
    for m in moves:
        head = m.description.split(" / ")
        got = set(head[0].removeprefix("add ").split(", "))
        sent = set(head[1].removeprefix("send ").split(", "))
        if got == set(get) and sent == set(give):
            return m
    return None


class TestSwapsAreVisible:
    def test_the_backfill_player_has_no_value_on_his_own(self):
        """The premise: judged alone, WR-x is worthless to us."""
        rosters, names, schedule = swap_league()
        ctx = SimulationContext(names=names, settings=SETTINGS, schedule=schedule, playoff_teams=2)
        wr_x = next(p for p in rosters[2] if p.player == "WR-x")
        assert marginal_gain(rosters[1], wr_x, ctx) == pytest.approx(0.0)

    def test_a_shallow_pool_cannot_reach_the_swap(self):
        """What the old default did: rank their players by standalone value and
        keep the top few. WR-x is never in the pool, so the deal cannot exist."""
        rosters, names, schedule = swap_league()
        shallow = find_trades(1, rosters, names, SETTINGS, schedule, 2, get_depth=4)
        assert found_deal(shallow, {"RB-C", "WR-x"}, {"WR3", "RB2"}) is None

    def test_full_depth_finds_it(self):
        rosters, names, schedule = swap_league()
        deep = find_trades(1, rosters, names, SETTINGS, schedule, 2)
        move = found_deal(deep, {"RB-C", "WR-x"}, {"WR3", "RB2"})
        assert move is not None, "the Metcalf-shaped swap should be reachable at full depth"
        assert move.weekly_points_change > 0 and move.counterparty_weekly_points_change > 0


class TestBothSidesAreMeasured:
    def _with_byes(self):
        rosters, names, schedule = swap_league()
        for p in rosters[1]:
            if p.player == "WR1":
                p.bye_week = 6  # our WR1 sits week 6
        for p in rosters[2]:
            if p.player == "RB-C":
                p.bye_week = 4
        return rosters, names, schedule

    def test_weekly_deltas_and_usage_come_back_for_both_sides(self):
        rosters, names, schedule = self._with_byes()
        ctx = SimulationContext(
            names=names, settings=SETTINGS, schedule=schedule, playoff_teams=2, current_week=3
        )
        get = [p for p in rosters[2] if p.player in ("RB-C", "WR-x")]
        give = [p for p in rosters[1] if p.player in ("WR3", "RB2")]
        m = evaluate_move(
            1,
            rosters,
            names,
            SETTINGS,
            schedule,
            2,
            add=get,
            drop=give,
            counterparty_id=2,
            context=ctx,
        )
        assert m.weeks == (3, 4, 5, 6, 7, 8)
        assert len(m.weekly_delta) == len(m.counterparty_weekly_delta) == 6
        assert {u.player for u in m.incoming} == {"RB-C", "WR-x"}
        assert {u.player for u in m.outgoing} == {"WR3", "RB2"}
        assert m.season_points_change == pytest.approx(sum(m.weekly_delta))
        # RB-C's bye lands in week 4 for us after the trade; that is our worst week.
        assert m.worst_week()[0] == 4
        rb_c = next(u for u in m.incoming if u.player == "RB-C")
        assert 4 not in rb_c.weeks_started

    def test_without_a_current_week_the_old_flat_behaviour_is_kept(self):
        rosters, names, schedule = swap_league()
        ctx = SimulationContext(names=names, settings=SETTINGS, schedule=schedule, playoff_teams=2)
        get = [p for p in rosters[2] if p.player == "RB-C"]
        give = [p for p in rosters[1] if p.player == "RB2"]
        m = evaluate_move(
            1,
            rosters,
            names,
            SETTINGS,
            schedule,
            2,
            add=get,
            drop=give,
            counterparty_id=2,
            context=ctx,
        )
        assert m.weeks == () and m.weekly_delta == () and m.incoming == ()


class TestOneEntryPerRealDeal:
    def _move(self, title, incoming, outgoing, cp=2):
        return MoveEvaluation(
            description=f"deal {title}",
            title_before=0.1,
            title_after=0.1 + title,
            weekly_points_change=1.0,
            counterparty_id=cp,
            incoming=tuple(PlayerUsage(n, r, (), 10) for n, r in incoming),
            outgoing=tuple(PlayerUsage(n, r, (), 10) for n, r in outgoing),
        )

    def test_throw_ins_that_never_start_collapse_into_one_deal(self):
        """Week 3 of 2026: four copies of Chase Brown for Collins and Javonte,
        padded with Caleb Williams, Charbonnet, the 49ers and Mahomes."""
        core_out = [("Collins", STARTER), ("Javonte", STARTER)]
        moves = [
            self._move(0.030, [("Chase Brown", STARTER), ("Caleb", DEPTH)], core_out),
            self._move(0.0295, [("Chase Brown", STARTER), ("Charbonnet", DEPTH)], core_out),
            self._move(0.0295, [("Chase Brown", STARTER), ("49ers D/ST", DEPTH)], core_out),
        ]
        kept = _distinct(moves)
        assert len(kept) == 1 and kept[0].title_delta == pytest.approx(0.030)

    def test_a_cover_player_is_part_of_the_deal_not_a_throw_in(self):
        """Cover plays some weeks. Only a player who never starts is noise."""
        out = [("Collins", STARTER)]
        a = self._move(0.02, [("Brown", STARTER), ("Tuten", COVER)], out)
        b = self._move(0.02, [("Brown", STARTER), ("Caleb", DEPTH)], out)
        assert len(_distinct([a, b])) == 2

    def test_different_partners_are_different_deals(self):
        inc = [("X", STARTER)]
        out = [("Y", STARTER)]
        assert len(_distinct([self._move(0.01, inc, out, 2), self._move(0.01, inc, out, 3)])) == 2


class TestPlayoffStrength:
    def _ctx(self, rosters, names, schedule):
        return SimulationContext(
            names=names, settings=SETTINGS, schedule=schedule, playoff_teams=2, current_week=3
        )

    def test_a_player_back_before_the_playoffs_counts_there(self):
        rosters, names, schedule = swap_league()
        ctx = self._ctx(rosters, names, schedule)
        full = ctx.playoff_points(rosters[1])
        for p in rosters[1]:
            if p.player == "WR1":
                p.return_week = 6  # back well before week 9's bracket
        assert ctx.playoff_points(rosters[1]) == pytest.approx(full)

    def test_a_player_out_past_the_playoffs_does_not(self):
        rosters, names, schedule = swap_league()
        ctx = self._ctx(rosters, names, schedule)
        full = ctx.playoff_points(rosters[1])
        for p in rosters[1]:
            if p.player == "WR1":
                p.return_week = 40
        assert ctx.playoff_points(rosters[1]) < full
