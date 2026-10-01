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


@dataclass
class Q:
    """A player with an id, so season stats can be attached to him."""

    player: str
    position: str
    mean: float
    espn_id: int
    sd: float = 5.0
    bye_week: int | None = None
    return_week: int | None = None


def perception_league():
    """Our WR3 projects 12 but has scored 25 a game; their RB-C projects 15 but has
    scored 10. By projection they lose the swap; by the box score they win it. The
    shape of Lamb for McCaffrey."""
    mine = [
        Q("QB", "QB", 18.0, 1),
        Q("RB1", "RB", 16.0, 2),
        Q("RB2", "RB", 4.0, 3),
        Q("WR1", "WR", 20.0, 4),
        Q("WR2", "WR", 19.0, 5),
        Q("HOT-WR", "WR", 12.0, 6),
        Q("WR4", "WR", 11.0, 7),
        Q("TE", "TE", 9.0, 8),
    ]
    theirs = [
        Q("QB2", "QB", 18.0, 11),
        Q("RB-A", "RB", 17.0, 12),
        Q("RB-B", "RB", 16.0, 13),
        Q("COLD-RB", "RB", 15.0, 14),
        Q("WR-x", "WR", 14.0, 15),
        Q("WR-y", "WR", 13.0, 16),
        Q("TE2", "TE", 9.0, 17),
    ]
    f3 = [Q(p.player, p.position, p.mean, 100 + i) for i, p in enumerate(filler("T3"))]
    f4 = [Q(p.player, p.position, p.mean, 200 + i) for i, p in enumerate(filler("T4"))]
    rosters = {1: mine, 2: theirs, 3: f3, 4: f4}
    names = {i: f"Team {i}" for i in rosters}
    _, _, schedule = swap_league()
    perceived = {6: 25.0, 14: 10.0}
    return rosters, names, schedule, perceived


class TestTheOtherManagersLens:
    def test_box_score_counts_skill_players_only(self):
        from fantasy_football.transactions.evaluate import box_score

        to_them = [Q("WR", "WR", 10, 1), Q("D", "D/ST", 8, 2)]
        from_them = [Q("RB", "RB", 10, 3), Q("K", "K", 9, 4)]
        assert box_score(to_them, from_them, {1: 20.0, 2: 12.0, 3: 15.0, 4: 11.0}) == (20.0, 15.0)
        assert box_score(to_them, from_them, None) is None

    def test_a_player_without_season_stats_makes_the_box_score_unknown(self):
        """Counted as zero, a package of unknowns passes at 0 against 0."""
        from fantasy_football.transactions.evaluate import box_score

        known, unknown = Q("A", "WR", 10, 1), Q("B", "RB", 10, 2)
        assert box_score([known], [unknown], {1: 20.0}) is None
        assert box_score([unknown], [known], {1: 20.0}) is None

    def test_a_defence_needs_no_stats_to_be_left_out(self):
        from fantasy_football.transactions.evaluate import box_score

        assert box_score(
            [Q("A", "WR", 10, 1), Q("D", "D/ST", 8, 9)], [Q("B", "RB", 10, 2)], {1: 20.0, 2: 12.0}
        ) == (20.0, 12.0)

    def test_season_stats_open_deals_our_model_says_they_lose(self):
        """Selling a player whose box score runs ahead of his projection: our model
        says the buyer loses, the buyer's own reading of his lineup says he wins."""
        rosters, names, schedule, perceived = perception_league()
        found = find_trades(1, rosters, names, SETTINGS, schedule, 2, perceived=perceived)
        via_box = [m for m in found if m.accepted_on == "box score"]
        assert via_box, "the perception lane should produce deals"
        for move in via_box:
            assert not move.model_accepts
            assert move.box_lineup_change >= 1.0
        # Among them, the gap being exploited: selling the receiver whose box score
        # runs far ahead of his projection.
        # (Usage labels need a current week, which this fixture does not set, so the
        # traded players are read from the description.)
        assert any("HOT-WR" in m.description.split(" / ")[1] for m in via_box)

    def test_a_box_score_yes_on_a_lopsided_deal_is_still_refused(self):
        """The toy league swings hard: the model has the buyer losing ~29% on the
        one-for-one. He would read it as a win, and the cap refuses it anyway."""
        rosters, names, schedule, perceived = perception_league()
        ctx = SimulationContext(names=names, settings=SETTINGS, schedule=schedule, playoff_teams=2)
        hot = next(p for p in rosters[1] if p.player == "HOT-WR")
        cold = next(p for p in rosters[2] if p.player == "COLD-RB")
        m = evaluate_move(
            1,
            rosters,
            names,
            SETTINGS,
            schedule,
            2,
            add=[cold],
            drop=[hot],
            counterparty_id=2,
            context=ctx,
            perceived=perceived,
        )
        assert (m.box_they_get, m.box_they_give) == (25.0, 10.0)
        assert m.box_score_accepts and m.counterparty_delta < -0.01
        assert not m.mutually_acceptable

    def test_without_season_stats_only_model_approved_deals_appear(self):
        """The old behaviour, and the reason Lamb for McCaffrey never appeared."""
        rosters, names, schedule, _ = perception_league()
        found = find_trades(1, rosters, names, SETTINGS, schedule, 2)
        assert all(m.model_accepts for m in found)
        assert found_deal(found, {"COLD-RB"}, {"HOT-WR"}) is None

    def test_it_must_still_be_good_for_us(self):
        """Their lens decides whether they say yes. Ours decides whether we ask."""
        m = MoveEvaluation(
            description="x",
            title_before=0.2,
            title_after=0.199,
            weekly_points_change=0.0,
            title_stderr=0.002,
            counterparty_id=2,
            box_lineup_before=100.0,
            box_lineup_after=110.0,
        )
        assert m.box_score_accepts and not m.mutually_acceptable

    def test_accepted_on_names_every_lens_that_said_yes(self):
        both = MoveEvaluation(
            description="x",
            title_before=0.1,
            title_after=0.13,
            weekly_points_change=1.0,
            title_stderr=0.002,
            counterparty_id=2,
            counterparty_before=0.1,
            counterparty_after=0.12,
            counterparty_stderr=0.002,
            box_lineup_before=100.0,
            box_lineup_after=105.0,
        )
        assert both.accepted_on == "model + box score"

    def test_a_tie_is_not_a_yes(self):
        """A manager needs a reason to move; a lineup a quarter-point better is not one."""
        m = MoveEvaluation(
            description="x",
            title_before=0.1,
            title_after=0.2,
            weekly_points_change=1.0,
            counterparty_id=2,
            box_lineup_before=100.0,
            box_lineup_after=100.25,
        )
        assert m.box_score_accepts is False

    def test_equal_traded_totals_that_gut_their_lineup_are_a_no(self):
        """What the first version got wrong: McCaffrey and Cook for Collins and
        Javonte looked like 38.0 for 36.1 and passed, while taking both of T Money's
        starting running backs. Judged by his lineup, it is plainly a loss to him."""
        from fantasy_football.transactions.evaluate import box_score, perceived_lineup

        theirs = [
            Q("QB", "QB", 18.0, 1),
            Q("RB-1", "RB", 19.0, 2),
            Q("RB-2", "RB", 17.0, 3),
            Q("WR-1", "WR", 16.0, 4),
            Q("WR-2", "WR", 15.0, 5),
            Q("TE", "TE", 9.0, 6),
            Q("WR-3", "WR", 12.0, 7),
        ]
        incoming = [Q("WR-in", "WR", 13.0, 8), Q("WR-in2", "WR", 13.0, 9)]
        seen = {2: 19.0, 3: 17.0, 8: 18.5, 9: 18.5}
        gives = [p for p in theirs if p.position == "RB"]
        assert box_score(incoming, gives, seen) == (37.0, 36.0)  # "passes" on the sum
        before = perceived_lineup(theirs, seen, SETTINGS)
        after = perceived_lineup([p for p in theirs if p not in gives] + incoming, seen, SETTINGS)
        assert after - before < 0  # no running backs left to start


class TestTheBoxScoreCannotHideALargeLoss:
    def _move(self, their_delta):
        return MoveEvaluation(
            description="x",
            title_before=0.2,
            title_after=0.25,
            weekly_points_change=3.0,
            title_stderr=0.003,
            counterparty_id=2,
            counterparty_before=0.1,
            counterparty_after=0.1 + their_delta,
            counterparty_stderr=0.002,
            box_lineup_before=100.0,
            box_lineup_after=104.0,
        )

    def test_a_small_model_loss_can_be_a_box_score_yes(self):
        """Lamb for McCaffrey: -0.87% to T Money by the model."""
        assert self._move(-0.0087).mutually_acceptable

    def test_a_large_model_loss_cannot(self):
        """McCaffrey and Cook for Warren and LaPorta: -8.9% by the model."""
        assert not self._move(-0.089).mutually_acceptable


class TestTheyKnowWhoIsInjured:
    def test_an_injured_player_is_not_a_starter_in_their_eyes_until_he_returns(self):
        """Selling A.J. Brown (IR until week 8) must not read to the buyer as
        handing him a healthy starter for the next four weeks."""
        from fantasy_football.transactions.evaluate import perceived_lineup

        roster = [
            Q("QB", "QB", 18.0, 1),
            Q("RB1", "RB", 15.0, 2),
            Q("RB2", "RB", 14.0, 3),
            Q("WR1", "WR", 15.0, 4),
            Q("WR2", "WR", 12.0, 5),
            Q("TE", "TE", 9.0, 6),
        ]
        weeks = list(range(4, 12))
        injured = Q("Brown", "WR", 20.0, 7, return_week=8)
        healthy = Q("Brown", "WR", 20.0, 7)
        base = perceived_lineup(roster, {}, SETTINGS, weeks, 4)
        with_injured = perceived_lineup([*roster, injured], {}, SETTINGS, weeks, 4)
        with_healthy = perceived_lineup([*roster, healthy], {}, SETTINGS, weeks, 4)
        # Out weeks 4-7 of 4-11: half the span, so worth half as much.
        assert with_injured - base == pytest.approx((with_healthy - base) / 2)
