from dataclasses import dataclass

from fantasy_football.transactions.evaluate import (
    MoveEvaluation,
    SimulationContext,
    evaluate_move,
    find_trades,
    marginal_cost,
)


class Settings:
    starting_slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1}
    team_count = 4
    playoff_team_count = 2


SETTINGS = Settings()


@dataclass
class P:
    player: str
    position: str
    mean: float
    sd: float


def roster(prefix, quality):
    return [
        P(f"{prefix}-QB", "QB", 18.0 * quality, 6.0),
        P(f"{prefix}-RB1", "RB", 16.0 * quality, 7.0),
        P(f"{prefix}-RB2", "RB", 12.0 * quality, 6.0),
        P(f"{prefix}-WR1", "WR", 15.0 * quality, 7.0),
        P(f"{prefix}-WR2", "WR", 11.0 * quality, 6.0),
        P(f"{prefix}-TE", "TE", 9.0 * quality, 5.0),
        P(f"{prefix}-FLEX", "WR", 8.0 * quality, 5.0),
    ]


def league():
    rosters = {i: roster(f"T{i}", 1.0) for i in range(1, 5)}
    names = {i: f"Team {i}" for i in range(1, 5)}
    schedule = {
        1: [2, 3, 4],
        2: [1, 4, 3],
        3: [4, 1, 2],
        4: [3, 2, 1],
    }
    return rosters, names, schedule


class TestWaiverClaim:
    def test_adding_a_better_player_raises_title_odds(self):
        rosters, names, schedule = league()
        star = P("Star", "RB", 30.0, 8.0)
        weakest = min(rosters[1], key=lambda p: p.mean)
        move = evaluate_move(
            1, rosters, names, SETTINGS, schedule, 2, add=[star], drop=[weakest], trials=300
        )
        assert move.title_delta > 0
        assert move.weekly_points_change > 0

    def test_a_waiver_claim_leaves_other_rosters_alone(self):
        rosters, names, schedule = league()
        before = [p.player for p in rosters[2]]
        evaluate_move(
            1,
            rosters,
            names,
            SETTINGS,
            schedule,
            2,
            add=[P("Star", "RB", 30.0, 8.0)],
            drop=[rosters[1][-1]],
            trials=100,
        )
        assert [p.player for p in rosters[2]] == before

    def test_summary_reads_cleanly(self):
        rosters, names, schedule = league()
        move = evaluate_move(
            1,
            rosters,
            names,
            SETTINGS,
            schedule,
            2,
            add=[P("Star", "RB", 30.0, 8.0)],
            drop=[rosters[1][-1]],
            trials=100,
        )
        assert "title" in move.summary() and "pts/week" in move.summary()


class TestTradeIsTwoSided:
    def test_the_counterparty_actually_loses_the_player(self):
        """The defect this test exists to catch: without a counterparty the
        acquired player would score for both teams at once."""
        rosters, names, schedule = league()
        target = max(rosters[2], key=lambda p: p.mean)
        spare = min(rosters[1], key=lambda p: p.mean)

        one_sided = evaluate_move(
            1, rosters, names, SETTINGS, schedule, 2, add=[target], drop=[spare], trials=400
        )
        two_sided = evaluate_move(
            1,
            rosters,
            names,
            SETTINGS,
            schedule,
            2,
            add=[target],
            drop=[spare],
            trials=400,
            counterparty_id=2,
        )
        # Stripping a rival should help at least as much as a free acquisition.
        assert two_sided.title_delta >= one_sided.title_delta - 1e-9

    def test_counterparty_odds_are_reported(self):
        rosters, names, schedule = league()
        move = evaluate_move(
            1,
            rosters,
            names,
            SETTINGS,
            schedule,
            2,
            add=[max(rosters[2], key=lambda p: p.mean)],
            drop=[min(rosters[1], key=lambda p: p.mean)],
            trials=300,
            counterparty_id=2,
        )
        assert move.counterparty_id == 2
        assert move.counterparty_before > 0

    def test_a_lopsided_trade_is_not_mutually_acceptable(self):
        rosters, names, schedule = league()
        move = evaluate_move(
            1,
            rosters,
            names,
            SETTINGS,
            schedule,
            2,
            add=[max(rosters[2], key=lambda p: p.mean)],
            drop=[P("Junk", "WR", 1.0, 1.0)],
            trials=400,
            counterparty_id=2,
        )
        assert not move.mutually_acceptable

    def test_label_says_send_not_drop_for_a_trade(self):
        rosters, names, schedule = league()
        move = evaluate_move(
            1,
            rosters,
            names,
            SETTINGS,
            schedule,
            2,
            add=[rosters[2][0]],
            drop=[rosters[1][-1]],
            trials=100,
            counterparty_id=2,
        )
        assert "send" in move.description


class TestTradeSearch:
    def test_only_returns_mutually_beneficial_deals(self):
        rosters, names, schedule = league()
        found = find_trades(
            1, rosters, names, SETTINGS, schedule, 2, give_depth=2, get_depth=2, trials=150
        )
        assert all(t.mutually_acceptable for t in found)

    def test_sorted_by_our_own_benefit(self):
        rosters, names, schedule = league()
        found = find_trades(
            1, rosters, names, SETTINGS, schedule, 2, give_depth=2, get_depth=2, trials=150
        )
        deltas = [t.title_delta for t in found]
        assert deltas == sorted(deltas, reverse=True)

    def test_empty_roster_finds_nothing(self):
        rosters, names, schedule = league()
        rosters[1] = []
        assert find_trades(1, rosters, names, SETTINGS, schedule, 2, trials=50) == []


class TestSurplusIsAboutSlotsNotPoints:
    """The bug this exists to catch: offering away the only startable RB.

    On a receiver-heavy roster the best running back can sit below six
    receivers on raw projection while being the only player who can fill a
    starting RB slot. Ranking "spare" by points offered him up; ranking by what
    his removal costs the lineup does not.
    """

    def _lopsided(self):
        """One startable RB, six receivers who cannot all start."""
        return [
            P("QB", "QB", 18.0, 6.0),
            P("WR1", "WR", 20.0, 8.0),
            P("WR2", "WR", 19.0, 8.0),
            P("WR3", "WR", 18.5, 8.0),
            P("WR4", "WR", 18.0, 8.0),
            P("WR5", "WR", 17.5, 8.0),
            P("WR6", "WR", 17.0, 8.0),
            P("RB1", "RB", 12.0, 6.0),
            P("TE", "TE", 9.0, 5.0),
        ]

    def test_the_only_startable_rb_is_not_surplus(self):
        rosters = {1: self._lopsided(), 2: roster("T2", 1.0)}
        context = SimulationContext(
            names={1: "mine", 2: "theirs"},
            settings=SETTINGS,
            schedule={1: [2], 2: [1]},
            playoff_teams=2,
        )
        rb_cost = marginal_cost(rosters[1], rosters[1][7], context)
        spare_wr_cost = marginal_cost(rosters[1], rosters[1][6], context)
        assert rb_cost > 0
        assert spare_wr_cost == 0
        # ...even though the running back scores far fewer points.
        assert rosters[1][7].mean < rosters[1][6].mean

    def test_a_sixth_receiver_who_never_starts_is_free_to_trade(self):
        rosters = {1: self._lopsided()}
        context = SimulationContext(
            names={1: "mine"}, settings=SETTINGS, schedule={1: []}, playoff_teams=1
        )
        cheapest = min(rosters[1], key=lambda p: marginal_cost(rosters[1], p, context))
        assert cheapest.position == "WR"


class TestNoiseIsNotReportedAsSignal:
    def test_a_move_that_changes_nothing_is_not_worth_doing(self):
        rosters, names, schedule = league()
        duplicate = P("Clone", "WR", rosters[1][-1].mean, rosters[1][-1].sd)
        move = evaluate_move(
            1, rosters, names, SETTINGS, schedule, 2, add=[duplicate], drop=[rosters[1][-1]]
        )
        assert move.title_delta == 0.0
        assert not move.worth_doing

    def test_a_delta_inside_its_own_error_is_refused(self):
        tiny = MoveEvaluation(
            description="noise",
            title_before=0.055,
            title_after=0.060,
            weekly_points_change=-0.4,
            title_stderr=0.016,
        )
        # The exact shape of the trade this repo used to recommend: +0.50% at a
        # standard error of 1.6 points of probability.
        assert tiny.title_delta < tiny.title_stderr
        assert not tiny.worth_doing
        assert not tiny.significant

    def test_the_error_bar_is_printed(self):
        rosters, names, schedule = league()
        move = evaluate_move(
            1,
            rosters,
            names,
            SETTINGS,
            schedule,
            2,
            add=[P("Star", "RB", 30.0, 8.0)],
            drop=[rosters[1][-1]],
        )
        assert "+/-" in move.summary()


class TestPackageTrades:
    def test_two_for_two_deals_are_reachable(self):
        """A receiver-rich roster trading for running backs is a two-for-two.

        No sequence of one-for-ones gets there: each leg on its own leaves the
        roster worse or the counterparty unwilling.
        """
        mine = [
            P("QB", "QB", 18.0, 6.0),
            P("WR1", "WR", 20.0, 8.0),
            P("WR2", "WR", 19.0, 8.0),
            P("WR3", "WR", 18.0, 8.0),
            P("WR4", "WR", 17.0, 8.0),
            P("RB1", "RB", 8.0, 5.0),
            P("RB2", "RB", 7.0, 5.0),
            P("TE", "TE", 9.0, 5.0),
        ]
        theirs = [
            P("QB2", "QB", 18.0, 6.0),
            P("RB-A", "RB", 19.0, 7.0),
            P("RB-B", "RB", 18.0, 7.0),
            P("RB-C", "RB", 17.0, 7.0),
            P("RB-D", "RB", 16.0, 7.0),
            P("WR-x", "WR", 7.0, 5.0),
            P("WR-y", "WR", 6.0, 5.0),
            P("TE2", "TE", 9.0, 5.0),
        ]
        # Four teams, two playoff places: with only two teams the title is
        # zero-sum and no trade can ever help both sides.
        rosters = {1: mine, 2: theirs, 3: roster("T3", 1.0), 4: roster("T4", 1.0)}
        names = {i: f"Team {i}" for i in range(1, 5)}
        _, _, schedule = league()
        found = find_trades(1, rosters, names, SETTINGS, schedule, 2)
        assert found, "a mutually positive deal should be reachable"
        assert all(t.weekly_points_change > 0 and t.counterparty_delta > 0 for t in found)

        def package_size(move):
            return len(move.description.split("add ")[1].split(" / ")[0].split(", "))

        # The search is allowed to prefer a one-for-one when one is better; what
        # it must not do is be unable to see a two-for-two at all.
        assert any(package_size(t) == 2 for t in found)

    def test_restricting_to_singles_loses_the_package_deals(self):
        """Guards the claim above: the two-player deals really do come from
        allowing packages, not from the search finding them anyway."""
        rosters, names, schedule = league()
        singles = find_trades(1, rosters, names, SETTINGS, schedule, 2, package_sizes=(1,))
        assert all(
            len(t.description.split("add ")[1].split(" / ")[0].split(", ")) == 1 for t in singles
        )

    def test_singles_only_still_works(self):
        rosters, names, schedule = league()
        found = find_trades(1, rosters, names, SETTINGS, schedule, 2, package_sizes=(1,))
        assert all(t.mutually_acceptable for t in found)
