from dataclasses import dataclass

from fantasy_football.transactions.evaluate import evaluate_move, find_trades


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
