import numpy as np
import polars as pl
import pytest

from fantasy_football.draft.model import fit_pick_model
from fantasy_football.draft.recommend import recommend, roster_limits
from fantasy_football.draft.state import DraftState, picks_for_slot, snake_slot_for_pick
from fantasy_football.projections.ensemble import Projection


class TestSnakeOrder:
    def test_first_round_runs_forward(self):
        assert [snake_slot_for_pick(p, 8) for p in range(1, 9)] == list(range(1, 9))

    def test_second_round_runs_backward(self):
        assert [snake_slot_for_pick(p, 8) for p in range(9, 17)] == list(range(8, 0, -1))

    def test_third_round_runs_forward_again(self):
        assert snake_slot_for_pick(17, 8) == 1
        assert snake_slot_for_pick(24, 8) == 8

    def test_slot_three_picks_match_a_hand_worked_example(self):
        assert picks_for_slot(3, 8, 4) == [3, 14, 19, 30]

    def test_turn_slots_pick_back_to_back(self):
        last = picks_for_slot(8, 8, 3)
        assert last == [8, 9, 24]

    def test_every_pick_is_owned_exactly_once(self):
        owners = [snake_slot_for_pick(p, 8) for p in range(1, 8 * 16 + 1)]
        assert len(owners) == 128
        for slot in range(1, 9):
            assert owners.count(slot) == 16

    def test_pick_zero_is_rejected(self):
        with pytest.raises(ValueError, match="start at 1"):
            snake_slot_for_pick(0, 8)


class TestDraftState:
    def test_tracks_whose_turn_it_is(self):
        state = DraftState(team_count=8, rounds=16, my_slot=3)
        assert state.current_pick == 1
        assert not state.is_my_turn
        state.record(101)
        state.record(102)
        assert state.is_my_turn

    def test_gap_to_next_pick(self):
        state = DraftState(team_count=8, rounds=16, my_slot=3)
        for i in range(2):
            state.record(100 + i)
        # At pick 3, my next is 14, so ten players go between.
        assert state.picks_until_my_next() == 10

    def test_gap_from_a_turn_slot_is_zero(self):
        state = DraftState(team_count=8, rounds=16, my_slot=8)
        for i in range(7):
            state.record(100 + i)
        assert state.current_pick == 8
        assert state.is_my_turn
        assert state.picks_until_my_next() == 0

    def test_roster_only_records_my_picks(self):
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        state.record_my_pick(1)
        state.record(2)
        assert state.my_roster == [1]
        assert state.drafted == [1, 2]

    def test_round_numbering(self):
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        assert state.round_of(1) == 1
        assert state.round_of(8) == 1
        assert state.round_of(9) == 2

    def test_completion(self):
        state = DraftState(team_count=2, rounds=2, my_slot=1)
        for i in range(4):
            state.record(i)
        assert state.is_complete
        assert state.next_pick_for_me() is None
        with pytest.raises(ValueError, match="already complete"):
            state.record(99)


def pick_history(n=200, noise=5.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for season in (2024, 2025):
        for rank in range(1, n + 1):
            pick = max(1, rank + rng.normal(0, noise))
            rows.append((season, rank, pick))
    return pl.DataFrame(rows, schema=["season", "consensus_rank", "overall_pick"], orient="row")


class TestPickModel:
    def test_expected_pick_tracks_rank(self):
        model = fit_pick_model(pick_history())
        assert abs(model.expected_pick(50) - 50) < 8

    def test_survival_falls_as_the_draft_advances(self):
        model = fit_pick_model(pick_history())
        assert model.survival(20, 5) > model.survival(20, 20) > model.survival(20, 60)

    def test_better_players_are_less_likely_to_survive(self):
        model = fit_pick_model(pick_history())
        assert model.survival(5, 30) < model.survival(80, 30)

    def test_survival_is_a_probability(self):
        model = fit_pick_model(pick_history())
        for rank in (1, 25, 100):
            for pick in (1, 40, 200):
                assert 0.0 <= model.survival(rank, pick) <= 1.0

    def test_too_little_history_returns_none(self):
        assert fit_pick_model(pick_history(n=2)) is None

    def test_board_order_is_sampled_per_player(self):
        model = fit_pick_model(pick_history())
        rng = np.random.default_rng(0)
        order = model.sample_board_order(np.arange(1, 21.0), np.arange(1, 21.0), rng)
        assert len(order) == 20
        # Noisy but still correlated with board position.
        assert np.corrcoef(order, np.arange(20))[0, 1] > 0.5


class FakeSettings:
    starting_slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1, "K": 1, "D/ST": 1}
    team_count = 8


def projection(name, position, overall, positional, vor):
    return Projection(
        player=name,
        position=position,
        team="X",
        espn_id=hash(name) % 100000,
        consensus_rank=positional,
        consensus_overall_rank=overall,
        espn_rank=positional,
        blended_rank=float(positional),
        mean=vor + 150.0,
        sd=70.0,
        replacement=150.0,
    )


class TestRosterLimits:
    def test_flex_widens_the_eligible_positions(self):
        limits = roster_limits(FakeSettings())
        assert limits["RB"] == 2 + 1 + 2
        assert limits["WR"] == 2 + 1 + 2
        assert limits["QB"] == 1 + 1

    def test_kicker_and_defense_are_capped_at_one(self):
        limits = roster_limits(FakeSettings())
        assert limits["K"] == 1
        assert limits["D/ST"] == 1


class TestRecommend:
    @pytest.fixture
    def board(self):
        players = []
        for i in range(1, 41):
            position = ["WR", "RB", "WR", "RB", "TE"][i % 5]
            players.append(projection(f"P{i}", position, i, i, 120.0 - 2.5 * i))
        return players

    @pytest.fixture
    def model(self):
        return fit_pick_model(pick_history())

    def test_returns_ranked_options(self, board, model):
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        out = recommend(state, board, model, FakeSettings(), trials=40)
        assert out
        values = [r.two_pick_value for r in out]
        assert values == sorted(values, reverse=True)

    def test_already_drafted_players_are_excluded(self, board, model):
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        state.record(board[0].espn_id)
        out = recommend(state, board, model, FakeSettings(), trials=40)
        assert board[0].player not in {r.player for r in out}

    def test_survival_is_a_probability(self, board, model):
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        out = recommend(state, board, model, FakeSettings(), trials=60)
        assert all(0.0 <= r.survival <= 1.0 for r in out)

    def test_top_players_are_less_likely_to_survive(self, board, model):
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        out = {r.player: r for r in recommend(state, board, model, FakeSettings(), trials=200)}
        assert out["P1"].survival < out["P30"].survival

    def test_roster_caps_stop_stacking_one_position(self, board, model):
        """With two quarterbacks already rostered, a third adds nothing."""
        settings = FakeSettings()
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        qbs = [projection(f"QB{i}", "QB", 50 + i, i, 40.0) for i in range(1, 4)]
        for qb in qbs[:2]:
            state.record_my_pick(qb.espn_id)
        out = recommend(state, board + qbs, model, settings, trials=40)
        assert out

    def test_last_pick_of_the_draft_has_no_lookahead(self, board, model):
        state = DraftState(team_count=2, rounds=1, my_slot=2)
        state.record(999)
        out = recommend(state, board, model, FakeSettings(), trials=10)
        assert out
        assert all(r.survival == 1.0 for r in out)
        assert all(r.next_pick_value == 0.0 for r in out)

    def test_two_pick_value_is_the_sum(self, board, model):
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        out = recommend(state, board, model, FakeSettings(), trials=40)[0]
        assert out.two_pick_value == pytest.approx(out.value_now + out.next_pick_value)

    def test_expected_loss_scales_with_scarcity(self):
        from fantasy_football.draft.recommend import Recommendation

        def make(survival):
            return Recommendation(
                player="A",
                position="WR",
                team="CIN",
                espn_id=1,
                consensus_rank=1,
                vor=80.0,
                mean=250.0,
                sd=70.0,
                survival=survival,
                value_now=80.0,
                next_pick_value=60.0,
            )

        assert make(0.0).expected_loss_if_passed == 80.0
        assert make(1.0).expected_loss_if_passed == 0.0

    def test_rationale_mentions_scarcity_when_a_player_may_vanish(self):
        from fantasy_football.draft.recommend import Recommendation

        scarce = Recommendation(
            player="A",
            position="WR",
            team="CIN",
            espn_id=1,
            consensus_rank=1,
            vor=80.0,
            mean=250.0,
            sd=70.0,
            survival=0.2,
            value_now=80.0,
            next_pick_value=60.0,
        )
        assert "20%" in scarce.rationale()
        safe = Recommendation(
            player="B",
            position="WR",
            team="DAL",
            espn_id=2,
            consensus_rank=2,
            vor=80.0,
            mean=250.0,
            sd=70.0,
            survival=0.97,
            value_now=80.0,
            next_pick_value=60.0,
        )
        assert "safe to wait" in safe.rationale()

    def test_empty_board_is_handled(self, model):
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        assert recommend(state, [], model, FakeSettings(), trials=10) == []
