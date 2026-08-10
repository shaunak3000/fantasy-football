import numpy as np
import polars as pl
import pytest

from fantasy_football.draft.model import fit_pick_model
from fantasy_football.draft.script import forecast, simulate_draft, target_survival
from fantasy_football.draft.state import DraftState
from fantasy_football.projections.ensemble import Projection


class Settings:
    starting_slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1, "K": 1, "D/ST": 1}
    team_count = 8


SETTINGS = Settings()


def board(n=140):
    positions = ["WR", "RB", "WR", "RB", "TE", "QB"]
    return [
        Projection(
            player=f"P{i}",
            position=positions[i % len(positions)],
            team="X",
            espn_id=i,
            consensus_rank=i,
            consensus_overall_rank=i,
            espn_rank=i,
            blended_rank=float(i),
            mean=300.0 - 1.5 * i,
            sd=60.0,
            replacement=150.0,
        )
        for i in range(1, n + 1)
    ]


@pytest.fixture
def model():
    rng = np.random.default_rng(0)
    rows = [
        (2024, rank, max(1, rank + rng.normal(0, 8))) for rank in range(1, 140) for _ in range(2)
    ]
    return fit_pick_model(
        pl.DataFrame(rows, schema=["season", "consensus_rank", "overall_pick"], orient="row")
    )


class TestSimulateDraft:
    def test_returns_one_player_per_pick(self, model):
        rng = np.random.default_rng(0)
        drafted = simulate_draft(8, board(), model, SETTINGS, rounds=16, rng=rng)
        assert len(drafted) == 16

    def test_never_drafts_the_same_player_twice(self, model):
        rng = np.random.default_rng(1)
        drafted = simulate_draft(3, board(), model, SETTINGS, rounds=16, rng=rng)
        assert len({p.espn_id for p in drafted}) == len(drafted)

    def test_respects_roster_caps(self, model):
        rng = np.random.default_rng(2)
        drafted = simulate_draft(1, board(), model, SETTINGS, rounds=16, rng=rng)
        counts = {}
        for player in drafted:
            counts[player.position] = counts.get(player.position, 0) + 1
        assert counts.get("QB", 0) <= 2
        assert counts.get("TE", 0) <= 3

    def test_early_slots_get_better_players(self, model):
        rng = np.random.default_rng(3)
        first = simulate_draft(1, board(), model, SETTINGS, rounds=16, rng=rng)
        last = simulate_draft(8, board(), model, SETTINGS, rounds=16, rng=rng)
        assert first[0].consensus_overall_rank <= last[0].consensus_overall_rank


class TestForecast:
    @pytest.fixture
    def plan(self, model):
        return forecast(8, board(), model, SETTINGS, rounds=16, trials=40)

    def test_covers_every_pick_the_slot_owns(self, plan):
        expected = DraftState(team_count=8, rounds=16, my_slot=8).my_picks
        assert [f.pick_number for f in plan] == expected

    def test_probabilities_are_shares(self, plan):
        for item in plan:
            assert all(0.0 < share <= 1.0 for _, _, share in item.likely)
            assert sum(item.position_mix.values()) == pytest.approx(1.0, abs=0.01)

    def test_value_declines_through_the_draft(self, plan):
        assert plan[0].mean_vor > plan[-1].mean_vor

    def test_rounds_are_labelled_correctly(self, plan):
        assert plan[0].round_number == 1
        assert plan[1].round_number == 2

    def test_headline_names_a_player(self, plan):
        assert "%" in plan[0].headline()


class TestTargetSurvival:
    def test_returns_probabilities_for_the_top_of_the_board(self, model):
        rows = target_survival(8, board(), model, SETTINGS, rounds=16, trials=60, top=10)
        assert len(rows) == 10
        assert all(0.0 <= survival <= 1.0 for _, _, _, survival in rows)

    def test_best_players_are_least_likely_to_survive(self, model):
        rows = target_survival(8, board(), model, SETTINGS, rounds=16, trials=200, top=20)
        assert rows[0][3] < rows[-1][3]

    def test_first_slot_sees_everyone(self, model):
        rows = target_survival(1, board(), model, SETTINGS, rounds=16, trials=30, top=5)
        assert all(survival == 1.0 for _, _, _, survival in rows)
