import numpy as np
import polars as pl
import pytest

from fantasy_football.draft.board_view import board_view, survival_odds
from fantasy_football.draft.model import fit_pick_model
from fantasy_football.draft.state import DraftState
from fantasy_football.projections.ensemble import Projection


class Settings:
    starting_slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1, "K": 1, "D/ST": 1}
    team_count = 8


def projection(name, position, overall, mean=200.0):
    return Projection(
        player=name,
        position=position,
        team="X",
        espn_id=overall,
        consensus_rank=overall,
        consensus_overall_rank=overall,
        espn_rank=overall,
        blended_rank=float(overall),
        mean=mean,
        sd=60.0,
        replacement=150.0,
    )


@pytest.fixture
def board():
    positions = ["WR", "RB", "WR", "RB", "TE", "QB"]
    return [
        projection(f"P{i}", positions[i % len(positions)], i, mean=300.0 - 2.0 * i)
        for i in range(1, 61)
    ]


@pytest.fixture
def model():
    rng = np.random.default_rng(0)
    rows = [
        (2024, rank, max(1, rank + rng.normal(0, 6))) for rank in range(1, 150) for _ in range(2)
    ]
    return fit_pick_model(
        pl.DataFrame(rows, schema=["season", "consensus_rank", "overall_pick"], orient="row")
    )


class TestSurvivalOdds:
    def test_all_probabilities(self, board, model):
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        odds = survival_odds(state, board, model, trials=100)
        assert all(0.0 <= v <= 1.0 for v in odds.values())

    def test_highly_ranked_players_survive_less(self, board, model):
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        odds = survival_odds(state, board, model, trials=300)
        assert odds[0] < odds[40]

    def test_everyone_survives_a_back_to_back_pick(self, board, model):
        state = DraftState(team_count=8, rounds=16, my_slot=8)
        for i in range(7):
            state.record(-i - 1)
        assert state.picks_until_my_next() == 0
        odds = survival_odds(state, board, model, trials=50)
        assert set(odds.values()) == {1.0}


class TestBoardView:
    def test_recommends_the_top_ranked_player_at_a_needed_position(self, board, model):
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        rows = board_view(state, board, model, Settings(), trials=50)
        best = next(r for r in rows if r.recommended)
        assert best.overall_rank == min(r.overall_rank for r in rows)

    def test_skips_positions_that_are_already_full(self, board, model):
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        # Fill every quarterback slot; no QB should be recommended after that.
        for projection_ in [p for p in board if p.position == "QB"][:2]:
            state.record_my_pick(projection_.espn_id)
        rows = board_view(state, board, model, Settings(), trials=50)
        best = next(r for r in rows if r.recommended)
        assert best.position != "QB"

    def test_drafted_players_disappear(self, board, model):
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        state.record(board[0].espn_id)
        rows = board_view(state, board, model, Settings(), trials=50)
        assert board[0].player not in {r.player for r in rows}

    def test_exactly_one_recommendation(self, board, model):
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        rows = board_view(state, board, model, Settings(), trials=50)
        assert sum(1 for r in rows if r.recommended) == 1

    def test_rationale_reads_as_a_sentence(self, board, model):
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        rows = board_view(state, board, model, Settings(), trials=50)
        text = rows[0].rationale()
        assert "consensus #" in text and text.endswith(".")

    def test_empty_board_returns_nothing(self, model):
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        assert board_view(state, [], model, Settings(), trials=10) == []
