import pytest

from fantasy_football.draft.state import DraftState
from fantasy_football.draft.strategies import best_available_at_need
from fantasy_football.projections.ensemble import Projection


class Settings:
    starting_slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1, "K": 1, "D/ST": 1}
    team_count = 8


SETTINGS = Settings()


def player(name, position, overall, vor, sd=50.0, mean=None):
    return Projection(
        player=name,
        position=position,
        team="X",
        espn_id=overall,
        consensus_rank=overall,
        consensus_overall_rank=overall,
        espn_rank=overall,
        blended_rank=float(overall),
        mean=vor + 150.0 if mean is None else mean,
        sd=sd,
        replacement=150.0,
    )


class TestEarlyRounds:
    def test_takes_the_highest_ranked_player_at_a_need(self):
        board = [
            player("Best", "WR", 1, 80.0),
            player("Second", "RB", 2, 75.0),
        ]
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        assert best_available_at_need(state, board, SETTINGS, {p.espn_id: p for p in board}) == 1

    def test_ceiling_is_ignored_while_value_remains(self):
        """A high-variance player must not jump a better one early on."""
        board = [
            player("Solid", "WR", 1, 80.0, sd=20.0),
            player("Volatile", "WR", 2, 70.0, sd=200.0),
        ]
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        chosen = best_available_at_need(state, board, SETTINGS, {p.espn_id: p for p in board})
        assert chosen == 1


class TestLateRoundLottery:
    def test_switches_to_ceiling_once_everything_is_below_replacement(self):
        """Past replacement level nothing improves the lineup in expectation,
        so the only thing left worth choosing on is upside."""
        board = [
            player("Safe Veteran", "WR", 1, -10.0, sd=15.0),
            player("Boom Rookie", "WR", 2, -12.0, sd=90.0),
        ]
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        chosen = best_available_at_need(state, board, SETTINGS, {p.espn_id: p for p in board})
        assert chosen == 2

    def test_a_higher_ceiling_still_loses_to_a_much_better_mean(self):
        board = [
            player("Clearly Better", "WR", 1, -2.0, sd=10.0, mean=200.0),
            player("Wild", "WR", 2, -40.0, sd=30.0, mean=110.0),
        ]
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        chosen = best_available_at_need(state, board, SETTINGS, {p.espn_id: p for p in board})
        assert chosen == 1

    def test_roster_needs_still_bind_in_the_lottery_rounds(self):
        board = [
            player("Extra QB", "QB", 1, -5.0, sd=99.0),
            player("Needed WR", "WR", 2, -8.0, sd=20.0),
        ]
        by_id = {p.espn_id: p for p in board}
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        for _ in range(2):
            state.record_my_pick(1)  # two quarterbacks fills the cap
        chosen = best_available_at_need(state, board, SETTINGS, by_id)
        assert chosen == 2


def test_empty_board_returns_none():
    state = DraftState(team_count=8, rounds=16, my_slot=1)
    assert best_available_at_need(state, [], SETTINGS, {}) is None


def test_defenses_are_draftable():
    """Regression: D/ST was silently absent from the board entirely."""
    board = [player("Some D/ST", "DST", 1, 20.0)]
    state = DraftState(team_count=8, rounds=16, my_slot=1)
    chosen = best_available_at_need(state, board, SETTINGS, {p.espn_id: p for p in board})
    assert chosen == 1


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_every_position_can_be_chosen(position):
    board = [player(f"{position} guy", position, 1, 20.0)]
    state = DraftState(team_count=8, rounds=16, my_slot=1)
    assert best_available_at_need(state, board, SETTINGS, {p.espn_id: p for p in board}) == 1
