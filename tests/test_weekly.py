import numpy as np
import polars as pl
import pytest

from fantasy_football.projections.ensemble import Projection
from fantasy_football.projections.weekly import (
    NFL_REGULAR_SEASON_GAMES,
    WeeklyModel,
    WeeklyOutlook,
    bye_weeks_from_board,
)

SCHEMA = [
    "pos",
    "pos_rank",
    "actual_points",
    "season",
    "games_played",
    "weekly_sd",
    "points_per_game",
]


def training(seasons=(2021, 2022, 2023, 2024, 2025), n_ranks=30):
    rng = np.random.default_rng(0)
    rows = []
    for season in seasons:
        for rank in range(1, n_ranks + 1):
            games = int(rng.integers(10, 18))
            ppg = max(1.0, 20.0 - 0.4 * rank + rng.normal(0, 2.0))
            rows.append(("RB", rank, ppg * games, season, games, ppg * 0.5, ppg))
    return pl.DataFrame(rows, schema=SCHEMA, orient="row")


def projection(rank=1.0, position="RB", player="Alpha"):
    return Projection(
        player=player,
        position=position,
        team="DET",
        espn_id=1,
        consensus_rank=int(rank),
        consensus_overall_rank=int(rank),
        espn_rank=int(rank),
        blended_rank=rank,
        mean=250.0,
        sd=80.0,
        replacement=150.0,
    )


@pytest.fixture
def model():
    return WeeklyModel.fit(training())


class TestFit:
    def test_better_rank_produces_more_per_game(self, model):
        assert model.per_game["RB"].points_at(1) > model.per_game["RB"].points_at(25)

    def test_per_game_is_not_the_season_total(self, model):
        """A per-game number must be plausible for one game, not seventeen."""
        assert 5.0 < model.per_game["RB"].points_at(1) < 40.0

    def test_expected_games_is_capped_at_the_season_length(self, model):
        assert model.expected_games("RB", 1) <= NFL_REGULAR_SEASON_GAMES

    def test_unknown_position_falls_back_to_a_full_season(self, model):
        assert model.expected_games("QB", 1) == float(NFL_REGULAR_SEASON_GAMES)


class TestOutlook:
    def test_produces_a_usable_week(self, model):
        out = model.outlook(projection(), week=1)
        assert out.week == 1
        assert out.mean > 0 and out.sd > 0
        assert out.is_startable

    def test_bye_week_zeroes_the_projection(self, model):
        out = model.outlook(projection(), week=7, bye_week=7)
        assert out.on_bye
        assert out.mean == 0.0 and out.sd == 0.0
        assert not out.is_startable

    def test_other_weeks_are_unaffected_by_the_bye(self, model):
        out = model.outlook(projection(), week=8, bye_week=7)
        assert not out.on_bye and out.mean > 0

    def test_no_bye_information_means_never_on_bye(self, model):
        assert not model.outlook(projection(), week=7, bye_week=None).on_bye

    def test_unknown_position_yields_a_zero_outlook(self, model):
        out = model.outlook(projection(position="K"), week=1)
        assert out.mean == 0.0
        assert not out.is_startable

    def test_season_outlooks_cover_every_player_week(self, model):
        outs = model.season_outlooks([projection(), projection(player="Bravo")], {}, range(1, 5))
        assert len(outs) == 8


class TestProbabilityOver:
    def test_threshold_at_the_mean_is_a_coin_flip(self):
        out = WeeklyOutlook("A", "RB", 1, mean=15.0, sd=6.0, on_bye=False)
        assert out.probability_over(15.0) == pytest.approx(0.5, abs=0.01)

    def test_higher_threshold_is_less_likely(self):
        out = WeeklyOutlook("A", "RB", 1, mean=15.0, sd=6.0, on_bye=False)
        assert out.probability_over(25.0) < out.probability_over(15.0)

    def test_bye_week_can_never_clear_a_threshold(self):
        out = WeeklyOutlook("A", "RB", 1, mean=0.0, sd=0.0, on_bye=True)
        assert out.probability_over(1.0) == 0.0

    def test_zero_spread_is_deterministic(self):
        out = WeeklyOutlook("A", "RB", 1, mean=20.0, sd=0.0, on_bye=False)
        assert out.probability_over(10.0) == 1.0
        assert out.probability_over(30.0) == 0.0


class TestByeWeeks:
    def test_reads_byes_off_the_board(self):
        board = pl.DataFrame([("Alpha", 5), ("Bravo", 9)], schema=["player", "bye"], orient="row")
        assert bye_weeks_from_board(board) == {"Alpha": 5, "Bravo": 9}

    def test_missing_column_is_not_an_error(self):
        assert bye_weeks_from_board(pl.DataFrame({"player": ["Alpha"]})) == {}

    def test_null_byes_are_skipped(self):
        board = pl.DataFrame(
            [("Alpha", None), ("Bravo", 9)], schema=["player", "bye"], orient="row"
        )
        assert bye_weeks_from_board(board) == {"Bravo": 9}
