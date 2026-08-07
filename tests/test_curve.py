import numpy as np
import polars as pl
import pytest

from fantasy_football.projections.curve import (
    RankCurve,
    fit_all,
    fit_rank_curve,
    replacement_points,
    replacement_ranks,
)


def training(rows):
    return pl.DataFrame(rows, schema=["pos", "pos_rank", "actual_points", "season"], orient="row")


def synthetic(position="RB", n_ranks=40, seasons=(2021, 2022, 2023, 2024, 2025), noise=0.0):
    """Points decay with rank, so a correct fit must recover a decreasing curve."""
    rng = np.random.default_rng(0)
    rows = []
    for season in seasons:
        for rank in range(1, n_ranks + 1):
            base = 300.0 - 4.0 * rank
            rows.append((position, rank, base + rng.normal(0, noise), season))
    return training(rows)


class TestFit:
    def test_expected_points_decrease_with_rank(self):
        curve = fit_rank_curve(synthetic(), "RB")
        assert curve is not None
        assert curve.points_at(1) > curve.points_at(10) > curve.points_at(30)

    def test_curve_is_monotone_everywhere(self):
        curve = fit_rank_curve(synthetic(noise=60.0), "RB")
        assert np.all(np.diff(curve.expected) <= 1e-9)

    def test_records_provenance(self):
        curve = fit_rank_curve(synthetic(), "RB")
        assert curve.n_observations == 200
        assert curve.seasons == (2021, 2022, 2023, 2024, 2025)

    def test_too_little_data_returns_none(self):
        assert fit_rank_curve(training([("RB", 1, 200.0, 2025)]), "RB") is None

    def test_unknown_position_returns_none(self):
        assert fit_rank_curve(synthetic(), "QB") is None

    def test_fit_all_covers_every_position_present(self):
        frame = pl.concat([synthetic("RB"), synthetic("WR")])
        curves = fit_all(frame)
        assert set(curves) == {"RB", "WR"}


class TestLookup:
    @pytest.fixture
    def curve(self):
        return RankCurve(
            position="RB",
            ranks=np.array([1, 2, 3, 4]),
            expected=np.array([200.0, 180.0, 160.0, 140.0]),
            spread=np.array([50.0, 45.0, 40.0, 35.0]),
        )

    def test_interpolates_fractional_ranks(self, curve):
        assert curve.points_at(1.5) == pytest.approx(190.0)
        assert curve.spread_at(2.5) == pytest.approx(42.5)

    def test_clamps_beyond_the_fitted_range(self, curve):
        assert curve.points_at(0) == pytest.approx(200.0)
        assert curve.points_at(99) == pytest.approx(140.0)

    def test_exact_ranks_are_exact(self, curve):
        assert curve.points_at(3) == pytest.approx(160.0)


class FakeSettings:
    def __init__(self, slots, team_count=8):
        self.starting_slots = slots
        self.team_count = team_count


class TestReplacement:
    @pytest.fixture
    def curves(self):
        return fit_all(pl.concat([synthetic("RB"), synthetic("WR"), synthetic("TE")]))

    def test_dedicated_slots_multiply_by_league_size(self, curves):
        settings = FakeSettings({"RB": 2, "WR": 2, "TE": 1}, team_count=8)
        ranks = replacement_ranks(settings, curves)
        assert ranks["RB"] == 16
        assert ranks["WR"] == 16
        assert ranks["TE"] == 8

    def test_dst_is_not_treated_as_a_flex_despite_the_slash(self, curves):
        settings = FakeSettings({"RB": 2, "WR": 2, "D/ST": 1}, team_count=8)
        ranks = replacement_ranks(settings, curves)
        assert ranks["RB"] == 16
        assert ranks["WR"] == 16

    def test_flex_slots_are_allocated_on_top_of_dedicated_demand(self, curves):
        settings = FakeSettings({"RB": 2, "WR": 2, "RB/WR/TE": 1}, team_count=8)
        ranks = replacement_ranks(settings, curves)
        assert ranks["RB"] + ranks["WR"] + ranks["TE"] == 16 + 16 + 8

    def test_replacement_points_fall_as_league_size_grows(self, curves):
        small = replacement_points(FakeSettings({"RB": 2}, team_count=8), curves)
        large = replacement_points(FakeSettings({"RB": 2}, team_count=14), curves)
        assert large["RB"] < small["RB"]
