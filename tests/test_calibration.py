import numpy as np
import polars as pl
import pytest

from fantasy_football.projections.calibration import NOMINAL_COVERAGES, backtest
from fantasy_football.projections.curve import QUANTILE_LEVELS, fit_rank_curve


def synthetic(seasons=(2021, 2022, 2023, 2024, 2025), n_ranks=40, noise=40.0, seed=0):
    """Points decay with rank plus real noise, so intervals have something to cover."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for rank in range(1, n_ranks + 1):
            points = max(0.0, 300.0 - 5.0 * rank + rng.normal(0, noise))
            rows.append(("RB", rank, points, season))
    return pl.DataFrame(rows, schema=["pos", "pos_rank", "actual_points", "season"], orient="row")


class TestQuantiles:
    @pytest.fixture
    def curve(self):
        return fit_rank_curve(synthetic(), "RB")

    def test_quantiles_are_ordered_within_a_rank(self, curve):
        values = [curve.quantile_at(10, level) for level in QUANTILE_LEVELS]
        assert values == sorted(values)

    def test_quantiles_decrease_with_worse_rank(self, curve):
        assert curve.quantile_at(1, 0.5) > curve.quantile_at(35, 0.5)

    def test_interval_brackets_the_median(self, curve):
        low, high = curve.interval_at(10, 0.80)
        assert low < curve.quantile_at(10, 0.5) < high

    def test_wider_coverage_gives_a_wider_interval(self, curve):
        narrow = curve.interval_at(10, 0.50)
        wide = curve.interval_at(10, 0.90)
        assert wide[0] <= narrow[0] and wide[1] >= narrow[1]

    def test_unknown_quantile_level_is_rejected(self, curve):
        with pytest.raises(ValueError, match="level must be one of"):
            curve.quantile_at(10, 0.42)

    def test_probability_below_is_monotone_in_points(self, curve):
        low = curve.probability_below(10, 50.0)
        high = curve.probability_below(10, 400.0)
        assert 0.0 <= low <= high <= 1.0

    def test_curve_without_quantiles_raises(self):
        from fantasy_football.projections.curve import RankCurve

        bare = RankCurve(
            position="RB",
            ranks=np.array([1, 2]),
            expected=np.array([100.0, 90.0]),
            spread=np.array([10.0, 10.0]),
        )
        with pytest.raises(ValueError, match="no quantiles"):
            bare.quantile_at(1, 0.5)


class TestBacktest:
    @pytest.fixture
    def report(self):
        return backtest(synthetic(), positions=("RB",))

    def test_holds_out_every_season(self, report):
        assert [s.season for s in report.seasons] == [2021, 2022, 2023, 2024, 2025]

    def test_beats_the_naive_baseline(self, report):
        assert report.mean_skill > 0
        for season in report.seasons:
            assert season.mae < season.baseline_mae

    def test_ordering_is_recovered(self, report):
        assert report.mean_spearman > 0.5

    def test_coverage_is_near_nominal_on_well_behaved_data(self, report):
        for level, actual in report.overall_coverage.items():
            assert abs(actual - level) < 0.12, f"{level} coverage was {actual}"

    def test_pit_values_stay_in_range(self, report):
        assert report.pit_values.min() >= 0.0
        assert report.pit_values.max() <= 1.0

    def test_histogram_normalizes(self, report):
        freqs, edges = report.pit_histogram(bins=10)
        assert len(freqs) == 10
        assert freqs.sum() == pytest.approx(1.0)

    def test_reports_every_nominal_level(self, report):
        for season in report.seasons:
            assert set(season.coverage) == set(NOMINAL_COVERAGES)


def test_single_season_cannot_be_backtested():
    assert backtest(synthetic(seasons=(2025,)), positions=("RB",)).seasons == []
