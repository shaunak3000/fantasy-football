import numpy as np
import pytest
from numpy.random import default_rng

from fantasy_football.season.simulator import (
    TeamSeason,
    _bracket_winners,
    paired_delta,
    simulate,
)


def league(strengths, sd=20.0):
    return [
        TeamSeason(team_id=i + 1, name=f"T{i + 1}", weekly_mean=mean, weekly_sd=sd)
        for i, mean in enumerate(strengths)
    ]


def round_robin(team_ids, weeks):
    """Each team plays the next one along, rotating each week."""
    schedule = {tid: [] for tid in team_ids}
    n = len(team_ids)
    for week in range(weeks):
        for index, tid in enumerate(team_ids):
            schedule[tid].append(team_ids[(index + week + 1) % n])
    return schedule


class TestSimulate:
    @pytest.fixture
    def outcome(self):
        teams = league([120, 115, 110, 105, 100, 95, 90, 85])
        ids = [t.team_id for t in teams]
        return simulate(teams, round_robin(ids, 13), playoff_teams=4, trials=400)

    def test_championship_probabilities_sum_to_one(self, outcome):
        assert sum(outcome.championship.values()) == pytest.approx(1.0, abs=1e-6)

    def test_playoff_berths_sum_to_the_bracket_size(self, outcome):
        assert sum(outcome.playoffs.values()) == pytest.approx(4.0, abs=0.05)

    def test_stronger_teams_win_more_titles(self, outcome):
        assert outcome.championship[1] > outcome.championship[8]

    def test_every_team_has_some_chance(self, outcome):
        assert all(p >= 0.0 for p in outcome.championship.values())

    def test_mean_wins_are_plausible(self, outcome):
        for wins in outcome.mean_wins.values():
            assert 0.0 <= wins <= 13.0

    def test_qualifying_is_easier_than_winning(self, outcome):
        """Four of eight make the playoffs, so a berth is far from a title."""
        for tid in outcome.championship:
            assert outcome.playoffs[tid] >= outcome.championship[tid]


class TestRiskMatters:
    def test_a_volatile_underdog_wins_more_titles_than_a_steady_one(self):
        """Variance is worth real title probability when you are behind.

        Two identical rosters by expected points, one volatile; the volatile one
        should win more championships in a bracket it is not favoured in.
        """
        steady = TeamSeason(team_id=1, name="steady", weekly_mean=95.0, weekly_sd=5.0)
        volatile = TeamSeason(team_id=2, name="volatile", weekly_mean=95.0, weekly_sd=35.0)
        strong = [
            TeamSeason(team_id=i, name=f"S{i}", weekly_mean=115.0, weekly_sd=15.0)
            for i in range(3, 9)
        ]
        teams = [steady, volatile, *strong]
        ids = [t.team_id for t in teams]
        outcome = simulate(teams, round_robin(ids, 13), playoff_teams=4, trials=1500, seed=7)
        assert outcome.championship[2] > outcome.championship[1]


class TestBankedResults:
    def test_existing_wins_carry_into_the_simulation(self):
        teams = league([100] * 8)
        teams[0].wins = 10
        ids = [t.team_id for t in teams]
        outcome = simulate(teams, round_robin(ids, 3), playoff_teams=4, trials=400)
        assert outcome.playoffs[1] > outcome.playoffs[2]

    def test_no_remaining_games_still_produces_a_champion(self):
        teams = league([100] * 4)
        schedule = {t.team_id: [] for t in teams}
        outcome = simulate(teams, schedule, playoff_teams=2, trials=50)
        assert sum(outcome.championship.values()) == pytest.approx(1.0, abs=1e-6)


class TestBracket:
    """The bracket was previously an unreadable zip with two break conditions.

    It happened to be right for four teams and silently dropped a team for odd
    sizes. These pin the pairing down directly rather than inferring it from
    championship totals.
    """

    def _pairing(self, strengths, trials=4000, seed=1):
        """Champion shares when seeds are separated by huge scoring gaps."""
        means = np.array(strengths, dtype=float)
        sds = np.full(len(strengths), 1e-3)
        qualifiers = np.tile(np.arange(len(strengths))[:, None], (1, trials))
        winners = _bracket_winners(
            qualifiers, np.tile(means[:, None], (1, trials)), sds, default_rng(seed)
        )
        return {i: float((winners == i).mean()) for i in range(len(strengths))}

    def test_the_top_seed_plays_the_bottom_seed(self):
        """1v4 and 2v3, so seed 2 reaches the final when seed 1 is unbeatable."""
        shares = self._pairing([1000.0, 500.0, 400.0, 1.0])
        assert shares[0] == pytest.approx(1.0)
        assert shares[1] == 0.0 and shares[2] == 0.0 and shares[3] == 0.0

    def test_pairing_is_one_v_four_and_not_one_v_two(self):
        """Three equal teams and one hopeless one separate the two bracketings.

        Under the correct 1v4 / 2v3 the top seed draws the walkover and reaches
        the final every time, taking half the titles while seeds 2 and 3 split
        the rest. Under a 1v2 / 3v4 bracket those roles swap and the *third*
        seed is the one handed the free pass. Champion totals alone cannot tell
        the two apart in most setups, which is why this one is shaped so they
        can.
        """
        shares = self._pairing([100.0, 100.0, 100.0, 1.0], trials=40000)
        assert shares[0] == pytest.approx(0.5, abs=0.02)
        assert shares[1] == pytest.approx(0.25, abs=0.02)
        assert shares[2] == pytest.approx(0.25, abs=0.02)
        assert shares[3] == pytest.approx(0.0, abs=0.01)

    def test_a_stronger_seed_wins_more_often(self):
        shares = self._pairing([120.0, 119.0, 118.0, 117.0], trials=20000)
        assert shares[0] > shares[3]
        assert sum(shares.values()) == pytest.approx(1.0)

    def test_an_odd_bracket_gives_the_top_seed_a_bye_and_drops_nobody(self):
        """Three qualifiers: the old loop eliminated the second seed silently."""
        shares = self._pairing([100.0, 100.0, 100.0], trials=20000)
        assert sum(shares.values()) == pytest.approx(1.0)
        # Seed 1 sits out the first round, so it wins more than the two who play.
        assert shares[0] > shares[1] > 0.0
        assert shares[2] > 0.0


class TestDeltasCarryTheirOwnError:
    def _league(self):
        teams = league([120, 118, 116, 114, 112, 110, 108, 106])
        return teams, round_robin([t.team_id for t in teams], 13)

    def test_an_identical_rerun_is_exactly_zero(self):
        """Common random numbers: no change must mean no measured change."""
        teams, schedule = self._league()
        before = simulate(teams, schedule, 4, trials=2000, seed=3)
        after = simulate(teams, schedule, 4, trials=2000, seed=3)
        assert paired_delta(before, after, 1) == (0.0, 0.0)

    def test_a_real_improvement_clears_its_own_error(self):
        teams, schedule = self._league()
        before = simulate(teams, schedule, 4, trials=20000, seed=3)
        teams[7].weekly_mean += 15.0
        after = simulate(teams, schedule, 4, trials=20000, seed=3)
        delta, stderr = paired_delta(before, after, 8)
        assert delta > stderr > 0.0

    def test_the_error_shrinks_as_trials_grow(self):
        teams, schedule = self._league()
        errors = []
        for trials in (500, 20000):
            before = simulate(teams, schedule, 4, trials=trials, seed=5)
            teams[7].weekly_mean += 2.0
            after = simulate(teams, schedule, 4, trials=trials, seed=5)
            teams[7].weekly_mean -= 2.0
            errors.append(paired_delta(before, after, 8)[1])
        assert errors[1] < errors[0]


class TestMeanUncertainty:
    def test_uncertain_strength_pulls_probabilities_toward_even(self):
        """The fix for the overconfidence `check_simulator` measured.

        Not knowing how good the teams really are must make a strong team's
        title less of a foregone conclusion.
        """
        ids = list(range(1, 9))
        schedule = round_robin(ids, 13)

        def odds(uncertainty):
            teams = [
                TeamSeason(
                    team_id=i + 1,
                    name=f"T{i + 1}",
                    weekly_mean=mean,
                    weekly_sd=20.0,
                    mean_uncertainty=uncertainty,
                )
                for i, mean in enumerate([140, 120, 118, 116, 114, 112, 110, 100])
            ]
            return simulate(teams, schedule, 4, trials=20000, seed=11).championship[1]

        assert odds(15.0) < odds(0.0)

    def test_it_is_off_by_default(self):
        team = TeamSeason(team_id=1, name="x", weekly_mean=100.0, weekly_sd=20.0)
        assert team.mean_uncertainty == 0.0
