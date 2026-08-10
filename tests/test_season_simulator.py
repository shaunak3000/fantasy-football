import pytest

from fantasy_football.season.simulator import TeamSeason, simulate


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
