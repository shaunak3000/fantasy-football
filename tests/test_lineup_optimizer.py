from dataclasses import dataclass

import pytest

from fantasy_football.lineup.optimizer import Lineup, best_lineup_against, optimize, slots_for


class Settings:
    starting_slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1}
    team_count = 8


SETTINGS = Settings()


@dataclass
class Player:
    player: str
    position: str
    mean: float
    sd: float


def roster():
    return [
        Player("QB1", "QB", 20.0, 6.0),
        Player("QB2", "QB", 15.0, 5.0),
        Player("RB1", "RB", 18.0, 8.0),
        Player("RB2", "RB", 14.0, 6.0),
        Player("RB3", "RB", 10.0, 5.0),
        Player("WR1", "WR", 17.0, 9.0),
        Player("WR2", "WR", 13.0, 7.0),
        Player("WR3", "WR", 11.0, 6.0),
        Player("TE1", "TE", 9.0, 5.0),
        Player("TE2", "TE", 6.0, 3.0),
    ]


class TestSlots:
    def test_expands_counts_into_individual_slots(self):
        slots = slots_for(SETTINGS)
        assert len(slots) == 7
        assert sum(1 for s in slots if s.name == "RB") == 2

    def test_flex_carries_its_eligibility(self):
        flex = next(s for s in slots_for(SETTINGS) if s.name == "RB/WR/TE")
        assert set(flex.eligible) == {"RB", "WR", "TE"}


class TestOptimize:
    def test_fills_every_slot_exactly_once(self):
        lineup = optimize(roster(), SETTINGS)
        assert len(lineup.players) == 7
        assert len({p.player for p in lineup.players}) == 7

    def test_respects_slot_eligibility(self):
        lineup = optimize(roster(), SETTINGS)
        for slot, players in lineup.starters.items():
            for player in players:
                if slot == "RB/WR/TE":
                    assert player.position in {"RB", "WR", "TE"}
                else:
                    assert player.position == slot

    def test_starts_the_best_players_at_neutral_risk(self):
        lineup = optimize(roster(), SETTINGS)
        names = {p.player for p in lineup.players}
        assert "QB1" in names and "QB2" not in names
        assert "TE2" not in names

    def test_never_starts_a_player_twice(self):
        lineup = optimize(roster(), SETTINGS)
        assert len(lineup.players) == len(set(id(p) for p in lineup.players))

    def test_positive_risk_prefers_volatility(self):
        safe = optimize(roster(), SETTINGS, risk=-1.5)
        wild = optimize(roster(), SETTINGS, risk=1.5)
        assert wild.sd >= safe.sd

    def test_mean_and_sd_match_the_chosen_players(self):
        lineup = optimize(roster(), SETTINGS)
        assert lineup.mean == pytest.approx(sum(p.mean for p in lineup.players))
        assert lineup.sd == pytest.approx(sum(p.sd**2 for p in lineup.players) ** 0.5)

    def test_a_short_roster_still_produces_a_legal_lineup(self):
        lineup = optimize([Player("QB1", "QB", 20.0, 6.0)], SETTINGS)
        assert len(lineup.players) == 1

    def test_empty_roster_is_handled(self):
        assert optimize([], SETTINGS).players == []


class TestWinProbability:
    def test_even_matchup_is_a_coin_flip(self):
        lineup = Lineup(mean=100.0, sd=20.0)
        assert lineup.win_probability(100.0, 20.0) == pytest.approx(0.5, abs=1e-6)

    def test_favourite_wins_more_often(self):
        lineup = Lineup(mean=120.0, sd=20.0)
        assert lineup.win_probability(100.0, 20.0) > 0.5

    def test_zero_variance_is_deterministic(self):
        assert Lineup(mean=110.0, sd=0.0).win_probability(100.0, 0.0) == 1.0


class TestAgainstAnOpponent:
    def test_returns_a_legal_lineup_and_a_probability(self):
        lineup, probability = best_lineup_against(roster(), SETTINGS, 95.0, 20.0)
        assert len(lineup.players) == 7
        assert 0.0 <= probability <= 1.0

    def test_heavy_underdog_takes_more_risk_than_heavy_favourite(self):
        """The whole reason this is not just expected-points maximization."""
        underdog, _ = best_lineup_against(roster(), SETTINGS, 200.0, 15.0)
        favourite, _ = best_lineup_against(roster(), SETTINGS, 40.0, 15.0)
        assert underdog.sd >= favourite.sd

    def test_beats_or_matches_the_plain_expected_points_lineup(self):
        neutral = optimize(roster(), SETTINGS)
        chosen, probability = best_lineup_against(roster(), SETTINGS, 200.0, 15.0)
        assert probability >= neutral.win_probability(200.0, 15.0) - 1e-9
