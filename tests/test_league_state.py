from dataclasses import dataclass

import pytest

from fantasy_football.lineup.optimizer import optimize
from fantasy_football.season.league_state import RosterPlayer


class Settings:
    starting_slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1}
    team_count = 8


SETTINGS = Settings()


@dataclass
class Stub:
    position: str
    blended_rank: float
    consensus_rank: int


def player(name, position, mean, sd, bye=None, current_week=1):
    on_bye = bye is not None and bye == current_week
    return RosterPlayer(
        player=name,
        position=position,
        espn_id=abs(hash(name)) % 10000,
        mean=0.0 if on_bye else mean,
        sd=0.0 if on_bye else sd,
        bye_week=bye,
        on_bye=on_bye,
    )


class TestByeHandling:
    def test_a_player_on_bye_is_worth_nothing(self):
        benched = player("Bye Guy", "RB", 20.0, 8.0, bye=5, current_week=5)
        assert benched.on_bye
        assert benched.mean == 0.0 and benched.sd == 0.0

    def test_a_player_not_on_bye_keeps_his_projection(self):
        active = player("Active", "RB", 20.0, 8.0, bye=9, current_week=5)
        assert not active.on_bye
        assert active.mean == 20.0

    def test_optimizer_prefers_an_available_player_over_one_on_bye(self):
        """The bug this exists to prevent: starting a player who scores zero."""
        roster = [
            player("QB", "QB", 18.0, 6.0),
            player("Star RB", "RB", 22.0, 8.0, bye=5, current_week=5),
            player("Ok RB", "RB", 11.0, 6.0),
            player("Other RB", "RB", 10.0, 5.0),
            player("Third RB", "RB", 9.0, 5.0),
            player("WR1", "WR", 15.0, 7.0),
            player("WR2", "WR", 13.0, 6.0),
            player("WR3", "WR", 12.0, 6.0),
            player("TE", "TE", 8.0, 4.0),
        ]
        lineup = optimize(roster, SETTINGS)
        started = {p.player for p in lineup.players}
        assert "Star RB" not in started
        assert "Ok RB" in started

    def test_no_bye_information_leaves_everyone_available(self):
        assert not player("Unknown", "WR", 14.0, 7.0).on_bye

    def test_bye_week_is_retained_for_lookahead(self):
        ahead = player("Later", "WR", 14.0, 7.0, bye=9, current_week=5)
        assert ahead.bye_week == 9
        assert not ahead.on_bye

    def test_started_flag_reads_the_espn_slot(self):
        bench = RosterPlayer("X", "RB", 1, 10.0, 5.0, slot_id=20)
        starter = RosterPlayer("Y", "RB", 2, 10.0, 5.0, slot_id=2)
        assert not bench.started
        assert starter.started


class TestLineupIntegrity:
    def test_a_slot_with_only_a_bye_player_is_left_empty(self):
        """Starting a zero is worth exactly as much as starting nobody.

        The solver leaves the slot unfilled rather than rostering a player who
        cannot score, which is right on points and is a signal in its own
        right: an empty slot means go make a waiver claim.
        """
        roster = [
            player("QB", "QB", 18.0, 6.0, bye=5, current_week=5),
            player("RB1", "RB", 14.0, 6.0),
            player("RB2", "RB", 12.0, 6.0),
            player("WR1", "WR", 15.0, 7.0),
            player("WR2", "WR", 13.0, 6.0),
            player("WR3", "WR", 11.0, 6.0),
            player("TE", "TE", 8.0, 4.0),
        ]
        lineup = optimize(roster, SETTINGS)
        assert "QB" not in lineup.starters
        assert len(lineup.players) == 6
        assert lineup.mean == pytest.approx(sum(p.mean for p in lineup.players))

    def test_unfilled_slots_are_detectable_for_reporting(self):
        roster = [player("RB1", "RB", 14.0, 6.0)]
        lineup = optimize(roster, SETTINGS)
        filled = {slot for slot, players in lineup.starters.items() if players}
        assert "QB" not in filled and "TE" not in filled
