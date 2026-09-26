"""The fast lineup valuation has to agree with the solver, or it is just a guess.

`optimize` is the authority — it guarantees legality by construction. `best_points`
exists only because screening trades needs a lineup value roughly a thousand times
cheaper. Its whole licence to exist is that the two answers are identical for this
league's slots, so that is tested against randomised rosters rather than asserted.
"""

import random
from dataclasses import dataclass

import pytest

from fantasy_football.lineup.fast import best_points, is_greedy_exact, select
from fantasy_football.lineup.optimizer import optimize


class Settings:
    """This league: one flex over RB/WR/TE, so greedy is provably exact."""

    starting_slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1, "D/ST": 1, "K": 1}


class TwoFlexSettings:
    starting_slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1, "WR/TE": 1}


SETTINGS = Settings()


@dataclass
class P:
    player: str
    position: str
    mean: float
    sd: float = 0.0


def random_roster(rng, size=16):
    positions = ["QB", "RB", "WR", "TE", "D/ST", "K"]
    return [
        P(f"p{i}", rng.choice(positions), round(rng.uniform(0.0, 25.0), 2)) for i in range(size)
    ]


class TestItMatchesTheSolver:
    @pytest.mark.parametrize("seed", range(40))
    def test_random_rosters_agree(self, seed):
        rng = random.Random(seed)
        players = random_roster(rng)
        assert best_points(players, SETTINGS) == pytest.approx(
            optimize(players, SETTINGS).mean, abs=1e-6
        )

    @pytest.mark.parametrize("seed", range(15))
    def test_it_agrees_on_thin_rosters_too(self, seed):
        """Byes and injuries leave positions empty; the slot then goes unfilled."""
        rng = random.Random(1000 + seed)
        players = random_roster(rng, size=rng.randint(1, 8))
        assert best_points(players, SETTINGS) == pytest.approx(
            optimize(players, SETTINGS).mean, abs=1e-6
        )

    def test_an_empty_roster_scores_nothing(self):
        assert best_points([], SETTINGS) == 0.0


class TestTheFlexIsFilledCorrectly:
    def test_the_flex_takes_the_best_leftover(self):
        players = [
            P("qb", "QB", 20.0),
            P("rb1", "RB", 15.0),
            P("rb2", "RB", 14.0),
            P("wr1", "WR", 13.0),
            P("wr2", "WR", 12.0),
            P("te", "TE", 8.0),
            P("rb3", "RB", 11.0),  # best remaining flex-eligible
            P("wr3", "WR", 10.0),
        ]
        assert best_points(players, SETTINGS) == pytest.approx(20 + 15 + 14 + 13 + 12 + 8 + 11)

    def test_a_surplus_receiver_who_cannot_start_adds_nothing(self):
        base = [
            P("qb", "QB", 20.0),
            P("rb1", "RB", 15.0),
            P("rb2", "RB", 14.0),
            P("wr1", "WR", 18.0),
            P("wr2", "WR", 17.0),
            P("te", "TE", 8.0),
            P("wr3", "WR", 16.0),
        ]
        before = best_points(base, SETTINGS)
        after = best_points([*base, P("wr4", "WR", 15.0)], SETTINGS)
        assert after == before

    def test_a_position_with_no_slot_never_starts(self):
        players = [P("qb", "QB", 20.0), P("punter", "P", 99.0)]
        assert best_points(players, SETTINGS) == pytest.approx(20.0)


class TestItKnowsWhenItIsAllowedToBeUsed:
    def test_one_flex_qualifies(self):
        assert is_greedy_exact(SETTINGS)

    def test_two_flex_slots_do_not(self):
        """With overlapping flex slots the assignment interacts and greedy would
        have to guess, so callers must fall back to the solver."""
        assert not is_greedy_exact(TwoFlexSettings())

    def test_no_flex_at_all_qualifies(self):
        class NoFlex:
            starting_slots = {"QB": 1, "RB": 2, "WR": 2}

        assert is_greedy_exact(NoFlex())


class TestTheStartersItReturnsAreLegal:
    """`select` is now load-bearing: "does the incoming player start, and in
    which weeks" is read straight off it. So the lineup has to be a real one."""

    @pytest.mark.parametrize("seed", range(30))
    def test_every_slot_is_respected(self, seed):
        rng = random.Random(500 + seed)
        players = random_roster(rng)
        starters = select(players, SETTINGS)
        counts = {}
        for player in starters:
            counts[player.position] = counts.get(player.position, 0) + 1
        for position, cap in (("QB", 1), ("D/ST", 1), ("K", 1)):
            assert counts.get(position, 0) <= cap
        # RB/WR/TE share one flex, so each is at most its dedicated count plus one,
        # and the three together at most five.
        assert counts.get("RB", 0) <= 3 and counts.get("WR", 0) <= 3 and counts.get("TE", 0) <= 2
        assert counts.get("RB", 0) + counts.get("WR", 0) + counts.get("TE", 0) <= 6
        assert len(starters) <= 9
        assert len({id(p) for p in starters}) == len(starters)

    @pytest.mark.parametrize("seed", range(30))
    def test_it_picks_the_same_total_as_the_solver(self, seed):
        rng = random.Random(900 + seed)
        players = random_roster(rng)
        assert sum(p.mean for p in select(players, SETTINGS)) == pytest.approx(
            optimize(players, SETTINGS).mean, abs=1e-6
        )
