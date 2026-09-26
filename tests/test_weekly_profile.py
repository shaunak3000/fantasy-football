"""A trade is judged week by week, so byes have to fall in the right weeks.

The case this exists for: week 6 of 2026 took Amon-Ra St. Brown, Sam LaPorta and
Joe Burrow at once, and a trade worth +3.93% on the season cost 3.7 points that
week. Nothing in the old search could see a week.
"""

from dataclasses import dataclass

import pytest

from fantasy_football.season.weekly_profile import (
    COVER,
    DEPTH,
    STARTER,
    available,
    usage,
    weekly_lineups,
    weekly_points,
)


class Settings:
    starting_slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1}


SETTINGS = Settings()
WEEKS = [4, 5, 6, 7]


@dataclass
class P:
    player: str
    position: str
    mean: float
    sd: float = 1.0
    bye_week: int | None = None
    unavailable: bool = False
    on_bye: bool = False
    return_week: int | None = None


def roster():
    return [
        P("qb", "QB", 20.0, bye_week=9),
        P("rb1", "RB", 15.0, bye_week=9),
        P("rb2", "RB", 13.0, bye_week=9),
        P("wr1", "WR", 20.0, bye_week=6),  # the St. Brown of this roster
        P("wr2", "WR", 17.0, bye_week=9),
        P("wr3", "WR", 15.0, bye_week=9),
        P("te1", "TE", 11.0, bye_week=6),  # the LaPorta
        P("te2", "TE", 9.0, bye_week=9),
    ]


class TestByesLandInTheRightWeek:
    def test_a_bye_removes_a_player_that_week_only(self):
        wr1 = P("wr1", "WR", 20.0, bye_week=6)
        assert not available(wr1, 6, current_week=4)
        assert available(wr1, 5, current_week=4)
        assert available(wr1, 7, current_week=4)

    def test_ruled_out_this_week_is_out_this_week_only(self):
        """IR in September is usually back by December — the repo-wide assumption."""
        hurt = P("hurt", "WR", 15.0, unavailable=True)
        assert not available(hurt, 4, current_week=4)
        assert available(hurt, 5, current_week=4)

    def test_the_bye_week_costs_exactly_what_it_should(self):
        points = dict(zip(WEEKS, weekly_points(roster(), WEEKS, 4, SETTINGS), strict=True))
        full = 20 + 15 + 13 + 20 + 17 + 11 + 15  # wr3 in the flex
        assert points[4] == points[5] == points[7] == pytest.approx(full)
        # wk6: wr1 and te1 out -> wr3 moves up, te2 in, flex falls to nobody better
        assert points[6] == pytest.approx(20 + 15 + 13 + 17 + 15 + 9)

    def test_a_roster_with_no_byes_in_range_is_flat(self):
        flat = [P("qb", "QB", 20.0), P("rb", "RB", 10.0)]
        assert len(set(weekly_points(flat, WEEKS, 4, SETTINGS))) == 1


class TestWhoActuallyStarts:
    def test_a_full_strength_starter_is_a_starter(self):
        r = roster()
        u = usage(r[0], r, WEEKS, 4, SETTINGS)
        assert u.role == STARTER and u.weeks_started == tuple(WEEKS)

    def test_a_player_who_only_plays_the_bye_week_is_cover(self):
        """The distinction you asked for: te2 plays week 6 and nothing else."""
        r = roster()
        te2 = r[-1]
        u = usage(te2, r, WEEKS, 4, SETTINGS)
        assert u.role == COVER
        assert u.weeks_started == (6,)
        assert "cover only" in u.summary() and "wk 6" in u.summary()

    def test_a_player_who_never_plays_is_depth(self):
        r = [*roster(), P("rb3", "RB", 12.0, bye_week=9), P("wr9", "WR", 2.0)]
        assert usage(r[-1], r, WEEKS, 4, SETTINGS).role == DEPTH

    def test_in_a_heavy_bye_week_even_deep_depth_plays(self):
        """With wr1 and te1 both out in week 6 and no spare RB, a 2-point receiver
        is the only flex-eligible body left. That is what a bye crunch looks like,
        and it is why the week shows up in trade values at all."""
        r = [*roster(), P("wr9", "WR", 2.0)]
        u = usage(r[-1], r, WEEKS, 4, SETTINGS)
        assert u.role == COVER and u.weeks_started == (6,)

    def test_a_starter_who_sits_his_own_bye_is_still_a_starter(self):
        r = roster()
        u = usage(r[3], r, WEEKS, 4, SETTINGS)  # wr1, bye in 6
        assert u.role == STARTER and 6 not in u.weeks_started
        assert "3 of 4" in u.summary() and "sits 1" in u.summary()


class TestTheMemoIsInvisible:
    def test_cached_weeks_match_a_fresh_computation(self):
        r = roster()
        together = weekly_lineups(r, WEEKS, 4, SETTINGS)
        one_at_a_time = [weekly_lineups(r, [w], 4, SETTINGS)[0] for w in WEEKS]
        assert [x.points for x in together] == [x.points for x in one_at_a_time]
        assert [x.week for x in together] == WEEKS


class TestTheFastSeasonTotal:
    """`season_total` is what the trade screen ranks on, 300,000 times a search.
    It has to equal the week-by-week sum exactly, or the screen ranks on a guess."""

    def test_it_equals_the_weekly_sum(self):
        from fantasy_football.season.weekly_profile import season_total

        r = roster()
        assert season_total(r, WEEKS, 4, SETTINGS) == pytest.approx(
            sum(weekly_points(r, WEEKS, 4, SETTINGS))
        )

    @pytest.mark.parametrize("seed", range(25))
    def test_it_equals_the_weekly_sum_on_random_rosters(self, seed):
        import random

        from fantasy_football.season.weekly_profile import season_total

        rng = random.Random(seed)
        weeks = list(range(3, 15))
        players = [
            P(
                f"p{i}",
                rng.choice(["QB", "RB", "WR", "TE"]),
                round(rng.uniform(0, 22), 2),
                bye_week=rng.choice([5, 6, 7, 8, 9, 10, 11, 12, 14, None]),
                unavailable=rng.random() < 0.1,
                return_week=rng.choice([None, None, None, 5, 8, 12, 20]),
            )
            for i in range(16)
        ]
        assert season_total(players, weeks, 3, SETTINGS) == pytest.approx(
            sum(weekly_points(players, weeks, 3, SETTINGS)), abs=1e-6
        )


class TestLongAbsences:
    """A.J. Brown was out until week 8 and the repo thought week 4. A return week
    has to remove a player from every week before it, and from none after."""

    def test_out_until_week_8_misses_every_week_before_it(self):
        brown = P("brown", "WR", 16.0, return_week=8)
        weeks = list(range(3, 11))
        missing = [w for w in weeks if not available(brown, w, current_week=3)]
        assert missing == [3, 4, 5, 6, 7]

    def test_the_backfill_receiver_is_cover_until_the_starter_returns(self):
        """The Metcalf case. With Brown out until week 8, the fourth receiver
        plays weeks 3-7 and then goes back to the bench — cover, not depth."""
        weeks = list(range(3, 11))
        r = [
            P("qb", "QB", 20.0),
            P("rb1", "RB", 16.0),
            P("rb2", "RB", 15.0),
            P("wr1", "WR", 20.0),
            P("wr2", "WR", 18.0),
            P("brown", "WR", 16.0, return_week=8),
            P("te", "TE", 9.0),
            P("metcalf", "WR", 12.0),
        ]
        u = usage(r[-1], r, weeks, 3, SETTINGS)
        assert u.role == COVER
        assert u.weeks_started == (3, 4, 5, 6, 7)

    def test_a_long_absence_matches_the_weekly_sum(self):
        from fantasy_football.season.weekly_profile import season_total

        weeks = list(range(3, 15))
        r = [*roster(), P("brown", "WR", 25.0, return_week=8, bye_week=11)]
        assert season_total(r, weeks, 3, SETTINGS) == pytest.approx(
            sum(weekly_points(r, weeks, 3, SETTINGS))
        )
