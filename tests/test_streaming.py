"""Defence is the one position where a weekly matchup swap is worth a roster move.

The threshold is the point of this module. Suggesting a swap for half a point
would spend a transaction on noise; suppressing one worth a point and a half
gives back the only edge the position has. Both failures are silent, so they are
pinned here.
"""

from dataclasses import dataclass

from fantasy_football.transactions.streaming import DEFAULT_THRESHOLD, find_stream


@dataclass(frozen=True)
class Projection:
    """Stands in for `espn.WeeklyProjection`, which is all `find_stream` reads."""

    espn_id: int
    name: str
    position: str
    points: float
    injury_status: str | None = "NORMAL"

    @property
    def unavailable(self) -> bool:
        return (self.injury_status or "").upper() in {"OUT", "INJURY_RESERVE", "DOUBTFUL"}


def pool(*projections):
    return {p.espn_id: p for p in projections}


EAGLES = Projection(1, "Eagles D/ST", "D/ST", 8.2)


class TestTheThreshold:
    def test_a_clear_upgrade_is_reported(self):
        """Week 3 of 2026: Lions 9.5 free against our Eagles 8.2."""
        lions = Projection(2, "Lions D/ST", "D/ST", 9.5)
        advice = find_stream(pool(EAGLES, lions), my_ids={1}, rostered_ids={1})
        assert advice is not None
        assert advice.candidate == "Lions D/ST"
        assert round(advice.gain, 1) == 1.3

    def test_half_a_point_is_not_worth_a_roster_move(self):
        """Week 2 of 2026: Buccaneers 9.8 against Eagles 9.3. Correctly skipped."""
        eagles = Projection(1, "Eagles D/ST", "D/ST", 9.3)
        bucs = Projection(2, "Buccaneers D/ST", "D/ST", 9.8)
        assert find_stream(pool(eagles, bucs), my_ids={1}, rostered_ids={1}) is None

    def test_the_threshold_is_configurable(self):
        bucs = Projection(2, "Buccaneers D/ST", "D/ST", 9.8)
        eagles = Projection(1, "Eagles D/ST", "D/ST", 9.3)
        assert find_stream(pool(eagles, bucs), my_ids={1}, rostered_ids={1}, threshold=0.4)

    def test_a_worse_free_agent_is_never_suggested(self):
        worse = Projection(2, "Broncos D/ST", "D/ST", 6.5)
        assert find_stream(pool(EAGLES, worse), my_ids={1}, rostered_ids={1}) is None


class TestItOnlyOffersWhatCanActuallyBeStarted:
    def test_a_defence_on_someone_elses_roster_is_not_available(self):
        owned = Projection(2, "49ers D/ST", "D/ST", 10.0)
        assert find_stream(pool(EAGLES, owned), my_ids={1}, rostered_ids={1, 2}) is None

    def test_a_defence_on_bye_projects_zero_and_is_skipped(self):
        bye = Projection(2, "Lions D/ST", "D/ST", 0.0)
        assert find_stream(pool(EAGLES, bye), my_ids={1}, rostered_ids={1}) is None

    def test_a_ruled_out_candidate_is_skipped(self):
        out = Projection(2, "Lions D/ST", "D/ST", 9.5, injury_status="OUT")
        assert find_stream(pool(EAGLES, out), my_ids={1}, rostered_ids={1}) is None

    def test_our_own_defence_on_bye_does_not_count_as_what_we_hold(self):
        """A zero is not a baseline to beat — with nothing startable there is
        nothing to compare, and the waiver section owns that case."""
        onbye = Projection(1, "Eagles D/ST", "D/ST", 0.0)
        lions = Projection(2, "Lions D/ST", "D/ST", 9.5)
        assert find_stream(pool(onbye, lions), my_ids={1}, rostered_ids={1}) is None

    def test_other_positions_are_ignored(self):
        kicker = Projection(2, "Harrison Mevis", "K", 20.0)
        assert find_stream(pool(EAGLES, kicker), my_ids={1}, rostered_ids={1}) is None

    def test_it_can_be_pointed_at_another_one_slot_position(self):
        mine = Projection(1, "Eddy Pineiro", "K", 10.4)
        theirs = Projection(2, "Harrison Mevis", "K", 11.3)
        advice = find_stream(
            pool(mine, theirs), my_ids={1}, rostered_ids={1}, position="K", threshold=0.5
        )
        assert advice is not None and advice.position == "K"


class TestWhatItReports:
    def test_runners_up_are_listed_in_order(self):
        advice = find_stream(
            pool(
                EAGLES,
                Projection(2, "Lions D/ST", "D/ST", 9.5),
                Projection(3, "Giants D/ST", "D/ST", 9.2),
                Projection(4, "Bengals D/ST", "D/ST", 9.1),
            ),
            my_ids={1},
            rostered_ids={1},
        )
        assert [name for name, _ in advice.runners_up] == ["Giants D/ST", "Bengals D/ST"]

    def test_the_summary_names_both_sides_and_the_gain(self):
        advice = find_stream(
            pool(EAGLES, Projection(2, "Lions D/ST", "D/ST", 9.5)), my_ids={1}, rostered_ids={1}
        )
        assert "Lions D/ST" in advice.summary() and "Eagles D/ST" in advice.summary()
        assert "+1.3" in advice.summary()

    def test_an_empty_wire_is_not_an_error(self):
        assert find_stream(pool(EAGLES), my_ids={1}, rostered_ids={1}) is None

    def test_the_threshold_default_is_documented(self):
        assert DEFAULT_THRESHOLD == 1.0
