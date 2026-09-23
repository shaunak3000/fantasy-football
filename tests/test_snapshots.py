"""The guards matter more than the writes.

ESPN serves only the current week's roster, so these files are the only record
of what anybody started. A bug that silently overwrites a good snapshot with a
stale one destroys data no rerun can recover, which makes `save` the most
safety-critical function in the repo.
"""

import json

import pytest

from fantasy_football.data import snapshots
from fantasy_football.data.snapshots import PlayerSnapshot, WeekSnapshot


@pytest.fixture(autouse=True)
def isolated_snapshots(tmp_path, monkeypatch):
    """Never touch the real captures while testing."""
    monkeypatch.setattr(
        snapshots, "snapshot_path", lambda *parts: _under(tmp_path, *parts), raising=True
    )
    return tmp_path


def _under(root, *parts):
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def player(espn_id, slot_id, actual=None, name=None):
    return PlayerSnapshot(
        espn_id=espn_id,
        name=name or f"P{espn_id}",
        position="RB",
        slot_id=slot_id,
        injury_status="ACTIVE",
        projected=10.0,
        actual=actual,
    )


def snapshot(week, nfl_week, players, season=2026, results_week="same"):
    return WeekSnapshot(
        season=season,
        week=week,
        captured_at=f"2026-09-0{week}T12:00:00+00:00",
        nfl_week_at_capture=nfl_week,
        results_week=nfl_week if results_week == "same" else results_week,
        schema=1,
        teams={1: players},
        records={1: (0, 0, 0.0)},
        names={1: "Team 1"},
    )


class TestItRefusesToDestroyData:
    def test_a_stale_capture_never_lands(self):
        """Running the report for week 3 in week 10 must not overwrite week 3.

        ESPN would hand back the week-10 roster, and writing it would replace
        the only copy of the real week-3 lineups with a worthless one.
        """
        path, _ = snapshots.save(snapshot(3, 3, [player(1, 2), player(2, 20)]))
        assert path is not None

        stale = snapshot(3, 10, [player(1, 20), player(2, 2)])
        path, reason = snapshots.save(stale)
        assert path is None
        assert "refused" in reason

        kept = snapshots.load(2026, 3)
        assert kept.teams[1][0].slot_id == 2, "the original lineup must survive"

    def test_a_results_free_capture_does_not_replace_one_with_results(self):
        snapshots.save(snapshot(4, 4, [player(1, 2, actual=21.0)]))
        path, reason = snapshots.save(snapshot(4, 4, [player(1, 2, actual=None)]))
        assert path is None and "refused" in reason
        assert snapshots.load(2026, 4).teams[1][0].actual == 21.0

    def test_force_overrides_the_guard(self):
        snapshots.save(snapshot(5, 5, [player(1, 2)]))
        path, _ = snapshots.save(snapshot(5, 99, [player(1, 20)]), force=True)
        assert path is not None
        assert snapshots.load(2026, 5).teams[1][0].slot_id == 20


class TestResultsArriveLate:
    def test_a_later_capture_fills_in_scores_without_moving_anyone(self):
        """Sunday's capture has the lineups and half the scores; Tuesday's has
        all the scores and the wrong lineups. Take one field from each."""
        snapshots.save(snapshot(6, 6, [player(1, 2, actual=None), player(2, 20, actual=None)]))

        # Two weeks later ESPN reports a different roster, but real stat lines.
        late = snapshot(6, 8, [player(1, 20, actual=14.0), player(2, 2, actual=3.0)])
        path, reason = snapshots.save(late)
        assert path is not None, reason

        merged = snapshots.load(2026, 6)
        by_id = {p.espn_id: p for p in merged.teams[1]}
        assert by_id[1].actual == 14.0 and by_id[2].actual == 3.0
        # ...and the lineup is still the one recorded while the week was live.
        assert by_id[1].slot_id == 2 and by_id[2].slot_id == 20

    def test_a_stale_capture_with_no_scores_is_still_refused(self):
        snapshots.save(snapshot(7, 7, [player(1, 2)]))
        path, reason = snapshots.save(snapshot(7, 9, [player(1, 20)]))
        assert path is None and "refused" in reason


class TestKnowingWhatIsStillMissing:
    """A week captured live has lineups and no scores, and something has to come
    back for them. Deciding when to stop coming back is the whole difficulty."""

    def test_scores_pulled_while_the_week_was_live_are_not_final(self):
        partial = snapshot(3, 3, [player(1, 2, actual=8.0)], results_week=3)
        assert partial.has_results
        assert not partial.results_final

    def test_scores_pulled_after_the_week_ended_are_final(self):
        assert snapshot(3, 3, [player(1, 2, actual=8.0)], results_week=4).results_final

    def test_a_started_player_who_never_played_does_not_hold_a_week_open(self):
        """ESPN publishes no stat line at all for someone started while inactive
        — three of them in week 2 of 2026. Waiting for those waits forever."""
        never = snapshot(3, 3, [player(1, 2, actual=None)], results_week=5)
        assert never.results_final

    def test_a_legacy_file_with_no_recorded_pull_gets_one_refetch(self):
        assert not snapshot(3, 3, [player(1, 2, actual=8.0)], results_week=None).results_final

    def test_it_lists_finished_weeks_whose_scores_are_not_yet_final(self):
        snapshots.save(snapshot(1, 1, [player(1, 2, actual=5.0)], results_week=2))
        snapshots.save(snapshot(2, 2, [player(1, 2, actual=None)], results_week=2))
        snapshots.save(snapshot(3, 3, [player(1, 2, actual=None)], results_week=3))
        assert snapshots.weeks_awaiting_results(2026, before_week=3) == [2]

    def test_the_live_week_is_never_listed(self):
        snapshots.save(snapshot(4, 4, [player(1, 2, actual=None)], results_week=4))
        assert snapshots.weeks_awaiting_results(2026, before_week=4) == []

    def test_a_refetched_week_stops_being_listed(self):
        snapshots.save(snapshot(2, 2, [player(1, 2, actual=None)], results_week=2))
        assert snapshots.weeks_awaiting_results(2026, before_week=5) == [2]
        snapshots.save(snapshot(2, 5, [player(1, 2, actual=11.0)]))
        assert snapshots.weeks_awaiting_results(2026, before_week=5) == []

    def test_nothing_captured_means_nothing_to_collect(self):
        assert snapshots.weeks_awaiting_results(2026, before_week=5) == []


class TestRoundTrip:
    def test_what_is_written_is_what_is_read(self):
        original = snapshot(2, 2, [player(1, 4, actual=12.5), player(2, 20)])
        snapshots.save(original)
        loaded = snapshots.load(2026, 2)
        assert loaded.week == 2
        assert loaded.trustworthy and loaded.has_results
        assert [p.slot_id for p in loaded.teams[1]] == [4, 20]
        assert loaded.names[1] == "Team 1"

    def test_started_reads_the_slot(self):
        assert player(1, 4).started
        for bench in (20, 21, 24):
            assert not player(1, bench).started
        assert not player(1, None).started

    def test_a_missing_week_is_none_not_an_error(self):
        assert snapshots.load(2026, 13) is None

    def test_load_season_returns_weeks_in_order(self):
        for week in (3, 1, 2):
            snapshots.save(snapshot(week, week, [player(1, 2)]))
        assert [s.week for s in snapshots.load_season(2026)] == [1, 2, 3]

    def test_an_empty_season_is_an_empty_list(self):
        assert snapshots.load_season(2019) == []

    def test_the_file_is_stable_json(self):
        """Committed to git, so it must diff cleanly week to week."""
        path, _ = snapshots.save(snapshot(8, 8, [player(2, 20), player(1, 2)]))
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert json.loads(text)["week"] == 8
        assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"


class TestCaptureNeverBreaksTheReport:
    def test_a_failure_is_reported_not_raised(self):
        class Broken:
            year = 2026
            nfl_week = 1

            @property
            def espn_request(self):
                raise RuntimeError("ESPN is down")

        assert "snapshot failed" in snapshots.capture_and_save(Broken(), 1, 2026)
