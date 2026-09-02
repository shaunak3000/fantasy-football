"""ADP is context, not authority — these tests pin down both halves of that."""

import json

from test_board_view import Settings, board, model, projection  # noqa: F401

from fantasy_football.data.espn import fetch_adp
from fantasy_football.draft.board_view import ADP_WAIT_CUSHION, BoardRow, board_view
from fantasy_football.draft.state import DraftState


class FakeRequest:
    """Stands in for espn_request, capturing the filter it was handed."""

    def __init__(self, players):
        self._players = players
        self.headers = None

    def league_get(self, params=None, headers=None):
        self.headers = headers
        return {"players": self._players}


class FakeLeague:
    def __init__(self, players):
        self.espn_request = FakeRequest(players)


def _player(player_id, adp):
    return {"player": {"id": player_id, "ownership": {"averageDraftPosition": adp}}}


class TestFetchAdp:
    def test_returns_ordinal_ranks_not_raw_pick_numbers(self):
        """The whole point: ESPN's raw ADP comes from 10- and 12-team drafts."""
        league = FakeLeague([_player(10, 45.6), _player(11, 3.2), _player(12, 120.9)])
        assert fetch_adp(league) == {11: 1, 10: 2, 12: 3}

    def test_undrafted_players_are_excluded_not_ranked_first(self):
        """ESPN reports 0.0 for players nobody drafts; sorting those in would
        put the least-wanted players at the top of the board."""
        league = FakeLeague([_player(1, 0.0), _player(2, 8.0), _player(3, None)])
        assert fetch_adp(league) == {2: 1}

    def test_players_without_an_id_are_skipped(self):
        league = FakeLeague([{"player": {"ownership": {"averageDraftPosition": 5.0}}}])
        assert fetch_adp(league) == {}

    def test_requests_ppr_draft_ranks(self):
        league = FakeLeague([])
        fetch_adp(league, limit=25)
        sent = json.loads(league.espn_request.headers["x-fantasy-filter"])
        assert sent["players"]["limit"] == 25
        assert sent["players"]["sortDraftRanks"]["value"] == "PPR"


def row(adp_rank=None, adp_slack=None):
    return BoardRow(
        player="P",
        position="WR",
        team="X",
        espn_id=1,
        overall_rank=1,
        mean=200.0,
        sd=50.0,
        vor=40.0,
        survival=0.5,
        fills_a_need=True,
        adp_rank=adp_rank,
        adp_slack=adp_slack,
    )


class TestCanWait:
    def test_needs_a_full_round_of_daylight(self):
        assert not row(adp_rank=30, adp_slack=ADP_WAIT_CUSHION - 1).can_wait
        assert row(adp_rank=30, adp_slack=ADP_WAIT_CUSHION).can_wait

    def test_absent_adp_never_claims_you_can_wait(self):
        assert not row().can_wait
        assert not row(adp_rank=30).can_wait

    def test_a_player_going_before_your_next_pick_cannot_wait(self):
        assert not row(adp_rank=5, adp_slack=-20).can_wait


class TestBoardViewAdp:
    def test_slack_is_measured_against_your_next_pick(self, board, model):  # noqa: F811
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        adp = {p.espn_id: p.consensus_overall_rank for p in board}
        rows = board_view(state, board, model, Settings(), adp=adp)
        next_pick = state.next_pick_for_me()
        for r in rows:
            if r.adp_rank is not None:
                assert r.adp_slack == r.adp_rank - next_pick

    def test_board_runs_without_adp(self, board, model):  # noqa: F811
        """A failed ADP fetch must degrade, never break the draft."""
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        rows = board_view(state, board, model, Settings(), adp=None)
        assert rows
        assert all(r.adp_rank is None and not r.can_wait for r in rows)

    def test_adp_does_not_change_the_recommendation(self, board, model):  # noqa: F811
        """The rehearsal killed six cleverer rules; ADP is untested and stays
        out of the decision entirely."""
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        without = board_view(state, board, model, Settings())
        # ADP that violently disagrees with consensus, to make any leakage show.
        hostile = {p.espn_id: 200 - p.consensus_overall_rank for p in board}
        with_adp = board_view(state, board, model, Settings(), adp=hostile)
        pick = lambda rows: next(r.player for r in rows if r.recommended)  # noqa: E731
        assert pick(without) == pick(with_adp)
