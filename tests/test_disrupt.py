"""The adversarial plays, and the gates that keep them from costing anything."""

from fantasy_football.draft.disrupt import (
    DEFAULT_FIRST_ROUND,
    TIE_SLIDE,
    defer_for_adp,
    disruptive_pick,
    handcuff,
    pair_at_the_turn,
)
from fantasy_football.draft.state import DraftState
from fantasy_football.projections.ensemble import Projection


class Settings:
    starting_slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "D/ST": 1, "K": 1, "RB/WR/TE": 1}
    team_count = 8


def player(espn_id, name, position, team, overall, mean=200.0, replacement=150.0):
    return Projection(
        player=name,
        position=position,
        team=team,
        espn_id=espn_id,
        consensus_rank=overall,
        consensus_overall_rank=overall,
        espn_rank=overall,
        blended_rank=float(overall),
        mean=mean,
        sd=40.0,
        replacement=replacement,
    )


class TestGate:
    def test_nothing_fires_before_the_gate_round(self):
        """The early picks are where all the value is; disruption must not touch them."""
        board = [player(i, f"P{i}", "RB", "AAA", i) for i in range(1, 40)]
        by_id = {p.espn_id: p for p in board}
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        assert state.round_of(state.current_pick) < DEFAULT_FIRST_ROUND
        adp = {p.espn_id: p.consensus_overall_rank + 100 for p in board}
        assert disruptive_pick(state, board, Settings(), by_id, adp, current_id=1) is None


class TestDeferForAdp:
    def setup_method(self):
        # Our board rates these two within the tie window; ADP splits them wide.
        self.slider = player(1, "Slider", "WR", "AAA", 50)
        self.urgent = player(2, "Urgent", "WR", "BBB", 53)
        self.available = [self.slider, self.urgent]
        self.needed = {"WR"}

    def test_defers_the_player_the_room_ignores(self):
        adp = {1: 90, 2: 60}  # next pick 70: slider lasts, urgent does not
        play = defer_for_adp(self.available, self.needed, adp, 70, self.slider)
        assert play is not None and play.espn_id == 2
        assert "Slider" in play.why and "Urgent" in play.why

    def test_silent_when_our_choice_will_not_survive(self):
        adp = {1: 71, 2: 60}  # only one pick of slack, under the cushion
        assert defer_for_adp(self.available, self.needed, adp, 70, self.slider) is None

    def test_silent_when_the_alternative_also_survives(self):
        adp = {1: 95, 2: 92}
        assert defer_for_adp(self.available, self.needed, adp, 70, self.slider) is None

    def test_cushion_boundary(self):
        adp = {1: 70 + TIE_SLIDE, 2: 60}
        assert defer_for_adp(self.available, self.needed, adp, 70, self.slider) is not None

    def test_compares_against_the_rule_not_its_own_top(self):
        """The rule reserves late picks and breaks ties on ceiling, so its choice
        is not always our highest-ranked eligible player."""
        adp = {1: 90, 2: 60}
        # The rule is taking `urgent`; there is then nothing to defer for.
        assert defer_for_adp(self.available, self.needed, adp, 70, self.urgent) is None

    def test_no_adp_is_silent(self):
        assert defer_for_adp(self.available, self.needed, {}, 70, self.slider) is None


class TestHandcuff:
    def test_takes_the_backup_to_a_rival_starter(self):
        starter = player(10, "Starter", "RB", "JAC", 5)
        backup = player(11, "Backup", "RB", "JAC", 190)
        other = player(12, "Elsewhere", "RB", "ZZZ", 180)
        by_id = {p.espn_id: p for p in (starter, backup, other)}
        state = DraftState(team_count=2, rounds=4, my_slot=2)
        state.record(10)  # slot 1 takes the starter
        play = handcuff(state, [backup, other], by_id, Settings())
        assert play is not None and play.espn_id == 11
        assert "Starter" in play.why

    def test_ignores_our_own_players(self):
        mine = player(10, "Mine", "RB", "JAC", 5)
        backup = player(11, "Backup", "RB", "JAC", 190)
        by_id = {10: mine, 11: backup}
        state = DraftState(team_count=2, rounds=4, my_slot=1)
        state.record(10, mine=True)
        assert handcuff(state, [backup], by_id, Settings()) is None


class TestPairAtTheTurn:
    def test_fires_only_on_back_to_back_picks(self):
        a = player(1, "A", "TE", "AAA", 60)
        b = player(2, "B", "TE", "BBB", 62)
        state = DraftState(team_count=2, rounds=4, my_slot=2)
        state.record(99)  # slot 1 picks; slot 2 now picks twice across the turn
        assert state.picks_until_my_next() == 0
        play = pair_at_the_turn(state, [a, b], {"TE"})
        assert play is not None and play.espn_id == 1

    def test_silent_when_others_pick_in_between(self):
        a = player(1, "A", "TE", "AAA", 60)
        b = player(2, "B", "TE", "BBB", 62)
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        assert pair_at_the_turn(state, [a, b], {"TE"}) is None

    def test_silent_when_the_top_two_are_different_positions(self):
        a = player(1, "A", "TE", "AAA", 60)
        b = player(2, "B", "WR", "BBB", 62)
        state = DraftState(team_count=2, rounds=4, my_slot=2)
        state.record(99)
        assert pair_at_the_turn(state, [a, b], {"TE", "WR"}) is None


class TestAdvisoryOnly:
    """The plays are untested against check_rehearsal, so they never decide."""

    def test_board_view_takes_no_play_argument(self):
        """Overriding used to be possible here. It must not be reachable at all."""
        import inspect

        from fantasy_football.draft.board_view import board_view

        assert "play" not in inspect.signature(board_view).parameters

    def test_recommendation_is_the_rule_regardless_of_adp(self):
        import numpy as np
        import polars as pl

        from fantasy_football.draft.board_view import board_view
        from fantasy_football.draft.model import fit_pick_model
        from fantasy_football.draft.strategies import best_available_at_need

        board = [
            player(i, f"P{i}", ["WR", "RB", "TE", "QB"][i % 4], "AAA", i, mean=300.0 - 2.0 * i)
            for i in range(1, 61)
        ]
        by_id = {p.espn_id: p for p in board}
        rng = np.random.default_rng(0)
        model = fit_pick_model(
            pl.DataFrame(
                [(2024, r, max(1, r + rng.normal(0, 6))) for r in range(1, 150) for _ in range(2)],
                schema=["season", "consensus_rank", "overall_pick"],
                orient="row",
            )
        )
        state = DraftState(team_count=8, rounds=16, my_slot=1)
        hostile = {p.espn_id: 200 - p.consensus_overall_rank for p in board}
        rows = board_view(state, board, model, Settings(), adp=hostile)
        recommended = next(r.espn_id for r in rows if r.recommended)
        assert recommended == best_available_at_need(state, board, Settings(), by_id)


class TestSilentFeedThreshold:
    """The alarm must not trip on a slow opening pick."""

    def test_threshold_exceeds_one_pick_clock(self):
        from fantasy_football.live_draft import FEED_SILENT_POLLS, POLL_SECONDS

        # ESPN allows 90 seconds a pick; the alarm has to outlast that or it will
        # fire every draft before anyone has done anything wrong.
        assert FEED_SILENT_POLLS * POLL_SECONDS > 90
