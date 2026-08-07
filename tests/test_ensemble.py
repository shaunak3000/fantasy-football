import numpy as np
import polars as pl
import pytest

from fantasy_football.projections.curve import RankCurve
from fantasy_football.projections.ensemble import build_projections, to_frame

RANKS = np.arange(1, 21)


def flat_curve(position, top=300.0, step=10.0, spread=50.0):
    return RankCurve(
        position=position,
        ranks=RANKS,
        expected=top - step * (RANKS - 1),
        spread=np.full(len(RANKS), spread),
    )


CURVES = {"RB": flat_curve("RB"), "WR": flat_curve("WR", top=280.0)}
REPLACEMENT = {"RB": 150.0, "WR": 140.0}


def board(rows):
    return pl.DataFrame(rows, schema=["player", "pos", "team", "ecr", "espn_id"], orient="row")


SIMPLE = board(
    [
        ("Alpha", "RB", "DET", 1.0, 100),
        ("Bravo", "RB", "ATL", 2.0, 200),
        ("Charlie", "WR", "CIN", 3.0, 300),
    ]
)


class TestBlending:
    def test_agreeing_sources_leave_rank_unchanged(self):
        espn = {100: 250.0, 200: 200.0, 300: 240.0}
        out = {p.player: p for p in build_projections(SIMPLE, espn, CURVES, REPLACEMENT)}
        assert out["Alpha"].blended_rank == pytest.approx(1.0)
        assert out["Bravo"].blended_rank == pytest.approx(2.0)

    def test_disagreement_pulls_the_blended_rank(self):
        # ESPN prefers Bravo; consensus prefers Alpha.
        espn = {100: 200.0, 200: 250.0, 300: 240.0}
        out = {p.player: p for p in build_projections(SIMPLE, espn, CURVES, REPLACEMENT)}
        assert 1.0 < out["Alpha"].blended_rank < 2.0
        assert 1.0 < out["Bravo"].blended_rank < 2.0
        assert out["Alpha"].blended_rank < out["Bravo"].blended_rank

    def test_consensus_weight_controls_the_blend(self):
        espn = {100: 200.0, 200: 250.0, 300: 240.0}
        heavy = build_projections(SIMPLE, espn, CURVES, REPLACEMENT, consensus_weight=1.0)
        assert heavy[0].blended_rank == pytest.approx(1.0)

    def test_player_without_an_espn_projection_still_projects(self):
        out = {p.player: p for p in build_projections(SIMPLE, {}, CURVES, REPLACEMENT)}
        assert out["Alpha"].espn_rank is None
        assert out["Alpha"].blended_rank == pytest.approx(1.0)
        assert out["Alpha"].mean > 0

    def test_ranks_are_computed_within_position_not_overall(self):
        out = {p.player: p for p in build_projections(SIMPLE, {}, CURVES, REPLACEMENT)}
        # Charlie is 3rd overall but the only WR.
        assert out["Charlie"].consensus_rank == 1


class TestValues:
    def test_vor_is_mean_less_positional_replacement(self):
        out = build_projections(SIMPLE, {}, CURVES, REPLACEMENT)[0]
        assert out.vor == pytest.approx(out.mean - REPLACEMENT[out.position])

    def test_sorted_by_value_over_replacement(self):
        out = build_projections(SIMPLE, {}, CURVES, REPLACEMENT)
        assert [p.vor for p in out] == sorted((p.vor for p in out), reverse=True)

    def test_disagreement_widens_the_spread(self):
        agree = build_projections(
            board([("Solo", "RB", "DET", 1.0, 100)]), {100: 300.0}, CURVES, REPLACEMENT
        )[0]
        wide = board([("Solo", "RB", "DET", 1.0, 100), ("Other", "RB", "ATL", 2.0, 200)])
        # ESPN ranks Solo last of the two, consensus ranks him first.
        disagreeing = {
            p.player: p
            for p in build_projections(wide, {100: 100.0, 200: 300.0}, CURVES, REPLACEMENT)
        }
        assert disagreeing["Solo"].sd > agree.sd

    def test_unknown_position_is_skipped_not_crashed(self):
        odd = board([("Kicker", "K", "NE", 1.0, 900)])
        assert build_projections(odd, {}, CURVES, REPLACEMENT) == []


def test_frame_round_trips_every_projection():
    frame = to_frame(build_projections(SIMPLE, {}, CURVES, REPLACEMENT))
    assert frame.height == 3
    assert set(frame.columns) >= {"player", "pos", "mean", "sd", "vor", "blended_rank"}
