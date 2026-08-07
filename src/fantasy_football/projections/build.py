"""Assemble the current season's projections end to end."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from espn_api.football import League

from ..config import load_credentials
from ..data.espn import LeagueSettings, fetch_raw_settings, parse_settings
from ..data.ids import attach_espn_ids
from ..data.nflverse import load_consensus_board
from .curve import RankCurve, fit_all, replacement_points, replacement_ranks
from .ensemble import Projection, build_projections
from .history import training_table
from .scoring import ScoringEngine

TRAIN_SEASONS = (2021, 2022, 2023, 2024, 2025)

# ESPN paginates free agents; a fresh pre-draft league has everyone available.
FREE_AGENT_PAGE = 1000


@dataclass
class ProjectionSet:
    settings: LeagueSettings
    projections: list[Projection]
    curves: dict[str, RankCurve]
    replacement: dict[str, float]
    replacement_rank: dict[str, int]
    training_rows: int
    espn_covered: int


def fetch_espn_projections(league: League) -> dict[int, float]:
    """ESPN's own projected season totals, keyed by ESPN player id."""
    points: dict[int, float] = {}
    for player in league.free_agents(size=FREE_AGENT_PAGE):
        projected = getattr(player, "projected_total_points", None)
        if projected:
            points[player.playerId] = float(projected)

    for team in league.teams:
        for player in team.roster:
            projected = getattr(player, "projected_total_points", None)
            if projected:
                points[player.playerId] = float(projected)

    return points


def build(season: int = 2026, refresh: bool = False) -> ProjectionSet:
    creds = load_credentials()
    league = League(league_id=creds.league_id, year=season, espn_s2=creds.espn_s2, swid=creds.swid)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, season)
    engine = ScoringEngine.from_settings(settings)

    training, _ = training_table(list(TRAIN_SEASONS), engine, refresh=refresh)
    curves = fit_all(training)
    replacement = replacement_points(settings, curves)
    ranks = replacement_ranks(settings, curves)

    board = load_consensus_board(refresh=refresh)
    matched = attach_espn_ids(board).matched.filter(pl.col("pos") != "DST")

    espn_points = fetch_espn_projections(league)
    projections = build_projections(matched, espn_points, curves, replacement)

    covered = sum(1 for p in projections if p.espn_rank is not None)

    return ProjectionSet(
        settings=settings,
        projections=projections,
        curves=curves,
        replacement=replacement,
        replacement_rank=ranks,
        training_rows=training.height,
        espn_covered=covered,
    )
