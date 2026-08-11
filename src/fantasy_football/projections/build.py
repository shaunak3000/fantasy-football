"""Assemble the current season's projections end to end."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from espn_api.football import League

from ..config import load_credentials
from ..data.espn import LeagueSettings, fetch_raw_settings, parse_settings
from ..data.ids import attach_espn_ids, normalize_team
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


def dst_ids_by_team(league: League) -> dict[str, int]:
    """ESPN's player id for each team defense, keyed by team abbreviation.

    The identity bridge cannot help here: nflverse's player table contains
    people, not team defenses, so every D/ST comes back with a null id. They do
    have a natural key though — there is exactly one defense per team — so they
    match on team abbreviation instead.
    """
    ids: dict[str, int] = {}
    for player in league.free_agents(size=FREE_AGENT_PAGE):
        if getattr(player, "position", None) == "D/ST":
            team = normalize_team(getattr(player, "proTeam", None))
            if team:
                ids[team] = player.playerId
    return ids


def attach_dst_ids(board: pl.DataFrame, league: League) -> pl.DataFrame:
    """Fill in espn_id for team defenses, which the name-based bridge misses."""
    ids = dst_ids_by_team(league)
    if not ids:
        return board

    return board.with_columns(
        pl.when(pl.col("pos") == "DST")
        .then(
            pl.col("team").map_elements(lambda t: ids.get(normalize_team(t)), return_dtype=pl.Int64)
        )
        .otherwise(pl.col("espn_id"))
        .alias("espn_id")
    )


def _unfitted_replacement(
    board, espn_points: dict[int, float], curves, settings
) -> dict[str, float]:
    """Replacement level for positions with no historical curve.

    Computed the same way as everywhere else — the best player nobody is obliged
    to start — just from ESPN's projections instead of a fitted curve.
    """
    levels: dict[str, float] = {}
    positions = {row["pos"] for row in board.iter_rows(named=True)} - set(curves)

    for position in positions:
        scores = sorted(
            (
                espn_points[row["espn_id"]]
                for row in board.iter_rows(named=True)
                if row["pos"] == position and row.get("espn_id") in espn_points
            ),
            reverse=True,
        )
        if not scores:
            continue
        demand = settings.starting_slots.get(position, 1) * settings.team_count
        levels[position] = scores[min(demand, len(scores) - 1)]
    return levels


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
    # Team defenses stay on the board. They have no fitted curve, so they are
    # projected from ESPN alone — but a position the league forces you to start
    # must be draftable, and dropping it left the slot permanently unfillable.
    matched = attach_dst_ids(attach_espn_ids(board).matched, league)

    espn_points = fetch_espn_projections(league)
    replacement = dict(replacement)
    replacement.update(_unfitted_replacement(matched, espn_points, curves, settings))

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
