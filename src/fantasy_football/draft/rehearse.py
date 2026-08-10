"""Replay a past draft and score the board against what really happened.

This is the only test that answers the question that matters: would following
this tool have produced a better roster than the humans produced? Everything
else measures a component.

Leakage is the whole danger, and it is easy to introduce accidentally. The rank
curves must be fit only on seasons *before* the draft being replayed, the
opponent model only on drafts before it, and the consensus board must be the one
published in August, not a revised one. Any of those slipping would produce a
flattering number that means nothing.

Scoring uses actual points from the season that followed, so a roster is judged
by what its players really did — not by what the projections thought of them,
which would be marking our own homework.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl
from espn_api.football import League

from ..data.espn import fetch_raw_settings, parse_settings
from ..projections.curve import fit_all, replacement_points
from ..projections.ensemble import Projection, build_projections
from ..projections.history import season_actuals, training_table
from ..projections.scoring import ScoringEngine
from .history import board_with_ids, load_draft
from .model import fit_pick_model
from .recommend import recommend
from .state import DraftState

# Starting requirements scored head to head. Kickers and defenses are excluded
# because neither the board nor the humans meaningfully predict them, so
# including them would add noise without testing anything.
SCORED_SLOTS = (("QB", 1), ("RB", 2), ("WR", 2), ("TE", 1))
FLEX_POSITIONS = ("RB", "WR", "TE")
FLEX_COUNT = 1


@dataclass
class RosterScore:
    total: float
    starters: list[tuple[str, str, float]] = field(default_factory=list)

    def summary(self) -> str:
        return ", ".join(f"{name} {points:.0f}" for _, name, points in self.starters[:4])


@dataclass
class SlotResult:
    slot: int
    tool: RosterScore
    human: RosterScore

    @property
    def edge(self) -> float:
        return self.tool.total - self.human.total


def best_lineup(points_by_player: list[tuple[str, str, float]]) -> RosterScore:
    """Score a roster by its best legal starting lineup.

    A roster is only worth what you can start, so the comparison uses the best
    lineup the roster supports rather than the sum of everyone on it — otherwise
    hoarding six running backs would look like a triumph.
    """
    remaining = sorted(points_by_player, key=lambda row: -row[2])
    chosen: list[tuple[str, str, float]] = []

    for position, count in SCORED_SLOTS:
        picked = [row for row in remaining if row[0] == position][:count]
        chosen.extend(picked)
        remaining = [row for row in remaining if row not in picked]

    flex = [row for row in remaining if row[0] in FLEX_POSITIONS][:FLEX_COUNT]
    chosen.extend(flex)

    return RosterScore(total=sum(row[2] for row in chosen), starters=chosen)


def _actual_points_by_espn_id(season: int, engine, board: pl.DataFrame) -> dict[int, float]:
    actuals = season_actuals(season, engine)
    lookup = dict(
        zip(actuals["player_id"].to_list(), actuals["actual_points"].to_list(), strict=True)
    )
    return {
        int(row["espn_id"]): float(lookup.get(row["gsis_id"], 0.0))
        for row in board.iter_rows(named=True)
        if row.get("espn_id") is not None
    }


def build_historical_projections(
    season: int, engine, settings
) -> tuple[list[Projection], pl.DataFrame]:
    """Projections for `season`, fit only on what was knowable before it."""
    prior_seasons = [s for s in range(2021, season)]
    training, _ = training_table(prior_seasons, engine)
    curves = fit_all(training)
    replacement = replacement_points(settings, curves)

    board = board_with_ids(season)
    # ESPN's historical preseason projections are not retrievable, so the
    # rehearsal runs on consensus alone. That understates the live tool slightly.
    projections = build_projections(board, {}, curves, replacement)
    return projections, board


def rehearse_slot(
    slot: int,
    season: int,
    projections: list[Projection],
    pick_model,
    settings,
    actual_picks: list,
    points: dict[int, float],
    rounds: int,
    trials: int = 200,
) -> SlotResult:
    """Replay one draft, substituting the board's choices at one slot."""
    by_id = {p.espn_id: p for p in projections if p.espn_id is not None}

    # Keyed by pick number, not a queue. A shared queue silently shifts every
    # opponent forward by one as soon as the board takes somebody, handing them
    # players who really went earlier and stacking the test against the tool.
    historical_by_pick = {p.overall_pick: p.espn_id for p in actual_picks}
    fallback = [p.espn_id for p in actual_picks]

    state = DraftState(team_count=settings.team_count, rounds=rounds, my_slot=slot)
    human_roster = [p.espn_id for p in actual_picks if p.overall_pick in set(state.my_picks)]

    while not state.is_complete:
        if state.is_my_turn:
            options = recommend(state, projections, pick_model, settings, trials=trials)
            if not options:
                break
            state.record_my_pick(options[0].espn_id or -1)
            continue

        pick_number = state.current_pick
        wanted = historical_by_pick.get(pick_number)
        if wanted is not None and wanted not in state.drafted_set:
            state.record(wanted)
            continue

        # The board took the player this opponent wanted, so he settles for the
        # best player still on the board by his own historical preferences.
        replacement_pick = next((i for i in fallback if i not in state.drafted_set), None)
        if replacement_pick is None:
            break
        state.record(replacement_pick)

    def score(ids: list[int]) -> RosterScore:
        rows = [
            (by_id[i].position, by_id[i].player, points.get(i, 0.0))
            for i in ids
            if i in by_id and by_id[i].position in {p for p, _ in SCORED_SLOTS}
        ]
        return best_lineup(rows)

    return SlotResult(slot=slot, tool=score(state.my_roster), human=score(human_roster))


def rehearse(season: int, creds, rounds: int = 16, trials: int = 200) -> list[SlotResult]:
    league = League(league_id=creds.league_id, year=season, espn_s2=creds.espn_s2, swid=creds.swid)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, season)
    engine = ScoringEngine.from_settings(settings)

    projections, board = build_historical_projections(season, engine, settings)
    points = _actual_points_by_espn_id(season, engine, board)
    actual_picks = load_draft(league, season)

    # Opponent behaviour must come from drafts strictly before the one replayed.
    prior_training, _ = _pick_training_before(season, creds)
    pick_model = fit_pick_model(prior_training)
    if pick_model is None:
        raise ValueError(f"no draft history before {season} to fit an opponent model")

    return [
        rehearse_slot(
            slot, season, projections, pick_model, settings, actual_picks, points, rounds, trials
        )
        for slot in range(1, settings.team_count + 1)
    ]


def _pick_training_before(season: int, creds):
    from .history import pick_training

    earlier = [s for s in (2024, 2025) if s < season]
    if not earlier:
        raise ValueError(f"no draft seasons available before {season}")
    return pick_training(earlier, creds)
