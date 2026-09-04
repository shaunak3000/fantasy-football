"""Live draft feed.

ESPN pre-allocates 128 pick slots and is expected to fill in `playerId` as picks
are made, so the whole live problem reduces to polling one endpoint and noticing
which slots have acquired a player. You draft in ESPN as normal; this watches.

**Measured caveat (2026-09-04):** ESPN *mock* drafts never populate these slots.
A mock with `inProgress: True` and picks visibly landing in the draft room still
reports every slot as `playerId: -1`, and `mRoster`, `mTeam` and
`kona_league_communication` are all equally empty — the mock draft room is served
over a channel the REST API does not see. Whether a real league draft behaves the
same way is unverified, which is why `watch` now raises a stall alarm rather than
sitting silently on pick 1, and why `manual` exists.

Everything here is defensive on purpose. A draft happens once, under a 90-second
clock, and a tool that throws an exception in round 4 is worse than no tool at
all — so a failed poll returns the last known state rather than raising, and the
manual path stays available the whole time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from espn_api.football import League

from .state import DraftState

EMPTY_PICK = -1


@dataclass(frozen=True)
class LivePick:
    overall_pick: int
    round_num: int
    round_pick: int
    team_id: int
    player_id: int


@dataclass
class DraftFeed:
    """Reads draft state from ESPN, with the last good snapshot kept on failure."""

    league: League
    picks: list[LivePick] = field(default_factory=list)
    in_progress: bool = False
    complete: bool = False
    last_error: str | None = None

    def poll(self) -> bool:
        """Refresh from ESPN. Returns True if anything changed."""
        try:
            payload = self.league.espn_request.league_get(params={"view": "mDraftDetail"})
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

        detail = payload.get("draftDetail", {})
        made = [
            LivePick(
                overall_pick=p.get("overallPickNumber", 0),
                round_num=p.get("roundId", 0),
                round_pick=p.get("roundPickNumber", 0),
                team_id=p.get("teamId", 0),
                player_id=p.get("playerId", EMPTY_PICK),
            )
            for p in detail.get("picks", [])
            if p.get("playerId", EMPTY_PICK) != EMPTY_PICK
        ]
        made.sort(key=lambda p: p.overall_pick)

        changed = len(made) != len(self.picks)
        self.picks = made
        self.in_progress = bool(detail.get("inProgress", False))
        self.complete = bool(detail.get("drafted", False))
        self.last_error = None
        return changed

    @property
    def pick_count(self) -> int:
        return len(self.picks)


def draft_order(league: League) -> list[int]:
    """Team ids in draft-slot order, straight from the league settings."""
    payload = league.espn_request.league_get(params={"view": "mSettings"})
    return payload.get("settings", {}).get("draftSettings", {}).get("pickOrder", [])


def my_team_id(league: League, swid: str | None) -> int | None:
    """Find which team belongs to the logged-in user.

    ESPN tags each team's owners with the same SWID that authenticates the
    session, so the draft slot can be discovered rather than typed — one less
    thing to get wrong at the worst possible moment.
    """
    if not swid:
        return None
    target = swid.strip("{}").upper()

    for team in league.teams:
        owners = getattr(team, "owners", None) or []
        for owner in owners:
            identifier = owner.get("id") if isinstance(owner, dict) else owner
            if identifier and str(identifier).strip("{}").upper() == target:
                return team.team_id
    return None


def slot_for_team(order: list[int], team_id: int | None) -> int | None:
    if team_id is None or team_id not in order:
        return None
    return order.index(team_id) + 1


def sync_state(state: DraftState, feed: DraftFeed, my_id: int | None) -> DraftState:
    """Rebuild draft state from the feed.

    Rebuilt from scratch each time rather than appended to, so a missed poll or
    a correction on ESPN's side self-heals on the next cycle instead of leaving
    the board permanently out of step with reality.
    """
    state.drafted = [p.player_id for p in feed.picks]
    state.my_roster = [p.player_id for p in feed.picks if my_id is not None and p.team_id == my_id]
    return state
