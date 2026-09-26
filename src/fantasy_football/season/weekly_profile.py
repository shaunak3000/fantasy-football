"""What a roster can actually field in each remaining week, and who plays.

A single rest-of-season number per team hides the weeks that decide matchups.
In week 6 of 2026 this roster lost Amon-Ra St. Brown, Sam LaPorta and Joe Burrow
to byes at once, and a trade that was +3.93% on the season was -3.7 points that
week — found by hand, because nothing in the trade search could see a week. The
same blindness ran the other way: nothing could say whether an incoming player
would start most weeks or only cover one bye.

So each remaining week is valued separately. A player on bye that week is gone;
a player ruled out *this* week is gone for this week only — someone on injured
reserve in September is usually back by December, which is the same assumption
the rest of the repo makes and is a known optimism for long injuries.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..lineup.fast import select


def identity(player) -> object:
    return getattr(player, "espn_id", None) or getattr(player, "player", id(player))


def available(player, week: int, current_week: int | None) -> bool:
    """Whether this player can score in `week`."""
    if getattr(player, "bye_week", None) == week:
        return False
    returning = getattr(player, "return_week", None)
    if returning is not None and week < returning:
        return False
    if current_week is not None and week == current_week:
        return not (getattr(player, "on_bye", False) or getattr(player, "unavailable", False))
    return True


@dataclass(frozen=True)
class WeekLineup:
    week: int
    points: float
    starters: frozenset


def weekly_lineups(roster: list, weeks: list[int], current_week: int | None, settings) -> list:
    """The best lineup for each week in `weeks`, with byes and this week's injuries out.

    Weeks where nobody on the roster is missing share one computation, so a
    sixteen-man roster costs roughly one lineup per distinct bye week rather than
    one per week.
    """
    memo: dict[frozenset, WeekLineup] = {}
    out = []
    for week in weeks:
        playing = [p for p in roster if available(p, week, current_week)]
        key = frozenset(identity(p) for p in playing)
        cached = memo.get(key)
        if cached is None:
            starters = select(playing, settings)
            cached = WeekLineup(
                week=week,
                points=sum(p.mean for p in starters),
                starters=frozenset(identity(p) for p in starters),
            )
            memo[key] = cached
        out.append(WeekLineup(week=week, points=cached.points, starters=cached.starters))
    return out


def weekly_points(roster: list, weeks: list[int], current_week: int | None, settings) -> tuple:
    return tuple(lineup.points for lineup in weekly_lineups(roster, weeks, current_week, settings))


#: A player who is in the full-strength lineup is a starter. One who is not, but
#: plays when someone ahead of him is on bye, is cover. One who never plays is
#: depth — which in an eight-team league is also what the waiver wire offers free.
STARTER, COVER, DEPTH = "starter", "cover", "depth"


@dataclass(frozen=True)
class PlayerUsage:
    player: str
    role: str
    weeks_started: tuple[int, ...]
    weeks_total: int

    def summary(self) -> str:
        n, total = len(self.weeks_started), self.weeks_total
        if self.role == DEPTH:
            return f"{self.player}: depth — never starts"
        weeks = ", ".join(str(w) for w in self.weeks_started)
        if self.role == COVER:
            return f"{self.player}: cover only — starts wk {weeks} ({n} of {total})"
        missed = total - n
        tail = "" if missed == 0 else f", sits {missed}"
        return f"{self.player}: starter — {n} of {total} weeks{tail}"


def usage(
    player, roster: list, weeks: list[int], current_week: int | None, settings
) -> PlayerUsage:
    """How often `player` would start on `roster`, and whether only as bye cover.

    The distinction matters for trade value. A receiver who starts every week is
    an upgrade; one who only plays the week your WR1 is on bye is worth that one
    week, which a free agent usually covers for nothing.
    """
    me = identity(player)
    lineups = weekly_lineups(roster, weeks, current_week, settings)
    started = tuple(lineup.week for lineup in lineups if me in lineup.starters)
    full_strength = frozenset(identity(p) for p in select(roster, settings))
    if not started:
        role = DEPTH
    elif me in full_strength:
        role = STARTER
    else:
        role = COVER
    return PlayerUsage(
        player=getattr(player, "player", str(me)),
        role=role,
        weeks_started=started,
        weeks_total=len(weeks),
    )


def season_total(roster: list, weeks: list[int], current_week: int | None, settings) -> float:
    """Total lineup points over `weeks`, the fast way. Equal to `sum(weekly_points)`.

    The trade screen calls this around 300,000 times at full depth, on rosters it
    will never see again, so it avoids building a set per week. A week is one of
    three kinds: the current week (byes and this week's injuries out), a week
    in which some roster player is on bye (that player out), or a clean week
    (nobody out). Each kind is computed once and multiplied by how often it
    occurs — a sixteen-man roster costs one lineup per distinct bye week.
    """
    if not weeks:
        return 0.0
    in_range = set(weeks)
    bye_weeks: dict[int, list] = {}
    for player in roster:
        week = getattr(player, "bye_week", None)
        if week in in_range and week != current_week:
            bye_weeks.setdefault(week, []).append(player)
    # A player out until week N is missing from every week before it — the same
    # as a bye in each of those weeks — so those weeks join the special cases.
    for player in roster:
        returning = getattr(player, "return_week", None)
        if returning is None:
            continue
        for week in weeks:
            if week < returning and week != current_week:
                bye_weeks.setdefault(week, [])
                if player not in bye_weeks[week]:
                    bye_weeks[week].append(player)

    total = 0.0
    clean_count = len(weeks)
    if current_week in in_range:
        playing = [p for p in roster if available(p, current_week, current_week)]
        total += sum(p.mean for p in select(playing, settings))
        clean_count -= 1
    for out in bye_weeks.values():
        gone = {id(p) for p in out}
        playing = [p for p in roster if id(p) not in gone]
        total += sum(p.mean for p in select(playing, settings))
        clean_count -= 1
    if clean_count:
        total += clean_count * sum(p.mean for p in select(roster, settings))
    return total
