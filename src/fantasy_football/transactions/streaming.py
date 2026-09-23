"""Swapping a one-slot position for whoever has the better week.

Defence is the one position where the weekly matchup genuinely moves the number.
Measured across weeks 4-10 of 2026, where the schedule is known and nothing else
about a player has changed yet, ESPN's projections vary by about 2-3% for skill
players — 0.46 points of standard deviation on Amon-Ra St. Brown, 0.19 on Harold
Fannin. For team defences it is around 8%, because defensive scoring depends far
more on the offence you happen to face than on your own roster.

That gap is the whole justification. Chasing a matchup at running back is noise;
at defence it is a point or so a week, available for a free-agent add because
nobody else in an eight-team league carries a spare defence.

The rule is deliberately not "find the weakest opponent". ESPN's projection
already blends opponent strength with the defence's own quality, and the blend
beats either half: in week 3 of 2026 the Lions projected best of the free agents
(9.5) while facing the Jets' strong offence, because Detroit's defence carried
it. Ranking by the published projection is simpler and strictly better informed.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Below this, a swap is not worth a roster move. Week 2 of 2026 offered +0.5
#: (Buccaneers over Eagles) and was rightly skipped; week 3 offered +1.3 (Lions
#: over Eagles) and was not. The threshold sits between them on purpose.
DEFAULT_THRESHOLD = 1.0


@dataclass(frozen=True)
class StreamAdvice:
    """A better starter for a one-slot position, sitting on the wire."""

    position: str
    holding: str
    holding_points: float
    candidate: str
    candidate_points: float
    runners_up: tuple[tuple[str, float], ...] = ()

    @property
    def gain(self) -> float:
        return self.candidate_points - self.holding_points

    def summary(self) -> str:
        return (
            f"start {self.candidate} ({self.candidate_points:.1f}) over "
            f"{self.holding} ({self.holding_points:.1f}): {self.gain:+.1f} pts this week"
        )


def find_stream(
    weekly: dict[int, object],
    my_ids: set[int],
    rostered_ids: set[int],
    position: str = "D/ST",
    threshold: float = DEFAULT_THRESHOLD,
    runners_up: int = 3,
) -> StreamAdvice | None:
    """The best free agent at `position` if he beats what we hold by `threshold`.

    Returns None when the swap is not worth making, which is most weeks — the
    report prints nothing rather than manufacturing advice. `weekly` is
    `fetch_weekly_projections` output, which already covers rostered players and
    free agents in one request, so this needs no second call.

    A player projected at zero is on a bye, and one ESPN has ruled out will not
    score either; neither is a candidate, and neither counts as what we hold.
    """

    def usable(projection) -> bool:
        return (
            getattr(projection, "position", None) == position
            and getattr(projection, "points", 0.0) > 0
            and not getattr(projection, "unavailable", False)
        )

    mine = [p for pid, p in weekly.items() if pid in my_ids and usable(p)]
    available = sorted(
        (p for pid, p in weekly.items() if pid not in rostered_ids and usable(p)),
        key=lambda p: -p.points,
    )
    if not mine or not available:
        return None

    holding = max(mine, key=lambda p: p.points)
    best = available[0]
    if best.points - holding.points < threshold:
        return None

    return StreamAdvice(
        position=position,
        holding=holding.name,
        holding_points=holding.points,
        candidate=best.name,
        candidate_points=best.points,
        runners_up=tuple((p.name, p.points) for p in available[1 : 1 + runners_up]),
    )
