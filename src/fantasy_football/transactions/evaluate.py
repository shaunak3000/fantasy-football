"""Price a roster move in the only currency that matters: title probability.

A waiver claim that adds twelve points a week is worth a great deal in week 3
and almost nothing in week 13 to a team already locked into a playoff seed. A
trade that raises your floor helps a favourite and actively hurts an underdog
who needs variance. Points cannot express either of those; championship
probability can, which is why every move here is quoted as a delta in P(first)
rather than a delta in points.

Two disciplines make those numbers mean something.

**One baseline.** Every move is measured against a single simulation of the
season as it stands, computed once and shared. Re-simulating the baseline per
section, at different trial counts, produced three different answers for the
same team in the same week and made every delta meaningless.

**An error bar on every delta.** A title probability is a Monte Carlo estimate,
and the difference of two estimates is noisier than either. At the trial counts
this module used to run at, the standard error on a delta was around 1.5
percentage points — so a trade reported at "+0.50%" was indistinguishable from
no trade at all. Deltas now carry their own standard error and anything smaller
than it is refused rather than printed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from ..lineup.optimizer import optimize
from ..season.simulator import SeasonOutcome, TeamSeason, paired_delta, simulate

#: Enough trials that the noise floor on a delta sits near a tenth of a point of
#: title probability. The simulation is vectorized, so this costs milliseconds;
#: the old default of 200-600 was a holdover from when it was a Python loop.
DEFAULT_TRIALS = 20000


def _identity(player) -> object:
    return getattr(player, "espn_id", None) or getattr(player, "player", id(player))


@dataclass
class SimulationContext:
    """Everything a season simulation needs, plus a cache of lineup strengths.

    Solving a roster's best lineup costs about 20ms, and a trade search asks for
    the same seven untouched rosters hundreds of times. Caching by roster
    membership turns the search from minutes into seconds and is what makes it
    affordable to search packages rather than single players.
    """

    names: dict[int, str]
    settings: object
    schedule: dict[int, list[int | None]]
    playoff_teams: int
    banked: dict[int, tuple[int, int, float]] = field(default_factory=dict)
    trials: int = DEFAULT_TRIALS
    seed: int = 0
    #: How far a team's projected weekly mean is likely to be from the truth.
    #: Left at zero the simulation treats a projection as a known fact and comes
    #: out overconfident — see `check_simulator`.
    mean_uncertainty: float = 0.0
    _strengths: dict = field(default_factory=dict, repr=False)

    def strength(self, roster: list) -> tuple[float, float]:
        """Weekly mean and spread of the best lineup this roster can start."""
        key = tuple(sorted(map(str, (_identity(p) for p in roster))))
        cached = self._strengths.get(key)
        if cached is None:
            lineup = optimize(roster, self.settings)
            cached = (lineup.mean, lineup.sd)
            self._strengths[key] = cached
        return cached

    def outcome(self, rosters: dict[int, list]) -> SeasonOutcome:
        teams = []
        for team_id, roster in rosters.items():
            mean, sd = self.strength(roster)
            wins, losses, points = self.banked.get(team_id, (0, 0, 0.0))
            teams.append(
                TeamSeason(
                    team_id=team_id,
                    name=self.names.get(team_id, str(team_id)),
                    weekly_mean=mean,
                    weekly_sd=sd,
                    wins=wins,
                    losses=losses,
                    points_for=points,
                    mean_uncertainty=self.mean_uncertainty,
                )
            )
        return simulate(
            teams, self.schedule, self.playoff_teams, trials=self.trials, seed=self.seed
        )


@dataclass(frozen=True)
class MoveEvaluation:
    description: str
    title_before: float
    title_after: float
    weekly_points_change: float
    title_stderr: float = 0.0
    counterparty_id: int | None = None
    counterparty_before: float = 0.0
    counterparty_after: float = 0.0
    counterparty_stderr: float = 0.0

    @property
    def title_delta(self) -> float:
        return self.title_after - self.title_before

    @property
    def counterparty_delta(self) -> float:
        """What the trade does to the other side's title odds.

        A trade they will refuse is worth nothing, so the number that decides
        whether a proposal is worth sending is theirs, not yours.
        """
        return self.counterparty_after - self.counterparty_before

    @property
    def significant(self) -> bool:
        """Whether the move clears its own Monte Carlo noise."""
        return self.title_delta > self.title_stderr

    @property
    def worth_doing(self) -> bool:
        return self.significant

    @property
    def mutually_acceptable(self) -> bool:
        return self.significant and self.counterparty_delta > self.counterparty_stderr

    def summary(self) -> str:
        direction = "+" if self.title_delta >= 0 else ""
        return (
            f"{self.description}: title {self.title_before:.1%} -> {self.title_after:.1%} "
            f"({direction}{self.title_delta:.2%} +/-{self.title_stderr:.2%}), "
            f"{self.weekly_points_change:+.1f} pts/week"
        )


def roster_strength(roster: list, settings) -> tuple[float, float]:
    """Weekly mean and spread of the best lineup this roster can start."""
    lineup = optimize(roster, settings)
    return lineup.mean, lineup.sd


def apply_move(
    rosters: dict[int, list],
    my_team_id: int,
    add: list,
    drop: list,
    counterparty_id: int | None,
) -> dict[int, list]:
    """Rosters as they would be after the move, leaving the originals untouched."""
    dropped = {_identity(p) for p in drop}
    acquired = {_identity(p) for p in add}

    updated = {tid: list(roster) for tid, roster in rosters.items()}
    updated[my_team_id] = [p for p in rosters[my_team_id] if _identity(p) not in dropped] + list(
        add
    )

    # A trade is two-sided: whoever owned the incoming players must lose them,
    # and must gain whatever went the other way. Without this the acquired
    # player scores for both teams at once, which silently overstates every
    # trade — and misses the real prize in an eight-team league, that taking a
    # player off a rival weakens a direct competitor for one of four berths.
    if counterparty_id is not None and counterparty_id in updated:
        updated[counterparty_id] = [
            p for p in rosters[counterparty_id] if _identity(p) not in acquired
        ] + list(drop)

    return updated


def evaluate_move(
    my_team_id: int,
    rosters: dict[int, list],
    names: dict[int, str],
    settings,
    schedule: dict[int, list[int | None]],
    playoff_teams: int,
    add: list | None = None,
    drop: list | None = None,
    banked: dict[int, tuple[int, int, float]] | None = None,
    trials: int = DEFAULT_TRIALS,
    seed: int = 0,
    counterparty_id: int | None = None,
    context: SimulationContext | None = None,
    baseline: SeasonOutcome | None = None,
) -> MoveEvaluation:
    """Simulate the season with and without a proposed move.

    Leave `counterparty_id` unset for a waiver claim, where the incoming player
    genuinely comes from outside the league. Set it for a trade, so the other
    side's roster changes too — see the note in `apply_move` on why that matters.

    The same random seed is used for both simulations so the comparison isolates
    the move rather than the luck of the draw. That pairing is also what lets
    the delta carry an honest error bar: the two runs are differenced trial by
    trial rather than compared as two summary numbers.

    Pass `context` and `baseline` to share one pre-computed view of the season
    across many candidate moves — that is what keeps a report's sections quoting
    the same starting odds.
    """
    add = add or []
    drop = drop or []
    context = context or SimulationContext(
        names=names,
        settings=settings,
        schedule=schedule,
        playoff_teams=playoff_teams,
        banked=banked or {},
        trials=trials,
        seed=seed,
    )

    before = baseline if baseline is not None else context.outcome(rosters)
    before_mean, _ = context.strength(rosters[my_team_id])

    updated = apply_move(rosters, my_team_id, add, drop, counterparty_id)
    after = context.outcome(updated)
    after_mean, _ = context.strength(updated[my_team_id])

    delta, stderr = paired_delta(before, after, my_team_id)
    counterparty_delta, counterparty_stderr = (
        paired_delta(before, after, counterparty_id) if counterparty_id is not None else (0.0, 0.0)
    )

    labels = []
    if add:
        labels.append("add " + ", ".join(getattr(p, "player", "?") for p in add))
    if drop:
        labels.append(
            ("send " if counterparty_id is not None else "drop ")
            + ", ".join(getattr(p, "player", "?") for p in drop)
        )

    title_before = before.championship.get(my_team_id, 0.0)
    counterparty_before = (
        before.championship.get(counterparty_id, 0.0) if counterparty_id is not None else 0.0
    )
    return MoveEvaluation(
        description=" / ".join(labels) or "no change",
        title_before=title_before,
        title_after=title_before + delta,
        title_stderr=stderr,
        weekly_points_change=after_mean - before_mean,
        counterparty_id=counterparty_id,
        counterparty_before=counterparty_before,
        counterparty_after=counterparty_before + counterparty_delta,
        counterparty_stderr=counterparty_stderr,
    )


def marginal_cost(roster: list, player, context: SimulationContext) -> float:
    """Points a week lost by removing this player from the roster.

    This is what "surplus" actually means, and it is not the same as being low
    scoring. On a receiver-heavy roster the best running back can sit below six
    receivers on raw projection while being the only player who can fill a
    starting slot — ranking by points alone offered him up as spare. Removing
    him costs real points; removing the sixth receiver, who never starts, costs
    nothing.
    """
    without = [p for p in roster if _identity(p) != _identity(player)]
    return context.strength(roster)[0] - context.strength(without)[0]


def marginal_gain(roster: list, player, context: SimulationContext) -> float:
    """Points a week this player would add to the roster, before any drop."""
    return context.strength([*roster, player])[0] - context.strength(roster)[0]


def find_trades(
    my_team_id: int,
    rosters: dict[int, list],
    names: dict[int, str],
    settings,
    schedule: dict[int, list[int | None]],
    playoff_teams: int,
    banked: dict[int, tuple[int, int, float]] | None = None,
    give_depth: int = 5,
    get_depth: int = 4,
    trials: int = DEFAULT_TRIALS,
    context: SimulationContext | None = None,
    baseline: SeasonOutcome | None = None,
    package_sizes: tuple[int, ...] = (1, 2),
    shortlist: int = 12,
) -> list[MoveEvaluation]:
    """Search opponent rosters for swaps that help both sides.

    Two things the earlier version could not do, both of which it needed to.

    *Packages, not just singles.* Restricting the search to one-for-one made
    the deals this roster most needs unreachable: trading two of six startable
    receivers for two startable running backs is a two-for-two, and no sequence
    of one-for-ones gets there through a roster with no spare slot.

    *A points screen before a probability screen.* Simulating every candidate
    was unaffordable, so the search was kept narrow and its answers were read
    off a few hundred trials — well inside the noise. Candidates are now ranked
    by the change in both sides' best lineup, which is fast and deterministic,
    and only the survivors are simulated, at a trial count where the resulting
    title delta means something.
    """
    my_roster = rosters.get(my_team_id, [])
    if not my_roster:
        return []

    context = context or SimulationContext(
        names=names,
        settings=settings,
        schedule=schedule,
        playoff_teams=playoff_teams,
        banked=banked or {},
        trials=trials,
    )
    baseline = baseline if baseline is not None else context.outcome(rosters)
    my_before = context.strength(my_roster)[0]

    # Offer what costs least, not what scores least.
    give_pool = sorted(my_roster, key=lambda p: marginal_cost(my_roster, p, context))[:give_depth]

    screened: list[tuple[float, dict, int, list, list]] = []
    for team_id, roster in rosters.items():
        if team_id == my_team_id or not roster:
            continue
        their_before = context.strength(roster)[0]
        get_pool = sorted(roster, key=lambda p: marginal_gain(my_roster, p, context), reverse=True)[
            :get_depth
        ]

        for size in package_sizes:
            if size > len(give_pool) or size > len(get_pool):
                continue
            for give in combinations(give_pool, size):
                for get in combinations(get_pool, size):
                    updated = apply_move(rosters, my_team_id, list(get), list(give), team_id)
                    my_gain = context.strength(updated[my_team_id])[0] - my_before
                    their_gain = context.strength(updated[team_id])[0] - their_before
                    # A proposal they refuse is worthless, so both sides must
                    # gain on points before it is worth a simulation.
                    if my_gain <= 0 or their_gain <= 0:
                        continue
                    # Rank by whichever side gains *least*. Sorting by our own
                    # gain fills the shortlist with deals that are wonderful for
                    # us and ruinous for them — they screen in on points because
                    # the incoming player improves their lineup, then score
                    # -16% on their title odds because they have just handed a
                    # direct rival the division. Those get refused, and they
                    # crowd out the balanced deals that would not have been.
                    screened.append(
                        (min(my_gain, their_gain), updated, team_id, list(get), list(give))
                    )

    screened.sort(key=lambda row: -row[0])

    accepted = []
    for _, _, team_id, get, give in screened[:shortlist]:
        move = evaluate_move(
            my_team_id,
            rosters,
            names,
            settings,
            schedule,
            playoff_teams,
            add=get,
            drop=give,
            banked=banked,
            counterparty_id=team_id,
            context=context,
            baseline=baseline,
        )
        if move.mutually_acceptable:
            accepted.append(move)

    accepted.sort(key=lambda p: -p.title_delta)
    return accepted


@dataclass(frozen=True)
class WaiverAdvice:
    """A ranked claim, plus whether it justifies spending waiver priority."""

    move: MoveEvaluation
    points_gain: float
    burn_priority: bool

    @property
    def verdict(self) -> str:
        if self.burn_priority:
            return "worth the claim"
        if self.points_gain <= 0:
            return "bench depth only — does not start, hold priority"
        return "below the noise floor — hold priority"


def rank_waiver_targets(
    my_team_id: int,
    rosters: dict[int, list],
    names: dict[int, str],
    settings,
    schedule: dict[int, list[int | None]],
    playoff_teams: int,
    free_agents: list,
    droppable: list,
    banked: dict[int, tuple[int, int, float]] | None = None,
    top: int = 8,
    trials: int = DEFAULT_TRIALS,
    context: SimulationContext | None = None,
    baseline: SeasonOutcome | None = None,
    shortlist: int = 5,
) -> list[WaiverAdvice]:
    """Best available add/drop pairs, ranked by championship impact.

    Free agents are screened on whether they would actually crack the starting
    lineup before any of them is simulated. Taking the first few names ESPN
    happens to return produced four suggestions in a row worth +0.00%, which is
    not a ranking of anything: a bench add is worth approximately zero by
    construction, and a list of them tells you nothing about the one claim that
    might matter.

    This league awards waivers by **priority order, not FAAB**, so a claim is
    not a price paid but a position spent — once used it goes to the back of the
    queue. A move that cannot be distinguished from zero is never worth that,
    however free it looks, which is what `burn_priority` records.
    """
    my_roster = rosters.get(my_team_id, [])
    if not my_roster or not free_agents:
        return []

    context = context or SimulationContext(
        names=names,
        settings=settings,
        schedule=schedule,
        playoff_teams=playoff_teams,
        banked=banked or {},
        trials=trials,
    )
    baseline = baseline if baseline is not None else context.outcome(rosters)

    worst = min(droppable, key=lambda p: p.mean, default=None)
    if worst is None:
        return []

    screened = []
    for candidate in free_agents[:top]:
        updated = apply_move(rosters, my_team_id, [candidate], [worst], None)
        gain = context.strength(updated[my_team_id])[0] - context.strength(my_roster)[0]
        screened.append((gain, candidate))
    screened.sort(key=lambda row: -row[0])

    advice = []
    for gain, candidate in screened[:shortlist]:
        move = evaluate_move(
            my_team_id,
            rosters,
            names,
            settings,
            schedule,
            playoff_teams,
            add=[candidate],
            drop=[worst],
            banked=banked,
            context=context,
            baseline=baseline,
        )
        advice.append(
            WaiverAdvice(move=move, points_gain=gain, burn_priority=gain > 0 and move.significant)
        )
    return advice
