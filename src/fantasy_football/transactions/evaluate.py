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

from ..lineup.fast import best_points, is_greedy_exact
from ..lineup.optimizer import optimize
from ..season.simulator import SeasonOutcome, TeamSeason, paired_delta, simulate
from ..season.weekly_profile import PlayerUsage, season_total, usage, weekly_points

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
    #: The NFL week the schedule starts from. With it, every remaining week is
    #: valued separately — byes fall where they fall — and the simulator plays
    #: each week at its own strength. Without it, the old flat valuation.
    current_week: int | None = None
    _strengths: dict = field(default_factory=dict, repr=False)
    _points: dict = field(default_factory=dict, repr=False)
    _profiles: dict = field(default_factory=dict, repr=False)

    @property
    def weeks(self) -> list[int]:
        """NFL week numbers of the remaining schedule, in order."""
        if self.current_week is None:
            return []
        length = max((len(games) for games in self.schedule.values()), default=0)
        return [self.current_week + offset for offset in range(length)]

    def _key(self, roster: list) -> tuple:
        return tuple(sorted(map(str, (_identity(p) for p in roster))))

    def points(self, roster: list) -> float:
        """Full-strength lineup points, the fast way.

        Identical to `strength(roster)[0]` — `tests/test_fast_lineup.py` holds
        the two to the same answer on randomised rosters — but about a thousand
        times cheaper, which is what lets the trade search go to full depth.
        Leagues where greedy filling is not provably exact use the solver.
        """
        key = self._key(roster)
        cached = self._points.get(key)
        if cached is None:
            if is_greedy_exact(self.settings):
                cached = best_points(roster, self.settings)
            else:
                cached = self.strength(roster)[0]
            self._points[key] = cached
        return cached

    def profile(self, roster: list) -> tuple[float, ...]:
        """Lineup points in each remaining week, with byes and injuries out."""
        if self.current_week is None or not is_greedy_exact(self.settings):
            return ()
        key = self._key(roster)
        cached = self._profiles.get(key)
        if cached is None:
            cached = weekly_points(roster, self.weeks, self.current_week, self.settings)
            self._profiles[key] = cached
        return cached

    @property
    def first_playoff_week(self) -> int | None:
        if self.current_week is None:
            return None
        regular = getattr(self.settings, "regular_season_weeks", None)
        if regular:
            return int(regular) + 1
        return (self.weeks[-1] + 1) if self.weeks else None

    def playoff_points(self, roster: list) -> float:
        """Full-strength points for the bracket, without anyone still injured then.

        A player out until week 8 is part of the playoff lineup; one out for the
        season is not, and counting him would overstate a team in exactly the
        games that decide the title.
        """
        first = self.first_playoff_week
        if first is None:
            return self.strength(roster)[0]
        healthy = [
            p for p in roster if getattr(p, "return_week", None) is None or p.return_week <= first
        ]
        if len(healthy) == len(roster):
            return self.strength(roster)[0]
        return self.points(healthy)

    def season_points(self, roster: list) -> float:
        """Total points over the remaining season, byes included.

        This is what the trade screen ranks on. A trade that is a point a week
        better at full strength but loses the week three starters are on bye is
        not the same trade as one that is a point a week better every week.
        """
        weekly = self.profile(roster)
        if weekly:
            return float(sum(weekly))
        return self.points(roster) * max(len(self.weeks), 1)

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
                    # Each week at its own strength; the bracket at full strength,
                    # since NFL byes are over by the fantasy playoffs — minus anyone
                    # whose injury runs past them.
                    weekly_means=self.profile(roster) or None,
                    playoff_mean=self.playoff_points(roster),
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
    counterparty_weekly_points_change: float = 0.0
    #: NFL week numbers the two profiles below refer to.
    weeks: tuple[int, ...] = ()
    #: Lineup points gained in each remaining week, byes and injuries included.
    weekly_delta: tuple[float, ...] = ()
    counterparty_weekly_delta: tuple[float, ...] = ()
    #: How often each incoming player would start for us, and in which weeks.
    incoming: tuple[PlayerUsage, ...] = ()
    #: The same for what we send, on the other side's roster. A proposal they
    #: refuse is worthless, and "your guy only covers one of my byes" is the most
    #: common reason to refuse — so it is measured the same way on both sides.
    outgoing: tuple[PlayerUsage, ...] = ()

    @property
    def season_points_change(self) -> float:
        return float(sum(self.weekly_delta))

    @property
    def counterparty_season_points_change(self) -> float:
        return float(sum(self.counterparty_weekly_delta))

    def worst_week(self, side: str = "us") -> tuple[int, float] | None:
        """The week this move hurts most, which is often a bye week."""
        delta = self.weekly_delta if side == "us" else self.counterparty_weekly_delta
        if not delta:
            return None
        index = min(range(len(delta)), key=lambda i: delta[i])
        return self.weeks[index], delta[index]

    def best_week(self, side: str = "us") -> tuple[int, float] | None:
        delta = self.weekly_delta if side == "us" else self.counterparty_weekly_delta
        if not delta:
            return None
        index = max(range(len(delta)), key=lambda i: delta[i])
        return self.weeks[index], delta[index]

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
    before_mean = context.points(rosters[my_team_id])

    updated = apply_move(rosters, my_team_id, add, drop, counterparty_id)
    after = context.outcome(updated)
    after_mean = context.points(updated[my_team_id])

    # Both sides, the same way. What matters to them is what matters to us: the
    # points, the weeks, and whether the players changing hands actually play.
    weeks = tuple(context.weeks)
    weekly_delta = _profile_delta(context, rosters[my_team_id], updated[my_team_id])
    counterparty_weekly_delta: tuple[float, ...] = ()
    counterparty_change = 0.0
    incoming: tuple[PlayerUsage, ...] = ()
    outgoing: tuple[PlayerUsage, ...] = ()
    if weeks:
        incoming = tuple(
            usage(p, updated[my_team_id], list(weeks), context.current_week, settings) for p in add
        )
    if counterparty_id is not None:
        counterparty_change = context.points(updated[counterparty_id]) - context.points(
            rosters[counterparty_id]
        )
        counterparty_weekly_delta = _profile_delta(
            context, rosters[counterparty_id], updated[counterparty_id]
        )
        if weeks:
            outgoing = tuple(
                usage(p, updated[counterparty_id], list(weeks), context.current_week, settings)
                for p in drop
            )

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
        counterparty_weekly_points_change=counterparty_change,
        weeks=weeks,
        weekly_delta=weekly_delta,
        counterparty_weekly_delta=counterparty_weekly_delta,
        incoming=incoming,
        outgoing=outgoing,
    )


def _profile_delta(context: SimulationContext, before: list, after: list) -> tuple[float, ...]:
    """Points gained in each remaining week by going from `before` to `after`."""
    a, b = context.profile(before), context.profile(after)
    if not a or not b:
        return ()
    return tuple(round(y - x, 4) for x, y in zip(a, b, strict=True))


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
    return context.points(roster) - context.points(without)


def marginal_gain(roster: list, player, context: SimulationContext) -> float:
    """Points a week this player would add to the roster, before any drop."""
    return context.points([*roster, player]) - context.points(roster)


def find_trades(
    my_team_id: int,
    rosters: dict[int, list],
    names: dict[int, str],
    settings,
    schedule: dict[int, list[int | None]],
    playoff_teams: int,
    banked: dict[int, tuple[int, int, float]] | None = None,
    give_depth: int | None = None,
    get_depth: int | None = None,
    trials: int = DEFAULT_TRIALS,
    context: SimulationContext | None = None,
    baseline: SeasonOutcome | None = None,
    package_sizes: tuple[int, ...] = (1, 2),
    shortlist: int = 40,
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

    # Screen on the whole remaining season, byes included, not one flat week. A
    # deal that is +2.3 a week at full strength but -3.7 the week three starters
    # are on bye is a different deal, and only the season total can rank them.
    weeks, current = context.weeks, context.current_week

    def season(roster: list) -> float:
        if weeks and is_greedy_exact(context.settings):
            return season_total(roster, weeks, current, context.settings)
        return context.points(roster)

    my_before = season(my_roster)

    # The whole roster by default. The pools used to be cut to the top few by
    # standalone value, which is blind to swaps: DK Metcalf adds nothing to a
    # lineup that already starts three better receivers, so he never entered the
    # pool — yet Metcalf *for* Nico Collins, whom he replaces, was the best deal
    # on the board. The package is scored as a whole below; the pool must not
    # pre-judge its members one at a time. Depth caps remain for callers who want
    # a quicker, narrower search.
    give_pool = sorted(my_roster, key=lambda p: marginal_cost(my_roster, p, context))
    if give_depth is not None:
        give_pool = give_pool[:give_depth]

    screened: list[tuple[float, None, int, list, list]] = []
    for team_id, roster in rosters.items():
        if team_id == my_team_id or not roster:
            continue
        their_before = season(roster)
        get_pool = sorted(roster, key=lambda p: marginal_gain(my_roster, p, context), reverse=True)
        if get_depth is not None:
            get_pool = get_pool[:get_depth]

        for size in package_sizes:
            if size > len(give_pool) or size > len(get_pool):
                continue
            give_packages = list(combinations(give_pool, size))
            for get in combinations(get_pool, size):
                get_ids = {id(p) for p in get}
                their_kept = [p for p in roster if id(p) not in get_ids]
                for give in give_packages:
                    give_ids = {id(p) for p in give}
                    my_after = [p for p in my_roster if id(p) not in give_ids] + list(get)
                    my_gain = season(my_after) - my_before
                    if my_gain <= 0:
                        continue
                    their_gain = season(their_kept + list(give)) - their_before
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
                        (min(my_gain, their_gain), None, team_id, list(get), list(give))
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
    return _distinct(accepted)


def _distinct(moves: list[MoveEvaluation]) -> list[MoveEvaluation]:
    """One entry per real deal, dropping copies that differ only by a throw-in.

    Packages are built at equal sizes, so a two-for-one arrives several times,
    each padded with a different player who would never start — the top four
    results in week 3 of 2026 were all Chase Brown for Collins and Javonte, with
    Caleb Williams, Zach Charbonnet, the 49ers defence or Patrick Mahomes
    attached. Those are the same trade. The best-scoring copy is kept.

    Without usage data there is nothing to judge a throw-in by, so every move is
    kept as its own deal.
    """
    kept: dict[tuple, MoveEvaluation] = {}
    for move in moves:
        if not move.incoming and not move.outgoing:
            kept[(id(move),)] = move
            continue
        core = (
            move.counterparty_id,
            frozenset(u.player for u in move.incoming if u.role != "depth"),
            frozenset(u.player for u in move.outgoing if u.role != "depth"),
        )
        if core not in kept or move.title_delta > kept[core].title_delta:
            kept[core] = move
    return sorted(kept.values(), key=lambda m: -m.title_delta)


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
