"""Convert a stat line into this league's fantasy points.

The engine works in ESPN's stat-id space rather than in stat names. That is a
deliberate choice: espn-api's `PLAYER_STATS_MAP` maps several ids onto the same
name — `passingYards` is both id 3 and id 22, `receivingReceptions` is both 41
and 53 — so a name cannot be reversed to an id unambiguously. ESPN publishes the
duplicate ids alongside the originals and scores only one of each pair, which
resolves cleanly in id space (the unscored twin simply has no rule) and not at
all in name space.

Everything comes from the league's own settings. There is no football knowledge
hardcoded here — change the league's scoring and this engine follows.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..data.espn import LeagueSettings, ScoringRule


@dataclass(frozen=True)
class ScoringEngine:
    rules: Mapping[int, ScoringRule]

    @classmethod
    def from_settings(cls, settings: LeagueSettings) -> ScoringEngine:
        return cls(rules={rule.stat_id: rule for rule in settings.scoring_rules})

    def points_for_stat(self, stat_id: int, position_id: int | None = None) -> float:
        rule = self.rules.get(stat_id)
        return 0.0 if rule is None else rule.points_for(position_id)

    def score(self, stats: Mapping[int, float], position_id: int | None = None) -> float:
        return sum(
            value * self.points_for_stat(stat_id, position_id) for stat_id, value in stats.items()
        )

    def explain(
        self, stats: Mapping[int, float], position_id: int | None = None
    ) -> dict[int, float]:
        """Per-stat point contributions, omitting stats worth nothing.

        Used to localize a mismatch to the offending stat rather than reporting
        only that a total came out wrong.
        """
        contributions = {}
        for stat_id, value in stats.items():
            points = value * self.points_for_stat(stat_id, position_id)
            if points:
                contributions[stat_id] = points
        return contributions

    @property
    def active_stat_ids(self) -> set[int]:
        return {sid for sid, rule in self.rules.items() if rule.is_active}


# ESPN emits some stats twice under different ids and scores only one copy. If a
# league ever scored both, every affected player would silently double-count.
DUPLICATE_STAT_IDS = ((3, 22), (24, 40), (41, 53), (42, 61), (120, 187), (205, 206))


def find_double_counted_stats(engine: ScoringEngine) -> list[tuple[int, int]]:
    active = engine.active_stat_ids
    return [pair for pair in DUPLICATE_STAT_IDS if pair[0] in active and pair[1] in active]
