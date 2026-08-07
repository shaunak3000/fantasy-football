import pytest

from fantasy_football.data.espn import DST_POSITION_ID, ScoringRule, parse_settings
from fantasy_football.projections.scoring import ScoringEngine, find_double_counted_stats

# Stat ids used below, in ESPN's numbering.
PASS_YDS, PASS_TD, PASS_INT = 3, 4, 20
RUSH_YDS, RUSH_TD = 24, 25
RECEPTIONS, REC_YDS = 53, 42
PASS_YDS_DUP, RUSH_YDS_DUP = 22, 40
DST_YARDS_ALLOWED = 133

RAW = {
    "name": "Rice Ball",
    "size": 8,
    "rosterSettings": {"lineupSlotCounts": {"0": 1, "2": 2, "4": 2, "6": 1, "20": 7}},
    "scheduleSettings": {
        "matchupPeriodCount": 13,
        "playoffTeamCount": 4,
        "playoffMatchupPeriodLength": 1,
        "matchupPeriods": {},
        "playoffSeedingRule": "TOTAL_POINTS_SCORED",
    },
    "scoringSettings": {
        "scoringType": "H2H_POINTS",
        "scoringItems": [
            {"statId": PASS_YDS, "points": 0.04},
            {"statId": PASS_TD, "points": 4},
            {"statId": PASS_INT, "points": -2},
            {"statId": RUSH_YDS, "points": 0.1},
            {"statId": RUSH_TD, "points": 6},
            {"statId": RECEPTIONS, "points": 1},
            {"statId": REC_YDS, "points": 0.1},
            # Worth nothing to a skill player, -2 to a team defense.
            {"statId": DST_YARDS_ALLOWED, "points": 0, "pointsOverrides": {"16": -2.0}},
        ],
    },
    "draftSettings": {"keeperCount": 0},
    "acquisitionSettings": {"isUsingAcquisitionBudget": False, "acquisitionBudget": 0},
    "tradeSettings": {"vetoVotesRequired": 4},
}


@pytest.fixture
def engine():
    return ScoringEngine.from_settings(parse_settings(RAW, league_id=1, season=2026))


class TestScoring:
    def test_reproduces_a_real_stat_line(self, engine):
        """Josh Allen, 2025 week 3: 213 pass yds, 3 pass TD, 25 rush yds = 23.02."""
        stats = {PASS_YDS: 213.0, PASS_TD: 3.0, RUSH_YDS: 25.0}
        assert engine.score(stats) == pytest.approx(23.02)

    def test_full_ppr_counts_every_reception(self, engine):
        stats = {RECEPTIONS: 9.0, REC_YDS: 120.0}
        assert engine.score(stats) == pytest.approx(21.0)

    def test_negative_stats_subtract(self, engine):
        assert engine.score({PASS_INT: 2.0}) == pytest.approx(-4.0)

    def test_empty_stat_line_scores_zero(self, engine):
        assert engine.score({}) == 0.0

    def test_unscored_stats_contribute_nothing(self, engine):
        """Attempts and completions are reported but carry no rule."""
        assert engine.score({0: 28.0, 1: 22.0, 999: 5.0}) == 0.0

    def test_duplicate_stat_ids_are_not_double_counted(self, engine):
        """ESPN reports passing yards under both id 3 and id 22, scoring only 3."""
        stats = {PASS_YDS: 213.0, PASS_YDS_DUP: 213.0, RUSH_YDS: 25.0, RUSH_YDS_DUP: 25.0}
        assert engine.score(stats) == pytest.approx(213 * 0.04 + 25 * 0.1)


class TestPositionOverrides:
    def test_override_applies_only_to_the_named_position(self, engine):
        stats = {DST_YARDS_ALLOWED: 1.0}
        assert engine.score(stats, position_id=DST_POSITION_ID) == pytest.approx(-2.0)

    def test_skill_players_keep_the_base_value(self, engine):
        stats = {DST_YARDS_ALLOWED: 1.0}
        assert engine.score(stats, position_id=2) == 0.0
        assert engine.score(stats) == 0.0

    def test_rule_reports_itself_active_despite_zero_base(self):
        rule = ScoringRule(
            stat_id=133, abbr="YA449", label="x", points=0.0, position_overrides={16: -2.0}
        )
        assert rule.is_active
        assert rule.points_for(16) == -2.0
        assert rule.points_for(2) == 0.0
        assert rule.points_for() == 0.0

    def test_inactive_rule(self):
        assert not ScoringRule(stat_id=58, abbr="RET", label="x", points=0.0).is_active


class TestExplain:
    def test_lists_only_contributing_stats(self, engine):
        got = engine.explain({PASS_YDS: 213.0, PASS_TD: 3.0, 0: 28.0})
        assert got == {PASS_YDS: pytest.approx(8.52), PASS_TD: pytest.approx(12.0)}

    def test_contributions_sum_to_the_total(self, engine):
        stats = {PASS_YDS: 300.0, PASS_TD: 2.0, PASS_INT: 1.0, RUSH_YDS: 14.0}
        assert sum(engine.explain(stats).values()) == pytest.approx(engine.score(stats))


class TestDoubleCountGuard:
    def test_clean_league_reports_nothing(self, engine):
        assert find_double_counted_stats(engine) == []

    def test_detects_a_league_scoring_both_copies(self):
        raw = {
            **RAW,
            "scoringSettings": {
                **RAW["scoringSettings"],
                "scoringItems": [
                    {"statId": PASS_YDS, "points": 0.04},
                    {"statId": PASS_YDS_DUP, "points": 0.04},
                ],
            },
        }
        engine = ScoringEngine.from_settings(parse_settings(raw, 1, 2026))
        assert find_double_counted_stats(engine) == [(PASS_YDS, PASS_YDS_DUP)]
