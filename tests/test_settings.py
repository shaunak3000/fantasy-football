from fantasy_football.data.espn import parse_settings

# Slot ids deliberately out of order: espn-api's own parser maps these
# positionally, so this payload is what would silently mislabel the roster.
RAW = {
    "name": "Family League",
    "size": 10,
    "rosterSettings": {
        "lineupSlotCounts": {
            "4": 2,  # WR
            "0": 1,  # QB
            "20": 7,  # BE
            "2": 2,  # RB
            "6": 1,  # TE
            "23": 1,  # FLEX
            "17": 1,  # K
            "16": 1,  # D/ST
            "21": 1,  # IR
            "1": 0,  # TQB, unused
        }
    },
    "scheduleSettings": {
        "matchupPeriodCount": 14,
        "playoffTeamCount": 6,
        "playoffMatchupPeriodLength": 1,
        "matchupPeriods": {},
        "playoffSeedingRule": "TOTAL_POINTS_SCORED",
    },
    "scoringSettings": {
        "scoringType": "H2H_POINTS",
        "scoringItems": [
            {"statId": 53, "points": 0.5},  # each reception
            {"statId": 42, "points": 0.1},  # receiving yards
            {"statId": 43, "points": 6},  # receiving TD
            {"statId": 58, "points": 0},  # targets, off
        ],
    },
    "draftSettings": {"keeperCount": 2},
    "acquisitionSettings": {"isUsingAcquisitionBudget": True, "acquisitionBudget": 100},
    "tradeSettings": {"vetoVotesRequired": 4},
}


def settings():
    return parse_settings(RAW, league_id=1815614957, season=2026)


def test_slots_map_by_id_not_position():
    s = settings()
    assert s.roster_slots == {
        "WR": 2,
        "QB": 1,
        "BE": 7,
        "RB": 2,
        "TE": 1,
        "RB/WR/TE": 1,
        "K": 1,
        "D/ST": 1,
        "IR": 1,
    }


def test_zero_count_slots_are_dropped():
    assert "TQB" not in settings().roster_slots


def test_bench_and_ir_excluded_from_starters():
    s = settings()
    assert "BE" not in s.starting_slots
    assert "IR" not in s.starting_slots
    assert s.starters_per_team == 9
    assert s.bench_size == 7
    assert s.ir_slots == 1


def test_ppr_detected_from_reception_stat():
    s = settings()
    assert s.points_per_reception == 0.5
    assert s.scoring_label == "0.5 PPR"


def test_standard_scoring_when_receptions_score_nothing():
    raw = {**RAW, "scoringSettings": {**RAW["scoringSettings"], "scoringItems": []}}
    assert parse_settings(raw, 1, 2026).scoring_label == "standard"


def test_league_shape_fields():
    s = settings()
    assert s.team_count == 10
    assert s.keeper_count == 2
    assert s.playoff_team_count == 6
    assert s.regular_season_weeks == 14
    assert s.uses_faab is True
    assert s.acquisition_budget == 100


def test_scoring_rules_carry_labels():
    rules = {r.abbr: r for r in settings().scoring_rules}
    assert rules["REC"].points == 0.5
    assert rules["RETD"].label == "TD Reception"
