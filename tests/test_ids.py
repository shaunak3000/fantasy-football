import polars as pl
import pytest

from fantasy_football.data import ids as ids_mod
from fantasy_football.data.ids import attach_espn_ids, normalize_name, normalize_team, surname

BRIDGE_ROWS = [
    # name, position, team, espn_id, gsis_id, fantasypros_id, sleeper_id
    ("Ja'Marr Chase", "WR", "CIN", 1, "g1", 11, 111),
    ("Amon-Ra St. Brown", "WR", "DET", 2, "g2", 12, 112),
    ("James Cook III", "RB", "BUF", 3, "g3", 13, 113),
    ("Kenneth Gainwell", "RB", "PHI", 4, "g4", 14, 114),
    ("Marquise Brown", "WR", "KC", 5, "g5", 15, 115),
    # Two same-surname WRs on one team: any surname-based match here is ambiguous.
    ("Noah Wilson", "WR", "NYJ", 6, "g6", 16, 116),
    ("Cedric Wilson", "WR", "NYJ", 7, "g7", 17, 117),
]


@pytest.fixture(autouse=True)
def fake_bridge(monkeypatch):
    frame = pl.DataFrame(
        BRIDGE_ROWS,
        schema=["name", "position", "team", "espn_id", "gsis_id", "fantasypros_id", "sleeper_id"],
        orient="row",
    )
    monkeypatch.setattr(ids_mod, "load_player_ids", lambda refresh=False: frame)


def board(rows):
    return pl.DataFrame(rows, schema=["player", "pos", "team"], orient="row")


class TestNormalizeName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Ja'Marr Chase", "jamarr chase"),
            ("A.J. Brown", "aj brown"),
            ("Amon-Ra St. Brown", "amonra st brown"),
            ("James Cook III", "james cook"),
            ("Marvin Harrison Jr.", "marvin harrison"),
            ("  Puka   Nacua  ", "puka nacua"),
            ("JAXON SMITH-NJIGBA", "jaxon smithnjigba"),
        ],
    )
    def test_normalizes(self, raw, expected):
        assert normalize_name(raw) == expected

    def test_empty_and_none(self):
        assert normalize_name(None) == ""
        assert normalize_name("") == ""

    def test_is_idempotent(self):
        once = normalize_name("Amon-Ra St. Brown")
        assert normalize_name(once) == once

    def test_applies_nickname_alias(self):
        assert normalize_name("Hollywood Brown") == "marquise brown"
        assert normalize_name("Kenny Gainwell") == "kenneth gainwell"


class TestNormalizeTeam:
    def test_aliases_divergent_abbreviations(self):
        assert normalize_team("WSH") == "WAS"
        assert normalize_team("JAC") == "JAX"
        assert normalize_team("OAK") == "LV"

    def test_passes_through_and_uppercases(self):
        assert normalize_team("cin") == "CIN"
        assert normalize_team(None) == ""


def test_surname_takes_last_token_ignoring_suffix():
    assert surname("James Cook III") == "cook"
    assert surname("Ja'Marr Chase") == "chase"
    assert surname("") == ""


class TestAttachEspnIds:
    def test_exact_name_and_position(self):
        r = attach_espn_ids(board([("Ja'Marr Chase", "WR", "CIN")]))
        assert r.matched["espn_id"].to_list() == [1]
        assert r.unmatched.height == 0

    def test_punctuation_and_suffix_differences(self):
        r = attach_espn_ids(board([("James Cook", "RB", "BUF"), ("Amon-Ra St Brown", "WR", "DET")]))
        assert r.matched["espn_id"].to_list() == [3, 2]

    def test_nickname_resolves_through_alias(self):
        r = attach_espn_ids(board([("Hollywood Brown", "WR", "KC")]))
        assert r.matched["espn_id"].to_list() == [5]

    def test_surname_rescues_player_on_a_stale_team(self):
        """FantasyPros has Gainwell on TB, the id table still says PHI."""
        r = attach_espn_ids(board([("Kenneth Gainwell", "RB", "TB")]))
        assert r.matched["espn_id"].to_list() == [4]

    def test_ambiguous_surname_is_left_unmatched_not_guessed(self):
        r = attach_espn_ids(board([("Zach Wilson", "WR", "NYJ")]))
        assert r.matched["espn_id"].to_list() == [None]
        assert r.unmatched.height == 1

    def test_unknown_player_is_reported(self):
        r = attach_espn_ids(board([("Nobody Atall", "RB", "FA")]))
        assert r.unmatched.height == 1
        assert r.match_rate == 0.0

    def test_team_defenses_excluded_from_unmatched(self):
        r = attach_espn_ids(board([("Houston Texans", "DST", "HOU")]))
        assert r.unmatched.height == 0

    def test_report_counts_every_input_row(self):
        r = attach_espn_ids(board([("Ja'Marr Chase", "WR", "CIN"), ("Nobody Atall", "RB", "FA")]))
        assert r.total == 2
        assert r.match_rate == 0.5
        assert "1/2 matched" in r.summary()
