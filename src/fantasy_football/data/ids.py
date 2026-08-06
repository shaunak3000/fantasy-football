"""ESPN <-> nflverse player identity bridge.

Fantasy sources agree on players and disagree on how to spell them: FantasyPros
publishes "Ja'Marr Chase", nflverse stores "jamarr chase", ESPN has its own id
space entirely. Nothing in this repo can ensemble two sources until a player
means the same thing in both, so all name matching funnels through here — and
every unmatched player is reported rather than silently dropped, because a
silent drop looks exactly like a player who simply isn't ranked.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import polars as pl

from .nflverse import load_player_ids

# Generational suffixes are inconsistently included across sources.
_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})

# Players whose public name is not a shortening of their roster name, so no
# amount of string normalization will connect the two. Keys and values are both
# already-normalized. Kept explicit rather than fuzzy-matched: these are few, and
# a wrong guess here silently corrupts a player's whole projection.
NICKNAME_ALIASES = {
    "hollywood brown": "marquise brown",
    "kenny gainwell": "kenneth gainwell",
    "chig okonkwo": "chigoziem okonkwo",
    "mitch tinsley": "mitchell tinsley",
    "gabe davis": "gabriel davis",
    "josh palmer": "joshua palmer",
    "cam ward": "cameron ward",
    "mike thomas": "michael thomas",
    "chris rodriguez": "christopher rodriguez",
    "will shipley": "william shipley",
}

# Team defenses have no entry in the player id table, so they match on team
# abbreviation instead. These are the abbreviations that differ between sources.
TEAM_ALIASES = {
    "WSH": "WAS",
    "JAC": "JAX",
    "LA": "LAR",
    "STL": "LAR",
    "SD": "LAC",
    "OAK": "LV",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
}


def normalize_name(name: str | None) -> str:
    """Collapse a player name to a comparable key.

    Applied to both sides of every join rather than trusting any source's own
    normalization, so the two sides cannot drift apart.
    """
    if not name:
        return ""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    stripped = re.sub(r"[^a-z0-9\s]", "", ascii_name.lower())
    parts = [p for p in stripped.split() if p not in _SUFFIXES]
    key = " ".join(parts)
    return NICKNAME_ALIASES.get(key, key)


def normalize_team(team: str | None) -> str:
    if not team:
        return ""
    upper = team.strip().upper()
    return TEAM_ALIASES.get(upper, upper)


@dataclass(frozen=True)
class MatchReport:
    """Outcome of a join, kept alongside the data so coverage is never assumed."""

    matched: pl.DataFrame
    unmatched: pl.DataFrame
    total: int

    @property
    def match_rate(self) -> float:
        return 0.0 if self.total == 0 else (self.total - self.unmatched.height) / self.total

    def summary(self) -> str:
        n = self.total - self.unmatched.height
        return f"{n}/{self.total} matched ({100 * self.match_rate:.1f}%)"


def surname(name: str | None) -> str:
    """Last token of a normalized name.

    Sources disagree on given names far more than surnames — FantasyPros says
    "Hollywood Brown" and "Kenny Gainwell" where nflverse says "Marquise Brown"
    and "Kenneth Gainwell". Surname survives the nickname; the given name does not.
    """
    parts = normalize_name(name).split()
    return parts[-1] if parts else ""


def build_bridge(refresh: bool = False) -> pl.DataFrame:
    """Identity table keyed by normalized name, one row per player."""
    ids = load_player_ids(refresh=refresh)

    bridge = (
        ids.select(
            ["name", "position", "team", "espn_id", "gsis_id", "fantasypros_id", "sleeper_id"]
        )
        .with_columns(
            pl.col("name").map_elements(normalize_name, return_dtype=pl.String).alias("norm_name"),
            pl.col("name").map_elements(surname, return_dtype=pl.String).alias("surname"),
            pl.col("team").map_elements(normalize_team, return_dtype=pl.String).alias("norm_team"),
        )
        .filter(pl.col("norm_name") != "")
    )

    # Prefer rows that actually carry an ESPN id, so a duplicate name does not
    # resolve to the copy that would be useless downstream.
    return (
        bridge.with_columns(pl.col("espn_id").is_null().alias("_no_espn"))
        .sort(["norm_name", "position", "_no_espn"])
        .unique(subset=["norm_name", "position"], keep="first")
        .drop("_no_espn")
    )


def attach_espn_ids(
    frame: pl.DataFrame,
    name_col: str = "player",
    pos_col: str = "pos",
    team_col: str = "team",
    refresh: bool = False,
) -> MatchReport:
    """Attach `espn_id` and `gsis_id` to any frame carrying player names.

    Matches on normalized name plus position first, then falls back to name alone
    for players whose position disagrees between sources (a TE listed as WR, say).
    Team defenses resolve by team abbreviation, since they have no player row.
    """
    bridge = build_bridge(refresh=refresh)

    keyed = frame.with_columns(
        pl.col(name_col).map_elements(normalize_name, return_dtype=pl.String).alias("norm_name"),
        pl.col(pos_col).cast(pl.String).str.to_uppercase().alias("norm_pos"),
        pl.col(team_col).map_elements(normalize_team, return_dtype=pl.String).alias("norm_team"),
    )

    on_name_pos = bridge.select(
        ["norm_name", "position", "espn_id", "gsis_id", "fantasypros_id"]
    ).rename({"position": "norm_pos"})

    matched = keyed.join(on_name_pos, on=["norm_name", "norm_pos"], how="left")

    # Name-only fallback, restricted to names that are unambiguous in the bridge.
    unique_names = (
        bridge.group_by("norm_name")
        .agg(
            pl.len().alias("n"),
            pl.first("espn_id"),
            pl.first("gsis_id"),
            pl.first("fantasypros_id"),
        )
        .filter(pl.col("n") == 1)
        .drop("n")
        .rename({"espn_id": "espn_id_fb", "gsis_id": "gsis_id_fb", "fantasypros_id": "fp_id_fb"})
    )

    matched = (
        matched.join(unique_names, on="norm_name", how="left")
        .with_columns(
            pl.coalesce(["espn_id", "espn_id_fb"]).alias("espn_id"),
            pl.coalesce(["gsis_id", "gsis_id_fb"]).alias("gsis_id"),
            pl.coalesce(["fantasypros_id", "fp_id_fb"]).alias("fantasypros_id"),
        )
        .drop(["espn_id_fb", "gsis_id_fb", "fp_id_fb"])
    )

    # Third tier: surname + team + position, which survives nicknames. Restricted
    # to combinations that resolve to exactly one player — an ambiguous surname is
    # left unmatched deliberately, because a wrong id is far worse than a missing
    # one. A missing player is visible in the report; a mismatched player quietly
    # poisons every projection built on top of it.
    matched = matched.with_columns(
        pl.col(name_col).map_elements(surname, return_dtype=pl.String).alias("surname")
    )

    # Team first, then position alone — sources also disagree about which team a
    # player is on, so the team-qualified pass catches the clean cases and the
    # looser pass rescues players who changed teams between the two snapshots.
    for keys in (["surname", "norm_team", "norm_pos"], ["surname", "norm_pos"]):
        bridge_keys = [k if k != "norm_pos" else "position" for k in keys]
        candidates = (
            bridge.group_by(bridge_keys)
            .agg(
                pl.len().alias("n"),
                pl.first("espn_id").alias("espn_id_sn"),
                pl.first("gsis_id").alias("gsis_id_sn"),
                pl.first("fantasypros_id").alias("fp_id_sn"),
            )
            .filter(pl.col("n") == 1)
            .drop("n")
        )
        if "position" in bridge_keys:
            candidates = candidates.rename({"position": "norm_pos"})

        matched = (
            matched.join(candidates, on=keys, how="left")
            .with_columns(
                pl.coalesce(["espn_id", "espn_id_sn"]).alias("espn_id"),
                pl.coalesce(["gsis_id", "gsis_id_sn"]).alias("gsis_id"),
                pl.coalesce(["fantasypros_id", "fp_id_sn"]).alias("fantasypros_id"),
            )
            .drop(["espn_id_sn", "gsis_id_sn", "fp_id_sn"])
        )

    unmatched = matched.filter(pl.col("espn_id").is_null() & (pl.col("norm_pos") != "DST"))

    return MatchReport(matched=matched, unmatched=unmatched, total=frame.height)
