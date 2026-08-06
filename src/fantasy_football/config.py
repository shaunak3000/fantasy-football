"""Environment-backed configuration for league access and local paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

_CREDENTIALS_HELP = (
    "Missing ESPN credentials. Copy .env.example to .env and fill in "
    "ESPN_LEAGUE_ID, ESPN_S2, and ESPN_SWID. See the README for how to pull "
    "the two cookies out of browser devtools."
)


@dataclass(frozen=True)
class EspnCredentials:
    league_id: int
    season: int
    espn_s2: str | None = None
    swid: str | None = None

    @property
    def is_private(self) -> bool:
        return bool(self.espn_s2 and self.swid)


def _clean(value: str | None) -> str | None:
    """Strip whitespace and stray quotes picked up when pasting out of devtools."""
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    return value or None


def _normalize_swid(swid: str | None) -> str | None:
    """ESPN rejects a SWID without its braces, which are easy to lose when copying."""
    if swid is None:
        return None
    return swid if swid.startswith("{") else "{" + swid.strip("{}") + "}"


def load_credentials(require_private: bool = True) -> EspnCredentials:
    load_dotenv(REPO_ROOT / ".env")

    league_id = _clean(os.getenv("ESPN_LEAGUE_ID"))
    if not league_id:
        raise RuntimeError(_CREDENTIALS_HELP)

    creds = EspnCredentials(
        league_id=int(league_id),
        season=int(_clean(os.getenv("ESPN_SEASON")) or 2026),
        espn_s2=_clean(os.getenv("ESPN_S2")),
        swid=_normalize_swid(_clean(os.getenv("ESPN_SWID"))),
    )

    if require_private and not creds.is_private:
        raise RuntimeError(_CREDENTIALS_HELP)

    return creds


def cache_path(*parts: str) -> Path:
    path = CACHE_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
