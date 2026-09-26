"""ESPN's public NFL injury report: status and an estimated return date per player.

The fantasy league API never publishes a return date — only a status — which is
why injured reserve used to mean "back next week" and A.J. Brown, out until
November, was valued as a starter from week 4. ESPN's own injury page does
carry an estimated return for all ~390 injured players, keyed by the same player
id the fantasy API uses, so no name matching is needed.

It is not an API. The JSON endpoints behind it answer 403 and 404; the data is
the blob the page embeds as `window['__espnfitt__']`, read here as JSON rather
than scraped from the table. That makes it sturdier than HTML parsing but not
guaranteed: if ESPN changes the page, `parse_report` raises, the caller falls
back to the hand-maintained overrides and the IR floor, and the report says so.

Dates come without a year ("Nov 1"). August to December are this season; January
to July are the next calendar year, which in practice means after the fantasy
season — ESPN writes "Feb" for a season-ending injury.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime

from ..config import cache_path

REPORT_URL = "https://www.espn.com/nfl/injuries"
MAX_AGE_HOURS = 6.0
_BLOB = "window['__espnfitt__']="
_MONTHS = {
    m: i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
    )
}

#: ESPN logo codes that differ from nflverse's schedule codes.
TEAM_ALIASES = {"LAR": "LA", "WSH": "WAS"}

#: Report statuses whose return date is honoured. QUESTIONABLE is left alone on
#: purpose — those players mostly play, and the repo does not bench them.
DATED_STATUSES = frozenset({"Injured Reserve", "Out", "Doubtful", "Suspension"})


@dataclass(frozen=True)
class ReportEntry:
    espn_id: int
    name: str
    team: str
    status: str
    return_date: date | None
    description: str


def infer_date(text: str | None, season: int) -> date | None:
    """ "Nov 1" in the 2026 season is 2026-11-01; "Feb 10" is 2027-02-10."""
    match = re.fullmatch(r"\s*([A-Z][a-z]{2})\s+(\d{1,2})\s*", text or "")
    if not match or match.group(1) not in _MONTHS:
        return None
    month = _MONTHS[match.group(1)]
    year = season if month >= 8 else season + 1
    try:
        return date(year, month, int(match.group(2)))
    except ValueError:
        return None


def _team_code(logo: str | None) -> str | None:
    match = re.search(r"/nfl/500/([a-z]+)\.png", logo or "")
    if not match:
        return None
    code = match.group(1).upper()
    return TEAM_ALIASES.get(code, code)


def parse_report(html: str, season: int) -> list[ReportEntry]:
    """Every entry on the page. Raises ValueError if the page is not recognisable."""
    start = html.find(_BLOB)
    if start < 0:
        raise ValueError("injury page no longer embeds __espnfitt__")
    blob, _ = json.JSONDecoder().raw_decode(html[start + len(_BLOB) :])
    try:
        groups = blob["page"]["content"]["injuries"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"injury page layout changed: {exc}") from exc

    entries = []
    for group in groups:
        team = _team_code(group.get("logo"))
        for item in group.get("items", []):
            athlete = item.get("athlete") or {}
            found = re.search(r"/id/(\d+)", athlete.get("href") or "")
            if not found:
                continue
            entries.append(
                ReportEntry(
                    espn_id=int(found.group(1)),
                    name=athlete.get("name") or "?",
                    team=team or "",
                    status=item.get("statusDesc") or "",
                    return_date=infer_date(item.get("date"), season),
                    description=item.get("description") or "",
                )
            )
    return entries


def _download() -> str:
    import requests

    response = requests.get(REPORT_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    response.raise_for_status()
    return response.text


@dataclass(frozen=True)
class Report:
    entries: dict[int, ReportEntry]
    fetched_at: str
    from_cache: bool

    def describe(self) -> str:
        how = "cached" if self.from_cache else "fetched"
        return f"ESPN injury report, {how} {self.fetched_at}, {len(self.entries)} players"


def load_report(
    season: int, max_age_hours: float = MAX_AGE_HOURS, download=_download, now=time.time
) -> Report:
    """The injury report, from a cache no older than `max_age_hours` or freshly.

    Raises on any failure — network, layout, parsing — so the caller can decide
    to fall back rather than silently run on nothing.
    """
    path = cache_path(f"espn_injuries_{season}.json")
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if now() - cached.get("fetched_epoch", 0) < max_age_hours * 3600:
            return Report(_decode(cached["entries"]), cached["fetched_at"], True)

    entries = parse_report(download(), season)
    stamp = datetime.fromtimestamp(now()).strftime("%Y-%m-%d %H:%M")
    payload = {
        "fetched_epoch": now(),
        "fetched_at": stamp,
        "entries": [
            {**asdict(e), "return_date": e.return_date.isoformat() if e.return_date else None}
            for e in entries
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return Report({e.espn_id: e for e in entries}, stamp, False)


def _decode(rows: list[dict]) -> dict[int, ReportEntry]:
    out = {}
    for row in rows:
        when = date.fromisoformat(row["return_date"]) if row.get("return_date") else None
        entry = ReportEntry(**{**row, "return_date": when})
        out[entry.espn_id] = entry
    return out
