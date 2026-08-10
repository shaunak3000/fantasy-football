"""Precomputed draft-night bundle.

Fitting the projections and the opponent model takes the better part of a
minute and hits the network several times. That is fine on a Tuesday and
unacceptable with a 90-second pick clock running, so everything that does not
depend on live picks is computed once beforehand and loaded from disk.

Preparing ahead of time also means the draft still works if ESPN's projection
endpoints are slow or the wifi is bad when it matters.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..config import cache_path
from ..projections.build import ProjectionSet
from .model import PickModel

BUNDLE_VERSION = 1


@dataclass
class DraftBundle:
    version: int
    prepared_at: str
    season: int
    projections: ProjectionSet
    pick_model: PickModel

    @property
    def age_hours(self) -> float:
        prepared = datetime.fromisoformat(self.prepared_at)
        return (datetime.now(UTC) - prepared).total_seconds() / 3600.0

    @property
    def is_stale(self) -> bool:
        """Rankings move daily in August; a bundle older than a day is suspect."""
        return self.age_hours > 24.0


def bundle_path(season: int) -> Path:
    return cache_path(f"draft_bundle_{season}.pkl")


def save_bundle(season: int, projections: ProjectionSet, pick_model: PickModel) -> Path:
    bundle = DraftBundle(
        version=BUNDLE_VERSION,
        prepared_at=datetime.now(UTC).isoformat(),
        season=season,
        projections=projections,
        pick_model=pick_model,
    )
    path = bundle_path(season)
    path.write_bytes(pickle.dumps(bundle))
    return path


def load_bundle(season: int) -> DraftBundle | None:
    path = bundle_path(season)
    if not path.exists():
        return None
    try:
        bundle = pickle.loads(path.read_bytes())
    except Exception:
        return None
    if not isinstance(bundle, DraftBundle) or bundle.version != BUNDLE_VERSION:
        return None
    return bundle
