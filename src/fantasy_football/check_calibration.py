"""Backtest the projections season by season.

uv run python -m fantasy_football.check_calibration
"""

from __future__ import annotations

from espn_api.football import League

from .config import load_credentials
from .data.espn import fetch_raw_settings, parse_settings
from .projections.calibration import NOMINAL_COVERAGES, backtest
from .projections.history import training_table
from .projections.scoring import ScoringEngine

TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]


def main() -> int:
    creds = load_credentials()
    league = League(league_id=creds.league_id, year=2026, espn_s2=creds.espn_s2, swid=creds.swid)
    settings = parse_settings(fetch_raw_settings(league), creds.league_id, 2026)
    engine = ScoringEngine.from_settings(settings)

    training, _ = training_table(TRAIN_SEASONS, engine)
    report = backtest(training)

    print("Leave-one-season-out backtest — fit on the other four, scored on the held-out year\n")
    header = (
        f"{'season':>7} {'n':>5} {'MAE':>7} {'naive':>7} {'skill':>7} {'bias':>7} {'rank r':>7}"
    )
    header += "".join(f" {int(c * 100):>4}%" for c in NOMINAL_COVERAGES)
    print(header)

    for s in report.seasons:
        row = (
            f"{s.season:>7} {s.n:>5} {s.mae:>7.1f} {s.baseline_mae:>7.1f} "
            f"{s.skill:>6.1%} {s.bias:>7.1f} {s.spearman:>7.3f}"
        )
        row += "".join(f" {s.coverage[c]:>5.1%}" for c in NOMINAL_COVERAGES)
        print(row)

    print(f"\nMean rank correlation: {report.mean_spearman:.3f}")
    print(f"Mean skill vs naive:   {report.mean_skill:.1%}")

    print("\nInterval coverage (nominal -> actual):")
    for level, actual in report.overall_coverage.items():
        gap = actual - level
        verdict = (
            "well calibrated" if abs(gap) <= 0.05 else ("too wide" if gap > 0 else "TOO NARROW")
        )
        print(f"  {level:>5.0%} -> {actual:>6.1%}   ({gap:+.1%}, {verdict})")

    print("\nPIT histogram (flat = calibrated; peaked = overconfident):")
    freqs, edges = report.pit_histogram()
    for i, freq in enumerate(freqs):
        bar = "#" * int(round(freq * 200))
        print(f"  {edges[i]:.1f}-{edges[i + 1]:.1f} {freq:>6.1%} {bar}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
