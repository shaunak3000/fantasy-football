"""Replay the 2025 draft and score the board against the humans.

    uv run python -m fantasy_football.check_rehearsal [season]

Fit only on seasons before the one replayed, so nothing the board knows was
unknowable in August of that year.
"""

from __future__ import annotations

import sys

from .config import load_credentials
from .draft.rehearse import rehearse


def main(argv: list[str]) -> int:
    season = int(argv[0]) if argv else 2025

    creds = load_credentials()
    print(f"Replaying the {season} draft with projections fit on 2021-{season - 1}")
    print(f"and opponent behaviour fit on drafts before {season}.\n")

    results = rehearse(season, creds)

    header = (
        f"  {'slot':>5} {'board':>9} {'adp':>9} {'adp+need':>9} {'human':>9} "
        f"{'vs adp':>8} {'vs human':>9}"
    )
    print(header)
    for result in results:
        print(
            f"  {result.slot:>5} {result.tool.total:>9.1f} "
            f"{(result.adp.total if result.adp else 0):>9.1f} "
            f"{(result.adp_needs.total if result.adp_needs else 0):>9.1f} "
            f"{result.human.total:>9.1f} "
            f"{result.edge_over_adp:>+8.1f} {result.edge:>+9.1f}"
        )

    def stats(values: list[float]) -> tuple[float, float, int]:
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
        return mean, (variance**0.5) / (n**0.5) if n else 0.0, sum(1 for v in values if v > 0)

    print("\n  The comparison that decides whether the modelling earns its keep:")
    for label, values in (
        ("vs pure ADP bot     ", [r.edge_over_adp for r in results]),
        ("vs ADP + roster caps", [r.edge_over_adp_needs for r in results]),
        ("vs the humans       ", [r.edge for r in results]),
    ):
        mean, stderr, wins_here = stats(values)
        sigma = abs(mean) / stderr if stderr else 0.0
        print(
            f"    {label}: {mean:>+8.1f} +/- {stderr:>5.0f}  "
            f"({wins_here}/{len(values)} slots, {sigma:.1f} SE from zero)"
        )

    edges = [r.edge for r in results]
    wins = sum(1 for e in edges if e > 0)
    mean_edge = sum(edges) / len(edges)

    # One season of fantasy outcomes is mostly luck, so the spread across slots
    # matters as much as the average. Without it a reader cannot tell a real
    # deficit from a coin landing badly eight times.
    n = len(edges)
    variance = sum((e - mean_edge) ** 2 for e in edges) / (n - 1) if n > 1 else 0.0
    sd = variance**0.5
    stderr = sd / (n**0.5) if n else 0.0

    print(f"\n  Board beat the human roster in {wins}/{n} slots")
    print(f"  Mean edge: {mean_edge:+.1f} points over the season ({mean_edge / 17:+.2f}/week)")
    print(f"  Spread across slots: sd {sd:.0f}, standard error {stderr:.0f}")
    if stderr:
        print(f"  Mean is {abs(mean_edge) / stderr:.1f} standard errors from zero")
    print(
        f"  Rough 95% range for the true edge: {mean_edge - 2 * stderr:+.0f} "
        f"to {mean_edge + 2 * stderr:+.0f}"
    )

    best = max(results, key=lambda r: r.edge)
    worst = min(results, key=lambda r: r.edge)
    print(f"\n  Best slot  {best.slot}: {best.edge:+.1f}")
    print(f"    board: {best.tool.summary()}")
    print(f"    human: {best.human.summary()}")
    print(f"\n  Worst slot {worst.slot}: {worst.edge:+.1f}")
    print(f"    board: {worst.tool.summary()}")
    print(f"    human: {worst.human.summary()}")

    print("\n  Scored on actual points that season, best legal lineup, QB/RB/WR/TE only.")
    print("  Kickers and defenses are excluded — nobody predicts them, so including")
    print("  them would add noise without testing anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
