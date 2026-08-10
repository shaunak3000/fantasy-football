"""Replay a past draft and score every strategy against the humans.

    uv run python -m fantasy_football.check_rehearsal [season]

Everything is fit only on seasons before the one replayed, so nothing any
strategy knows was unknowable in August of that year.
"""

from __future__ import annotations

import sys

from .config import load_credentials
from .draft.rehearse import rehearse

STRATEGIES = [
    ("adp", "pure best-available"),
    ("need", "best available at need"),
    ("tier", "tier-aware"),
    ("board", "monte carlo board"),
    ("rollout", "full-draft rollout"),
    ("champ", "rollout, championship objective"),
    ("human", "what the humans did"),
]


def stats(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return mean, (variance**0.5) / (n**0.5) if n else 0.0


def main(argv: list[str]) -> int:
    season = int(argv[0]) if argv else 2025

    creds = load_credentials()
    print(f"Replaying the {season} draft. Projections fit on 2021-{season - 1};")
    print(f"opponent behaviour fit on drafts before {season}.\n")

    results = rehearse(season, creds)
    names = [name for name, _ in STRATEGIES]

    print("  Starting-lineup points, by draft slot")
    print("  slot " + "".join(f"{name:>10}" for name in names))
    for result in results:
        row = f"  {result.slot:>4} " + "".join(f"{result.total(name):>10.0f}" for name in names)
        print(row)

    print("\n  " + "-" * 62)
    print(f"  {'strategy':<32}{'mean':>9}{'vs need':>10}{'vs human':>11}")
    baseline = [r.total("need") for r in results]

    ranked = []
    for name, label in STRATEGIES:
        mean, _ = stats([r.total(name) for r in results])
        deltas = [r.total(name) - b for r, b in zip(results, baseline, strict=True)]
        vs_need, se_need = stats(deltas)
        vs_human, _ = stats([r.edge(name) for r in results])
        ranked.append((mean, name, label, vs_need, se_need, vs_human))

    for mean, name, label, vs_need, se_need, vs_human in sorted(ranked, reverse=True):
        if name == "need":
            print(f"  {label:<32}{mean:>9.0f}{'--':>10}{vs_human:>+11.0f}   (baseline)")
            continue
        sigma = abs(vs_need) / se_need if se_need else 0.0
        flag = "  *" if (sigma >= 2 and vs_need > 0) else ("  x" if sigma >= 2 else "   ")
        print(
            f"  {label:<32}{mean:>9.0f}{vs_need:>+10.0f}{vs_human:>+11.0f}{flag} ({sigma:.1f} SE)"
        )

    print("\n  * beats the baseline by 2+ standard errors;  x loses by 2+.")
    print("  Scored on actual points that season, best legal lineup, QB/RB/WR/TE only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
