# Plan — fantasy-football

Full-season management tool for an ESPN fantasy football family league: draft kit, weekly lineup optimizer, and waiver/trade evaluation, run live all season.

**The objective is not expected points.** It is P(finishing first). Those are different optimizations and they diverge exactly when it matters — a manager trailing in week 14 should be seeking variance, and a manager holding the top seed should be buying floor. Every recommendation in this repo is scored in units of championship probability, and the season simulator that produces that number is the spine everything else hangs off.

The framing is credit-risk portfolio management applied to a roster: a projection is an expected return, boom/bust is volatility, a starting lineup is a portfolio selected under a target quantile rather than a target mean, and P(first) is the analog of a default probability that every decision is priced against. That framing is the differentiator versus every other fantasy repo on GitHub.

League format: **snake draft**. Draft date not yet scheduled as of 2026-08-06 — build to mid-August readiness so the tool is never the blocker.

## Scope

- **In:** private-league ESPN ingest via cookie auth; league-settings-driven scoring engine; ensemble projections with per-player distributions; snake draft kit with opponent modeling and a live draft-night board; MILP weekly lineup optimizer; waiver and trade evaluation; season Monte Carlo producing P(first) per manager; backtest against the 2024 and 2025 seasons.
- **Out:** auction drafts (the league is snake; revisit only if it switches); DFS; sportsbook or prediction-market integration; other platforms (Yahoo, Sleeper); LLM components; anything that auto-submits a transaction without review.

## Action items

- [ ] 1. **Scaffold** — uv + Python 3.12, `src/fantasy_football/{data,projections,draft,lineup,transactions,season}`, `.env.example`, gitleaks pre-commit, pytest, ruff.
- [ ] 2. **ESPN client** (`data/espn.py`) — authenticated `espn-api` wrapper reading league settings, rosters, free agents, schedule, and prior-season draft results. Everything downstream reads scoring rules, roster slots, league size, and keeper rules from here; nothing about the league is hardcoded. Snapshot responses to `data/cache/` so the whole test suite runs offline.
- [ ] 3. **nflverse ingest** (`data/nflverse.py`) — `load_ff_rankings()` for FantasyPros consensus draft projections, `load_ff_opportunity()` for expected-fantasy-points, `load_player_stats()` and `load_snap_counts()` for history, `load_ff_playerids()` for cross-platform identity. Cache to parquet.
- [ ] 4. **ID bridge + scoring engine** (`data/ids.py`, `projections/scoring.py`) — join ESPN player IDs to nflverse IDs; convert any raw stat line into *this league's* points using the settings from step 2. **Gate:** recompute every 2025 box score from raw stats and match ESPN's published totals exactly. No projection is trusted until this passes.
- [ ] 5. **Ensemble projections** (`projections/`) — blend ESPN's own projections, FantasyPros consensus, and an opportunity-based model into a per-player mean *and variance*, weekly and rest-of-season. Backtest 2024–2025 on rank correlation and on interval calibration — the variance estimate is load-bearing for everything risk-aware, so it gets validated as hard as the mean.
- [ ] 6. **Draft kit** (`draft/`) — value over replacement computed against this league's actual starting requirements; positional scarcity curves; an opponent model fit to prior-season draft behavior and public ADP; a survival simulator answering "what is the chance this player is still there at my next pick?"; a draft-night board that ingests picks live and re-ranks. Rehearse against the 2025 draft before draft night.
- [ ] 7. **Weekly lineup optimizer** (`lineup/`) — MILP via PuLP over the league's real slot and eligibility rules. Objective is selected by matchup state: maximize expected points when evenly matched, maximize P(beating this specific opponent) when a heavy underdog or heavy favorite. Property tests guarantee it can never return an illegal lineup.
- [ ] 8. **Waivers + trades** (`transactions/`) — rest-of-season projections drive add/drop and trade evaluation, with every candidate move priced as a delta in P(first), not a delta in points. FAAB/waiver-priority aware.
- [ ] 9. **Season simulator** (`season/`) — Monte Carlo the remaining schedule using the projection distributions to produce P(first) per manager, plus playoff seeding odds. This feeds the risk posture in steps 7 and 8 and is the repo's headline chart.
- [ ] 10. **Weekly run loop** — one command that refreshes data, regenerates projections, and emits a markdown report with the recommended lineup, waiver targets, and trade ideas, each annotated with its championship-probability impact.

## Validation

- The scoring engine reproduces 2025 ESPN box scores exactly before any projection is trusted (step 4 is a hard gate on steps 5–10).
- Projection intervals are calibration-checked on held-out 2025 weeks, not scored on RMSE alone.
- The optimizer is property-tested against the league's slot rules — an illegal lineup is a build failure, not a bug report.
- The draft kit is rehearsed end-to-end against the 2025 draft before it is used live.
- No recommendation ships without its P(first) delta attached.

## Open questions

1. Draft date not scheduled as of 2026-08-06 — mid-August readiness is the working target.
2. Keeper rules, scoring, and roster slots unknown until the 2026 league is created; all are read from the API rather than assumed.
3. Is the 2026 league created yet? Until it is, development runs against the prior season's league ID.
