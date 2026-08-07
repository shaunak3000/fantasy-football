# Plan — fantasy-football

Full-season management tool for an ESPN fantasy football family league: draft kit, weekly lineup optimizer, and waiver/trade evaluation, run live all season.

**The objective is not expected points.** It is P(finishing first). Those are different optimizations and they diverge exactly when it matters — a manager trailing in week 14 should be seeking variance, and a manager holding the top seed should be buying floor. Every recommendation in this repo is scored in units of championship probability, and the season simulator that produces that number is the spine everything else hangs off.

The framing is credit-risk portfolio management applied to a roster: a projection is an expected return, boom/bust is volatility, a starting lineup is a portfolio selected under a target quantile rather than a target mean, and P(first) is the analog of a default probability that every decision is priced against. That framing is the differentiator versus every other fantasy repo on GitHub.

## The league

League "Rice Ball" (id 1815614957), verified against the API on 2026-08-06:

- **8 teams**, snake draft, no keepers, waiver priority (no FAAB)
- **Full PPR**, 4pt passing TD, 0.04/passing yard, 0.1/rushing and receiving yard
- Starters (9): 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 D/ST, 1 K; 7 bench, 2 IR
- 13-week regular season, **4 of 8 make the playoffs**, 1-week playoff matchups

Three properties of this format drive the modeling and are easy to get wrong:

1. **Eight teams compresses positional scarcity.** Only 72 roster spots start each week, so replacement level is very high and the waiver wire stays deep all season — a streamed QB is worth about what QB8 is worth. The scarcity gradient that powers a normal VOR board is much flatter here, the real draft edge sits in RB/WR, and QB/TE/K/D/ST belong late. A board that recommends an early QB in this league is miscalibrated, not clever.
2. **Half the league makes the playoffs, and matchups are one week long.** P(first) ≈ P(top 4) × P(win semi) × P(win final), where qualifying is the cheap term and the title comes down to two single-game outcomes. Seeding is worth little beyond making it. So the regular season is played for **floor** — just qualify — and weeks 14–15 are played for **ceiling**. Season-long expected points is the wrong objective in this format, which is the whole reason the P(first) simulator is the spine rather than a nice extra.
3. **Waiver priority, not FAAB.** The transactions problem is sequential — when to spend a priority position — not a budget allocation.

Draft date not yet scheduled as of 2026-08-06 — build to mid-August readiness so the tool is never the blocker.

## Scope

- **In:** private-league ESPN ingest via cookie auth; league-settings-driven scoring engine; ensemble projections with per-player distributions; snake draft kit with opponent modeling and a live draft-night board; MILP weekly lineup optimizer; waiver and trade evaluation; season Monte Carlo producing P(first) per manager; backtest against the 2024 and 2025 seasons.
- **Out:** auction drafts (the league is snake; revisit only if it switches); DFS; sportsbook or prediction-market integration; other platforms (Yahoo, Sleeper); LLM components; anything that auto-submits a transaction without review.

## Action items

- [ ] 1. **Scaffold** — uv + Python 3.12, `src/fantasy_football/{data,projections,draft,lineup,transactions,season}`, `.env.example`, gitleaks pre-commit, pytest, ruff.
- [ ] 2. **ESPN client** (`data/espn.py`) — authenticated `espn-api` wrapper reading league settings, rosters, free agents, schedule, and prior-season draft results. Everything downstream reads scoring rules, roster slots, league size, and keeper rules from here; nothing about the league is hardcoded. Snapshot responses to `data/cache/` so the whole test suite runs offline.
- [x] 3. **nflverse ingest** (`data/nflverse.py`) — `load_ff_rankings()` for FantasyPros consensus, `load_ff_opportunity()` for expected-fantasy-points, `load_player_stats()` and `load_snap_counts()` for history, `load_ff_playerids()` for cross-platform identity. Cached to parquet.
- [x] 4. **ID bridge** (`data/ids.py`) — ESPN ids joined to nflverse by normalized name, with uniqueness-guarded surname fallbacks and an explicit alias map. Ambiguous matches are refused rather than guessed, and every miss is reported. 100% coverage through ECR 200; `check_bridge` reports it.
- [x] 5. **Scoring engine** (`projections/scoring.py`) — converts any raw stat line into this league's points, driven entirely by the settings from step 2. **Gate passed:** 1,922/1,922 player-weeks in 2025 and 2,165/2,165 in 2024 reproduced exactly (`check_scoring`). Works in ESPN's stat-id space, not stat names, because `PLAYER_STATS_MAP` maps several ids onto one name (`passingYards` is both 3 and 22) and cannot be inverted; ESPN publishes the duplicate and scores only one copy, which is unambiguous by id. Per-position scoring overrides are preserved rather than collapsed.
- [x] 6. **Ensemble projections** (`projections/`) — per-player mean and variance for the season, built on 2,581 historical player-seasons (2021–2025).

      **How it works.** `load_ff_rankings("all")` turned out to carry every consensus board back to 2019, including a preseason snapshot each August. That makes it possible to fit *preseason rank → actual outcome* directly, on five seasons, rather than the naive *final rank → points* curve. The distinction is not cosmetic: final rank is an order statistic, so it describes the player who happened to finish RB5 and is unachievable in expectation by anyone you can actually draft.

      The two sources are blended in **rank** space, not points space. ESPN projects healthy full seasons and so runs systematically high; averaging its point totals against a historically calibrated number would import that optimism. Instead both sources are reduced to a within-position ordering, blended 65/35 toward consensus, and mapped through the rank curve once — calibration happens in exactly one place.

      Variance comes from the same fit: the spread at each rank is the standard deviation of what players ranked there actually did, including the 11% who never played a snap. Those are deliberately kept as zeros, since a first-rounder tearing an ACL is precisely the downside the number exists to capture.

      Remaining: interval calibration on held-out seasons, and a weekly (not just season-long) projection for the lineup optimizer.
- [ ] 7. **Draft kit** (`draft/`) — value over replacement computed against this league's actual starting requirements; positional scarcity curves; an opponent model fit to the 2024 and 2025 drafts plus public ADP; a survival simulator answering "what is the chance this player is still there at my next pick?"; a draft-night board that ingests picks live and re-ranks. Rehearse against the 2025 draft before draft night.
- [ ] 8. **Weekly lineup optimizer** (`lineup/`) — MILP via PuLP over the league's real slot and eligibility rules. Objective is selected by matchup state: maximize expected points when evenly matched, maximize P(beating this specific opponent) when a heavy underdog or heavy favorite. Property tests guarantee it can never return an illegal lineup.
- [ ] 9. **Waivers + trades** (`transactions/`) — rest-of-season projections drive add/drop and trade evaluation, with every candidate move priced as a delta in P(first), not a delta in points. Waiver-priority aware (this league does not use FAAB).
- [ ] 10. **Season simulator** (`season/`) — Monte Carlo the remaining schedule using the projection distributions to produce P(first) per manager, plus playoff seeding odds. This feeds the risk posture in steps 8 and 9 and is the repo's headline chart.
- [ ] 11. **Weekly run loop** — one command that refreshes data, regenerates projections, and emits a markdown report with the recommended lineup, waiver targets, and trade ideas, each annotated with its championship-probability impact.

## Validation

- The scoring engine reproduces 2025 ESPN box scores exactly before any projection is trusted (step 5 is a hard gate on steps 6–11).
- The ID bridge refuses ambiguous matches rather than guessing, and reports every miss — a wrong player id silently corrupts everything built on it, while a missing one is visible.
- Projection intervals are calibration-checked on held-out 2025 weeks, not scored on RMSE alone.
- The optimizer is property-tested against the league's slot rules — an illegal lineup is a build failure, not a bug report.
- The draft kit is rehearsed end-to-end against the 2025 draft before it is used live.
- No recommendation ships without its P(first) delta attached.

## Open questions

1. Draft date not scheduled as of 2026-08-06 — mid-August readiness is the working target.
2. Does the flat scarcity gradient in an 8-team league leave enough draft-day edge to be worth the modeling, or does the real advantage sit in weekly lineups and waivers? Worth measuring explicitly in the step 7 rehearsal rather than assuming.
3. FantasyPros ECR carries no scoring-format variant in the nflverse feed, but this league is full PPR, which shifts RB/WR balance materially. Quantify how far the generic board sits from a PPR-correct one before leaning on it — the rank→points curve in step 6 is fit on this league's own scoring, so this may resolve itself.

**Resolved:** the league has 2024 and 2025 history reachable through the API (the web UI hides it from members who joined later), so both the box-score reconciliation gate and the draft opponent model have real data. 2024 ran 9 teams and a 14-week season; 2025 and 2026 are identical 8-team, 13-week, full-PPR configurations, so anything fit across both seasons must normalize for size.
