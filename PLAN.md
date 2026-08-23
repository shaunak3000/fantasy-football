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

**Draft: Friday 4 September 2026, 8:30 PM ET** (room opens 7:30 PM ET). 90 seconds per pick, 16 rounds, snake, slot 8 of 8 — so picks come in back-to-back pairs separated by fourteen selections. Trade deadline 2 December.

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

      **Calibration (backtested).** Leave-one-season-out over 2021–2025: nominal 50/80/90% intervals cover 52.3/80.5/88.7% of held-out outcomes, all within 2.3 points of nominal. Mean rank correlation 0.748; the projections remove 44% of the error a naive positional-average baseline makes. Intervals are read off empirical quantiles rather than a normal approximation, because season outcomes are right-skewed with a lump of zeros. Caveat: the backtest validates the *consensus-only* projection — ESPN's historical preseason projections are not retrievable, so the blend inherits this calibration rather than proving its own.

      **Weekly projections** (`projections/weekly.py`) separate three things a season total conflates: production per game *played*, week-to-week spread, and availability (byes plus the background rate of missing time). All three are fit with the same rank-curve machinery. Relative volatility rises as rank worsens and is highest at TE (coefficient of variation 0.58 at TE1 vs 0.40 at QB1) — which is what the risk-aware lineup objective will trade on.
- [x] 7. **Draft kit** (`draft/`) — opponent model fit on 1,185 ranked players across the 2024 and 2025 drafts; snake-order bookkeeping; a Monte Carlo survival and lookahead engine; a board that re-ranks as picks come in.

      **The decision rule is two-pick value, not VOR.** Taking the best available player ignores what it costs you later. Each candidate is scored by simulating the picks between now and your next turn, then asking what the board looks like when you choose again — so a high-VOR player who will still be there in ten picks correctly loses to a slightly cheaper one who will not.

      **Two ranks are kept strictly separate**, and conflating them was a real bug caught in testing: *value* follows positional rank, but *draft position* follows overall rank. Josh Allen is QB1 and Ja'Marr Chase is WR1 — both "rank 1", drafted thirty picks apart. Survival is measured from the same simulation that drives the lookahead so the two can never disagree, and board position is recomputed among players still available, so a run on one position correctly moves everyone else up.

      **Live draft night** (`live_draft.py`, `draft/live.py`, `draft/cache.py`) requires no typing. ESPN pre-allocates all 128 pick slots and fills in `playerId` as picks are made, so `watch` polls `mDraftDetail`, re-ranks, and says what to take — you draft in ESPN exactly as normal. The draft slot is detected from the SWID against `pickOrder` rather than typed. `prepare` precomputes the whole bundle beforehand, so nothing slow or network-dependent runs while a 90-second clock is going; a live re-rank takes 0.08s. `manual` drives the same board by typing picks, for an offline draft or a dead feed. State is rebuilt from the feed each poll rather than appended to, so a missed poll self-heals.

      **The rehearsal overturned the design.** Replaying the real 2025 draft at all 8 slots, fit only on prior seasons, and scoring each resulting roster by what its players actually did:

      | strategy | mean roster | vs baseline |
      |----------|------------|-------------|
      | **best available at need** | **1793** | baseline |
      | full-draft rollout | 1783 | −10 (0.3 SE) |
      | best VOR at need | 1767 | −27 (1.1 SE) |
      | rollout, championship objective | 1745 | −49 (0.8 SE) |
      | the humans | 1712 | −81 |
      | tier-aware | 1704 | −89 |
      | monte carlo lookahead board | 1553 | −241 (5.2 SE) |
      | pure best-available | 1422 | −371 (7.5 SE) |

      Nothing beat "take the highest-ranked player at a position you still need". The elaborate board *lost* by 241 points at 5.2 standard errors. The rollout confirmed the diagnosis — judging a pick by the finished roster rather than the roster so far moved it from −241 to −10 — and then produced the same picks as the one-liner at hundreds of times the cost.

      **The projections contribute nothing to the draft.** Both the winner (1793) and pure best-available (1422) order by raw consensus rank — the entire 371-point gap between them is the roster-cap filter, not any model. And `best VOR at need` losing to `best available at need` says the same thing from the other side: transforming consensus rank into our own value estimate *destroys* information, because ECR already aggregates injury news, depth charts and holdouts that a mechanical rank→points curve cannot recover.

      So the draft is settled: consensus rank plus roster discipline, with the simulation demoted to supplying survival odds for context (`board_view` separates information from authority on purpose). The projection pipeline still has one validated asset nothing yet uses — calibrated *distributions*, verified at 80.5% coverage on held-out seasons. That is an in-season instrument, not a draft one.

      Caveat on the championship objective: it was scored by expected points, which is the metric it is deliberately not optimizing, so its −49 is not evidence against it. Judging it fairly needs the season simulator.
- [x] 8. **Weekly lineup optimizer** (`lineup/optimizer.py`) — MILP via PuLP over the league's real slot rules, so an illegal lineup is impossible by construction. **This is where the edge actually lives.**

      `check_lineups` measured the prize: the league leaves **272 points a season on the bench** (20.9/week), against a total draft edge of 81. `check_optimizer` then replayed all of 2025 using only ESPN's contemporaneous weekly projections and scored the result on what actually happened: **+200 points a season, +15.4 per week, capturing 74% of the ceiling — and beating all eight managers**, by margins from +16 to +380.

      The matchup-aware objective (sweep a risk frontier, then score each candidate by exact win probability against that week's opponent) turned out to pick almost the same lineup as plain projection-maximizing — a +2 point difference over a season, with only two teams ever diverging. Honest read: with a full roster the variance differences between startable options are small next to the mean differences, so risk-adjustment has little room to work. Judging it properly needs head-to-head results rather than point totals, but the ceiling on its value is clearly small. **Simply starting the right players is where the 200 points come from.**
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
