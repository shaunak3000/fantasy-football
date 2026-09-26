# Plan — fantasy-football

Full-season management tool for an ESPN fantasy football family league: draft kit, weekly lineup optimizer, and waiver/trade evaluation, run live all season.

**The objective is not expected points.** It is P(finishing first). Those are different optimizations and they diverge exactly when it matters — a manager trailing in week 14 should be seeking variance, and a manager holding the top seed should be buying floor. Every recommendation in this repo is scored in units of championship probability, and the season simulator that produces that number is the spine everything else hangs off.

The framing is credit-risk portfolio management applied to a roster: a projection is an expected return, boom/bust is volatility, a starting lineup is a portfolio selected under a target quantile rather than a target mean, and P(first) is the analog of a default probability that every decision is priced against. That framing is the differentiator versus every other fantasy repo on GitHub.

## The league

League "Rice Ball" (id 1815614957), verified against the API on 2026-08-06:

- **8 teams**, snake draft, no keepers, waiver priority (no FAAB)
- **Full PPR**, 4pt passing TD, 0.04/passing yard, 0.1/rushing and receiving yard
- Starters (9): 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 D/ST, 1 K; 7 bench, 2 IR
- **14-week** regular season in 2026, **4 of 8 make the playoffs**, 1-week playoff matchups.
  (2025 ran 13 weeks; ESPN reports `matchupPeriodCount` 13 for 2025 and 14 for 2026, and the
  2026 schedule really does carry 14 rounds of regular-season matchups. Everything reads this
  from the API, so nothing needed changing — but the "13-week" figure repeated elsewhere in
  this plan was stale, and is corrected here.)

Three properties of this format drive the modeling and are easy to get wrong:

1. **Eight teams compresses positional scarcity.** Only 72 roster spots start each week, so replacement level is very high and the waiver wire stays deep all season — a streamed QB is worth about what QB8 is worth. The scarcity gradient that powers a normal VOR board is much flatter here, the real draft edge sits in RB/WR, and QB/TE/K/D/ST belong late. A board that recommends an early QB in this league is miscalibrated, not clever.
2. **Half the league makes the playoffs, and matchups are one week long.** P(first) ≈ P(top 4) × P(win semi) × P(win final), where qualifying is the cheap term and the title comes down to two single-game outcomes. Seeding is worth little beyond making it. So the regular season is played for **floor** — just qualify — and weeks 14–15 are played for **ceiling**. Season-long expected points is the wrong objective in this format, which is the whole reason the P(first) simulator is the spine rather than a nice extra.
3. **Waiver priority, not FAAB.** The transactions problem is sequential — when to spend a priority position — not a budget allocation.

**Draft: Friday 4 September 2026, 8:30 PM ET** (room opens 7:30 PM ET). 90 seconds per pick, 16 rounds, snake. **The draft slot is randomized one hour before kickoff**, so nothing slot-dependent can be settled in advance — `plan` prints the shape of all eight draws, and takes a slot argument once the real one is known. Trade deadline 2 December.

## Scope

- **In:** private-league ESPN ingest via cookie auth; league-settings-driven scoring engine; ensemble projections with per-player distributions; snake draft kit with opponent modeling and a live draft-night board; MILP weekly lineup optimizer; waiver and trade evaluation; season Monte Carlo producing P(first) per manager; backtest against the 2024 and 2025 seasons.
- **Out:** auction drafts (the league is snake; revisit only if it switches); DFS; sportsbook or prediction-market integration; other platforms (Yahoo, Sleeper); LLM components; anything that auto-submits a transaction without review.

## Action items

- [x] 1. **Scaffold** — uv + Python 3.12, `src/fantasy_football/{data,projections,draft,lineup,transactions,season}`, `.env.example`, gitleaks pre-commit, pytest, ruff.
- [x] 2. **ESPN client** (`data/espn.py`) — authenticated `espn-api` wrapper reading league settings, rosters, free agents, schedule, and prior-season draft results. Everything downstream reads scoring rules, roster slots, league size, and keeper rules from here; nothing about the league is hardcoded. Snapshot responses to `data/cache/` so the whole test suite runs offline.
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
      **ADP (added 2026-09-02) — context only, and deliberately kept out of the decision.**
      ESPN's `ownership.averageDraftPosition` is the one *behavioural* signal available:
      not another opinion about who is good, but a measurement of when the room actually
      takes people. It is also the freshest input in the bundle — ESPN restamps it daily,
      against a weekly consensus scrape.

      The gaps against our board are large and one-directional in places: Chris Godwin is
      consensus 72 and goes at ADP 123, Josh Downs 86 against 137, Brian Thomas Jr. 79
      against 114. Without ADP the board cannot see that it is reaching — it will spend an
      early pick on a player the room ignores for another fifty.

      So the board now prints an ADP column and, when the recommended player has a full
      round of daylight past your next turn, a note naming who will be gone instead. It
      does **not** enter the selection rule. Six cleverer rules lost the rehearsal, and
      unlike those, an ADP rule cannot even be scored: ESPN publishes a live snapshot, no
      historical preseason ADP exists in our data, so `check_rehearsal` has nothing to
      replay. Untestable signals stay advisory. Stored as an ordinal rank, never ESPN's raw
      pick number — their ADP comes from 10- and 12-team drafts running 160-192 picks
      against our 128, so the ordering transfers and the pick numbers do not.

- [x] 8. **Weekly lineup optimizer** (`lineup/optimizer.py`) — MILP via PuLP over the league's real slot rules, so an illegal lineup is impossible by construction. **This is where the edge actually lives.**

      **CORRECTION (2026-09-11) — the headline number was wrong, and the edge is much smaller than claimed.** Both `check_lineups` and `check_optimizer` mapped ESPN's `defaultPositionId` through the **lineup-slot** enumeration instead of the **player-position** one. The two collide only at 16 (D/ST), which is exactly why it survived: the scoring engine branches on nothing else, so its 1,922/1,922 reconciliation gate passed regardless. Everywhere else it was silently destructive — quarterbacks, receivers and kickers were dropped from every roster and tight ends came back labelled as receivers. The measured rosters were roughly half a team. `data/espn.PLAYER_POSITION_BY_ID` is now the single definition and both modules use it.

      With the enum fixed, `check_optimizer` reports +217 points a season. **That number is still not an edge**, because of a second and deeper problem: ESPN does not keep historical weekly lineups. A request for week 3 returns the roster and the lineup slots *as they stand today* — week 1 and week 13 come back byte-identical — which is why `box_scores()` raises `KeyError` on a finished season. So the "manager" column is not what the manager started; it is the lineup they finished with, scored against every week.

      Compared against what the managers **really** scored, from the scoreboard's own `pointsByScoringPeriod`, the projection-maximizing lineup is worth **+0.5 points a week** (spread 3.7 across the eight teams, so about 0.4 standard errors — indistinguishable from zero). The 2025 league averaged 128.3 points a week; projection-max on the same rosters averaged 128.8. And that comparison *flatters* the optimizer, since it gets to use the roster each manager finished with in every week of the season.

      **Honest position: the optimizer's in-season edge is unmeasured, not established.** It is not disproven either — the test that would settle it needs weekly roster snapshots taken live, which the 2026 season can start collecting and no replay can recover. The claim that "the edge is in-season lineups, not the draft" rests on this measurement and should be treated as open. What survives unharmed: the lineup is legal by construction, and it will never start a player who is out or on bye, which the old one did.

      The matchup-aware objective (sweep a risk frontier, then score each candidate by exact win probability against that week's opponent) turned out to pick almost the same lineup as plain projection-maximizing — a +2 point difference over a season, with only two teams ever diverging. Honest read: with a full roster the variance differences between startable options are small next to the mean differences, so risk-adjustment has little room to work. Judging it properly needs head-to-head results rather than point totals, but the ceiling on its value is clearly small. **Simply starting the right players is where the 200 points come from.**
- [x] 9. **Waivers + trades** (`transactions/`) — rest-of-season projections drive add/drop and trade evaluation, with every candidate move priced as a delta in P(first), not a delta in points. Waiver-priority aware (this league does not use FAAB).

      **Fixed 2026-09-11, after the first live report recommended giving away our best running back for a third quarterback.** Three separate faults, each of which alone made the output untrustworthy:

      *Surplus was defined as low-scoring.* On a receiver-heavy roster the best RB sits below six receivers on raw projection while being the only player who can fill a starting RB slot. `marginal_cost` now measures what removing a player actually costs the lineup: Jaylen Warren costs 0.44 points a week, the sixth receiver costs nothing.

      *The search was one-for-one only.* Trading two of six startable receivers for two startable running backs is a two-for-two, and no sequence of one-for-ones reaches it through a roster with no spare slot. Candidates are screened on points first — fast and deterministic — and only the survivors are simulated.

      *The deltas were noise.* At the 200–600 trials these ran at, the standard error on a title delta is 1.1–1.6 percentage points; the trade being recommended was "+0.50%". Every delta now carries the standard error of the *paired difference*, and anything inside it is refused rather than printed.

      Shortlisting is by whichever side gains **least**. Ranking by our own gain filled the list with deals that improve a rival's lineup on points while costing them 16 points of title probability — proposals they refuse, crowding out the ones they would not.

      **Rebuilt again 2026-09-25: full depth, every week, both sides.** Three blind spots surfaced in one week of live use.

      *The candidate pool pre-judged the package.* `get_pool` kept the other team's top few players by *standalone* value to us. DK Metcalf, on his own, adds nothing to a roster already starting three better receivers — so he never entered the pool, and Metcalf *for* Nico Collins, whom he replaces, could not be found. It was the best deal on the board and was found by hand. The package was always scored as a whole; the pool must not score its members one at a time. Both pools now default to the entire roster, and `tests/test_trade_search.py` pins the exact shape: unreachable at `get_depth=4`, found at full depth.

      *Depth was gated on the solver.* Full depth is ~146,000 candidate pairs, and every lineup was a PuLP solve — the wide search simply never returned. For this league's slots (one flex, over RB/WR/TE) greedy filling is provably optimal: fill each dedicated slot, then give the flex to the best eligible player left, and no reshuffle can beat it. `lineup/fast.py` does that, is held equal to the solver on 55 randomised rosters, and falls back to the solver in any league with more than one flex slot, where the assignment genuinely interacts. **The full-depth search runs in about 7 seconds.** The solver stays the authority for lineups a person acts on.

      *One number per team hid the week that matters.* Every trade is now valued week by week (`season/weekly_profile.py`): a player on bye that week is gone, a player ruled out *this* week is gone for this week only. The screen ranks on the bye-aware season total, and each shortlisted deal reports the per-week delta and the worst week — **for both sides**, because a bad week of their own is the commonest reason a counterparty refuses. The Doc trade shows why: +3.93% / +1.22% on the flat valuation, **+3.00% / +0.19%** once byes were counted, with a **-12.5-point week 11 for Doc** that nothing could previously show.

      *"Will we actually start him, and when?"* Each player changing hands — on both sides — is classified from the weekly lineups: **starter** (in the full-strength lineup; reports weeks started), **cover** (plays only when someone ahead of him is out; reports which weeks), or **depth** (never plays). Since packages are built at equal sizes, a two-for-one used to appear several times padded with a different never-starting throw-in — the top four results in week 3 were one deal. Results are now collapsed to one entry per core deal.

      **Injury return weeks (fixed 2026-09-25).** Injured reserve used to count as out for the current week only, so A.J. Brown — out until November — was a healthy 15.9-point starter from week 4, and every trade touching the receiver room was mis-priced. ESPN's league API publishes a status and nothing else: no return date on any view, checked. So return weeks come from two sources (`data/injuries.py`), in order:

      **Updated the same day: ESPN's injury page is the primary source.** https://www.espn.com/nfl/injuries carries an estimated return date for all ~390 injured players, keyed by the same player id as the fantasy API. It is not an API — the JSON endpoints behind it answer 403/404 — so `data/espn_injuries.py` reads the JSON blob the page embeds (`window['__espnfitt__']`), which is sturdier than scraping the table. Dates come without a year: August-December is this season, January-July the next, and ESPN writes "Feb" for season-ending injuries, which resolve past the last game and count as out for the season and the playoffs. The page is cached for six hours; if it cannot be read the report says `ESPN injury report UNAVAILABLE` and falls back. Sources, in order:

      1. **`data/injury_returns.json`** — corrections only, when the news is ahead of ESPN. It always wins, so a stale entry silently overrides a fresher date; delete entries once ESPN catches up. Empty by default (A.J. Brown's hand entry was removed once ESPN showed the same 1 November).
      2. **ESPN's estimated return**, for IR, OUT, DOUBTFUL and suspensions. QUESTIONABLE players are assumed to play, as elsewhere in the repo. OUT is not always one week — Zach Charbonnet was out until 11 October.
      3. **The NFL's four-game IR minimum** from first-seen-on-IR in the snapshots, only when ESPN has nothing. It ran two weeks short for Jonathon Brooks (floor week 7, ESPN 8 November → week 9).

      Every date resolves to the player's own team's first game on or after it, and the report prints each return week with its source.

      A player out until week N is gone from every week before N: in the weekly profile, the fast season total, and the "starts N of M" labels. Playoff strength drops anyone not back by the first playoff week. The report prints `BACK LATER:` with each player's return week and source, so the assumption is visible and correctable. Effect in week 3: title odds 17.9% → 17.5% (level with Back 2 Back), and the Doc deal fell again to **+2.07% / +0.47%** — weeks 4-7 turn negative for us because it sends away Collins, who was covering Brown's absence, and DK Metcalf is correctly labelled cover for week 6 only, since Pittman already covers Brown.

- [x] 10. **Season simulator** (`season/`) — Monte Carlo the remaining schedule using the projection distributions to produce P(first) per manager, plus playoff seeding odds. This feeds the risk posture in steps 8 and 9 and is the repo's headline chart.

      **Validated 2026-09-11 by `check_simulator`, which found it overconfident and then fixed it.** Replaying 2025 week by week, the original simulator *lost* to "rank the teams by their record so far" (Brier 0.2483 against 0.2393) and was badly miscalibrated in the middle: outcomes it called 58% happened 36% of the time.

      The cause was modelling one source of uncertainty where there are two. A team's weekly score varies around its mean — that is `weekly_sd` — but the mean itself is an estimate and can simply be wrong. Simulating only the first treats a projection as a known fact. `mean_uncertainty` now draws each team's true strength once per trial and holds it for the whole season, bracket included, so a thirteen-week run no longer assumes it knew the answer in week 3. Set to the standard error of the estimate, sd/sqrt(n) — **derived, not fitted** — it scores 0.2318 and beats every naive baseline. A constant tuned on the same season scores better still, which is exactly why it was not used.

      Also fixed: the bracket. The old pairing loop was a `zip` against its own reverse with two `break` conditions; it was correct for four teams by luck and silently eliminated a team at odd bracket sizes. It is now explicit, gives the top seed the bye, and is tested directly for 1v4 / 2v3 pairing rather than inferred from championship totals.

      The whole simulation is vectorized over trials, which is what makes the error bars affordable: 20,000 trials run in 0.04s against several seconds for the old Python loop, and the trade search evaluates hundreds of rosters.

      **Per-week means (2026-09-25).** `TeamSeason` now takes a mean for each remaining week, because byes make a team weaker in specific weeks and a single rest-of-season mean hid exactly the week that decides a matchup. The strength-uncertainty draw is an *offset* held per team per trial, so a bye week moves the level and leaves "how good is this team really" alone. The bracket uses `playoff_mean` — full strength, since NFL byes are over by the fantasy playoffs. The random draws are taken in the same order as before, so flat inputs reproduce the old results **bit for bit** (tested), and `check_simulator` still scores **0.2318** — the validated path is untouched. Live, the bye-aware baseline moved this team from 16.5% (2nd) to 17.9%, level with Back 2 Back at 17.6% — within noise of each other.

      **What it does not validate:** the roster projections feeding it. The replay takes each team's scoring distribution from its own past results, so it tests the schedule, seeding and bracket machinery, not the inputs. One season of one league is 8 teams and 13 weeks — a small Brier edge is suggestive, not settled.
- [x] 11. **Weekly run loop** (`weekly.py`) — one command that refreshes data, regenerates projections, and emits a markdown report with the recommended lineup, waiver targets, and trade ideas, each annotated with its championship-probability impact.

      **One baseline per report.** Sections used to re-simulate independently at different trial counts, so a single report quoted 8.4%, 6.3% and 5.5% as the same team's title odds in the same week, and every delta was measured against a moving target. The baseline is computed once, its Monte Carlo error is printed, and every section differences against it.

      **Two horizons, two valuations.** Start/sit runs on ESPN's published weekly projections — the input `check_optimizer` replays, and the only one that can see a matchup or a player ruled out an hour ago. The season simulator runs on the fitted rank curves, because it plays out the rest of the season and must not assume every week looks like this one. Collapsing the two was the original mistake.

      **Injuries are handled like byes.** `build_state` read every player on the roster regardless of status, so the first live report recommended starting A.J. Brown, who was on injured reserve carrying a 13.8-point ESPN projection. OUT, INJURY_RESERVE, DOUBTFUL and SUSPENSION are zeroed for the current week only — a player on IR in September is usually back before December, so his rest-of-season value is left alone. QUESTIONABLE is deliberately untouched; those players mostly play. Note that team defences report `NORMAL` rather than `ACTIVE`, so a whitelist of healthy statuses would bench every defence in the league.

      **The lineup comparison reads real ESPN slots.** `to_roster_player` never received one, so `slot_id` defaulted to -1, `started` was true for all sixteen players, and "your current lineup is already optimal" was comparing the best nine against the entire roster — a test nothing could fail. Week 2 of 2026 actually had 17.2 points sitting on the bench.

      **Defence streaming** (`transactions/streaming.py`, added 2026-09-23) — the one matchup worth chasing, and the one the optimizer structurally cannot find. `optimize` ranks only the defence already on the roster; it never looks at the wire for a better one, so three weeks running this was done by hand.

      The justification is measured, not assumed. Across weeks 4-10 of 2026 — where the schedule is known and nothing else about a player has changed yet, so the spread isolates the matchup adjustment — ESPN's weekly projections vary by **2-3% for skill players** (0.46 points of standard deviation on Amon-Ra St. Brown, 0.19 on Harold Fannin) and by about **8% for team defences**, because defensive scoring depends far more on the offence you face than on your own roster. Chasing a matchup at running back is noise; at defence it is worth a point or so a week, and it is available for a free-agent add because nobody in an eight-team league carries a spare defence.

      The rule is deliberately **not** "find the weakest opponent". ESPN's projection already blends opponent strength with the defence's own quality, and the blend beats either half: in week 3 the Lions projected best of the free agents at 9.5 while facing the Jets' strong offence, because Detroit's defence carried it. An independent opponent-strength ranking (sum of the opposing team's top-8 projected skill players) agreed with ESPN's ordering, which is a useful sanity check and also means there is no edge hiding in building one.

      A **1.0-point threshold** decides whether to report. It sits between the two real cases: week 2 offered +0.5 (Buccaneers over Eagles) and was rightly skipped, week 3 offered +1.3 (Lions over Eagles) and was not. Below the threshold the section prints nothing rather than manufacturing advice. `find_stream` takes a `position` argument, so the same machinery covers kicker if that ever proves worth it.

- [x] 12. **Weekly roster snapshots** (`data/snapshots.py`, `check_snapshots`) — capture what every manager actually started, every week, while it still exists.

      **This is the only irreplaceable data in the repo, and collecting it is a standing weekly duty.** ESPN serves exactly one roster per league: the current one. A request for a past week's `mRoster` returns today's players in today's lineup slots — week 1 and week 13 of a finished season come back byte-identical. So a week that passes uncaptured is gone permanently, and no amount of care afterwards recovers it.

      That gap is precisely why step 8's headline claim had to be withdrawn. `check_optimizer` could only compare against the lineup each manager *finished* with, because that is the only lineup ESPN would tell it. A season of these files is what makes that comparison honest, and it is the single thing standing between "the optimizer beats the league" being an open question and a settled one.

      `weekly.py` captures automatically on every run, always targeting the *live* scoring period rather than the week being reported on, and never raising — a failed snapshot must not take the report down. Files land in `data/snapshots/<season>/weekNN.json`, which is **tracked in git** rather than gitignored like `data/cache/`, so they travel between machines.

      The guards are the point. A stale capture never lands: running the report for week 3 in week 10 would otherwise overwrite the real week-3 lineups with the week-10 roster, silently replacing the only copy of the data with a worthless one. A capture without results never replaces one that has them. But results *are* allowed to arrive late — stat lines stay available per week forever, so a Tuesday capture fills in Sunday's scores while keeping the lineups recorded while the week was live.

      **Results are now collected, not merely permitted (fixed 2026-09-23).** The paragraph above was true and useless: late results were *allowed* to land, and nothing ever brought them. `weekly.py` only ever captured the live scoring period, which by definition has no scores in it — the games have not been played. So the files recorded what every manager started and not what it was worth, which is half of what the backtest needs. Weeks 1 and 2 of 2026 sat at 20 and 0 players scored. `weekly.py` now re-fetches finished weeks whose scores are not yet final; they filled in at 126 and 118.

      Deciding when to stop coming back was the whole difficulty, and the obvious test is wrong. "Every starter has a stat line" never completes: **ESPN publishes no stat line at all for a player who was started and did not play**, and week 2 has three of those — Puka Nacua, Nico Collins and Zay Flowers were all in lineups and all inactive — so that week would have been re-fetched forever. `has_results` fails in the other direction, being true the moment anybody scores, which a Thursday capture satisfies while missing 100+ lines; that is why week 1 reported results at 20 of 126.

      So the question is not what the file contains but when it was filled. Snapshots record `results_week`, the league's live scoring period at the moment stat lines were pulled. Once that is past the snapshot's own week, the week was over and the scores are as final as they will ever get. Files written before this get exactly one refetch to set it, so the fix is self-healing.

      `uv run python -m fantasy_football.check_snapshots [season]` prints coverage: which weeks are captured, which are missing and unrecoverable, and which still await results.

## Validation

- The scoring engine reproduces 2025 ESPN box scores exactly before any projection is trusted (step 5 is a hard gate on steps 6–11).
- The ID bridge refuses ambiguous matches rather than guessing, and reports every miss — a wrong player id silently corrupts everything built on it, while a missing one is visible.
- Projection intervals are calibration-checked on held-out 2025 weeks, not scored on RMSE alone.
- The optimizer is property-tested against the league's slot rules — an illegal lineup is a build failure, not a bug report.
- The draft kit is rehearsed end-to-end against the 2025 draft before it is used live.
- No recommendation ships without its P(first) delta attached, **and no delta ships without its own standard error**. `check_simulator` replays a finished season week by week, predicting the top-4 finish from each week's standpoint using only earlier scores, and scores those predictions against what happened. It must beat the naive baselines — rank by record, rank by strength, or call everything 50/50 — or the simulator is decoration.
- Historical ESPN weekly *lineups* do not exist (see step 8), so anything claiming to replay weekly decisions must say what it is actually replaying. Real weekly *team scores* do exist, via `pointsByScoringPeriod`, and are the honest backtest surface.
- Going forward they are being recorded as the season runs (step 12). Once a season of `data/snapshots/` exists, `check_optimizer` can finally be rerun against what managers really started, and the withdrawn step 8 claim can be settled either way.
- A recommendation that fires on noise is worse than none, because it spends a real transaction. The defence-streaming threshold (step 11) is set from the two cases that actually occurred rather than picked: +0.5 skipped, +1.3 taken. Anything below it prints nothing.

## Running it during the season

**Run `uv run python -m fantasy_football.weekly` at least once a week, every week, before kickoff.** It prints the report *and* captures that week's lineups, which is the part that cannot wait — see step 12. Lineups are only available while the week is live; scores are not, so the same run also goes back and fills in the stat lines for any finished week still missing them. One run a week is therefore enough, and a missed run costs only that week's lineups rather than its scores too.

Then `uv run python -m fantasy_football.check_snapshots` to confirm there are no gaps. A gap is permanent.


## Open questions

1. Does the flat scarcity gradient in an 8-team league leave enough draft-day edge to be worth the modeling, or does the real advantage sit in weekly lineups and waivers? Worth measuring explicitly in the step 7 rehearsal rather than assuming.
2. FantasyPros ECR carries no scoring-format variant in the nflverse feed, but this league is full PPR, which shifts RB/WR balance materially. Quantify how far the generic board sits from a PPR-correct one before leaning on it — the rank→points curve in step 6 is fit on this league's own scoring, so this may resolve itself.

**Resolved:** the draft happened on 4 September 2026 and the season is being run live — see "Running it during the season". Also, the league has 2024 and 2025 history reachable through the API (the web UI hides it from members who joined later), so both the box-score reconciliation gate and the draft opponent model have real data. 2024 ran 9 teams and a 14-week season; 2025 ran 8 teams over 13 weeks and 2026 runs 8 teams over 14, all full PPR, so anything fit across seasons must normalize for size and season length.
