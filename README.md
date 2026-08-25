# fantasy-football

A full-season manager for an ESPN fantasy league, built around one rule: **no strategy ships unless it beats a simple baseline on real historical data.** Several didn't, and were cut.

## What it does, and what that's worth

| | measured result |
|---|---|
| **Weekly lineups** | **+200 points a season** (+15.4/week), 74% of the achievable ceiling, beating all 8 managers in a 2025 backtest |
| **Draft** | +81 points a season over the humans, from a rule that fits on one line |
| **Waivers / trades** | advisory — priced in championship probability, not yet validated |

Both headline numbers come from replaying a completed season using only information available at the time, and scoring the result against what actually happened. `check_optimizer` and `check_rehearsal` reproduce them.

## Commands

```bash
uv sync
cp .env.example .env          # league id + two ESPN cookies; see below

# Draft
uv run python -m fantasy_football.live_draft prepare   # the morning of
uv run python -m fantasy_football.live_draft plan      # shape of every possible slot
uv run python -m fantasy_football.live_draft plan 5    # detail once your slot is drawn
uv run python -m fantasy_football.live_draft watch     # during: draft in ESPN, this advises
uv run python -m fantasy_football.live_draft manual    # fallback if the feed dies

# In season
uv run python -m fantasy_football.weekly               # lineup, waivers, trades

# Validation
uv run python -m fantasy_football.check_scoring        # engine vs ESPN, exact
uv run python -m fantasy_football.check_calibration    # are the intervals honest?
uv run python -m fantasy_football.check_rehearsal      # draft strategies, head to head
uv run python -m fantasy_football.check_optimizer      # lineup edge, backtested
```

## How it decides

**Scoring** is driven entirely by the league's own settings, in ESPN's stat-id space. It reproduces 4,087 historical player-weeks exactly — 100% across 2024 and 2025.

**Projections** blend expert consensus rank with ESPN's projections in *rank* space, then map through a curve fit on five seasons of preseason-rank-to-actual-outcome. Fitting on preseason rank rather than final rank matters: final rank is an order statistic describing whoever happened to finish RB5, and nobody can draft that in expectation. Intervals are backtested at 50/80/90% nominal → 52.3/80.5/88.7% actual.

**Drafting** takes the highest-ranked player at a position you still need. That is the entire rule, and it won a bench of six strategies — including a Monte Carlo lookahead board that lost by 241 points at 5.2 standard errors. Converting consensus rank into a derived value estimate made things *worse*, because expert rank already encodes injury and depth-chart information a curve cannot recover. The simulation still runs, but only to report who is likely to be gone by your next pick; `board_view` separates information from authority deliberately.

**Lineups** are solved as an assignment problem, so an illegal lineup is impossible by construction. This is where the real edge is: the league leaves 272 points a season on the bench, and simply starting the right players recovers three quarters of it.

**Roster moves** are priced as a change in P(finishing first), simulated through the actual playoff bracket. In a league where 4 of 8 qualify and the title is two single-game coin flips, points and championships genuinely diverge.

## ESPN credentials

Private leagues need a league ID and two browser cookies.

**League ID** — from the URL: `fantasy.espn.com/football/league?leagueId=123456`.

**Cookies** — `espn_s2` and `SWID` are `HttpOnly`, so `document.cookie` will not show them:

1. On any logged-in `espn.com` page, press **F12**
2. **Storage** tab (Firefox) or **Application** tab (Chrome/Edge)
3. **Cookies** → `https://fantasy.espn.com` (if the list looks short, check `.espn.com`)
4. Copy the full value of `espn_s2`, and of `SWID` **including its curly braces**

Put all three in `.env`. It is gitignored and a gitleaks pre-commit hook guards it. The cookies expire; a sudden 401 means pull them again.

## Notes

Fit on 2,581 historical player-seasons and 1,185 draft picks from this league's own history. See [PLAN.md](PLAN.md) for the full record, including the strategies that were tested and rejected.
