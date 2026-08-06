# fantasy-football

Full-season management tool for an ESPN fantasy football league — draft kit, weekly lineup optimizer, and waiver/trade evaluation.

It optimizes for **probability of finishing first**, not expected points. A Monte Carlo season simulator prices every recommendation in championship probability, which makes the tool seek variance when trailing and buy floor when leading. Roster construction is treated as portfolio selection under a target quantile: projections are expected returns, boom/bust is volatility.

Status: planning — see [PLAN.md](PLAN.md).

## Setup

```bash
uv sync
cp .env.example .env   # then fill in the values below
```

## ESPN credentials

Private leagues require a league ID and two browser cookies.

**League ID** — open your league on fantasy.espn.com and read it from the URL:

```
https://fantasy.espn.com/football/league?leagueId=123456
                                                  ^^^^^^
```

**Cookies** — `espn_s2` and `SWID` are `HttpOnly`, so `document.cookie` will not show them. Use the devtools storage panel:

1. On any `espn.com` page, logged in, press **F12**
2. **Application** tab (Chrome/Edge) or **Storage** tab (Firefox)
3. **Cookies** → `https://fantasy.espn.com` (if the list looks short, check the `.espn.com` entry instead — both cookies are set at the parent domain)
4. Copy the full **Value** of `espn_s2` — a long URL-encoded string
5. Copy the full **Value** of `SWID` — formatted `{ABC12345-6789-...}`, **including the curly braces**

Put all three in `.env`. They are secrets: `.env` is gitignored and a gitleaks pre-commit hook guards against committing them. The cookies expire — if requests start returning 401, pull them again.
