# WolfSportsAI V6 — Cloud Edition

A local NBA + NFL sports modeling dashboard.

## Included
- NBA and NFL
- Live internet odds refresh with The Odds API
- Moneyline, spreads and game totals
- Individual player search
- NBA player performance model using `nba_api`
- NFL player performance model using `nflreadpy` / nflverse
- Player prop Over/Under probabilities
- Best sportsbook price shown for each available line
- SQLite database that archives odds and completed scores
- Team market calibration models that can retrain as results accumulate
- Prop model that blends recent performance with no-vig sportsbook consensus
- Parlay builder from 2 through 16 legs
- Highest Chance, Best Value and Longshot optimization modes
- Player props can be mixed with team markets in parlays
- Same-game correlation protection is ON by default
- CSV exports

## Live odds setup
WolfSportsAI uses The Odds API for current betting markets.
Open the Settings tab, paste your own API key, save it and restart the app.

## NBA stats
`nba_api` requests NBA player game logs. If the NBA endpoint is temporarily unavailable
or throttles requests, the app falls back to market consensus instead of inventing data.

## NFL stats
`nflreadpy` loads nflverse weekly player statistics and requires no separate NFL stats key.

## Player prop model
For each available line:
1. pull the sportsbook consensus probability;
2. pull recent player performances when available;
3. calculate a recency-weighted average, standard deviation and hit rate;
4. estimate Over/Under probability;
5. blend the stats estimate with market consensus using sample-size reliability;
6. calculate edge and EV.

## Parlays
Available sizes:
2, 3, 4, 5, 6, 8, 10, 12 and 16 legs.

The 16-leg Longshot mode rewards payout/value while still heavily weighting projected win
probability. It will not invent extra legs when fewer than 16 qualifying selections exist.

By default only one leg per game is allowed. This reduces unmodeled same-game correlation.
You can override this manually.

## Self-updating
While the Streamlit app is open it can refresh every 5-60 minutes, archive the new market
snapshot and sync recently completed scores. Closing the Command Prompt stops the updater.

## Windows
Extract the ZIP and double-click:

run_windows.bat

Or run:

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py

No pick or parlay is guaranteed. Long parlays have compounding risk.


## V4 dashboard redesign

V4 keeps the V3 modeling backend and replaces the interface with a cleaner betting-terminal layout:
- dark sportsbook-style theme;
- sidebar navigation;
- pick cards with Model %, Market %, Edge and EV;
- live API status;
- easier market filters;
- improved individual-player workflow;
- dedicated prop scanner;
- streamlined 2–16 leg parlay builder;
- cleaner Model Lab and Settings pages.

The visual design is inspired by modern sports analytics dashboards, but uses original WolfSportsAI branding and layout.


## V5 additions

### Historical Bootstrap
Model Lab now has a one-click bootstrap:
- NBA: downloads real historical team game logs through nba_api.
- NFL: downloads nflverse schedules and final scores through nflreadpy.
- Calculates pre-game rolling Last 5 / Last 10 win rate, points for, points allowed and margin.
- Trains three historical-form models per league:
  - home win probability
  - expected scoring margin
  - expected game total
- Current spreads and totals are evaluated against the predicted margin/total.

NFL bootstrap also stores spread/total lines if those fields are available in the nflverse schedule dataset.

Exact historical sportsbook snapshots from The Odds API require a paid provider plan; the free bootstrap
does not claim historical ATS data when it does not have historical closing lines.

### Game Center
From Best Bets, NBA or NFL, click any matchup to open Game Center.

Game Center contains:
- H2H / Moneyline
- Spread
- Total O/U
- Head-to-head games
- Last 5 and Last 10 views
- current bookmaker prices
- model vs market probability
- rolling scoring/margin stats
- projected margin and total
- historical hit-rate vs today's line (clearly labeled as today's-line simulation, not historical ATS)


## V5.2 Autopilot

WolfSportsAI now starts an automatic AI-maintenance worker whenever Streamlit starts.

Default schedule:
- Every 15 minutes: refresh NBA/NFL odds and recent final scores.
- If historical training data is missing: bootstrap NBA/NFL history automatically.
- Every 12 hours: retrain historical-form models.
- Every 6 hours: run chronological holdout backtests and save the metrics.
- Save automatic backtest history in SQLite.
- Display AI Autopilot status, last retrain, last backtest and last refresh in the dashboard.

No Train or Backtest button is required. Manual Model Lab buttons remain only as force-run controls.

Important: the background worker lives inside the Streamlit process. If the Windows Command Prompt
or Streamlit process is closed, the worker stops. A future Windows Scheduled Task/service mode could
keep it running even when the dashboard is closed.


## V6 Cloud Edition
- Public-hosting entrypoint: `streamlit_app.py`
- Streamlit Secrets support for `ODDS_API_KEY`
- Real API keys are excluded by `.gitignore`
- Cloud-safe Streamlit config
- Automatic historical bootstrap/retraining remains enabled while the app instance is awake
- Local Windows mode continues to work
