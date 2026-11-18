# CBB Model (draft)

College basketball betting model: spreads, totals, player props. Built 11/2026 as a **rough draft that mirrors the CFB model** ([CFB-Model](https://github.com/jaredpatchett/CFB-Model)): same pipeline, same play grades, same dashboard ideas, with basketball stats in place of football ones.

**Status:** code runs end to end on demo data. The official rules in `config/rules.json` are **provisional** until the backtest lab runs on real data.

## CFB → CBB: what changed

| CFB model | CBB model |
|---|---|
| CollegeFootballData (CFBD) | CollegeBasketballData (CBBD): same company, same terms, commercial use allowed |
| SP+ / CORE (prior season) | Adjusted efficiency, prior season (`/ratings/adjusted`) |
| Scoring margin, PPA per play | Points per 100 possessions on offense and defense, opponent-adjusted, recent games weighted more |
| Plays per drive | Tempo: possessions per 40 minutes |
| Success rate | The four factors: eFG%, turnover rate, offensive rebound %, free-throw rate (offense and defense), plus 3-point attempt rate |
| Weather | None (indoors). Rest days and back-to-backs instead |
| — | CBBD's pre-game Elo |
| QB out | Not modeled yet (needs an injury feed) |
| Props: receptions, rec yds, rush yds | Props: points, rebounds, assists, threes, from minutes, usage, per-40 rates, tempo and opponent defense |

## Same as CFB

- **Leakage rules:** in-season numbers only use games before the one being predicted. Season ratings always come from the prior season.
- **Play grades:** 🔥 A (bet it), 🎲 B (degen play, not a bet), 🚫 C (don't play). The grade = edge-tier points (+ matchup for props), −1 if the line moved against us. A line moving *our* way never raises a grade; the play gets a "bet small" tag instead.
- **Official only at the current line:** every play is re-checked each run.
- **Moneylines:** shown but never official.
- **Play log** (`docs/data/play_log.json`): logs every official play each run, grades it from final scores or the box score, and builds the record by market and grade with history.
- **Phone alerts (ntfy):** new 🔥 A plays and A's that drop.

## Layout

```
config.py                    keys, paths, season calendar, CBB_DEMO switch
config/rules.json            official rules + grade tiers (PROVISIONAL)
src/data/cbbd_client.py      games, team + player box scores, lines, adjusted ratings
src/data/odds_api_client.py  today's spreads/totals/MLs + props (basketball_ncaab)
src/features/team_features.py    efficiency, tempo, four factors, Elo, rest (train + live)
src/features/player_features.py  minutes, usage, per-40 rates, environment (props)
src/models/gbm.py            gradient-boosted model (margin, total, each prop stat)
src/analysis/rules.py        official rules + grades
src/analysis/play_log.py     tracking, grading, record by grade, history
scripts/
  demo_data.py               fake data in the real feeds' shapes (no keys needed)
  fetch_historical_data.py   past seasons cached; this season topped up incrementally
  build_features.py          raw -> data/processed
  train_models.py            margin, total, props_{points,rebounds,assists,threes}
  run_backtest.py            walk-forward vs closing + opening lines -> docs/data/lab.json
  fetch_current_lines.py     today's lines + props
  export_dashboard_data.py   score the slate, rules, grades, play log -> latest.json
  build_dashboard.py         docs/dashboard.html (Edge Board, Totals, Props, Tracker, Performance)
  push_alerts.py             ntfy
.github/workflows/
  daily_run.yml              twice a day Nov-Apr (8am + 2pm Arizona)
  backtest_lab.yml           run first, and after model changes
  start_daily_run.yml        push this file to start a run
  alerts_test.yml            test ping
```

## Setup (one time)

1. **Secrets** (Settings → Secrets and variables → Actions):
   - `CBBD_API_KEY`: try your CollegeFootballData key first; if it's rejected, get one at collegebasketballdata.com/key. Your CFBD Patreon tier should cover the higher quota.
   - `ODDS_API_KEY`: the same key as the CFB model. College props are one call per game, so watch the quota.
   - `NTFY_TOPIC`: reuse the CFB topic, or make a new one for basketball-only alerts.
2. Actions → **Backtest lab** → Run. This pulls 4 past seasons (cached afterwards) and writes the edge tiers.
3. Set `config/rules.json` from `docs/data/lab.json`: keep the edge sizes that were profitable every season.
4. Actions → **Daily CBB model run** → Run. After that it runs on its own twice a day in season.
5. Dashboard: GitHub Pages on a **private** repo needs a paid GitHub plan. Without one, open `docs/dashboard.html` from the repo, or make the dashboard repo public (it holds no PFF data, unlike CFB).

## Try it locally with no keys

```bash
pip install -r requirements.txt
export CBB_DEMO=1
python scripts/demo_data.py && python scripts/build_features.py && python scripts/train_models.py
python scripts/run_backtest.py && python scripts/export_dashboard_data.py && python scripts/build_dashboard.py
open docs/dashboard.html
```

Every number in demo mode is made up. The demo "market" is noisier than a real one, so demo backtests look far too profitable.

## Not done yet

- Rules and grade tiers set from a **real** backtest.
- Injuries / players out. CBB has no free injury feed; this matters most for props.
- Prop backtests: CBBD has no past prop lines, so props are graded live from day one. Treat them as watch-only until they prove out.
- Conference tournaments and March Madness (neutral courts are handled; tournament-specific factors aren't).
- Team logos, power ratings tab, game-by-game player cards (the CFB dashboard has these).
