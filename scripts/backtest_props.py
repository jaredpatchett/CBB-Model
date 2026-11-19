#!/usr/bin/env python3
"""
Props backtest by grade, the CBB version of the CFB A/B/C prop results.

1. Past sportsbook prop lines from The Odds API's historical event-odds
   endpoint (props history starts May 2023; paid plans only). One snapshot
   per game, `--hours-before` tip-off (default 3: college props mostly post
   the day of the game). Raw responses are cached in data/raw/prop_history/
   so a re-run never pays twice.
2. Model projections with no leakage: the same prop models as the live run,
   trained only on games before the test window, using each player's
   pre-game features (data/processed/player_features.csv).
3. The live rules (config/rules.json) and grades (edge tier + soft/tough
   defense matchup). One snapshot per game, so there is no line-move
   adjustment.
4. Graded from the box score at the sportsbook price. Results by grade,
   official vs not, edge tier, side and stat go to docs/data/prop_backtest.json
   and docs/data/prop_backtest_detail.csv.

Cost: 10 credits per market per region per game that has props (4 markets,
region us = up to 40 a game) + 1 per day for the event list. `--max-credits`
stops before going over; `--dry-run` only lists games and the estimate.

Usage:
  python scripts/backtest_props.py --start 2026-11-11 --end 2026-11-17 --max-credits 8000
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

import config
from src.analysis import rules
from src.data.odds_api_client import PROP_BOOK_PRIORITY, PROP_MARKET_NAMES
from src.features import player_features as pf
from src.models.gbm import GBMModel

CACHE_DIR = f"{config.DATA_RAW_DIR}/prop_history"
OUT_JSON = f"{config.DOCS_DATA_DIR}/prop_backtest.json"
OUT_CSV = f"{config.DOCS_DATA_DIR}/prop_backtest_detail.csv"
EDGE_TIERS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 999)]


class Budget:
    def __init__(self, cap):
        self.cap, self.used, self.remaining = cap, 0, None

    def get(self, path, params, est):
        if self.used + est > self.cap:
            raise RuntimeError(f"credit cap {self.cap} reached (used {self.used})")
        r = requests.get(f"{config.ODDS_API_BASE_URL}{path}", params={"apiKey": config.ODDS_API_KEY, **params}, timeout=60)
        last = r.headers.get("x-requests-last")
        self.used += int(float(last)) if last else est
        self.remaining = r.headers.get("x-requests-remaining", self.remaining)
        if r.status_code in (404, 422):
            return None
        r.raise_for_status()
        return r.json()


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def events_for(day, budget):
    """Games that tip between 15:00 UTC on `day` and 15:00 UTC the next day
    (about 8am-8am Arizona), from the historical events list (1 credit)."""
    lo = datetime(day.year, day.month, day.day, 15, tzinfo=timezone.utc)
    hi = lo + timedelta(days=1)
    path = f"{CACHE_DIR}/events_{day:%Y%m%d}.json"
    if os.path.exists(path):
        data = json.load(open(path))
    else:
        data = budget.get(f"/historical/sports/{config.ODDS_API_SPORT}/events", {"date": iso(lo)}, 1) or {}
        json.dump(data, open(path, "w"))
    out = []
    for e in data.get("data") or []:
        k = pd.to_datetime(e.get("commence_time"), utc=True)
        if lo <= k < hi:
            out.append(e)
    return out


def props_for(event, hours_before, budget, regions="us"):
    path = f"{CACHE_DIR}/odds_{event['id']}.json"
    if os.path.exists(path):
        data = json.load(open(path))
    else:
        k = pd.to_datetime(event["commence_time"], utc=True) - pd.Timedelta(hours=hours_before)
        data = budget.get(f"/historical/sports/{config.ODDS_API_SPORT}/events/{event['id']}/odds",
                          {"date": iso(k), "regions": regions, "markets": ",".join(PROP_MARKET_NAMES),
                           "oddsFormat": "american"}, 10 * len(PROP_MARKET_NAMES) * len(regions.split(","))) or {}
        json.dump(data, open(path, "w"))
    game = data.get("data") or {}
    books = {b.get("key"): b for b in game.get("bookmakers", [])}
    merged = {}
    for bk in PROP_BOOK_PRIORITY + [b for b in books if b not in PROP_BOOK_PRIORITY]:
        for m in (books.get(bk) or {}).get("markets", []):
            mname = PROP_MARKET_NAMES.get(m.get("key"))
            for o in m.get("outcomes", []) if mname else []:
                side, line, player = o.get("name"), o.get("point"), o.get("description")
                if side not in ("Over", "Under") or line is None or not player:
                    continue
                slot = merged.setdefault((player, mname, line), {})
                f = "over_price" if side == "Over" else "under_price"
                if f not in slot:
                    slot[f] = o.get("price")
                    slot["book"] = bk
    return [{"event_id": event["id"], "kickoff": event["commence_time"], "snapshot": data.get("timestamp"),
             "home_team": event.get("home_team"), "away_team": event.get("away_team"), "player": p, "prop_market": m,
             "line": l, **s} for (p, m, l), s in merged.items()]


def norm_team(s):
    s = str(s).lower().replace("&", "and").replace(".", "").replace("'", "")
    s = re.sub(r"\bst\b", "state", s)
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def norm_player(s):
    s = re.sub(r"[^a-z ]", "", str(s).lower().replace("-", " ").replace(".", ""))
    return " ".join(re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s).split())


def team_matcher(names):
    names = sorted({n for n in names if isinstance(n, str)}, key=lambda n: -len(norm_team(n)))
    normed = [(n, norm_team(n)) for n in names]

    def match(o):
        o = norm_team(o)
        for n, nn in normed:
            if o == nn or o.startswith(nn + " "):
                return n
        return None
    return match


def ev_pct(p, price):
    try:
        price = float(price)
    except (TypeError, ValueError):
        return None
    pay = price / 100 if price > 0 else 100 / abs(price)
    return (p * pay - (1 - p)) * 100


def units(result, price):
    if result == "P":
        return 0.0
    if result == "L":
        return -1.0
    price = float(price)
    return price / 100 if price > 0 else 100 / abs(price)


def summ(d):
    if d.empty:
        return {"n": 0}
    w, l = int((d.result == "W").sum()), int((d.result == "L").sum())
    u = float(d.units.sum())
    return {"n": int(len(d)), "w": w, "l": l, "hit": round(w / (w + l), 3) if w + l else None,
            "units": round(u, 1), "roi": round(100 * u / len(d), 1)}


def main():
    ap = argparse.ArgumentParser()
    today = datetime.now(timezone.utc).date()
    ap.add_argument("--start", default=str(today - timedelta(days=7)))
    ap.add_argument("--end", default=str(today - timedelta(days=1)))
    ap.add_argument("--hours-before", type=float, default=3.0)
    ap.add_argument("--max-credits", type=int, default=8000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    config.require_keys("ODDS_API_KEY")
    os.makedirs(CACHE_DIR, exist_ok=True)
    budget = Budget(a.max_credits)
    days = pd.date_range(a.start, a.end, freq="D")

    # 1) lines
    events = []
    for d in days:
        ev = events_for(d.date(), budget)
        print(f"{d.date()}: {len(ev)} games")
        events += ev
    new = [e for e in events if not os.path.exists(f"{CACHE_DIR}/odds_{e['id']}.json")]
    print(f"{len(events)} games, {len(new)} not cached; worst case {40 * len(new)} credits "
          f"(games with no props cost less). Odds API credits remaining: {budget.remaining}")
    if a.dry_run:
        return
    rows, stopped = [], None
    for e in events:
        try:
            rows += props_for(e, a.hours_before, budget)
        except RuntimeError as err:
            stopped = str(err)
            print(f"[stop] {err}")
            break
    print(f"prop lines: {len(rows)} from {len({r['event_id'] for r in rows})} games | credits used this run {budget.used}, "
          f"remaining {budget.remaining}")
    lines = pd.DataFrame(rows)
    if lines.empty:
        print("no prop lines found")
        return

    # 2) projections from models trained only on games before the window
    p = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/player_features.csv", low_memory=False)
    p["start"] = pd.to_datetime(p["start"], utc=True, errors="coerce")
    w0 = pd.Timestamp(a.start, tz="UTC")
    train = p[(p["start"] < w0) & (p["p_games"] >= 3)]
    test = p[(p["start"] >= w0 - pd.Timedelta(days=1)) & (p["start"] < pd.Timestamp(a.end, tz="UTC") + pd.Timedelta(days=2))].copy()
    test["_pn"] = test["name"].map(norm_player)
    models = {}
    for stat in pf.STATS.values():
        ver = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config", "model.json"))).get("features", "base")
        models[stat] = GBMModel(pf.features_for(stat, ver), stat, n_estimators=200).fit(train, verbose=False)
        print(f"  {stat}: trained on {models[stat].n_train} player-games, sd {models[stat].residual_std:.2f}")
    defs = test["opp_adj_def"].dropna()
    cut_lo, cut_hi = (np.percentile(defs, 33), np.percentile(defs, 67)) if len(defs) else (None, None)
    match = team_matcher(list(p["team"].dropna().unique()))
    R = rules.load()

    # 3) grade every line
    out, unmatched, miss = [], 0, []
    for _, r in lines.iterrows():
        stat = pf.STATS.get(r["prop_market"])
        hs, as_ = match(r["home_team"]), match(r["away_team"])
        k = pd.to_datetime(r["kickoff"], utc=True)
        cand = test[(test["team"].isin([hs, as_])) & (test["_pn"] == norm_player(r["player"])) &
                    ((test["start"] - k).abs() < pd.Timedelta(hours=18))]
        why = ("team not matched" if not (hs and as_) else "one-sided line" if pd.isna(r.get("over_price")) or pd.isna(r.get("under_price"))
               else "player not in box score" if cand.empty else None)
        if why:
            unmatched += 1
            miss.append({**r.to_dict(), "home_cbbd": hs, "away_cbbd": as_, "why": why})
            continue
        row = cand.iloc[0]
        m = models[stat]
        mu = float(m.predict(cand.iloc[[0]])[0])
        p_over = float(1 - norm.cdf((float(r["line"]) - mu) / m.residual_std))
        evo, evu = ev_pct(p_over, r["over_price"]), ev_pct(1 - p_over, r["under_price"])
        side = "over" if evo >= evu else "under"
        evv = evo if side == "over" else evu
        d = row.get("opp_adj_def")
        soft = pd.notna(d) and cut_hi is not None and d >= cut_hi
        tough = pd.notna(d) and cut_lo is not None and d <= cut_lo
        mu_grade = 1 if ((side == "over" and soft) or (side == "under" and tough)) else (
            -1 if ((side == "over" and tough) or (side == "under" and soft)) else 0)
        ev = rules.evaluate_prop(r["prop_market"], r["line"], evv, side, int(row.get("p_games") or 0), R["prop"])
        g = rules.grade(ev["tier_pts"], 0, extra=max(mu_grade, 0) - (1 if mu_grade < 0 else 0))
        actual = float(row[stat])
        diff = (actual - float(r["line"])) * (1 if side == "over" else -1)
        res = "P" if diff == 0 else ("W" if diff > 0 else "L")
        price = r["over_price"] if side == "over" else r["under_price"]
        out.append({**r.to_dict(), "team": row["team"], "games": int(row.get("p_games") or 0), "model": round(mu, 1),
                    "p_over": round(p_over, 3), "side": side, "edge": round(evv, 1), "official": ev["official"],
                    "why_not": ev["why_not"], "matchup": "soft D" if soft else ("tough D" if tough else "average D"),
                    "grade": g, "price": price, "actual": actual, "result": res, "units": units(res, price)})
    df = pd.DataFrame(out)
    miss = pd.DataFrame(miss)
    reasons = miss["why"].value_counts().to_dict() if len(miss) else {}
    print(f"graded {len(df)} lines, {unmatched} unmatched: {reasons}")
    if len(miss):
        miss.to_csv(OUT_CSV.replace(".csv", "_unmatched.csv"), index=False)
        # diagnostics: box-score coverage by day, and what we have for missed games
        print("player-game rows by day:", test["start"].dt.strftime("%m-%d").value_counts().sort_index().to_dict())
        for _, mrow in miss[miss.why == "player not in box score"].drop_duplicates("event_id").head(4).iterrows():
            t = p[p["team"].isin([mrow["home_cbbd"], mrow["away_cbbd"]]) & (p["season"] == p["season"].max())]
            print(f"  missed {mrow['away_cbbd']} @ {mrow['home_cbbd']} {mrow['kickoff']} | {mrow['player']} "
                  f"(norm '{norm_player(mrow['player'])}') | that team's games: {sorted(t['start'].dt.strftime('%m-%d %H:%M').unique())[:8]}")
            names = sorted(t["name"].dropna().unique())[:6]
            print(f"    names on file: {names}")
    if df.empty:
        return
    off = df[df.official]
    res = {"generated_at": datetime.now(timezone.utc).isoformat(), "window": [a.start, a.end],
           "snapshot_hours_before": a.hours_before, "stopped_early": stopped, "credits_used": budget.used,
           "credits_remaining": budget.remaining, "lines": int(len(df)), "unmatched": unmatched, "unmatched_why": reasons,
           "official": summ(off), "all_lines": summ(df),
           "by_grade": {k: summ(off[off.grade == k]) for k in ("A", "B", "C")},
           "by_edge": [{"label": f"{lo}-{hi}%" if hi < 999 else f"{lo}%+", **summ(df[(df.edge >= lo) & (df.edge < hi)])}
                       for lo, hi in EDGE_TIERS],
           "by_side": {k: summ(v) for k, v in df.groupby("side")},
           "official_by_side": {k: summ(v) for k, v in off.groupby("side")},
           "official_by_stat": {k: summ(v) for k, v in off.groupby("prop_market")},
           "official_by_matchup": {k: summ(v) for k, v in off.groupby("matchup")}}
    os.makedirs(config.DOCS_DATA_DIR, exist_ok=True)
    json.dump(res, open(OUT_JSON, "w"), indent=1, default=str)
    df.to_csv(OUT_CSV, index=False)
    print(json.dumps({k: res[k] for k in ("official", "by_grade", "by_edge", "official_by_side", "official_by_stat")}, indent=1))


if __name__ == "__main__":
    main()
