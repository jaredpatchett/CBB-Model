#!/usr/bin/env python3
"""
Walk-forward backtest (the CBB "lab"): each season is predicted by models
trained only on earlier seasons, then bet against that game's real closing
and opening lines from CBBD /lines at -110.

Writes docs/data/lab.json -- what sets config/rules.json:
  spread / total, vs close and vs open:
    every game, each edge tier (record, ROI, ROI by season, every-season-up),
    a daily card (top N edges per day)
  totals also split by over / under
  props: model accuracy vs a naive "recent average" guess (no past prop
         lines in CBBD, so no prop ROI until lines are collected live)
and docs/data/lab_detail.csv (every graded bet).

Usage: python scripts/run_backtest.py
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd

import config
from src.data import load
from src.features import player_features as pf, team_features as tf
from src.models.gbm import GBMModel

SPREAD_TIERS = [(0, 2), (2, 3), (3, 5), (5, 7), (7, 10), (10, 99)]
TOTAL_TIERS = [(0, 3), (3, 5), (5, 7), (7, 10), (10, 99)]
ML_TIERS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 999)]   # expected value, % of stake


def summ(d):
    if d.empty:
        return {"n": 0}
    w, l = int((d.result == "W").sum()), int((d.result == "L").sum())
    by = {str(int(k)): round(float(g.units.mean()) * 100, 1) for k, g in d.groupby("season")}
    return {"n": int(len(d)), "w": w, "l": l, "units": round(float(d.units.sum()), 1),
            "roi": round(float(d.units.mean()) * 100, 1), "seasons": by, "every": bool(by) and all(v >= 0 for v in by.values())}


def grade(d, pred_col, line_col, market):
    d = d.dropna(subset=[pred_col, line_col]).copy()
    d["line"] = d[line_col].astype(float)
    if market == "spread":
        e = d[pred_col] + d["line"]                      # home covers by e
        d["side"] = np.where(e > 0, "home", "away")
        res = (d["margin"] + d["line"]) * np.where(e > 0, 1, -1)
    else:
        e = d[pred_col] - d["line"]
        d["side"] = np.where(e > 0, "over", "under")
        res = (d["total"] - d["line"]) * np.where(e > 0, 1, -1)
    d["edge"] = e.abs()
    d = d[d["edge"] > 0]
    res = res.loc[d.index]
    d["result"] = np.where(res == 0, "P", np.where(res > 0, "W", "L"))
    d["units"] = d["result"].map({"W": 100 / 110, "L": -1.0, "P": 0.0})
    return d


def ml_grade(d):
    """Moneylines at the real closing price. Model win% = P(margin > 0) from
    the margin model and its holdout sd; bet the side with the higher EV."""
    from scipy.stats import norm
    d = d.copy()
    for c in ("market_ml_home", "market_ml_away"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["p_margin", "market_ml_home", "market_ml_away", "ml_sd"])
    d = d[(d.market_ml_home.abs() < 5000) & (d.market_ml_away.abs() < 5000) & (d.market_ml_home.abs() >= 100) & (d.market_ml_away.abs() >= 100)]
    pay = lambda pr: np.where(pr > 0, pr / 100, 100 / pr.abs())
    p_home = norm.cdf(d.p_margin / d.ml_sd)
    ev_h = p_home * pay(d.market_ml_home) - (1 - p_home)
    ev_a = (1 - p_home) * pay(d.market_ml_away) - p_home
    home = ev_h >= ev_a
    d["side"] = np.where(home, "home", "away")
    d["price"] = np.where(home, d.market_ml_home, d.market_ml_away)
    d["model_p"] = np.where(home, p_home, 1 - p_home)
    d["edge"] = 100 * np.where(home, ev_h, ev_a)
    d["line"] = d["price"]
    d = d[d.edge > 0]
    won = np.where(d.side == "home", d.margin > 0, d.margin < 0)
    d["result"] = np.where(d.margin == 0, "P", np.where(won, "W", "L"))
    d["units"] = np.where(d.result == "W", pay(d.price.astype(float)), np.where(d.result == "L", -1.0, 0.0))
    return d


def ml_block(d):
    out = {"every_game": summ(d), "tiers": [], "favorites": summ(d[d.price < 0]), "underdogs": summ(d[d.price > 0]),
           "calibration": []}
    for lo, hi in ML_TIERS:
        s = summ(d[(d.edge >= lo) & (d.edge < hi)])
        s.update({"lo": lo, "hi": hi, "label": f"{lo}-{hi}% EV" if hi < 999 else f"{lo}%+ EV"})
        out["tiers"].append(s)
    for lo in (0.5, 0.6, 0.7, 0.8, 0.9):
        b = d[(d.model_p >= lo) & (d.model_p < lo + 0.1)]
        if len(b):
            out["calibration"].append({"band": f"{int(lo * 100)}-{int(lo * 100) + 10}%", "n": int(len(b)),
                                       "model_says": round(float(b.model_p.mean()) * 100, 1),
                                       "actually_won": round(float((b.result == "W").mean()) * 100, 1),
                                       "roi": round(float(b.units.mean()) * 100, 1)})
    return out


def block(d, tiers, per_day=8):
    out = {"every_game": summ(d), "tiers": []}
    for lo, hi in tiers:
        s = summ(d[(d.edge >= lo) & (d.edge < hi)])
        s.update({"lo": lo, "hi": hi, "label": f"{lo:g}-{hi:g} pts" if hi < 99 else f"{lo:g}+ pts"})
        out["tiers"].append(s)
    day = pd.to_datetime(d["startDate"], utc=True).dt.tz_convert("America/New_York").dt.date
    rank = d.groupby(day)["edge"].rank(ascending=False, method="first")
    out["daily_card"] = summ(d[rank <= per_day])
    return out


def model_cfg():
    try:
        return json.load(open(os.path.join(os.path.dirname(__file__), "..", "config", "model.json")))
    except OSError:
        return {"features": "base", "anchored": False}


def walk(g, j, test_seasons, features="base", anchored=False):
    """Walk-forward predictions. Plain: predict margin/total from team inputs.
    Anchored: predict how far the result lands from the market line (trained
    on closing lines), then add back the line being bet -- the opener for the
    vs-open test, the close for the vs-close test."""
    mf, tf_ = tf.FEATURE_SETS[features]
    parts = []
    for s in test_seasons:
        tr = j[j.season < s] if anchored else g[g.season < s]
        tr = tr[tr["home_games_prior"].fillna(0) + tr["away_games_prior"].fillna(0) >= 2].copy()
        te = j[j.season == s].copy()
        if len(tr) < 300 or te.empty:
            continue
        if anchored:
            tr = tr[tr.market_spread_home.notna() & tr.market_total.notna()].copy()
            tr["ats"], tr["ou"] = tr.margin + tr.market_spread_home, tr.total - tr.market_total
            tr["mkt_spread"], tr["mkt_total"] = tr.market_spread_home, tr.market_total
            mm = GBMModel(mf + ["mkt_spread"], "ats").fit(tr, verbose=False)
            tm = GBMModel(tf_ + ["mkt_total"], "ou").fit(tr, verbose=False)
            for ref, sl, tl_ in (("close", "market_spread_home", "market_total"), ("open", "market_spread_open_home", "market_total_open")):
                x = te.assign(mkt_spread=te[sl], mkt_total=te[tl_])
                te[f"p_margin_{ref}"] = -te[sl] + mm.predict(x)
                te[f"p_total_{ref}"] = te[tl_] + tm.predict(x)
            te["ml_sd"] = mm.residual_std
        else:
            mm = GBMModel(mf, "margin").fit(tr, verbose=False)
            pm = mm.predict(te)
            pt = GBMModel(tf_, "total").fit(tr, verbose=False).predict(te)
            te["p_margin_close"] = te["p_margin_open"] = pm
            te["p_total_close"] = te["p_total_open"] = pt
            te["ml_sd"] = mm.residual_std
        parts.append(te)
        print(f"  [{features}{' anchored' if anchored else ''}] {s}: trained on {len(tr)}, graded {len(te)}")
    sc = pd.concat(parts, ignore_index=True)
    sc = sc[sc["margin"].notna()]
    sc["p_margin"], sc["p_total"] = sc["p_margin_close"], sc["p_total_close"]
    return sc


def evaluate(sc):
    out, details = {"spread": {}, "total": {}}, []
    for lab, sl, tl_ in (("close", "market_spread_home", "market_total"), ("open", "market_spread_open_home", "market_total_open")):
        sp = grade(sc, f"p_margin_{lab}", sl, "spread")
        sp = sp[sp["line"].abs() < 20]
        tt = grade(sc, f"p_total_{lab}", tl_, "total")
        out["spread"][f"vs_{lab}"] = block(sp, SPREAD_TIERS)
        out["total"][f"vs_{lab}"] = block(tt, TOTAL_TIERS)
        out["total"][f"vs_{lab}"]["over"] = block(tt[tt.side == "over"], TOTAL_TIERS)
        out["total"][f"vs_{lab}"]["under"] = block(tt[tt.side == "under"], TOTAL_TIERS)
        for m, d in (("spread", sp), ("total", tt)):
            details.append(d.assign(market=m, vs=lab)[["market", "vs", "season", "startDate", "homeTeam", "awayTeam",
                                                         "line", "side", "edge", "result", "units"]])
    ml = ml_grade(sc)
    out["moneyline"] = {"vs_close": ml_block(ml),
                        "note": "closing moneylines only (CBBD has no opening moneyline); never official, like CFB"}
    details.append(ml.assign(market="moneyline", vs="close")[["market", "vs", "season", "startDate", "homeTeam", "awayTeam",
                                                               "line", "side", "edge", "result", "units"]])
    ok = sc.market_spread_home.notna() & sc.market_total.notna()
    out["accuracy"] = {"margin_mae": round(float((sc.p_margin - sc.margin).abs()[ok].mean()), 2),
                       "market_spread_mae": round(float((-sc.market_spread_home - sc.margin).abs()[ok].mean()), 2),
                       "total_mae": round(float((sc.p_total - sc.total).abs()[ok].mean()), 2),
                       "market_total_mae": round(float((sc.market_total - sc.total).abs()[ok].mean()), 2)}
    oo = sc.market_spread_open_home.notna()
    if oo.any():
        out["accuracy"]["margin_mae_vs_open_games"] = round(float((sc.p_margin_open - sc.margin).abs()[oo].mean()), 2)
        out["accuracy"]["market_open_spread_mae"] = round(float((-sc.market_spread_open_home - sc.margin).abs()[oo].mean()), 2)
    return out, details


def headline(out):
    """The numbers that decide whether a change is kept."""
    h = {"accuracy": out["accuracy"]}
    for m in ("spread", "total"):
        for lab in ("close", "open"):
            b = out[m][f"vs_{lab}"]
            h[f"{m}_vs_{lab}"] = {"all": b["every_game"].get("roi"),
                                  "tiers": {t["label"]: [t.get("n"), t.get("roi"), t.get("every")] for t in b["tiers"]}}
    for lab in ("close", "open"):
        u = out["total"][f"vs_{lab}"]["under"]["tiers"]
        h[f"unders_vs_{lab}"] = {t["label"]: [t.get("n"), t.get("roi"), t.get("every")] for t in u}
    ml = out["moneyline"]["vs_close"]
    h["moneyline"] = {"all": ml["every_game"].get("roi"), "calibration": [[c["model_says"], c["actually_won"]] for c in ml["calibration"]]}
    return h


def props_accuracy(test_seasons, version):
    """Prop model accuracy (MAE) vs a recent-average guess; "early" = a
    player's 2nd-3rd game of the season, where last season's numbers matter most."""
    pall = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/player_features.csv", low_memory=False)
    pall = pall[pall["p_games"] >= 1]
    p = pall[pall["p_games"] >= 3]
    props = {}
    for stat in pf.STATS.values():
        maes, early = [], []
        for s in test_seasons:
            tr, te = p[p.season < s], p[p.season == s]
            if len(tr) < 1000 or te.empty:
                continue
            m = GBMModel(pf.features_for(stat, version), stat, n_estimators=150).fit(tr, verbose=False)
            pred = m.predict(te)
            maes.append((float(np.abs(pred - te[stat]).mean()), float(np.abs(te[f"p_{stat}"] - te[stat]).mean()), len(te)))
            ee = pall[(pall.season == s) & (pall.p_games < 3)]
            if len(ee):
                early.append((float(np.abs(m.predict(ee) - ee[stat]).mean()), len(ee)))
        if maes:
            n = sum(x[2] for x in maes)
            props[stat] = {"model_mae": round(sum(x[0] * x[2] for x in maes) / n, 2),
                           "recent_avg_mae": round(sum(x[1] * x[2] for x in maes) / n, 2), "n": n}
            if early:
                ne = sum(x[1] for x in early)
                props[stat]["early_mae"] = round(sum(x[0] * x[1] for x in early) / ne, 2)
    return props


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="also test the other feature sets; writes docs/data/feature_test.json")
    a = ap.parse_args()
    cfg = model_cfg()
    seasons = load.saved_seasons()
    cur = config.current_season()
    g = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/game_features.csv", low_memory=False)
    lines = load.lines(seasons)
    j = g.merge(lines[["id", "market_spread_home", "market_spread_open_home", "market_total", "market_total_open",
                       "market_ml_home", "market_ml_away"]], on="id", how="inner")
    j = j[j["home_games_prior"].fillna(0) + j["away_games_prior"].fillna(0) >= 2]
    for c in ("market_spread_home", "market_spread_open_home", "market_total", "market_total_open", "market_ml_home", "market_ml_away"):
        j[c] = pd.to_numeric(j[c], errors="coerce")          # missing lines come through as None
    test_seasons = [s for s in seasons[1:] if s <= cur]

    if a.compare:
        variants = [("base", False), ("v2", False), ("v2", True)]
        res = {"generated_at": datetime.now(timezone.utc).isoformat(), "seasons": test_seasons, "variants": {}}
        for feats, anch in variants:
            name = feats + ("_anchored" if anch else "")
            o, _ = evaluate(walk(g, j, test_seasons, feats, anch))
            res["variants"][name] = headline(o)
            print(f"\n=== {name} ===\n" + json.dumps(res["variants"][name]["accuracy"]))
            for k in ("spread_vs_open", "spread_vs_close", "total_vs_open", "total_vs_close"):
                print(k, res["variants"][name][k]["all"], res["variants"][name][k]["tiers"])
            print("unders_vs_open", res["variants"][name]["unders_vs_open"])
            print("moneyline", res["variants"][name]["moneyline"])
        res["props_accuracy"] = {v: props_accuracy(test_seasons, v) for v in ("base", "v2")}
        print("\nprops", res["props_accuracy"])
        os.makedirs(config.DOCS_DATA_DIR, exist_ok=True)
        json.dump(res, open(f"{config.DOCS_DATA_DIR}/feature_test.json", "w"), indent=1)

    sc = walk(g, j, test_seasons, cfg.get("features", "base"), cfg.get("anchored", False))
    out, details = evaluate(sc)
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "seasons": test_seasons, "demo": config.DEMO,
           "model": cfg, "method": "walk-forward by season, -110, edge vs the line shown", **out}
    out["props_accuracy"] = props_accuracy(test_seasons, cfg.get("features", "base"))
    os.makedirs(config.DOCS_DATA_DIR, exist_ok=True)
    json.dump(out, open(f"{config.DOCS_DATA_DIR}/lab.json", "w"), indent=1)
    pd.concat(details, ignore_index=True).to_csv(f"{config.DOCS_DATA_DIR}/lab_detail.csv", index=False)
    for m in ("spread", "total"):
        for lab in ("close", "open"):
            b = out[m][f"vs_{lab}"]
            print(f"\n{m.upper()} vs {lab}: every game {b['every_game'].get('w')}-{b['every_game'].get('l')} "
                  f"{b['every_game'].get('roi')}% | daily card {b['daily_card'].get('roi')}%")
            for t in b["tiers"]:
                print(f"   {t['label']:<10} {t.get('w', 0)}-{t.get('l', 0):<5} {t.get('roi', 0):+6.1f}%  {t.get('seasons')}")
    print("\naccuracy:", out["accuracy"], "\nprops accuracy:", out["props_accuracy"])


if __name__ == "__main__":
    main()
