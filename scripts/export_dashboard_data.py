#!/usr/bin/env python3
"""
Score today's slate and write docs/data/latest.json for the dashboard -- the
CBB version of the CFB export. For every game with a posted line:
  spread   model margin vs the market spread -> side, edge, official, grade
  total    model total vs the market total   -> side, edge, official, grade
  ML       model win % vs the price (shown, never official)
  breakdown each team's numbers, the matchup, and what moves the model
For every prop: projection, hit chance, edge (EV%), matchup, official, grade.
Every official play goes into docs/data/play_log.json, which grades finished
games and builds the record by market and grade.
"""
import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from scipy.stats import norm

import config
from src.analysis import play_log, rules
from src.data import load
from src.features import player_features as pf, team_features as tf
from src.models.gbm import GBMModel

GROUPS = {
    "Efficiency (per 100 poss)": ["net_diff", "r_adj_off_diff", "r_adj_def_diff", "exp_margin", "exp_total",
                                  "home_r_adj_off", "home_r_adj_def", "away_r_adj_off", "away_r_adj_def"],
    "Tempo": ["r_tempo_diff", "exp_poss", "home_r_tempo", "away_r_tempo"],
    "Four factors": [c for c in tf.MARGIN_FEATURES + tf.TOTAL_FEATURES if "matchup" in c or c.startswith(("home_r_", "away_r_")) and
                     any(f in c for f in ("efg", "tov", "orb", "ftr", "three"))],
    "Elo": ["elo_diff"],
    "Last season": ["prior_net_diff", "home_prior_off", "home_prior_def", "away_prior_off", "away_prior_def"],
    "Rest": ["rest_days_diff", "back_to_back_diff", "home_rest_days", "away_rest_days"],
    "Home court": ["home_court"],
}


def norm_name(s):
    s = str(s).lower().replace("&", "and").replace(".", "").replace("'", "")
    s = re.sub(r"\bst\b", "state", s)
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def matcher(cbbd_names):
    names = sorted({n for n in cbbd_names if isinstance(n, str)}, key=lambda n: -len(norm_name(n)))
    normed = [(n, norm_name(n)) for n in names]

    def match(odds_name):
        o = norm_name(odds_name)
        for n, nn in normed:          # longest CBBD school name that starts the Odds API name
            if o == nn or o.startswith(nn + " "):
                return n
        return None
    return match


def explain(model, row: pd.DataFrame, base: float):
    out = []
    for name, cols in GROUPS.items():
        cols = [c for c in cols if c in model.feature_columns]
        if not cols:
            continue
        r2 = row.copy()
        for c in cols:
            r2[c] = model.medians.get(c)
        d = base - float(model.predict(r2)[0])
        if abs(d) >= 0.1:
            out.append([name, round(d, 1)])
    return sorted(out, key=lambda x: -abs(x[1]))


def ev_pct(p, price):
    try:
        price = float(price)
    except (TypeError, ValueError):
        return None
    pay = price / 100 if price > 0 else 100 / abs(price)
    return round((p * pay - (1 - p)) * 100, 1)


def r1(x, n=1):
    try:
        return None if x is None or pd.isna(x) else round(float(x), n)
    except (TypeError, ValueError):
        return None


def main():
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    cur = config.current_season(now)
    R = rules.load()
    games_cur = load.games([cur])
    gt_cur = load.game_teams([cur])
    pl_cur = load.players([cur])
    adj_prev = load.adjusted([cur - 1])
    states = tf.current_team_states(gt_cur, adj_prev, cur) if len(gt_cur) else {}
    pstates = pf.current_player_states(pl_cur, cur) if len(pl_cur) else {}
    models = {k: GBMModel.load(f"{config.MODELS_DIR}/{k}.joblib") for k in ("margin", "total")}
    pmodels = {s: GBMModel.load(f"{config.MODELS_DIR}/props_{s}.joblib") for s in pf.STATS.values()
               if os.path.exists(f"{config.MODELS_DIR}/props_{s}.joblib")}
    match = matcher(list(states) + list(games_cur.get("homeTeam", [])) + list(games_cur.get("awayTeam", [])))
    log = play_log.load()
    defs = sorted(v["r_adj_def"] for v in states.values() if v.get("r_adj_def") is not None)
    cut_lo, cut_hi = (np.percentile(defs, 33), np.percentile(defs, 67)) if defs else (None, None)

    lines = pd.read_csv(f"{config.DATA_CURRENT_DIR}/game_lines.csv") if os.path.exists(f"{config.DATA_CURRENT_DIR}/game_lines.csv") else pd.DataFrame()
    games_out, plays, unmatched = [], [], []
    for _, g in lines.iterrows():
        k = pd.to_datetime(g["commence_time"], utc=True)
        if k <= now:
            continue
        hs, as_ = match(g["home_team"]), match(g["away_team"])
        if not hs or not as_:
            unmatched.append(f"{g['away_team']} at {g['home_team']}")
            continue
        sched = games_cur[(games_cur.homeTeam == hs) & (games_cur.awayTeam == as_)] if len(games_cur) else pd.DataFrame()
        srow = sched.iloc[0] if len(sched) else {}
        neutral = bool(srow.get("neutralSite", False)) if len(sched) else False
        row = tf.live_game_row(hs, as_, states, k, neutral, srow.get("homeTeamEloStart") if len(sched) else None,
                               srow.get("awayTeamEloStart") if len(sched) else None)
        enough = all((states.get(t) or {}).get("games_prior", 0) >= 2 for t in (hs, as_))
        X = tf.finish_features(pd.DataFrame([row]))
        gm = {"home": g["home_team"], "away": g["away_team"], "home_school": hs, "away_school": as_,
              "kickoff": g["commence_time"], "book": g.get("book_used"), "neutral": neutral,
              "market_spread": r1(g.get("spread_home")), "market_total": r1(g.get("total")),
              "ml_home": r1(g.get("ml_home"), 0), "ml_away": r1(g.get("ml_away"), 0), "enough_games": enough}
        if enough:
            m = float(models["margin"].predict(X)[0])
            t = float(models["total"].predict(X)[0])
            sd = models["margin"].residual_std
            gm.update({"model_margin": round(m, 1), "model_spread": round(-m, 1), "model_total": round(t, 1),
                       "home_win_prob": round(float(norm.cdf(m / sd)), 3),
                       "drivers_spread": explain(models["margin"], X, m), "drivers_total": explain(models["total"], X, t)})
            for side in ("home", "away"):
                s = states.get(hs if side == "home" else as_) or {}
                gm[f"{side}_stats"] = {c: r1(s.get(c), 3 if c.startswith("r_") and c not in ("r_adj_off", "r_adj_def", "r_tempo") else 1)
                                       for c in ["r_adj_off", "r_adj_def", "r_tempo", "r_efg_o", "r_efg_d", "r_tov_o", "r_tov_d",
                                                 "r_orb_o", "r_orb_d", "r_ftr_o", "r_ftr_d", "r_three_rate_o", "prior_net", "games_prior"]}
                gm[f"{side}_rest"] = r1(row.get(f"{side}_rest_days"))
            # Spread
            ev = rules.evaluate_spread(m, gm["market_spread"], R["spread"])
            if ev:
                base = {"market": "spread", "home": gm["home"], "away": gm["away"], "home_cbbd": hs, "away_cbbd": as_,
                        "kickoff": gm["kickoff"], "side": ev["side"], "line": gm["market_spread"], "model": round(-m, 1)}
                mv = rules.line_move(ev["side"], gm["market_spread"], play_log.first_line(log, base))
                ev.update({"grade": rules.grade(ev["tier_pts"], mv), "move": mv})
                gm["spread_play"] = ev
                plays.append({**base, **ev, "cover_prob": round(float(norm.cdf(ev["edge"] / sd)), 3)})
            # Total
            ev = rules.evaluate_total(t, gm["market_total"], R["total"])
            if ev:
                base = {"market": "total", "home": gm["home"], "away": gm["away"], "home_cbbd": hs, "away_cbbd": as_,
                        "kickoff": gm["kickoff"], "side": ev["side"], "line": gm["market_total"], "model": round(t, 1)}
                mv = rules.line_move(ev["side"], gm["market_total"], play_log.first_line(log, base))
                ev.update({"grade": rules.grade(ev["tier_pts"], mv), "move": mv})
                gm["total_play"] = ev
                plays.append({**base, **ev})
        games_out.append(gm)

    # ---- Props
    props_out = []
    props = pd.read_csv(f"{config.DATA_CURRENT_DIR}/props.csv") if os.path.exists(f"{config.DATA_CURRENT_DIR}/props.csv") else pd.DataFrame()
    for _, p in props.iterrows():
        stat = pf.STATS.get(p["market_name"])
        if stat not in pmodels:
            continue
        hs, as_ = match(p["home_team"]), match(p["away_team"])
        team = st = None
        for t in (hs, as_):
            st = pstates.get((t, str(p["player_name"]).lower()))
            if st:
                team = t
                break
        if not st:
            continue
        opp = as_ if team == hs else hs
        ts, os_ = states.get(team) or {}, states.get(opp) or {}
        feat = {**{k: v for k, v in st.items() if k.startswith("p_")}, "team_tempo": ts.get("r_tempo"),
                "opp_tempo": os_.get("r_tempo"), "opp_adj_def": os_.get("r_adj_def"), "is_home": 1.0 if team == hs else 0.0}
        mdl = pmodels[stat]
        mu = float(mdl.predict(pd.DataFrame([feat]))[0])
        p_over = float(1 - norm.cdf((float(p["line"]) - mu) / mdl.residual_std))
        evo, evu = ev_pct(p_over, p["over_price"]), ev_pct(1 - p_over, p["under_price"])
        side = "over" if (evo or -99) >= (evu or -99) else "under"
        evv = evo if side == "over" else evu
        if evv is None:
            continue
        d = os_.get("r_adj_def")
        soft = d is not None and cut_hi is not None and d >= cut_hi
        tough = d is not None and cut_lo is not None and d <= cut_lo
        mu_grade = 1 if ((side == "over" and soft) or (side == "under" and tough)) else (-1 if ((side == "over" and tough) or (side == "under" and soft)) else 0)
        ev = rules.evaluate_prop(p["market_name"], p["line"], evv, side, int(st.get("p_games") or 0), R["prop"])
        base = {"market": "prop", "home": p["home_team"], "away": p["away_team"], "home_cbbd": hs, "away_cbbd": as_,
                "kickoff": p["start_time"], "side": side, "line": float(p["line"]), "player": p["player_name"],
                "prop_market": p["market_name"], "athleteId": st.get("athleteId"), "model": round(mu, 1),
                "price": p["over_price"] if side == "over" else p["under_price"]}
        mv = rules.line_move(side, float(p["line"]), play_log.first_line(log, base))
        ev.update({"grade": rules.grade(ev["tier_pts"], mv, extra=max(mu_grade, 0) - (1 if mu_grade < 0 else 0)), "move": mv,
                   "matchup": "soft D" if soft else ("tough D" if tough else "average D"), "mu_pts": mu_grade})
        row = {**base, **ev, "team": team, "opponent": opp, "hit_prob": round(p_over if side == "over" else 1 - p_over, 3),
               "minutes": r1(st.get("p_minutes")), "avg": r1(st.get(f"p_{stat}")), "games": int(st.get("p_games") or 0),
               "book": p.get("book_used")}
        props_out.append(row)
        plays.append(row)

    play_log.record_run(log, plays, now_iso)
    n = play_log.grade_results(log, games_cur, pl_cur)
    play_log.save(log)
    rec = play_log.record(log)
    lab = json.load(open(f"{config.DOCS_DATA_DIR}/lab.json")) if os.path.exists(f"{config.DOCS_DATA_DIR}/lab.json") else {}
    diag = json.load(open(f"{config.DOCS_DATA_DIR}/model_diagnostics.json")) if os.path.exists(f"{config.DOCS_DATA_DIR}/model_diagnostics.json") else {}
    order = {"A": 0, "B": 1, "C": 2}
    official = [p for p in plays if p["official"]]
    out = {"meta": {"generated_at": now_iso, "season": cur, "demo": config.DEMO, "rules": R, "unmatched": unmatched,
                    "spread_sd": round(models["margin"].residual_std, 2)},
           "games": games_out,
           "spreads": sorted([p for p in official if p["market"] == "spread"], key=lambda p: (order[p["grade"]], -p["edge"]))[:R["spread"]["card_size"]],
           "totals": sorted([p for p in official if p["market"] == "total"], key=lambda p: (order[p["grade"]], -p["edge"])),
           "props": sorted(props_out, key=lambda p: (not p["official"], order[p["grade"]], -p["edge"])),
           "record": rec, "lab": lab, "diagnostics": diag}
    os.makedirs(config.DOCS_DATA_DIR, exist_ok=True)
    json.dump(out, open(f"{config.DOCS_DATA_DIR}/latest.json", "w"), indent=1, default=str)
    print(f"slate: {len(games_out)} games | official: {len(out['spreads'])} spreads, {len(out['totals'])} totals, "
          f"{sum(p['official'] for p in props_out)} props | graded {n} | unmatched {len(unmatched)}")


if __name__ == "__main__":
    main()
