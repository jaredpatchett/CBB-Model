"""
Play log for every official play (spreads, totals, props) -- the CBB version
of the CFB model's totals_log / prop grade tracker in one file.

docs/data/play_log.json:
  plays[key] = {market, game, side, (player), first: {at, line}, h: [[at, line, model, edge, grade, official]],
                result, units, final_grade, final_line}
Each run appends one entry per play still live. A play's bet is its last
entry before tip-off; it is graded from final scores (spreads, totals) or
the box score (props) once the game is over. 1 unit at -110 for spreads and
totals; props at the price logged.
"""
import json
import os

import pandas as pd

LOG_PATH = "docs/data/play_log.json"
KEEP_DAYS = 200


def load(path=LOG_PATH):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"plays": {}, "updated": None}


def save(log, path=LOG_PATH):
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=KEEP_DAYS)
    keep = {}
    for k, v in log.get("plays", {}).items():
        t = pd.to_datetime(v.get("kickoff"), utc=True, errors="coerce")
        if pd.isna(t) or t >= cutoff:
            keep[k] = v
    log["plays"] = keep
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=1, default=str)


def key_of(row):
    parts = [row["market"], row["home"], row["away"], str(row["kickoff"]), row["side"]]
    if row.get("player"):
        parts += [row["player"], row.get("prop_market", "")]
    return "|".join(parts)


def first_line(log, row):
    p = log.get("plays", {}).get(key_of(row))
    return p["first"]["line"] if p else None


def record_run(log, rows, now_iso):
    """rows: every evaluated play this run (official or not). Only official
    plays start an entry; a logged play that stops being official gets an
    entry marked 0 (dropped at this line)."""
    plays = log.setdefault("plays", {})
    for r in rows:
        k = key_of(r)
        p = plays.get(k)
        if p is None and not r["official"]:
            continue
        if p is None:
            p = plays[k] = {"market": r["market"], "home": r["home"], "away": r["away"], "kickoff": r["kickoff"],
                            "side": r["side"], "player": r.get("player"), "prop_market": r.get("prop_market"),
                            "athleteId": r.get("athleteId"), "home_cbbd": r.get("home_cbbd"), "away_cbbd": r.get("away_cbbd"),
                            "first": {"at": now_iso, "line": r["line"]}, "h": []}
        p["h"].append([now_iso, r["line"], r.get("model"), r["edge"], r["grade"], 1 if r["official"] else 0, r.get("price")])
        r["new"] = len(p["h"]) == 1
    log["updated"] = now_iso


def _final(p):
    k = pd.to_datetime(p.get("kickoff"), utc=True, errors="coerce")
    h = [x for x in p.get("h") or [] if pd.isna(k) or pd.to_datetime(x[0], utc=True) < k]
    return h[-1] if h else None


def _units(result, price):
    if result == "P":
        return 0.0
    if result == "L":
        return -1.0
    try:
        price = float(price)
    except (TypeError, ValueError):
        price = -110.0
    return round(price / 100 if price > 0 else 100 / abs(price), 4)


def grade_results(log, games: pd.DataFrame, players: pd.DataFrame = None):
    """games: CBBD /games rows (homeTeam, awayTeam, homePoints, awayPoints,
    startDate). players: flattened /games/players rows for props."""
    if games is None or games.empty:
        return 0
    g = games.dropna(subset=["homePoints", "awayPoints"]).copy()
    g["_start"] = pd.to_datetime(g["startDate"], utc=True, errors="coerce")
    n = 0
    for p in log.get("plays", {}).values():
        if p.get("result"):
            continue
        f = _final(p)
        if not f or not f[5]:
            continue
        k = pd.to_datetime(p["kickoff"], utc=True)
        m = g[(g.homeTeam == p.get("home_cbbd", p["home"])) & (g.awayTeam == p.get("away_cbbd", p["away"]))]
        m = m[(m._start - k).abs() < pd.Timedelta(days=1)]
        if m.empty:
            continue
        hp, ap = float(m.iloc[0].homePoints), float(m.iloc[0].awayPoints)
        line = float(f[1])
        if p["market"] == "spread":
            cover = (hp - ap) + line                     # home spread
            diff = cover if p["side"] == "home" else -cover
        elif p["market"] == "total":
            diff = (hp + ap - line) * (1 if p["side"] == "over" else -1)
        elif p["market"] == "prop" and players is not None and not players.empty:
            stat = {"Points": "points", "Rebounds": "rebounds", "Assists": "assists", "Threes": "threes"}.get(p.get("prop_market"))
            pm = players[(players.get("athleteId") == p.get("athleteId"))] if p.get("athleteId") is not None else pd.DataFrame()
            pm = pm[(pd.to_datetime(pm["startDate"], utc=True) - k).abs() < pd.Timedelta(days=1)] if len(pm) else pm
            if pm.empty:
                continue                                  # did not play / not in box yet: void later if never
            val = float(pm.iloc[0][stat])
            p["actual"] = val
            diff = (val - line) * (1 if p["side"] == "over" else -1)
        else:
            continue
        p["result"] = "P" if diff == 0 else ("W" if diff > 0 else "L")
        p["units"] = _units(p["result"], f[6] if len(f) > 6 and p["market"] == "prop" else -110)
        p["final_grade"], p["final_line"] = f[4], line
        p["final_score"] = f"{int(ap)}-{int(hp)}"
        n += 1
    return n


def record(log):
    done = [p for p in log.get("plays", {}).values() if p.get("result")]

    def summ(lst):
        w = sum(p["result"] == "W" for p in lst)
        l = sum(p["result"] == "L" for p in lst)
        u = sum(float(p.get("units") or 0) for p in lst)
        return {"w": w, "l": l, "p": len(lst) - w - l, "units": round(u, 2), "roi": round(u / len(lst) * 100, 1) if lst else None}
    out = {"all": summ(done), "markets": {}, "since": None}
    for m in ("spread", "total", "prop"):
        lst = [p for p in done if p["market"] == m]
        out["markets"][m] = {"all": summ(lst), "grades": {g: summ([p for p in lst if p.get("final_grade") == g]) for g in "ABC"}}
    firsts = [p["first"]["at"] for p in log.get("plays", {}).values() if p.get("first")]
    out["since"] = min(firsts) if firsts else None
    out["history"] = sorted(({k: p.get(k) for k in ("market", "home", "away", "kickoff", "side", "player", "prop_market",
                                                     "final_line", "final_grade", "result", "units", "final_score", "actual")}
                             for p in done), key=lambda r: r["kickoff"], reverse=True)
    return out
