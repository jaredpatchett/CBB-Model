"""
Prop line archive: every sportsbook prop line the model scores, every run,
graded from the box score once the game is over.

Why: there is no free history of college prop lines, so props can't be
backtested for ROI the way spreads and totals are. This file builds that
history from today on (same idea as the CFB model's live prop grading).

docs/data/prop_archive.csv has one row per prop per run:
  run_at, kickoff, home, away, player, athleteId, prop_market, line,
  over_price, under_price, book, model, p_over, side, edge, games, matchup,
  official, grade, actual, result
The bet line for a prop is its LAST row before tip-off (`is_last`); the first
row is the earliest line we saw (the "opener" for this purpose).

docs/data/prop_lab.json summarizes graded last-before-tip rows by edge tier,
side, stat and grade, at the logged sportsbook price, plus first-seen lines
(bet early) for comparison.
"""
import json
import os

import pandas as pd

ARCHIVE_PATH = "docs/data/prop_archive.csv"
LAB_PATH = "docs/data/prop_lab.json"
STAT_COL = {"Points": "points", "Rebounds": "rebounds", "Assists": "assists", "Threes": "threes"}
COLS = ["run_at", "kickoff", "home", "away", "home_cbbd", "away_cbbd", "player", "athleteId", "prop_market", "line",
        "over_price", "under_price", "book", "model", "p_over", "side", "edge", "games", "matchup", "official", "grade",
        "actual", "result"]
EDGE_TIERS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 999)]


def load(path=ARCHIVE_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False) if os.path.exists(path) else pd.DataFrame(columns=COLS)
    df["result"] = df["result"].astype(object).where(df["result"].notna(), None)
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    return df


def append(rows: list, now_iso: str, path=ARCHIVE_PATH) -> pd.DataFrame:
    """rows: every scored prop this run (official or not)."""
    df = load(path)
    new = pd.DataFrame([{
        "run_at": now_iso, "kickoff": r.get("kickoff"), "home": r.get("home"), "away": r.get("away"),
        "home_cbbd": r.get("home_cbbd"), "away_cbbd": r.get("away_cbbd"), "player": r.get("player"),
        "athleteId": r.get("athleteId"), "prop_market": r.get("prop_market"), "line": r.get("line"),
        "over_price": r.get("over_price"), "under_price": r.get("under_price"), "book": r.get("book"),
        "model": r.get("model"), "p_over": r.get("p_over"), "side": r.get("side"), "edge": r.get("edge"),
        "games": r.get("games"), "matchup": r.get("matchup"), "official": int(bool(r.get("official"))),
        "grade": r.get("grade"), "actual": None, "result": None} for r in rows], columns=COLS)
    out = pd.concat([df, new], ignore_index=True) if len(df) else new
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    return out


def grade(players: pd.DataFrame, path=ARCHIVE_PATH) -> int:
    """Fill actual/result for finished games from the CBBD player box scores."""
    df = load(path)
    if df.empty or players is None or players.empty or "athleteId" not in players:
        return 0
    open_ = df["result"].isna()
    if not open_.any():
        return 0
    pl = players.copy()
    pl["_start"] = pd.to_datetime(pl["startDate"], utc=True, errors="coerce")
    pl["athleteId"] = pd.to_numeric(pl["athleteId"], errors="coerce")
    by_id = {k: g for k, g in pl.groupby("athleteId")}
    n = 0
    for i in df.index[open_]:
        r = df.loc[i]
        aid = pd.to_numeric(r["athleteId"], errors="coerce")
        g = by_id.get(aid)
        if g is None:
            continue
        k = pd.to_datetime(r["kickoff"], utc=True, errors="coerce")
        g = g[(g["_start"] - k).abs() < pd.Timedelta(days=1)]
        stat = STAT_COL.get(r["prop_market"])
        if g.empty or not stat or pd.isna(g.iloc[0].get(stat)):
            continue
        val = float(g.iloc[0][stat])
        line = float(r["line"])
        diff = (val - line) * (1 if r["side"] == "over" else -1)
        df.at[i, "actual"] = val
        df.at[i, "result"] = "P" if diff == 0 else ("W" if diff > 0 else "L")
        n += 1
    df.to_csv(path, index=False)
    return n


def _units(result, price):
    if result == "P":
        return 0.0
    if result == "L":
        return -1.0
    try:
        price = float(price)
    except (TypeError, ValueError):
        price = -110.0
    return price / 100 if price > 0 else 100 / abs(price)


def _summ(d: pd.DataFrame) -> dict:
    if d.empty:
        return {"n": 0}
    w, l = int((d.result == "W").sum()), int((d.result == "L").sum())
    u = float(d["units"].sum())
    return {"n": int(len(d)), "w": w, "l": l, "hit": round(w / (w + l), 3) if w + l else None,
            "units": round(u, 1), "roi": round(100 * u / len(d), 1)}


def _bucket(df: pd.DataFrame) -> dict:
    out = {"all": _summ(df), "official": _summ(df[df.official == 1]),
           "by_edge": [], "by_side": {}, "by_stat": {}, "by_grade": {}}
    for lo, hi in EDGE_TIERS:
        d = df[(df.edge >= lo) & (df.edge < hi)]
        out["by_edge"].append({"label": f"{lo}-{hi}%" if hi < 999 else f"{lo}%+", **_summ(d)})
    for k, d in df.groupby("side"):
        out["by_side"][k] = _summ(d)
    for k, d in df.groupby("prop_market"):
        out["by_stat"][k] = _summ(d)
    for k, d in df[df.official == 1].groupby("grade"):
        out["by_grade"][k] = _summ(d)
    return out


def lab(path=ARCHIVE_PATH, out_path=LAB_PATH) -> dict:
    df = load(path)
    res = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "rows": int(len(df))}
    if df.empty:
        json.dump(res, open(out_path, "w"), indent=1)
        return res
    df["_run"] = pd.to_datetime(df["run_at"], utc=True, errors="coerce")
    df["_k"] = pd.to_datetime(df["kickoff"], utc=True, errors="coerce")
    df = df[df["_run"] < df["_k"]]                      # pre-tip snapshots only
    key = ["kickoff", "player", "prop_market"]
    df = df.sort_values("_run")
    last = df.groupby(key, as_index=False).tail(1)
    first = df.groupby(key, as_index=False).head(1)
    res["props_seen"] = int(len(last))
    out = {}
    for name, d in (("at_tip", last), ("first_seen", first)):
        d = d[d.result.isin(["W", "L", "P"])].copy()
        d["units"] = [_units(r, o if s == "over" else u) for r, s, o, u in zip(d.result, d.side, d.over_price, d.under_price)]
        out[name] = _bucket(d)
    res.update(out)
    res["graded"] = out["at_tip"]["all"].get("n", 0)
    json.dump(res, open(out_path, "w"), indent=1, default=str)
    return res
