"""
Official-play rules and play grades for every market (11/2026 draft).

Same shape as the CFB model:
  * a play is official only if it clears its market's rule at the CURRENT
    line (re-checked every run)
  * grade = edge tier points (+ matchup for props), minus 1 if the line has
    moved against us since the play was first flagged; 2+ = A (bet it),
    0-1 = B (degen play, not a bet), below 0 = C (don't play)
  * a line moving OUR way never raises a grade; the play gets a "bet small"
    tag instead (the best number is gone -- CFB spreads: +14.7% at the flag,
    -4.4% at the moved number)
  * moneylines are unofficial (the CFB model's moneylines lost 14u in 2026)

The numbers live in config/rules.json, set 11/19/2026 from the walk-forward
backtest (scripts/run_backtest.py): totals are unders only ("sides").
"""
import json
import os

RULES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "rules.json")


def load():
    with open(RULES_PATH) as f:
        return json.load(f)


def tier_points(x, tiers):
    for lo, hi, pts in tiers:
        if lo <= x < hi:
            return pts, f"{lo:g}-{hi:g}" if hi < 99 else f"{lo:g}+"
    return None, None


def line_move(side, line_now, first_line):
    """+1 = moved toward our side (market agrees), -1 = against, 0 = none.
    For a spread `side` is 'home'/'away' and lines are the HOME spread."""
    if first_line is None or line_now is None or float(line_now) == float(first_line):
        return 0
    d = float(line_now) - float(first_line)
    if side in ("under",):
        return 1 if d < 0 else -1
    if side in ("over",):
        return 1 if d > 0 else -1
    if side == "home":            # home spread getting more negative = market likes home more
        return 1 if d < 0 else -1
    if side == "away":
        return 1 if d > 0 else -1
    return 0


def grade(points, move, extra=0):
    s = points + extra + min(move, 0)
    return "A" if s >= 2 else ("B" if s >= 0 else "C")


def evaluate_spread(model_margin, home_spread, R=None):
    """model_margin = predicted home minus away. Edge in points for the side
    the model likes. Returns dict or None if no line."""
    R = R or load()["spread"]
    if home_spread is None:
        return None
    edge_home = model_margin + home_spread          # >0: home covers by this much
    side = "home" if edge_home > 0 else "away"
    edge = abs(edge_home)
    pts, tier = tier_points(edge, R["tiers"])
    official = edge >= R["min_edge"] and abs(home_spread) < R["max_abs_spread"]
    why_not = "" if official else (f"edge {edge:.1f} (needs {R['min_edge']:g}+)" if edge < R["min_edge"] else f"spread {abs(home_spread):g}+ (faded)")
    return {"side": side, "edge": round(edge, 1), "tier": tier, "tier_pts": pts or 0, "official": official, "why_not": why_not}


def evaluate_total(pred, line, R=None):
    R = R or load()["total"]
    if line is None:
        return None
    e = pred - line
    side = "over" if e > 0 else "under"
    pts, tier = tier_points(abs(e), R["tiers"])
    sides = R.get("sides", ["over", "under"])
    big = abs(e) >= R["min_edge"]
    official = big and side in sides
    why = "" if official else (f"edge {abs(e):.1f} (needs {R['min_edge']:g}+)" if not big else f"{side}s never official (backtest)")
    return {"side": side, "edge": round(abs(e), 1), "tier": tier, "tier_pts": pts or 0, "official": official, "why_not": why}


def evaluate_prop(market, line, ev_pct, side, games, R=None):
    R = R or load()["prop"]
    pts, tier = tier_points(ev_pct, R["tiers"])
    min_line = R["min_line"].get(market, 0)
    ok = (ev_pct >= R["min_ev"] and ev_pct < R["max_ev"] and games >= R["min_games"] and float(line) >= min_line)
    why = "" if ok else ("edge too small" if ev_pct < R["min_ev"] else "edge 40%+ (usually a data problem)" if ev_pct >= R["max_ev"]
                         else "too few games" if games < R["min_games"] else "line too small")
    return {"side": side, "edge": round(ev_pct, 1), "tier": tier, "tier_pts": pts or 0, "official": ok, "why_not": why}


def cap_prop_grade(g, market, price, R=None):
    """Rod's 11/25 caps: no A on markets in `no_a_markets` (Points) or on a
    price worse than `a_worst_price` (-150 or worse can't be A). Capped A -> B."""
    R = R or load()["prop"]
    if g != "A":
        return g, ""
    if market in R.get("no_a_markets", []):
        return "B", f"{market} props capped at B"
    worst = R.get("a_worst_price")
    try:
        p = float(price)
    except (TypeError, ValueError):
        p = None
    if worst is not None and p is not None and p < 0 and p < worst:
        return "B", f"price {p:+.0f} capped at B"
    return g, ""
