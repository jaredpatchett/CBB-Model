"""
The Odds API client for college basketball (sport key basketball_ncaab).
Same pattern as the CFB model: game lines (spreads, totals, moneylines) in
one bulk call, player props one event at a time, real sportsbooks only.

Basketball prop markets requested: points, rebounds, assists, threes.
Books post far fewer NCAAB props than NFL/NBA -- expect props mainly on
bigger games and tournament games.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

import config

PROP_MARKET_NAMES = {
    "player_points": "Points",
    "player_rebounds": "Rebounds",
    "player_assists": "Assists",
    "player_threes": "Threes",
}
PROP_BOOK_PRIORITY = ["draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet", "betrivers",
                      "fanatics", "bovada", "betonlineag", "mybookieag", "ballybet", "hardrockbet"]


def _get(path, params):
    config.require_keys("ODDS_API_KEY")
    r = requests.get(f"{config.ODDS_API_BASE_URL}{path}", params={"apiKey": config.ODDS_API_KEY, **params}, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    if r.headers.get("x-requests-remaining") is not None:
        print(f"[odds_api] requests used {r.headers.get('x-requests-used')}, remaining {r.headers.get('x-requests-remaining')}")
    return r.json()


def get_game_odds(regions="us", markets="h2h,spreads,totals"):
    return _get(f"/sports/{config.ODDS_API_SPORT}/odds", {"regions": regions, "markets": markets, "oddsFormat": "american"}) or []


def odds_to_dataframe(raw: list, preferred_book: str = "draftkings") -> pd.DataFrame:
    rows = []
    for game in raw or []:
        books = {b["key"]: b for b in game.get("bookmakers", [])}
        book = books.get(preferred_book) or (list(books.values())[0] if books else None)
        row = {"game_id": game.get("id"), "commence_time": game.get("commence_time"),
               "home_team": game.get("home_team"), "away_team": game.get("away_team"),
               "book_used": book.get("key") if book else None}
        for m in (book or {}).get("markets", []):
            for o in m.get("outcomes", []):
                name = o.get("name", "")
                if m["key"] == "h2h":
                    row["ml_home" if name == game.get("home_team") else "ml_away"] = o.get("price")
                elif m["key"] == "spreads" and name == game.get("home_team"):
                    row["spread_home"] = o.get("point")
                elif m["key"] == "totals" and name == "Over":
                    row["total"] = o.get("point")
                    row["total_over_price"] = o.get("price")
                elif m["key"] == "totals" and name == "Under":
                    row["total_under_price"] = o.get("price")
        rows.append(row)
    return pd.DataFrame(rows)


def get_player_props(events: list, max_days_out: float = 2.0, regions="us,us2") -> pd.DataFrame:
    """One row per player / market / line with over and under prices (best-
    priority book per side), same shape as the CFB props feed."""
    now = datetime.now(timezone.utc)
    rows = []
    for game in events or []:
        try:
            k = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
        except Exception:
            continue
        if k < now or k > now + timedelta(days=max_days_out):
            continue
        data = _get(f"/sports/{config.ODDS_API_SPORT}/events/{game['id']}/odds",
                    {"regions": regions, "markets": ",".join(PROP_MARKET_NAMES), "oddsFormat": "american"}) or {}
        books = {b.get("key"): b for b in data.get("bookmakers", [])}
        merged = {}
        for bk in PROP_BOOK_PRIORITY:
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
                        slot["book_used"] = bk
        for (player, mname, line), s in merged.items():
            rows.append({"fixture_id": game["id"], "start_time": game["commence_time"], "home_team": game.get("home_team"),
                         "away_team": game.get("away_team"), "player_name": player, "market_name": mname, "line": line,
                         "over_price": s.get("over_price"), "under_price": s.get("under_price"), "book_used": s.get("book_used")})
    return pd.DataFrame(rows)
