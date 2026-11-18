"""
CollegeBasketballData.com (CBBD) client -- the basketball twin of the CFB
model's cfbd_client.py. Same company, same bearer-key auth, same terms
(commercial use allowed; your tier sets the call quota).

Endpoints used (field names checked against CBBD's official Python package,
cbbd, 11/2026):
  /games            schedule + scores        id, season, seasonType, startDate, neutralSite,
                                             homeTeam/awayTeam, homePoints/awayPoints,
                                             homeTeamEloStart/awayTeamEloStart (PRE-game Elo)
  /games/teams      team box score per game  gameId, team, opponent, isHome, pace,
                                             teamStats/opponentStats{possessions, points{total},
                                             fourFactors{effectiveFieldGoalPct, turnoverRatio,
                                             offensiveReboundPct, freeThrowRate},
                                             threePointFieldGoals{made, attempted}, rating}
  /games/players    player box score         players[{athleteId, name, minutes, points,
                                             rebounds, assists, threePointFieldGoals, usage, starter}]
  /lines            betting lines            lines[{provider, spread, overUnder, spreadOpen,
                                             overUnderOpen, homeMoneyline, awayMoneyline}]
  /ratings/adjusted adjusted efficiency      team, offensiveRating, defensiveRating, netRating

Season numbering: CBBD's `season` is checked on the first real fetch against
`seasonLabel` (printed by fetch_historical_data.py). This code assumes the
season is labeled by the year it ends (2026-27 = 2027).
"""
import time

import pandas as pd
import requests

import config

DEFAULT_LINE_PROVIDER_PREFERENCE = ["consensus", "ESPN Bet", "DraftKings", "Bovada", "William Hill (New Jersey)"]


def _get(path: str, params: dict = None, retries: int = 3):
    config.require_keys("CBBD_API_KEY")
    url = f"{config.CBBD_BASE_URL}{path}"
    headers = {"Authorization": f"Bearer {config.CBBD_API_KEY}", "Accept": "application/json"}
    for i in range(retries):
        r = requests.get(url, params=params or {}, headers=headers, timeout=60)
        if r.status_code == 429 and i < retries - 1:
            time.sleep(5 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    return []


# ---------------------------------------------------------------- games

ROW_CAP = 3000  # CBBD returns at most this many rows per call


def _windowed(path: str, season: int, extra: dict = None, key: str = "id") -> list:
    """Pull a whole season in two-week windows. One call per season stops at
    CBBD's 3,000-row cap (about early January), which silently cut every past
    season short in the first real backtest (11/18/2026)."""
    out, seen = [], set()
    for a, b in _date_chunks(season):
        data = _get(path, {"season": season, "startDateRange": a, "endDateRange": b, **(extra or {})}) or []
        if len(data) >= ROW_CAP:
            print(f"  [warn] {path} {a[:10]}..{b[:10]} hit the {ROW_CAP}-row cap; shrink the window")
        for row in data:
            k = row.get(key)
            if k is None or k not in seen:
                seen.add(k)
                out.append(row)
        time.sleep(0.3)
    return out


def get_games(season: int, season_type: str = None, start=None, end=None) -> pd.DataFrame:
    if not start and not end:
        extra = {"seasonType": season_type} if season_type else None
        return pd.json_normalize(_windowed("/games", season, extra))
    params = {"season": season}
    if season_type:
        params["seasonType"] = season_type
    if start:
        params["startDateRange"] = start
    if end:
        params["endDateRange"] = end
    return pd.json_normalize(_get("/games", params))


def _date_chunks(season: int, days: int = 14, since=None):
    """CBBD caps how many rows one call returns, so box scores are pulled in
    two-week windows across the season (Nov 1 - Apr 15). `since` limits it to
    recent windows (the daily run's incremental pull)."""
    start = pd.Timestamp(f"{season - 1}-11-01", tz="UTC")
    if since is not None:
        start = max(start, pd.Timestamp(since).tz_convert("UTC") if pd.Timestamp(since).tzinfo else pd.Timestamp(since, tz="UTC"))
    end = min(pd.Timestamp(f"{season}-04-15", tz="UTC"), pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1))
    while start < end:
        stop = min(start + pd.Timedelta(days=days), end)
        yield start.strftime("%Y-%m-%dT00:00:00Z"), stop.strftime("%Y-%m-%dT00:00:00Z")
        start = stop


def get_game_teams(season: int, since=None) -> pd.DataFrame:
    """One row per team per game: the team's box score and its opponent's,
    flattened (teamStats.possessions, opponentStats.fourFactors.effectiveFieldGoalPct, ...)."""
    frames = []
    for a, b in _date_chunks(season, since=since):
        data = _get("/games/teams", {"season": season, "startDateRange": a, "endDateRange": b})
        if data:
            frames.append(pd.json_normalize(data))
        time.sleep(0.3)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["gameId", "team"]) if frames else pd.DataFrame()


def get_game_players(season: int, since=None) -> pd.DataFrame:
    """One row per player per game (players list exploded)."""
    windows = list(_date_chunks(season, since=since))
    rows = []
    for a, b in windows:
        params = {"season": season}
        if a:
            params.update({"startDateRange": a, "endDateRange": b})
        for g in _get("/games/players", params) or []:
            base = {k: g.get(k) for k in ("gameId", "season", "startDate", "team", "opponent", "isHome",
                                          "neutralSite", "gamePace", "seasonType")}
            for p in g.get("players") or []:
                rows.append({**base, **flatten_player(p)})
        time.sleep(0.3)
    return pd.DataFrame(rows)


def _num(x, *keys):
    """Value from a number or a nested dict (CBBD returns some stats as
    objects, e.g. rebounds {offensive, defensive, total})."""
    for k in keys:
        if isinstance(x, dict):
            x = x.get(k)
    return x


def flatten_player(p: dict) -> dict:
    reb = p.get("rebounds")
    threes = p.get("threePointFieldGoals") or {}
    return {
        "athleteId": p.get("athleteId"), "name": p.get("name"), "position": p.get("position"),
        "starter": p.get("starter"), "minutes": p.get("minutes"), "points": p.get("points"),
        "rebounds": _num(reb, "total") if isinstance(reb, dict) else reb,
        "assists": p.get("assists"),
        "threes": threes.get("made") if isinstance(threes, dict) else threes,
        "threes_att": threes.get("attempted") if isinstance(threes, dict) else None,
        "usage": p.get("usage"), "fouls": p.get("fouls"),
    }


# ---------------------------------------------------------------- lines & ratings

def get_lines(season: int) -> list:
    return _windowed("/lines", season, key="gameId")


def lines_to_dataframe(raw: list, prefs: list = None) -> pd.DataFrame:
    """One row per game: one provider's spread/total (consensus first), with
    the opening numbers, same shape as the CFB model's historical lines."""
    prefs = prefs or DEFAULT_LINE_PROVIDER_PREFERENCE
    rows = []
    for g in raw or []:
        lines = g.get("lines") or []
        if not lines:
            continue
        by = {l.get("provider"): l for l in lines}
        chosen = next((by[p] for p in prefs if p in by), lines[0])
        rows.append({"id": g.get("gameId"), "season": g.get("season"), "startDate": g.get("startDate"),
                     "homeTeam": g.get("homeTeam"), "awayTeam": g.get("awayTeam"),
                     "homeScore": g.get("homeScore"), "awayScore": g.get("awayScore"),
                     "line_provider": chosen.get("provider"),
                     "market_spread_home": chosen.get("spread"), "market_spread_open_home": chosen.get("spreadOpen"),
                     "market_total": chosen.get("overUnder"), "market_total_open": chosen.get("overUnderOpen"),
                     "market_ml_home": chosen.get("homeMoneyline"), "market_ml_away": chosen.get("awayMoneyline")})
    return pd.DataFrame(rows)


def get_adjusted_ratings(season: int) -> pd.DataFrame:
    """Season-end adjusted efficiency (points per 100 possessions). Used only
    as the PRIOR season's rating (a preseason prior), never same-season --
    the same leakage rule the CFB model uses for SP+/CORE."""
    return pd.json_normalize(_get("/ratings/adjusted", {"season": season}))


def get_teams(season: int) -> pd.DataFrame:
    return pd.json_normalize(_get("/teams", {"season": season}))
