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


def _get(path: str, params: dict = None, retries: int = 5):
    config.require_keys("CBBD_API_KEY")
    url = f"{config.CBBD_BASE_URL}{path}"
    headers = {"Authorization": f"Bearer {config.CBBD_API_KEY}", "Accept": "application/json"}
    for i in range(retries):
        last = i == retries - 1
        try:
            r = requests.get(url, params=params or {}, headers=headers, timeout=60)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
            # dropped connections happen on big pulls; wait and retry
            if last:
                raise
            time.sleep(10 * (i + 1))
            continue
        if (r.status_code == 429 or r.status_code >= 500) and not last:
            time.sleep(10 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    return []


# ---------------------------------------------------------------- games

ROW_CAP = 3000  # CBBD returns at most this many rows per call


def _pull(path: str, params: dict, a: str, b: str) -> list:
    """One date window; if CBBD returns a full 3,000 rows the window was cut
    off, so split it in half and pull each half (down to one day)."""
    data = _get(path, {**params, "startDateRange": a, "endDateRange": b}) or []
    time.sleep(0.3)
    ta, tb = pd.Timestamp(a), pd.Timestamp(b)
    if len(data) < ROW_CAP:
        return data
    if tb - ta <= pd.Timedelta(days=1):
        print(f"  [warn] {path} {a[:10]}..{b[:10]} still hits the {ROW_CAP}-row cap in one day")
        return data
    mid = (ta + (tb - ta) / 2).floor("D").strftime("%Y-%m-%dT00:00:00Z")
    return _pull(path, params, a, mid) + _pull(path, params, mid, b)


def _season_rows(path: str, season: int, extra: dict = None, keys=("id",), since=None) -> list:
    """Pull a season window by window, splitting any window that hits CBBD's
    3,000-row cap. One call per season cut every past season off in early
    January, and two-week box-score windows cut the opening two weeks of
    each season (both found 11/18/2026)."""
    out, seen = [], set()
    for a, b in _date_chunks(season, since=since):
        for row in _pull(path, {"season": season, **(extra or {})}, a, b):
            k = tuple(row.get(x) for x in keys)
            if None in k or k not in seen:
                seen.add(k)
                out.append(row)
    return out


def _windowed(path: str, season: int, extra: dict = None, key: str = "id") -> list:
    return _season_rows(path, season, extra, keys=(key,))


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
    data = _season_rows("/games/teams", season, keys=("gameId", "team"), since=since)
    return pd.json_normalize(data).drop_duplicates(["gameId", "team"]) if data else pd.DataFrame()


def get_game_players(season: int, since=None) -> pd.DataFrame:
    """One row per player per game (players list exploded)."""
    rows = []
    for g in _season_rows("/games/players", season, keys=("gameId", "team"), since=since):
        base = {k: g.get(k) for k in ("gameId", "season", "startDate", "team", "opponent", "isHome",
                                      "neutralSite", "gamePace", "seasonType")}
        for p in g.get("players") or []:
            rows.append({**base, **flatten_player(p)})
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
