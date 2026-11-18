#!/usr/bin/env python3
"""
Pull past seasons from CollegeBasketballData.com into data/raw/ (cached:
a finished season is skipped once saved; the current season is always
re-pulled). Files per season: games, game_teams (box scores), players,
lines (raw JSON), adjusted (season-end adjusted efficiency).

Call budget: about 12 calls a season for box scores (two-week windows) +
12 for players + 3 others. Same key/quota as the CFB model's CFBD key.

Usage:
  python scripts/fetch_historical_data.py --seasons 2023 2024 2025 2026 2027
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd

import config
from src.data import cbbd_client as cbbd


def cached(season):
    return all(os.path.exists(f"{config.DATA_RAW_DIR}/{n}_{season}.csv") for n in ("games", "game_teams", "players", "adjusted"))


def incremental(season):
    """Current season, already saved: re-pull schedule + lines (one call
    each) and only the last few days of box scores, merged into the files."""
    print(f"--- {season}: incremental ---")
    old_t = pd.read_csv(f"{config.DATA_RAW_DIR}/game_teams_{season}.csv", low_memory=False)
    since = pd.to_datetime(old_t["startDate"], utc=True).max() - pd.Timedelta(days=3) if len(old_t) else None
    cbbd.get_games(season).to_csv(f"{config.DATA_RAW_DIR}/games_{season}.csv", index=False)
    json.dump(cbbd.get_lines(season), open(f"{config.DATA_RAW_DIR}/lines_raw_{season}.json", "w"))
    for name, fn, keys in (("game_teams", cbbd.get_game_teams, ["gameId", "team"]), ("players", cbbd.get_game_players, ["gameId", "athleteId"])):
        path = f"{config.DATA_RAW_DIR}/{name}_{season}.csv"
        old = pd.read_csv(path, low_memory=False)
        new = fn(season, since=since)
        both = pd.concat([old, new], ignore_index=True).drop_duplicates(keys, keep="last") if len(new) else old
        both.to_csv(path, index=False)
        print(f"  {name}: +{len(new)} rows (now {len(both)})")


def fetch(season):
    print(f"--- {season} ---")
    g = cbbd.get_games(season)
    g.to_csv(f"{config.DATA_RAW_DIR}/games_{season}.csv", index=False)
    if len(g) and "seasonLabel" in g.columns:
        print(f"  games: {len(g)}  (seasonLabel check: {g['seasonLabel'].iloc[0]})")
    cbbd.get_game_teams(season).to_csv(f"{config.DATA_RAW_DIR}/game_teams_{season}.csv", index=False)
    cbbd.get_game_players(season).to_csv(f"{config.DATA_RAW_DIR}/players_{season}.csv", index=False)
    json.dump(cbbd.get_lines(season), open(f"{config.DATA_RAW_DIR}/lines_raw_{season}.json", "w"))
    try:
        cbbd.get_adjusted_ratings(season).to_csv(f"{config.DATA_RAW_DIR}/adjusted_{season}.csv", index=False)
    except Exception as e:
        print(f"  [warn] adjusted ratings: {e}")
        pd.DataFrame(columns=["season", "team", "offensiveRating", "defensiveRating", "netRating"]).to_csv(
            f"{config.DATA_RAW_DIR}/adjusted_{season}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    cur = config.current_season()
    for s in a.seasons:
        if cached(s) and not a.force:
            if s == cur:
                incremental(s)
            else:
                print(f"--- {s}: cached ---")
            continue
        fetch(s)
