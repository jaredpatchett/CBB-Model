"""Load the saved raw files (real or demo) for a list of seasons."""
import glob
import json
import os

import pandas as pd

import config
from src.data.cbbd_client import lines_to_dataframe


def _cat(pattern, seasons):
    frames = []
    for s in seasons:
        p = f"{config.DATA_RAW_DIR}/{pattern}_{s}.csv"
        if os.path.exists(p) and os.path.getsize(p) > 2:
            frames.append(pd.read_csv(p, low_memory=False))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def games(seasons):
    return _cat("games", seasons)


def game_teams(seasons):
    return _cat("game_teams", seasons)


def players(seasons):
    return _cat("players", seasons)


def adjusted(seasons):
    return _cat("adjusted", seasons)


def lines(seasons):
    frames = []
    for s in seasons:
        p = f"{config.DATA_RAW_DIR}/lines_raw_{s}.json"
        if os.path.exists(p):
            df = lines_to_dataframe(json.load(open(p)))
            df["season"] = s
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def saved_seasons():
    return sorted(int(os.path.basename(p)[6:10]) for p in glob.glob(f"{config.DATA_RAW_DIR}/games_*.csv"))
