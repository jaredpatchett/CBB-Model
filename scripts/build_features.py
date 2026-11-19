#!/usr/bin/env python3
"""
Raw data -> model-ready tables (data/processed/):
  game_features.csv    one row per game (spread + totals models)
  player_features.csv  one row per player per game (props models)
  team_long.csv        one row per team per game, pre-game rolling values

Usage: python scripts/build_features.py [--seasons ...]   (default: every saved season)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.data import load
from src.features import team_features as tf, player_features as pf


def main(seasons):
    g, gt, pl = load.games(seasons), load.game_teams(seasons), load.players(seasons)
    adj = load.adjusted([s - 1 for s in seasons] + seasons)
    long = tf.add_rolling(tf.team_game_long(gt), tf.prior_ratings(adj), pl)
    feats = tf.build_game_features(g, gt, adj, long=long)
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    feats.to_csv(f"{config.DATA_PROCESSED_DIR}/game_features.csv", index=False)
    long.to_csv(f"{config.DATA_PROCESSED_DIR}/team_long.csv", index=False)
    p = pf.attach_environment(pf.add_player_rolling(pf.player_long(pl)), long)
    p.to_csv(f"{config.DATA_PROCESSED_DIR}/player_features.csv", index=False)
    print(f"game_features: {len(feats)} games | team_long: {len(long)} | player_features: {len(p)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*")
    a = ap.parse_args()
    main(a.seasons or load.saved_seasons())
