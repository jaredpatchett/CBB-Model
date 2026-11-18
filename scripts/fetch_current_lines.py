#!/usr/bin/env python3
"""
Today's lines from The Odds API -> data/current/game_lines.csv and
data/current/props.csv (props only for games within ~2 days, one call per
game, so it is the expensive part of the Odds API quota).
In demo mode (CBB_DEMO=1) the files written by demo_data.py are used.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.data import odds_api_client as odds

if __name__ == "__main__":
    if config.DEMO:
        print("demo mode: using data/current from demo_data.py")
        sys.exit(0)
    os.makedirs(config.DATA_CURRENT_DIR, exist_ok=True)
    raw = odds.get_game_odds()
    df = odds.odds_to_dataframe(raw)
    df.to_csv(f"{config.DATA_CURRENT_DIR}/game_lines.csv", index=False)
    print(f"game lines: {len(df)} games")
    props = odds.get_player_props(raw)
    props.to_csv(f"{config.DATA_CURRENT_DIR}/props.csv", index=False)
    print(f"props: {len(props)} lines")
