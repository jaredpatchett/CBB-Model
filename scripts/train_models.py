#!/usr/bin/env python3
"""
Train and save every model on FINISHED seasons (the current season is
excluded so live picks are always out-of-sample):
  models/margin.joblib            spread / moneyline (predicted home margin)
  models/total.joblib             game totals
  models/props_<stat>.joblib      points, rebounds, assists, threes
Writes docs/data/model_diagnostics.json (accuracy + top inputs).
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd

import config
from src.features import player_features as pf, team_features as tf
from src.models.gbm import GBMModel


def main():
    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config", "model.json")))
    ver = cfg.get("features", "base")
    mf, tfeat = tf.FEATURE_SETS[ver]
    cur = config.current_season()
    g = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/game_features.csv", low_memory=False)
    g = g[g["season"] < cur]
    g = g[g["home_games_prior"].fillna(0) + g["away_games_prior"].fillna(0) >= 2]
    diag = {}
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    for name, feats, target in (("margin", mf, "margin"), ("total", tfeat, "total")):
        m = GBMModel(feats, target).fit(g)
        m.save(f"{config.MODELS_DIR}/{name}.joblib")
        diag[name] = {"n": m.n_train, "holdout_mae": round(m.holdout_mae, 2), "sd": round(m.residual_std, 2), "top": m.importances(8)}
    p = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/player_features.csv", low_memory=False)
    p = p[(p["season"] < cur) & (p["p_games"] >= 3)]
    for stat in pf.STATS.values():
        m = GBMModel(pf.features_for(stat, ver), stat, n_estimators=200).fit(p)
        m.save(f"{config.MODELS_DIR}/props_{stat}.joblib")
        diag[f"props_{stat}"] = {"n": m.n_train, "holdout_mae": round(m.holdout_mae, 2), "sd": round(m.residual_std, 2), "top": m.importances(5)}
    os.makedirs(config.DOCS_DATA_DIR, exist_ok=True)
    json.dump(diag, open(f"{config.DOCS_DATA_DIR}/model_diagnostics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
