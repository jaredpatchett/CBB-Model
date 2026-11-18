"""
Player features for props (points, rebounds, assists, threes) -- the CBB
version of the CFB props inputs (routes/targets/usage -> minutes/usage).

Per player, before each game (shift(1), recent games weighted more):
  minutes, usage, starter share, each stat per game and per 40 minutes,
  games played; plus the game environment: team and opponent tempo and the
  opponent's adjusted defensive efficiency (from team_features), and home.
A projection is roughly minutes x per-minute rate x pace, adjusted for the
defense -- the GBM learns how to combine them.
"""
import numpy as np
import pandas as pd

STATS = {"Points": "points", "Rebounds": "rebounds", "Assists": "assists", "Threes": "threes"}
HALF_LIFE = 4


def player_long(players: pd.DataFrame) -> pd.DataFrame:
    d = players.copy()
    d["start"] = pd.to_datetime(d["startDate"], utc=True, errors="coerce")
    for c in ["minutes", "points", "rebounds", "assists", "threes", "threes_att", "usage"]:
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d["starter"] = d.get("starter", False).fillna(False).astype(float) if "starter" in d else 0.0
    return d[d["minutes"].fillna(0) > 0].sort_values(["athleteId", "start"])


def add_player_rolling(d: pd.DataFrame, shift=True) -> pd.DataFrame:
    d = d.sort_values(["athleteId", "start"]).copy()
    grp = d.groupby(["athleteId", "season"])
    sh = (lambda s: s.shift(1)) if shift else (lambda s: s)
    ew = lambda s: sh(s).ewm(halflife=HALF_LIFE, min_periods=1).mean()
    for c in ["minutes", "usage", "starter", "points", "rebounds", "assists", "threes", "threes_att"]:
        d[f"p_{c}"] = grp[c].transform(ew)
    for c in ["points", "rebounds", "assists", "threes"]:
        per40 = 40 * d[c] / d["minutes"].replace(0, np.nan)
        d[f"p_{c}_per40"] = per40.groupby([d["athleteId"], d["season"]]).transform(ew)
    d["p_games"] = grp.cumcount() if shift else grp.cumcount() + 1
    return d


PROP_FEATURES = ["p_minutes", "p_usage", "p_starter", "p_games", "team_tempo", "opp_tempo", "opp_adj_def",
                 "is_home"]


def features_for(stat: str):
    return PROP_FEATURES + [f"p_{stat}", f"p_{stat}_per40"] + (["p_threes_att"] if stat == "threes" else [])


def attach_environment(d: pd.DataFrame, team_long: pd.DataFrame) -> pd.DataFrame:
    """Team/opponent pre-game tempo and the opponent's adjusted defense, from
    team_features.add_rolling output (columns r_tempo, r_adj_def)."""
    t = team_long[["gameId", "team", "r_tempo", "r_adj_def"]]
    d = d.merge(t.rename(columns={"r_tempo": "team_tempo", "r_adj_def": "_own_def"}), on=["gameId", "team"], how="left")
    d = d.merge(t.rename(columns={"team": "opponent", "r_tempo": "opp_tempo", "r_adj_def": "opp_adj_def"}),
                on=["gameId", "opponent"], how="left")
    d["is_home"] = d.get("isHome", False).fillna(False).astype(float) if "isHome" in d else 0.0
    return d.drop(columns=["_own_def"], errors="ignore")


def current_player_states(players: pd.DataFrame, season: int) -> dict:
    """Live: (team, player name) -> rolling values going into the next game."""
    d = player_long(players)
    d = d[d["season"] == season]
    if d.empty:
        return {}
    d = add_player_rolling(d, shift=False)
    last = d.sort_values("start").groupby("athleteId").tail(1)
    out = {}
    for _, r in last.iterrows():
        out[(r["team"], str(r["name"]).lower())] = {c: r[c] for c in r.index if c.startswith("p_")} | {
            "athleteId": r["athleteId"], "name": r["name"], "team": r["team"]}
    return out
