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
    # v2: last season carried in (fixes the first weeks, when p_* rest on 1-4 games; transfers keep theirs)
    tot = d.groupby(["athleteId", "season"]).agg(_m=("minutes", "sum"), _g=("minutes", "size"),
                                                  **{f"_{c}": (c, "sum") for c in ["points", "rebounds", "assists", "threes"]})
    prev = pd.DataFrame({"athleteId": tot.index.get_level_values(0), "season": tot.index.get_level_values(1) + 1,
                         "p_prev_minutes": (tot["_m"] / tot["_g"]).values})
    for c in ["points", "rebounds", "assists", "threes"]:
        prev[f"p_prev_{c}_per40"] = (40 * tot[f"_{c}"] / tot["_m"].replace(0, np.nan)).values
    d = d.drop(columns=[c for c in prev.columns if c.startswith("p_prev_") and c in d.columns]).merge(prev, on=["athleteId", "season"], how="left")
    return d


PROP_FEATURES = ["p_minutes", "p_usage", "p_starter", "p_games", "team_tempo", "opp_tempo", "opp_adj_def",
                 "is_home"]


PROP_FEATURES_V2 = PROP_FEATURES + ["p_prev_minutes", "team_out_last"]


def features_for(stat: str, version: str = "base"):
    base = PROP_FEATURES if version == "base" else PROP_FEATURES_V2
    extra = [f"p_prev_{stat}_per40"] if version != "base" else []
    return base + [f"p_{stat}", f"p_{stat}_per40"] + extra + (["p_threes_att"] if stat == "threes" else [])


def attach_environment(d: pd.DataFrame, team_long: pd.DataFrame) -> pd.DataFrame:
    """Team/opponent pre-game tempo and the opponent's adjusted defense, from
    team_features.add_rolling output (columns r_tempo, r_adj_def)."""
    t = team_long[["gameId", "team", "r_tempo", "r_adj_def"] + (["out_last"] if "out_last" in team_long else [])]
    d = d.merge(t.rename(columns={"r_tempo": "team_tempo", "r_adj_def": "_own_def", "out_last": "team_out_last"}),
                on=["gameId", "team"], how="left")
    d = d.merge(t.drop(columns=["out_last"], errors="ignore").rename(
        columns={"team": "opponent", "r_tempo": "opp_tempo", "r_adj_def": "opp_adj_def"}), on=["gameId", "opponent"], how="left")
    d["is_home"] = d.get("isHome", False).fillna(False).astype(float) if "isHome" in d else 0.0
    return d.drop(columns=["_own_def"], errors="ignore")


def current_player_states(players: pd.DataFrame, season: int) -> dict:
    """Live: (team, player name) -> rolling values going into the next game.
    `players` may include last season too (for the p_prev_* inputs)."""
    d = player_long(players)
    d = d[d["season"].isin([season, season - 1])]
    if d.empty or not (d["season"] == season).any():
        return {}
    d = add_player_rolling(d, shift=False)
    d = d[d["season"] == season]
    last = d.sort_values("start").groupby("athleteId").tail(1)
    out = {}
    for _, r in last.iterrows():
        out[(r["team"], str(r["name"]).lower())] = {c: r[c] for c in r.index if c.startswith("p_")} | {
            "athleteId": r["athleteId"], "name": r["name"], "team": r["team"]}
    return out
