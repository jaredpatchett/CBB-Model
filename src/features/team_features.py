"""
Team features for college basketball -- the CBB version of the CFB model's
team_features.py + team_features_v2.py.

Football -> basketball translation:
  SP+ (prior season)          -> adjusted efficiency (prior season, CBBD /ratings/adjusted)
  scoring margin / PPA        -> points per 100 possessions on offense and defense,
                                 opponent-adjusted, recent games weighted more
  plays per drive (pace)      -> tempo: possessions per 40 minutes
  success rate                -> the four factors: eFG%, turnover rate, offensive
                                 rebound %, free-throw rate (offense and defense)
  weather                     -> nothing (indoor sport); rest days and back-to-backs instead
  QB out                      -> (draft) not modeled yet; needs an injury feed
  CORE / Elo                  -> CBBD's pre-game Elo (homeTeamEloStart) -- known before tip

Leakage rules (same as CFB): every in-season value uses only games BEFORE
the one being predicted (shift(1) before averaging), and the season-level
adjusted rating always comes from the PRIOR season.
"""
import numpy as np
import pandas as pd

HALF_LIFE_GAMES = 5
FACTORS = ["efg", "tov", "orb", "ftr", "three_rate"]


def _col(df, *names):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def team_game_long(game_teams: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with that game's raw efficiency numbers,
    from CBBD /games/teams flattened (teamStats.* / opponentStats.*)."""
    g = game_teams.copy()
    out = pd.DataFrame({
        "gameId": g["gameId"], "season": _col(g, "season"), "start": pd.to_datetime(g["startDate"], utc=True, errors="coerce"),
        "team": g["team"], "opponent": g["opponent"],
        "is_home": g.get("isHome", False).fillna(False).astype(bool) if "isHome" in g else False,
        "neutral": g.get("neutralSite", False).fillna(False).astype(bool) if "neutralSite" in g else False,
    })
    poss = _col(g, "teamStats.possessions")
    opp_poss = _col(g, "opponentStats.possessions")
    pts = _col(g, "teamStats.points.total", "teamStats.points")
    opp_pts = _col(g, "opponentStats.points.total", "opponentStats.points")
    minutes = _col(g, "gameMinutes").fillna(40.0)
    avg_poss = (poss + opp_poss.fillna(poss)) / 2
    out["poss"] = avg_poss
    out["tempo"] = avg_poss * 40.0 / minutes                     # possessions per 40 min (OT-neutral)
    out["pts"], out["opp_pts"] = pts, opp_pts
    out["off_eff"] = 100 * pts / avg_poss
    out["def_eff"] = 100 * opp_pts / avg_poss
    for side, pre in (("o", "teamStats"), ("d", "opponentStats")):
        out[f"efg_{side}"] = _col(g, f"{pre}.fourFactors.effectiveFieldGoalPct")
        out[f"tov_{side}"] = _col(g, f"{pre}.fourFactors.turnoverRatio")
        out[f"orb_{side}"] = _col(g, f"{pre}.fourFactors.offensiveReboundPct")
        out[f"ftr_{side}"] = _col(g, f"{pre}.fourFactors.freeThrowRate")
        fga = _col(g, f"{pre}.fieldGoals.attempted")
        out[f"three_rate_{side}"] = _col(g, f"{pre}.threePointFieldGoals.attempted") / fga.replace(0, np.nan)
    return out.dropna(subset=["off_eff", "def_eff"])


def prior_ratings(adjusted: pd.DataFrame) -> dict:
    """(season, team) -> prior season's adjusted off/def/net rating."""
    out = {}
    if adjusted is None or adjusted.empty:
        return out
    for _, r in adjusted.iterrows():
        out[(int(r["season"]) + 1, r["team"])] = {"off": r.get("offensiveRating"), "def": r.get("defensiveRating"),
                                                  "net": r.get("netRating")}
    return out


def add_rolling(long: pd.DataFrame, prior: dict) -> pd.DataFrame:
    """Pre-game rolling values per team (shift(1): never includes the game
    itself), opponent-adjusted with the opponent's PRIOR-season rating."""
    d = long.sort_values(["team", "start"]).copy()
    lg_off = np.nanmean([v["off"] for v in prior.values() if v.get("off") is not None]) if prior else d["off_eff"].mean()
    lg_def = np.nanmean([v["def"] for v in prior.values() if v.get("def") is not None]) if prior else d["def_eff"].mean()
    opp = [prior.get((int(s), o)) or {} for s, o in zip(d["season"].fillna(0), d["opponent"])]
    opp_def = pd.Series([x.get("def") for x in opp], index=d.index, dtype=float)
    opp_off = pd.Series([x.get("off") for x in opp], index=d.index, dtype=float)
    # Scoring 110 against a defense that is 8 points better than average
    # counts as 118 against an average one; allowing 110 to an offense 8
    # points better than average counts as 102.
    d["adj_off"] = d["off_eff"] + (lg_def - opp_def).fillna(0)
    d["adj_def"] = d["def_eff"] - (opp_off - lg_off).fillna(0)
    grp = d.groupby(["team", "season"])
    cols = ["adj_off", "adj_def", "tempo"] + [f"{f}_{s}" for f in FACTORS for s in ("o", "d")]
    for c in cols:
        d[f"r_{c}"] = grp[c].transform(lambda s: s.shift(1).ewm(halflife=HALF_LIFE_GAMES, min_periods=1).mean())
    d["games_prior"] = grp.cumcount()
    d["rest_days"] = grp["start"].transform(lambda s: (s - s.shift(1)).dt.total_seconds() / 86400).clip(upper=10)
    pr = [prior.get((int(s), t)) or {} for s, t in zip(d["season"].fillna(0), d["team"])]
    d["prior_off"] = [x.get("off") for x in pr]
    d["prior_def"] = [x.get("def") for x in pr]
    d["prior_net"] = [x.get("net") for x in pr]
    return d


ROLL = ["r_adj_off", "r_adj_def", "r_tempo"] + [f"r_{f}_{s}" for f in FACTORS for s in ("o", "d")] + \
       ["games_prior", "rest_days", "prior_off", "prior_def", "prior_net"]


def build_game_features(games: pd.DataFrame, game_teams: pd.DataFrame, adjusted: pd.DataFrame) -> pd.DataFrame:
    """One row per game: home_* / away_* team values, diffs, Elo, home court,
    and targets (margin = home - away, total)."""
    prior = prior_ratings(adjusted)
    long = add_rolling(team_game_long(game_teams), prior)
    side = long[["gameId", "team"] + ROLL]
    g = games.copy()
    g = g[pd.to_numeric(g.get("homePoints"), errors="coerce").notna()] if "homePoints" in g else g
    df = g.merge(side.add_prefix("home_").rename(columns={"home_gameId": "id", "home_team": "homeTeam"}),
                 on=["id", "homeTeam"], how="left")
    df = df.merge(side.add_prefix("away_").rename(columns={"away_gameId": "id", "away_team": "awayTeam"}),
                  on=["id", "awayTeam"], how="left")
    return finish_features(df)


def finish_features(df: pd.DataFrame) -> pd.DataFrame:
    """Diffs and derived inputs; shared by training and live scoring."""
    d = df.copy()
    n = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d.columns else pd.Series(np.nan, index=d.index)
    d["home_court"] = np.where(d.get("neutralSite", False) == True, 0, 1)
    d["elo_diff"] = n("homeTeamEloStart") - n("awayTeamEloStart")
    for c in ["r_adj_off", "r_adj_def", "r_tempo", "prior_net", "rest_days"] + [f"r_{f}_{s}" for f in FACTORS for s in ("o", "d")]:
        d[f"{c}_diff"] = n(f"home_{c}") - n(f"away_{c}")
    d["net_diff"] = (n("home_r_adj_off") - n("home_r_adj_def")) - (n("away_r_adj_off") - n("away_r_adj_def"))
    # Matchup edges: each offense's four factors against the other defense's.
    for f in ("efg", "tov", "orb", "ftr"):
        d[f"home_{f}_matchup"] = n(f"home_r_{f}_o") - n(f"away_r_{f}_d")
        d[f"away_{f}_matchup"] = n(f"away_r_{f}_o") - n(f"home_r_{f}_d")
    d["exp_poss"] = (n("home_r_tempo") + n("away_r_tempo")) / 2
    d["exp_total"] = d["exp_poss"] * ((n("home_r_adj_off") + n("away_r_adj_def")) / 2
                                      + (n("away_r_adj_off") + n("home_r_adj_def")) / 2) / 100
    d["exp_margin"] = d["exp_poss"] * ((n("home_r_adj_off") - n("away_r_adj_def")) - (n("away_r_adj_off") - n("home_r_adj_def"))) / 200 \
        + 3.0 * d["home_court"]
    d["back_to_back_diff"] = (n("home_rest_days") <= 1).astype(float) - (n("away_rest_days") <= 1).astype(float)
    if "homePoints" in d.columns:
        hp, ap = pd.to_numeric(d["homePoints"], errors="coerce"), pd.to_numeric(d["awayPoints"], errors="coerce")
        d["margin"], d["total"] = hp - ap, hp + ap
    return d


MARGIN_FEATURES = ["home_court", "elo_diff", "net_diff", "r_adj_off_diff", "r_adj_def_diff", "r_tempo_diff",
                   "prior_net_diff", "exp_margin", "rest_days_diff", "back_to_back_diff",
                   "home_efg_matchup", "away_efg_matchup", "home_tov_matchup", "away_tov_matchup",
                   "home_orb_matchup", "away_orb_matchup", "home_ftr_matchup", "away_ftr_matchup",
                   "home_games_prior", "away_games_prior"]
TOTAL_FEATURES = ["exp_total", "exp_poss", "home_r_tempo", "away_r_tempo", "home_r_adj_off", "home_r_adj_def",
                  "away_r_adj_off", "away_r_adj_def", "home_prior_off", "home_prior_def", "away_prior_off",
                  "away_prior_def", "home_r_three_rate_o", "away_r_three_rate_o", "home_r_three_rate_d",
                  "away_r_three_rate_d", "home_r_ftr_o", "away_r_ftr_o", "home_r_orb_o", "away_r_orb_o",
                  "home_r_tov_o", "away_r_tov_o", "home_court", "home_games_prior", "away_games_prior",
                  "home_rest_days", "away_rest_days"]


def current_team_states(game_teams: pd.DataFrame, adjusted_prior: pd.DataFrame, season: int) -> dict:
    """Live: each team's values going into its NEXT game (all completed games
    this season, no shift), keyed by team name. Same math as add_rolling."""
    prior = prior_ratings(adjusted_prior)
    long = team_game_long(game_teams)
    long = long[long["season"] == season]
    if long.empty:
        return {}
    d = add_rolling(long, prior)
    out = {}
    lg_off = np.nanmean([v["off"] for v in prior.values() if v.get("off") is not None]) if prior else d["off_eff"].mean()
    for team, g in d.sort_values("start").groupby("team"):
        e = {"games_prior": int(len(g)), "last_start": g["start"].max()}
        for c in ["adj_off", "adj_def", "tempo"] + [f"{f}_{s}" for f in FACTORS for s in ("o", "d")]:
            v = pd.to_numeric(g[c], errors="coerce").dropna()
            e[f"r_{c}"] = float(v.ewm(halflife=HALF_LIFE_GAMES).mean().iloc[-1]) if len(v) else None
        p = prior.get((season, team)) or {}
        e.update({"prior_off": p.get("off"), "prior_def": p.get("def"), "prior_net": p.get("net")})
        out[team] = e
    return out


def live_game_row(home: str, away: str, states: dict, kickoff, neutral=False, elo_home=None, elo_away=None) -> dict:
    """One feature row for an upcoming game from current_team_states."""
    row = {"homeTeam": home, "awayTeam": away, "neutralSite": bool(neutral),
           "homeTeamEloStart": elo_home, "awayTeamEloStart": elo_away}
    k = pd.Timestamp(kickoff)
    k = k.tz_localize("UTC") if k.tzinfo is None else k
    for side, t in (("home", home), ("away", away)):
        s = states.get(t) or {}
        for c in ROLL:
            if c == "rest_days":
                row[f"{side}_rest_days"] = min(max((k - s["last_start"]).total_seconds() / 86400, 0), 10) if s.get("last_start") is not None else 4.0
            else:
                row[f"{side}_{c}"] = s.get(c)
    return row
