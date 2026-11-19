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
        tpa = _col(g, f"{pre}.threePointFieldGoals.attempted")
        out[f"three_rate_{side}"] = tpa / fga.replace(0, np.nan)
        out[f"tpa_{side}"] = tpa
        out[f"tpm_{side}"] = _col(g, f"{pre}.threePointFieldGoals.made")
        out[f"three_pct_{side}"] = out[f"tpm_{side}"] / tpa.replace(0, np.nan)
    return out.dropna(subset=["off_eff", "def_eff"])


# ------------------------------------------------------------------ v2 inputs (11/2026)
# Opponents' 3-point % is mostly luck over a few games, and markets overreact
# to hot and cold shooting. "x" efficiency swaps each game's 3P% for the
# league average: all of it on defense (the defense barely controls how the
# other team shoots once the shot is taken), half of it on offense (shooters
# do have skill).
OFF_3P_KEEP = 0.5


def add_three_luck(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    lg = d.groupby("season").apply(lambda x: x["tpm_o"].sum() / x["tpa_o"].sum() if x["tpa_o"].sum() else np.nan)
    lg3 = d["season"].map(lg).fillna(0.34)
    excess_o = (d["tpm_o"] - d["tpa_o"] * lg3) * 3          # points above an average-shooting night
    excess_d = (d["tpm_d"] - d["tpa_d"] * lg3) * 3
    d["xoff_eff"] = d["off_eff"] - 100 * (1 - OFF_3P_KEEP) * excess_o.fillna(0) / d["poss"]
    d["xdef_eff"] = d["def_eff"] - 100 * excess_d.fillna(0) / d["poss"]
    return d


def roster_inputs(long: pd.DataFrame, players: pd.DataFrame, prior: dict) -> pd.DataFrame:
    """Per team-game minute totals for the roster features: minutes played by
    players with 300+ D1 minutes last season (any school), by players back
    from THIS team, and minutes weighted by the strength (prior net rating)
    of the school each player played for last season. Transfers follow the
    player: CBBD athlete IDs carry across schools."""
    cols = ["gameId", "team", "m_all", "m_ret", "m_same", "m_tal", "m_taln"]
    if players is None or players.empty or "athleteId" not in players:
        return pd.DataFrame(columns=cols)
    p = players.copy()
    p["minutes"] = pd.to_numeric(p["minutes"], errors="coerce").fillna(0)
    p = p[p["minutes"] > 0]
    p["season"] = pd.to_numeric(p["season"], errors="coerce")
    tot = p.groupby(["athleteId", "season", "team"], as_index=False)["minutes"].sum()
    season_min = tot.groupby(["athleteId", "season"])["minutes"].sum().rename("prev_min")
    best = tot.sort_values("minutes").groupby(["athleteId", "season"]).tail(1).set_index(["athleteId", "season"])["team"].rename("prev_team")
    prev = pd.concat([season_min, best], axis=1).reset_index()
    prev["season"] = prev["season"] + 1
    p = p.merge(prev, on=["athleteId", "season"], how="left")
    ret = p["prev_min"].fillna(0) >= 300
    p["m_all"] = p["minutes"]
    p["m_ret"] = p["minutes"] * ret
    p["m_same"] = p["minutes"] * (ret & (p["prev_team"] == p["team"]))
    tal = pd.Series([(prior.get((int(s), t)) or {}).get("net") if isinstance(t, str) and pd.notna(s) else np.nan
                     for s, t in zip(p["season"], p["prev_team"])], index=p.index, dtype=float)
    tal = tal.where(ret)
    p["m_tal"] = (p["minutes"] * tal).fillna(0)
    p["m_taln"] = p["minutes"] * tal.notna()
    return p.groupby(["gameId", "team"], as_index=False)[["m_all", "m_ret", "m_same", "m_tal", "m_taln"]].sum()


def availability(long: pd.DataFrame, players: pd.DataFrame) -> pd.Series:
    """Share of the team's minutes (out of 200) from regulars (20+ min a game
    over 2+ games) who did NOT play in the team's previous game -- a rough
    injury/suspension flag, known before tip."""
    out = pd.Series(np.nan, index=long.index)
    if players is None or players.empty:
        return out
    p = players[pd.to_numeric(players["minutes"], errors="coerce").fillna(0) > 0]
    who = p.groupby(["gameId", "team"])["athleteId"].apply(set).to_dict()
    mins = p.groupby(["gameId", "team", "athleteId"])["minutes"].sum().to_dict()
    for (team, season), g in long.sort_values("start").groupby(["team", "season"]):
        tot, n, nxt = {}, {}, None
        for idx, gid in zip(g.index, g["gameId"]):
            out[idx] = nxt
            here = who.get((gid, team))
            if here is None:          # no box score (or the live placeholder row)
                nxt = None
                continue
            regs = {a: tot[a] / n[a] for a in tot if n[a] >= 2 and tot[a] / n[a] >= 20}
            nxt = sum(m for a, m in regs.items() if a not in here) / 200.0
            for a in here:
                tot[a] = tot.get(a, 0) + float(mins.get((gid, team, a), 0))
                n[a] = n.get(a, 0) + 1
    return out


def prior_ratings(adjusted: pd.DataFrame) -> dict:
    """(season, team) -> prior season's adjusted off/def/net rating."""
    out = {}
    if adjusted is None or adjusted.empty:
        return out
    for _, r in adjusted.iterrows():
        out[(int(r["season"]) + 1, r["team"])] = {"off": r.get("offensiveRating"), "def": r.get("defensiveRating"),
                                                  "net": r.get("netRating")}
    return out


def add_rolling(long: pd.DataFrame, prior: dict, players: pd.DataFrame = None) -> pd.DataFrame:
    """Pre-game rolling values per team (shift(1): never includes the game
    itself), opponent-adjusted with the opponent's PRIOR-season rating.
    `players` (box scores, this season and last) adds the roster and
    availability inputs."""
    d = add_three_luck(long).sort_values(["team", "start"])
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
    d["adj_xoff"] = d["xoff_eff"] + (lg_def - opp_def).fillna(0)
    d["adj_xdef"] = d["xdef_eff"] - (opp_off - lg_off).fillna(0)
    grp = d.groupby(["team", "season"])
    cols = ["adj_off", "adj_def", "tempo"] + [f"{f}_{s}" for f in FACTORS for s in ("o", "d")] + V2_ROLL_RAW
    for c in cols:
        d[f"r_{c}"] = grp[c].transform(lambda s: s.shift(1).ewm(halflife=HALF_LIFE_GAMES, min_periods=1).mean())
    d["games_prior"] = grp.cumcount()
    d["rest_days"] = grp["start"].transform(lambda s: (s - s.shift(1)).dt.total_seconds() / 86400).clip(upper=10)
    # schedule spots: consecutive games away from home before this one, games in the last 7 days
    away = (~d["is_home"].astype(bool)).astype(int)
    d["road_streak"] = away.groupby([d["team"], d["season"]]).transform(
        lambda s: s.shift(1).fillna(0).groupby((s.shift(1).fillna(0) == 0).cumsum()).cumsum())
    d["games_7d"] = grp["start"].transform(lambda s: pd.Series(
        [((s.iloc[:i] > t - pd.Timedelta(days=7))).sum() for i, t in enumerate(s)], index=s.index))
    # roster continuity and talent, from the minutes played so far this season
    r = roster_inputs(d, players, prior)
    d = d.drop(columns=[c for c in ("m_all", "m_ret", "m_same", "m_tal", "m_taln") if c in d.columns])
    d = d.merge(r, on=["gameId", "team"], how="left") if len(r) else d.assign(m_all=np.nan, m_ret=np.nan, m_same=np.nan, m_tal=np.nan, m_taln=np.nan)
    d = d.sort_values(["team", "start"])
    g2 = d.groupby(["team", "season"])
    cum = {c: g2[c].transform(lambda s: s.fillna(0).cumsum().shift(1)) for c in ("m_all", "m_ret", "m_same", "m_tal", "m_taln")}
    d["roster_ret"] = cum["m_ret"] / cum["m_all"].replace(0, np.nan)
    d["roster_same"] = cum["m_same"] / cum["m_all"].replace(0, np.nan)
    d["roster_talent"] = cum["m_tal"] / cum["m_taln"].replace(0, np.nan)
    d["out_last"] = availability(d, players)
    grp = d.groupby(["team", "season"])
    pr = [prior.get((int(s), t)) or {} for s, t in zip(d["season"].fillna(0), d["team"])]
    d["prior_off"] = [x.get("off") for x in pr]
    d["prior_def"] = [x.get("def") for x in pr]
    d["prior_net"] = [x.get("net") for x in pr]
    return d


V2_ROLL_RAW = ["adj_xoff", "adj_xdef", "three_pct_o", "three_pct_d"]
V2_TEAM = ["r_adj_xoff", "r_adj_xdef", "r_three_pct_o", "r_three_pct_d", "roster_ret", "roster_same", "roster_talent",
           "out_last", "road_streak", "games_7d"]
ROLL = ["r_adj_off", "r_adj_def", "r_tempo"] + [f"r_{f}_{s}" for f in FACTORS for s in ("o", "d")] + \
       ["games_prior", "rest_days", "prior_off", "prior_def", "prior_net"] + V2_TEAM


def build_game_features(games: pd.DataFrame, game_teams: pd.DataFrame, adjusted: pd.DataFrame,
                        players: pd.DataFrame = None, long: pd.DataFrame = None) -> pd.DataFrame:
    """One row per game: home_* / away_* team values, diffs, Elo, home court,
    and targets (margin = home - away, total)."""
    prior = prior_ratings(adjusted)
    if long is None:
        long = add_rolling(team_game_long(game_teams), prior, players)
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
    # v2: 3-point-luck-adjusted efficiency, roster, availability, schedule
    for c in V2_TEAM:
        d[f"{c}_diff"] = n(f"home_{c}") - n(f"away_{c}")
    d["xnet_diff"] = (n("home_r_adj_xoff") - n("home_r_adj_xdef")) - (n("away_r_adj_xoff") - n("away_r_adj_xdef"))
    d["exp_total_x"] = d["exp_poss"] * ((n("home_r_adj_xoff") + n("away_r_adj_xdef")) / 2
                                        + (n("away_r_adj_xoff") + n("home_r_adj_xdef")) / 2) / 100
    d["exp_margin_x"] = d["exp_poss"] * ((n("home_r_adj_xoff") - n("away_r_adj_xdef")) - (n("away_r_adj_xoff") - n("home_r_adj_xdef"))) / 200 \
        + 3.0 * d["home_court"]
    if "homePoints" in d.columns:
        hp, ap = pd.to_numeric(d["homePoints"], errors="coerce"), pd.to_numeric(d["awayPoints"], errors="coerce")
        d["margin"], d["total"] = hp - ap, hp + ap
    return d


MARGIN_FEATURES = ["home_court", "elo_diff", "net_diff", "r_adj_off_diff", "r_adj_def_diff", "r_tempo_diff",
                   "prior_net_diff", "exp_margin", "rest_days_diff", "back_to_back_diff",
                   "home_efg_matchup", "away_efg_matchup", "home_tov_matchup", "away_tov_matchup",
                   "home_orb_matchup", "away_orb_matchup", "home_ftr_matchup", "away_ftr_matchup",
                   "home_games_prior", "away_games_prior"]
TOTAL_FEATURES_BASE = None  # set below
TOTAL_FEATURES = ["exp_total", "exp_poss", "home_r_tempo", "away_r_tempo", "home_r_adj_off", "home_r_adj_def",
                  "away_r_adj_off", "away_r_adj_def", "home_prior_off", "home_prior_def", "away_prior_off",
                  "away_prior_def", "home_r_three_rate_o", "away_r_three_rate_o", "home_r_three_rate_d",
                  "away_r_three_rate_d", "home_r_ftr_o", "away_r_ftr_o", "home_r_orb_o", "away_r_orb_o",
                  "home_r_tov_o", "away_r_tov_o", "home_court", "home_games_prior", "away_games_prior",
                  "home_rest_days", "away_rest_days"]


MARGIN_FEATURES_BASE, TOTAL_FEATURES_BASE = list(MARGIN_FEATURES), list(TOTAL_FEATURES)
MARGIN_FEATURES_V2 = MARGIN_FEATURES_BASE + ["xnet_diff", "exp_margin_x", "r_adj_xoff_diff", "r_adj_xdef_diff",
                                             "roster_ret_diff", "roster_same_diff", "roster_talent_diff",
                                             "home_out_last", "away_out_last", "road_streak_diff", "games_7d_diff",
                                             "home_roster_talent", "away_roster_talent"]
TOTAL_FEATURES_V2 = TOTAL_FEATURES_BASE + ["exp_total_x", "home_r_adj_xoff", "home_r_adj_xdef", "away_r_adj_xoff",
                                           "away_r_adj_xdef", "home_r_three_pct_o", "away_r_three_pct_o",
                                           "home_r_three_pct_d", "away_r_three_pct_d", "home_roster_ret",
                                           "away_roster_ret", "home_out_last", "away_out_last", "home_games_7d",
                                           "away_games_7d"]
FEATURE_SETS = {"base": (MARGIN_FEATURES_BASE, TOTAL_FEATURES_BASE), "v2": (MARGIN_FEATURES_V2, TOTAL_FEATURES_V2)}


def current_team_states(game_teams: pd.DataFrame, adjusted_prior: pd.DataFrame, season: int,
                        players: pd.DataFrame = None) -> dict:
    """Live: each team's values going into its NEXT game, keyed by team name.
    Adds a placeholder "next game" row per team and runs the exact training
    pipeline (add_rolling), so live inputs can't drift from training inputs.
    `players`: box scores for this season and last (roster inputs)."""
    prior = prior_ratings(adjusted_prior)
    long = team_game_long(game_teams)
    long = long[long["season"] == season]
    if long.empty:
        return {}
    now = pd.Timestamp.now(tz="UTC")
    last = long.groupby("team")["start"].max()
    ph = pd.DataFrame({"gameId": [-(i + 1) for i in range(len(last))], "season": season, "team": last.index,
                       "start": [max(now, t + pd.Timedelta(hours=1)) for t in last], "opponent": None,
                       "is_home": False, "neutral": False})
    d = add_rolling(pd.concat([long, ph], ignore_index=True), prior, players)
    d = d[d["gameId"] < 0]
    out = {}
    for _, r in d.iterrows():
        e = {c: (None if pd.isna(r.get(c)) else float(r.get(c))) for c in ROLL if c not in ("rest_days",)}
        e["games_prior"] = int(r["games_prior"])
        e["last_start"] = last[r["team"]]
        out[r["team"]] = e
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
