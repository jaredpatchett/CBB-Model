#!/usr/bin/env python3
"""
Demo data (11/2026): writes SYNTHETIC college basketball data in exactly the
shapes the real feeds produce (CBBD /games, /games/teams flattened,
/games/players, /lines, /ratings/adjusted; The Odds API game lines and
props), so the whole pipeline -- features, training, backtest, export,
dashboard, alerts -- runs with no API keys.

Every number here is made up. It exists to build and test the code and to
preview the dashboard. Run the real pipeline (GitHub Actions, with keys)
for real numbers.

Usage:
  CBB_DEMO=1 python scripts/demo_data.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd

import config

RNG = np.random.default_rng(7)
TEAMS = {
    "Duke": "Blue Devils", "North Carolina": "Tar Heels", "Kansas": "Jayhawks", "Kentucky": "Wildcats",
    "Houston": "Cougars", "Purdue": "Boilermakers", "UConn": "Huskies", "Arizona": "Wildcats",
    "Arizona State": "Sun Devils", "Grand Canyon": "Antelopes", "Gonzaga": "Bulldogs", "Baylor": "Bears",
    "Tennessee": "Volunteers", "Auburn": "Tigers", "Alabama": "Crimson Tide", "Florida": "Gators",
    "Michigan State": "Spartans", "Michigan": "Wolverines", "Illinois": "Fighting Illini", "Wisconsin": "Badgers",
    "Indiana": "Hoosiers", "Iowa": "Hawkeyes", "Ohio State": "Buckeyes", "Marquette": "Golden Eagles",
    "Creighton": "Bluejays", "Villanova": "Wildcats", "St. John's": "Red Storm", "Xavier": "Musketeers",
    "Texas": "Longhorns", "Texas Tech": "Red Raiders", "Iowa State": "Cyclones", "BYU": "Cougars",
    "UCLA": "Bruins", "USC": "Trojans", "Oregon": "Ducks", "Utah": "Utes",
    "San Diego State": "Aztecs", "Saint Mary's": "Gaels", "Memphis": "Tigers", "Florida Atlantic": "Owls",
    "Virginia": "Cavaliers", "Clemson": "Tigers", "Miami": "Hurricanes", "Louisville": "Cardinals",
    "Wake Forest": "Demon Deacons", "NC State": "Wolfpack", "Pittsburgh": "Panthers", "Syracuse": "Orange",
    "Arkansas": "Razorbacks", "LSU": "Tigers", "Ole Miss": "Rebels", "Mississippi State": "Bulldogs",
    "Missouri": "Tigers", "Texas A&M": "Aggies", "Oklahoma": "Sooners", "Cincinnati": "Bearcats",
    "Dayton": "Flyers", "VCU": "Rams", "New Mexico": "Lobos", "Boise State": "Broncos",
    "Nevada": "Wolf Pack", "Colorado State": "Rams", "Drake": "Bulldogs", "McNeese": "Cowboys",
}
NAMES = list(TEAMS)
FIRST = ["Jalen", "Marcus", "Tyrese", "Caleb", "Andre", "Isaiah", "Jordan", "Malik", "Cam", "Darius", "Kobe", "Trey",
         "Elijah", "Devin", "Jaylen", "Zion", "Amari", "Keon", "Nolan", "Ryan"]
LAST = ["Walker", "Johnson", "Brooks", "Carter", "Mitchell", "Hayes", "Foster", "Coleman", "Reed", "Bryant", "Price",
        "Hill", "Ward", "Ellis", "Barnes", "Fields", "Hunt", "Shaw", "Grant", "Wells"]


def team_truth(season):
    t = {}
    for n in NAMES:
        base = RNG.normal(0, 7)
        t[n] = {"off": 108 + base / 2 + RNG.normal(0, 4), "def": 104 - base / 2 + RNG.normal(0, 4),
                "tempo": RNG.normal(68, 3.0), "three": RNG.uniform(0.32, 0.45)}
    return t


def roster(team):
    rng = np.random.default_rng(abs(hash(team)) % 2**32)
    players = []
    shares = np.array([0.24, 0.21, 0.17, 0.14, 0.11, 0.07, 0.04, 0.02])
    mins = np.array([34, 32, 30, 28, 26, 20, 16, 10])
    for i in range(8):
        players.append({"athleteId": abs(hash(team + str(i))) % 10**7, "name": f"{FIRST[(i * 7 + len(team)) % len(FIRST)]} {LAST[(i * 3 + sum(map(ord, team))) % len(LAST)]}",
                        "position": ["G", "G", "F", "F", "C", "G", "F", "C"][i], "share": shares[i], "minutes": mins[i],
                        "reb_rate": [0.07, 0.08, 0.14, 0.16, 0.22, 0.07, 0.12, 0.18][i],
                        "ast_rate": [0.30, 0.20, 0.12, 0.10, 0.06, 0.15, 0.06, 0.04][i], "three_share": [0.3, 0.25, 0.2, 0.12, 0.02, 0.2, 0.1, 0.0][i]})
    return players


def simulate_season(season, truth, start, end, played_through=None):
    days = pd.date_range(start, end, freq="D", tz="UTC")
    games, gteams, players, lines = [], [], [], []
    gid = season * 100000
    elo = {n: 1500 + (truth[n]["off"] - truth[n]["def"]) * 12 for n in NAMES}
    for day in days:
        if RNG.random() < 0.35:
            continue
        today = RNG.permutation(NAMES)[:int(RNG.integers(8, 24))]
        for i in range(0, len(today) - 1, 2):
            h, a = today[i], today[i + 1]
            gid += 1
            neutral = RNG.random() < 0.08
            th, ta = truth[h], truth[a]
            poss = (th["tempo"] + ta["tempo"]) / 2 + RNG.normal(0, 3)
            hc = 0 if neutral else 1.6
            h_eff = (th["off"] + ta["def"]) / 2 + hc + RNG.normal(0, 9)
            a_eff = (ta["off"] + th["def"]) / 2 - hc + RNG.normal(0, 9)
            hp, ap = round(poss * h_eff / 100), round(poss * a_eff / 100)
            if hp == ap:
                hp += 1
            start_dt = day + pd.Timedelta(hours=int(RNG.integers(23, 27)))
            future = played_through is not None and start_dt > played_through
            exp_m = poss * ((th["off"] - ta["def"]) - (ta["off"] - th["def"])) / 200 + 2 * hc
            exp_t = poss * (th["off"] + ta["def"] + ta["off"] + th["def"]) / 200
            mkt_spread = -round((exp_m + RNG.normal(0, 2.2)) * 2) / 2
            mkt_total = round((exp_t + RNG.normal(0, 3.0)) * 2) / 2
            games.append({"id": gid, "season": season, "seasonLabel": f"{season-1}{season}", "seasonType": "regular",
                          "startDate": start_dt.isoformat(), "neutralSite": neutral, "status": "scheduled" if future else "final",
                          "homeTeam": h, "awayTeam": a, "homePoints": None if future else hp, "awayPoints": None if future else ap,
                          "homeTeamEloStart": round(elo[h]), "awayTeamEloStart": round(elo[a])})
            lines.append({"gameId": gid, "season": season, "startDate": start_dt.isoformat(), "homeTeam": h, "awayTeam": a,
                          "homeScore": None if future else hp, "awayScore": None if future else ap,
                          "lines": [{"provider": "consensus", "spread": mkt_spread, "overUnder": mkt_total,
                                     "spreadOpen": mkt_spread + RNG.choice([-1, -0.5, 0, 0, 0.5, 1]),
                                     "overUnderOpen": mkt_total + RNG.choice([-2, -1, 0, 0, 1, 2]),
                                     "homeMoneyline": None, "awayMoneyline": None}]})
            if future:
                continue
            exp_h = 1 / (1 + 10 ** ((elo[a] - elo[h] - 60 * hc) / 400))
            elo[h] += 20 * ((hp > ap) - exp_h)
            elo[a] -= 20 * ((hp > ap) - exp_h)
            for team, opp, pts, opts, t, o in ((h, a, hp, ap, th, ta), (a, h, ap, hp, ta, th)):
                fga = poss * 0.88
                three_att = fga * t["three"] * RNG.uniform(0.85, 1.15)
                row = {"gameId": gid, "season": season, "seasonType": "regular", "startDate": start_dt.isoformat(),
                       "team": team, "opponent": opp, "isHome": team == h and not neutral, "neutralSite": neutral,
                       "gameMinutes": 40, "pace": poss,
                       "teamStats.possessions": round(poss + RNG.normal(0, 0.8), 1),
                       "opponentStats.possessions": round(poss + RNG.normal(0, 0.8), 1),
                       "teamStats.points.total": pts, "opponentStats.points.total": opts,
                       "teamStats.fieldGoals.attempted": round(fga), "opponentStats.fieldGoals.attempted": round(fga),
                       "teamStats.threePointFieldGoals.attempted": round(three_att),
                       "opponentStats.threePointFieldGoals.attempted": round(fga * o["three"])}
                for side, eff in (("teamStats", pts / poss), ("opponentStats", opts / poss)):
                    row[f"{side}.fourFactors.effectiveFieldGoalPct"] = round(0.25 + 0.27 * eff + RNG.normal(0, 0.02), 3)
                    row[f"{side}.fourFactors.turnoverRatio"] = round(RNG.normal(0.17, 0.03) - 0.04 * (eff - 1.05), 3)
                    row[f"{side}.fourFactors.offensiveReboundPct"] = round(RNG.normal(0.30, 0.05), 3)
                    row[f"{side}.fourFactors.freeThrowRate"] = round(RNG.normal(0.32, 0.07), 3)
                gteams.append(row)
                for p in roster(team):
                    m = max(0, p["minutes"] + RNG.normal(0, 4))
                    ppts = max(0, round(pts * p["share"] * m / p["minutes"] + RNG.normal(0, 3)))
                    players.append({"gameId": gid, "season": season, "startDate": start_dt.isoformat(), "team": team,
                                    "opponent": opp, "isHome": team == h, "neutralSite": neutral, "gamePace": poss,
                                    "seasonType": "regular", "athleteId": p["athleteId"], "name": p["name"],
                                    "position": p["position"], "starter": p["minutes"] >= 26, "minutes": round(m, 1),
                                    "points": ppts, "rebounds": max(0, round(poss * 0.75 * p["reb_rate"] * m / 40 + RNG.normal(0, 1.5))),
                                    "assists": max(0, round(pts / 2.3 * p["ast_rate"] * m / p["minutes"] + RNG.normal(0, 1.2))),
                                    "threes": int(RNG.binomial(max(int(three_att * p["three_share"]), 0), 0.35)),
                                    "threes_att": int(three_att * p["three_share"]), "usage": round(p["share"] * 100 / 1.1, 1), "fouls": int(RNG.integers(0, 5))})
    return games, gteams, players, lines, truth


def main():
    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    os.makedirs(config.DATA_CURRENT_DIR, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC").floor("h")
    cur = config.current_season(now.to_pydatetime())
    prev_truth = None
    for season in range(cur - 3, cur + 1):
        truth = team_truth(season)
        if prev_truth:   # teams carry over some strength year to year
            for n in NAMES:
                for k in ("off", "def", "tempo"):
                    truth[n][k] = 0.6 * prev_truth[n][k] + 0.4 * truth[n][k]
        if season < cur:
            prev_players = None
            g, t, p, l, _ = simulate_season(season, truth, f"{season-1}-11-04", f"{season}-03-08")
        else:
            g, t, p, l, _ = simulate_season(season, truth, f"{season-1}-11-03", (now + pd.Timedelta(days=2)).strftime("%Y-%m-%d"), played_through=now)
        pd.DataFrame(g).to_csv(f"{config.DATA_RAW_DIR}/games_{season}.csv", index=False)
        pd.DataFrame(t).to_csv(f"{config.DATA_RAW_DIR}/game_teams_{season}.csv", index=False)
        pd.DataFrame(p).to_csv(f"{config.DATA_RAW_DIR}/players_{season}.csv", index=False)
        json.dump(l, open(f"{config.DATA_RAW_DIR}/lines_raw_{season}.json", "w"))
        adj = [{"season": season, "team": n, "offensiveRating": round(truth[n]["off"] + RNG.normal(0, 1.5), 1),
                "defensiveRating": round(truth[n]["def"] + RNG.normal(0, 1.5), 1)} for n in NAMES]
        for a in adj:
            a["netRating"] = round(a["offensiveRating"] - a["defensiveRating"], 1)
        pd.DataFrame(adj).to_csv(f"{config.DATA_RAW_DIR}/adjusted_{season}.csv", index=False)
        print(f"  demo season {season}: {len(g)} games")
        prev_truth = truth
        if season < cur:
            prev_players = p
    # Current slate in The Odds API's shape (+ props), from this season's unplayed games.
    games = pd.DataFrame(g)
    lines = {x["gameId"]: x["lines"][0] for x in l}
    up = games[games["homePoints"].isna()]
    rows, props = [], []
    pall = pd.concat([pd.DataFrame(pp) for pp in (prev_players, p)], ignore_index=True)
    pall = pall.sort_values("startDate").groupby("athleteId").tail(8)
    avg = pall.groupby("athleteId")[["points", "rebounds", "assists", "threes"]].mean().to_dict("index")
    for _, r in up.iterrows():
        ln = lines[r["id"]]
        rows.append({"game_id": f"demo{r['id']}", "commence_time": r["startDate"], "home_team": f"{r['homeTeam']} {TEAMS[r['homeTeam']]}",
                     "away_team": f"{r['awayTeam']} {TEAMS[r['awayTeam']]}", "book_used": "draftkings",
                     "spread_home": ln["spread"], "total": ln["overUnder"], "total_over_price": -110, "total_under_price": -110,
                     "ml_home": None, "ml_away": None})
        if RNG.random() < 0.6:
            for team in (r["homeTeam"], r["awayTeam"]):
                for p in roster(team)[:4]:
                    a = avg.get(p["athleteId"]) or {}
                    for mk, col, noise in (("Points", "points", 1.5), ("Rebounds", "rebounds", 0.8),
                                           ("Assists", "assists", 0.6), ("Threes", "threes", 0.4)):
                        if a.get(col) is None:
                            continue
                        line = max(0.5, np.floor(a[col] + RNG.normal(0, noise)) + 0.5)
                        props.append({"fixture_id": f"demo{r['id']}", "start_time": r["startDate"],
                                      "home_team": f"{r['homeTeam']} {TEAMS[r['homeTeam']]}", "away_team": f"{r['awayTeam']} {TEAMS[r['awayTeam']]}",
                                      "player_name": p["name"], "market_name": mk, "line": line,
                                      "over_price": int(RNG.choice([-125, -115, -110, -105, 100, 105])),
                                      "under_price": int(RNG.choice([-125, -115, -110, -105, 100, 105])), "book_used": "draftkings"})
    pd.DataFrame(rows).to_csv(f"{config.DATA_CURRENT_DIR}/game_lines.csv", index=False)
    pd.DataFrame(props).to_csv(f"{config.DATA_CURRENT_DIR}/props.csv", index=False)
    print(f"  demo slate: {len(rows)} upcoming games, {len(props)} prop lines")


if __name__ == "__main__":
    main()
