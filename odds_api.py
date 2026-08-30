
from datetime import datetime, timezone
import requests
import pandas as pd

BASE="https://api.the-odds-api.com/v4"
SPORT_KEYS={"NBA":"basketball_nba","NFL":"americanfootball_nfl"}

NBA_PROPS=[
 "player_points","player_rebounds","player_assists","player_threes",
 "player_points_rebounds_assists","player_points_rebounds",
 "player_points_assists","player_rebounds_assists","player_blocks",
 "player_steals","player_turnovers"
]
NFL_PROPS=[
 "player_pass_yds","player_pass_tds","player_pass_attempts",
 "player_pass_completions","player_pass_interceptions","player_rush_yds",
 "player_rush_attempts","player_receptions","player_reception_yds",
 "player_anytime_td"
]

def _get(path,key,params=None):
    if not key:
        raise RuntimeError("Add your The Odds API key in Settings.")
    p=dict(params or {})
    p["apiKey"]=key
    r=requests.get(BASE+path,params=p,timeout=30)
    if not r.ok:
        raise RuntimeError(f"Odds API {r.status_code}: {r.text[:400]}")
    return r.json(),r.headers

def featured(sport,key,regions="us,us2"):
    data,h=_get(f"/sports/{SPORT_KEYS[sport]}/odds",key,{
      "regions":regions,"markets":"h2h,spreads,totals",
      "oddsFormat":"american","dateFormat":"iso"
    })
    now=datetime.now(timezone.utc).isoformat()
    rows=[]
    for ev in data:
        for book in ev.get("bookmakers",[]):
            for market in book.get("markets",[]):
                for out in market.get("outcomes",[]):
                    rows.append({
                      "captured_at":now,"sport":sport,"event_id":ev.get("id"),
                      "commence_time":ev.get("commence_time"),
                      "home_team":ev.get("home_team"),"away_team":ev.get("away_team"),
                      "bookmaker":book.get("title"),"market":market.get("key"),
                      "outcome":out.get("name"),"point":out.get("point"),
                      "price":out.get("price")
                    })
    return pd.DataFrame(rows),h

def scores(sport,key):
    data,h=_get(f"/sports/{SPORT_KEYS[sport]}/scores",key,{
      "daysFrom":3,"dateFormat":"iso"
    })
    now=datetime.now(timezone.utc).isoformat()
    rows=[]
    for ev in data:
        sm={x.get("name"):x.get("score") for x in (ev.get("scores") or [])}
        home,away=ev.get("home_team"),ev.get("away_team")
        rows.append({
          "event_id":ev.get("id"),"sport":sport,"commence_time":ev.get("commence_time"),
          "home_team":home,"away_team":away,
          "home_score":pd.to_numeric(sm.get(home),errors="coerce"),
          "away_score":pd.to_numeric(sm.get(away),errors="coerce"),
          "completed":int(bool(ev.get("completed"))),"updated_at":now
        })
    return pd.DataFrame(rows),h

def events(sport,key):
    return _get(f"/sports/{SPORT_KEYS[sport]}/events",key,{"dateFormat":"iso"})

def event_props(sport,event_id,key,regions="us,us2"):
    markets=NBA_PROPS if sport=="NBA" else NFL_PROPS
    data,h=_get(f"/sports/{SPORT_KEYS[sport]}/events/{event_id}/odds",key,{
      "regions":regions,"markets":",".join(markets),
      "oddsFormat":"american","dateFormat":"iso"
    })
    now=datetime.now(timezone.utc).isoformat()
    rows=[]
    for book in data.get("bookmakers",[]):
        for market in book.get("markets",[]):
            for out in market.get("outcomes",[]):
                rows.append({
                  "captured_at":now,"sport":sport,"event_id":data.get("id"),
                  "commence_time":data.get("commence_time"),
                  "home_team":data.get("home_team"),"away_team":data.get("away_team"),
                  "bookmaker":book.get("title"),"market":market.get("key"),
                  "player_name":out.get("description") or "Unknown",
                  "side":out.get("name"),"point":out.get("point"),"price":out.get("price")
                })
    return pd.DataFrame(rows),h
