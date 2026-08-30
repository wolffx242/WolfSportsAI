
from __future__ import annotations
import numpy as np
import pandas as pd
from database import read_sql
from historical_bootstrap import predict_matchup

def history_for_team(sport,team,n=10):
    q=read_sql("""SELECT * FROM historical_games
                  WHERE sport=? AND (home_team=? OR away_team=?)
                  ORDER BY game_date DESC LIMIT ?""",(sport,team,team,int(n)))
    if q.empty:return q
    home=q.home_team.eq(team)
    q=q.copy()
    q["team_score"]=np.where(home,q.home_score,q.away_score).astype(float)
    q["opp_score"]=np.where(home,q.away_score,q.home_score).astype(float)
    q["margin"]=q.team_score-q.opp_score
    q["win"]=q.margin>0
    q["opponent"]=np.where(home,q.away_team,q.home_team)
    return q

def h2h_history(sport,home,away,n=10):
    q=read_sql("""SELECT * FROM historical_games
                  WHERE sport=? AND ((home_team=? AND away_team=?) OR (home_team=? AND away_team=?))
                  ORDER BY game_date DESC LIMIT ?""",(sport,home,away,away,home,int(n)))
    return q

def team_summary(sport,team,n):
    q=history_for_team(sport,team,n)
    if q.empty:return {"games":0}
    return {
      "games":len(q),"win_pct":float(q.win.mean()),"pf":float(q.team_score.mean()),
      "pa":float(q.opp_score.mean()),"margin":float(q.margin.mean()),
      "last_score":float(q.team_score.iloc[0])
    }

def current_line_hit_rates(sport,team,market,line,n):
    q=history_for_team(sport,team,n)
    if q.empty:return {"games":0,"hit_rate":np.nan}
    if market=="Moneyline":
        hit=q.win.astype(bool)
    elif market=="Spread":
        hit=(q.margin+float(line))>0
    else:
        hit=(q.team_score+q.opp_score)>float(line)
    return {"games":len(q),"hit_rate":float(hit.mean())}

def matchup_packet(sport,home,away):
    return {
      "model":predict_matchup(sport,home,away),
      "home_l5":team_summary(sport,home,5),
      "away_l5":team_summary(sport,away,5),
      "home_l10":team_summary(sport,home,10),
      "away_l10":team_summary(sport,away,10),
      "h2h5":h2h_history(sport,home,away,5),
      "h2h10":h2h_history(sport,home,away,10)
    }
