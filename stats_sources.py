
from datetime import datetime
import difflib
import pandas as pd

NBA_SINGLE={
 "player_points":"PTS","player_rebounds":"REB","player_assists":"AST",
 "player_threes":"FG3M","player_blocks":"BLK","player_steals":"STL",
 "player_turnovers":"TOV"
}
NFL_COLS={
 "player_pass_yds":"passing_yards","player_pass_tds":"passing_tds",
 "player_pass_attempts":"attempts","player_pass_completions":"completions",
 "player_pass_interceptions":"interceptions","player_rush_yds":"rushing_yards",
 "player_rush_attempts":"carries","player_receptions":"receptions",
 "player_reception_yds":"receiving_yards"
}

def nba_season():
    n=datetime.now()
    return f"{n.year}-{str(n.year+1)[-2:]}" if n.month>=10 else f"{n.year-1}-{str(n.year)[-2:]}"

def _nba_series(df,m):
    if m=="player_points_rebounds_assists":
        return df.PTS+df.REB+df.AST
    if m=="player_points_rebounds":
        return df.PTS+df.REB
    if m=="player_points_assists":
        return df.PTS+df.AST
    if m=="player_rebounds_assists":
        return df.REB+df.AST
    c=NBA_SINGLE.get(m)
    return pd.to_numeric(df[c],errors="coerce") if c in df.columns else pd.Series(dtype=float)

def nba_logs(name,n=20):
    from nba_api.stats.static import players
    from nba_api.stats.endpoints import playergamelog
    plist=players.get_players()
    names=[p["full_name"] for p in plist]
    exact=[p for p in plist if p["full_name"].lower()==name.lower()]
    if not exact:
        m=difflib.get_close_matches(name,names,n=1,cutoff=.55)
        if not m: raise RuntimeError("NBA player not found.")
        exact=[p for p in plist if p["full_name"]==m[0]]
    p=exact[0]
    season=nba_season()
    try:
        df=playergamelog.PlayerGameLog(
          player_id=p["id"],season=season,season_type_all_star="Regular Season",timeout=20
        ).get_data_frames()[0]
    except Exception:
        y=int(season[:4])-1
        prev=f"{y}-{str(y+1)[-2:]}"
        df=playergamelog.PlayerGameLog(
          player_id=p["id"],season=prev,season_type_all_star="Regular Season",timeout=20
        ).get_data_frames()[0]
    if df.empty: raise RuntimeError("No NBA logs returned.")
    return df.head(n).copy(),p["full_name"]

def nfl_logs(name,n=20):
    import nflreadpy as nfl
    df=nfl.load_player_stats(seasons=None,summary_level="week").to_pandas()
    nc=next((c for c in ["player_display_name","player_name","name"] if c in df.columns),None)
    if not nc: raise RuntimeError("NFL player-name column unavailable.")
    names=df[nc].dropna().astype(str).unique().tolist()
    exact=[x for x in names if x.lower()==name.lower()]
    resolved=exact[0] if exact else (difflib.get_close_matches(name,names,n=1,cutoff=.55) or [None])[0]
    if not resolved: raise RuntimeError("NFL player not found.")
    x=df[df[nc].astype(str).eq(resolved)].copy()
    sort=[c for c in ["season","week"] if c in x.columns]
    if sort:x=x.sort_values(sort,ascending=False)
    return x.head(n),resolved

def stat_series(sport,name,market,n=20):
    if sport=="NBA":
        df,res=nba_logs(name,n)
        return pd.to_numeric(_nba_series(df,market),errors="coerce").dropna(),res
    df,res=nfl_logs(name,n)
    c=NFL_COLS.get(market)
    if not c or c not in df.columns:return pd.Series(dtype=float),res
    return pd.to_numeric(df[c],errors="coerce").dropna(),res
