
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from database import upsert_historical_games, replace_historical_features, read_sql

MODEL_DIR=Path(__file__).parent/"data"/"models"
MODEL_DIR.mkdir(parents=True,exist_ok=True)

FEATURES=[
 "home_l5_win","away_l5_win","home_l10_win","away_l10_win",
 "home_l5_pf","away_l5_pf","home_l10_pf","away_l10_pf",
 "home_l5_pa","away_l5_pa","home_l10_pa","away_l10_pa",
 "home_l5_margin","away_l5_margin","home_l10_margin","away_l10_margin"
]

def _season_label(start_year):
    return f"{start_year}-{str(start_year+1)[-2:]}"

def fetch_nba_history(seasons=4):
    from nba_api.stats.endpoints import leaguegamelog
    now=datetime.now()
    current_start=now.year if now.month>=10 else now.year-1
    frames=[]
    for y in range(current_start-seasons+1,current_start+1):
        season=_season_label(y)
        for season_type in ["Regular Season","Playoffs"]:
            try:
                df=leaguegamelog.LeagueGameLog(
                    season=season,
                    season_type_all_star=season_type,
                    player_or_team_abbreviation="T",
                    sorter="DATE",
                    direction="ASC",
                    timeout=30
                ).get_data_frames()[0]
                if len(df):
                    df["BOOTSTRAP_SEASON"]=season
                    frames.append(df)
            except Exception:
                continue
    if not frames:
        raise RuntimeError("NBA Stats did not return historical games. It may be temporarily rate-limiting requests.")
    raw=pd.concat(frames,ignore_index=True)
    rows=[]
    for gid,g in raw.groupby("GAME_ID"):
        if len(g)<2: continue
        home_rows=g[g.MATCHUP.astype(str).str.contains("vs.",regex=False)]
        away_rows=g[g.MATCHUP.astype(str).str.contains("@",regex=False)]
        if home_rows.empty or away_rows.empty:
            continue
        h=home_rows.iloc[0];a=away_rows.iloc[0]
        rows.append({
            "game_key":f"NBA_{gid}","sport":"NBA","season":h.BOOTSTRAP_SEASON,
            "game_date":str(h.GAME_DATE),"home_team":str(h.TEAM_ABBREVIATION),
            "away_team":str(a.TEAM_ABBREVIATION),"home_score":float(h.PTS),
            "away_score":float(a.PTS),"source":"nba_api",
            "spread_line":np.nan,"total_line":np.nan
        })
    return pd.DataFrame(rows).drop_duplicates("game_key")

def fetch_nfl_history(seasons=6):
    import nflreadpy as nfl
    now=datetime.now()
    current=now.year
    yrs=list(range(current-seasons+1,current+1))
    try:
        df=nfl.load_schedules(seasons=yrs).to_pandas()
    except Exception as e:
        raise RuntimeError(f"Could not load nflverse schedules: {e}")
    def col(*names):
        return next((n for n in names if n in df.columns),None)
    game=col("game_id","game_key")
    date=col("gameday","game_date","date")
    home=col("home_team")
    away=col("away_team")
    hs=col("home_score")
    aas=col("away_score")
    season_col=col("season")
    spread=col("spread_line","spread")
    total=col("total_line","total")
    need=[game,date,home,away,hs,aas]
    if any(x is None for x in need):
        raise RuntimeError("nflverse schedule columns changed and required fields were not found.")
    x=df.dropna(subset=[hs,aas]).copy()
    out=pd.DataFrame({
        "game_key":"NFL_"+x[game].astype(str),
        "sport":"NFL",
        "season":x[season_col].astype(str) if season_col else "",
        "game_date":x[date].astype(str),
        "home_team":x[home].astype(str),
        "away_team":x[away].astype(str),
        "home_score":pd.to_numeric(x[hs],errors="coerce"),
        "away_score":pd.to_numeric(x[aas],errors="coerce"),
        "source":"nflverse",
        "spread_line":pd.to_numeric(x[spread],errors="coerce") if spread else np.nan,
        "total_line":pd.to_numeric(x[total],errors="coerce") if total else np.nan,
    })
    return out.dropna(subset=["home_score","away_score"]).drop_duplicates("game_key")

def _team_history_games(games,team,before_date):
    g=games[(games.game_date<before_date)&((games.home_team==team)|(games.away_team==team))].copy()
    return g.sort_values("game_date")

def _team_stats(hist,team,n):
    x=hist.tail(n)
    if x.empty:
        return dict(win=.5,pf=np.nan,pa=np.nan,margin=np.nan)
    home=x.home_team.eq(team)
    pf=np.where(home,x.home_score,x.away_score).astype(float)
    pa=np.where(home,x.away_score,x.home_score).astype(float)
    return {
        "win":float(np.mean(pf>pa)),
        "pf":float(np.mean(pf)),
        "pa":float(np.mean(pa)),
        "margin":float(np.mean(pf-pa))
    }

def build_features(games):
    if games is None or games.empty:
        return pd.DataFrame()
    g=games.copy()
    g["game_date"]=pd.to_datetime(g.game_date,errors="coerce")
    g=g.dropna(subset=["game_date","home_score","away_score"]).sort_values("game_date")
    rows=[]
    for _,r in g.iterrows():
        hh=_team_history_games(g,r.home_team,r.game_date)
        ah=_team_history_games(g,r.away_team,r.game_date)
        h5=_team_stats(hh,r.home_team,5);a5=_team_stats(ah,r.away_team,5)
        h10=_team_stats(hh,r.home_team,10);a10=_team_stats(ah,r.away_team,10)
        rows.append({
          "game_key":r.game_key,"sport":r.sport,"season":r.season,
          "game_date":r.game_date.strftime("%Y-%m-%d"),
          "home_team":r.home_team,"away_team":r.away_team,
          "home_score":float(r.home_score),"away_score":float(r.away_score),
          "home_win":int(r.home_score>r.away_score),
          "margin":float(r.home_score-r.away_score),
          "total_points":float(r.home_score+r.away_score),
          "home_l5_win":h5["win"],"away_l5_win":a5["win"],
          "home_l10_win":h10["win"],"away_l10_win":a10["win"],
          "home_l5_pf":h5["pf"],"away_l5_pf":a5["pf"],
          "home_l10_pf":h10["pf"],"away_l10_pf":a10["pf"],
          "home_l5_pa":h5["pa"],"away_l5_pa":a5["pa"],
          "home_l10_pa":h10["pa"],"away_l10_pa":a10["pa"],
          "home_l5_margin":h5["margin"],"away_l5_margin":a5["margin"],
          "home_l10_margin":h10["margin"],"away_l10_margin":a10["margin"],
          "spread_line":r.get("spread_line",np.nan),
          "total_line":r.get("total_line",np.nan)
        })
    return pd.DataFrame(rows)

def bootstrap_sport(sport,seasons):
    games=fetch_nba_history(seasons) if sport=="NBA" else fetch_nfl_history(seasons)
    upsert_historical_games(games)
    feats=build_features(games)
    replace_historical_features(feats,sport)
    return games,feats

def train_bootstrap_models(sport,min_rows=150):
    d=read_sql("SELECT * FROM historical_features WHERE sport=? ORDER BY game_date",(sport,))
    if len(d)<min_rows:
        return None,{"status":f"Need at least {min_rows} historical feature rows.","rows":len(d)}
    d=d.copy()
    # Drop first games with insufficient rolling history by requiring most features to exist.
    d=d[d[FEATURES].notna().sum(axis=1)>=12].reset_index(drop=True)
    if len(d)<min_rows:
        return None,{"status":f"Only {len(d)} rows have enough rolling history.","rows":len(d)}
    cut=max(int(len(d)*.8),1)
    tr=d.iloc[:cut];te=d.iloc[cut:]
    Xtr=tr[FEATURES];Xte=te[FEATURES]

    imputer=SimpleImputer(strategy="median")
    Xtri=imputer.fit_transform(Xtr);Xtei=imputer.transform(Xte)

    clf=HistGradientBoostingClassifier(max_iter=220,learning_rate=.055,max_leaf_nodes=15,l2_regularization=2.0,random_state=42)
    margin=HistGradientBoostingRegressor(max_iter=260,learning_rate=.05,max_leaf_nodes=15,l2_regularization=2.0,random_state=42)
    total=HistGradientBoostingRegressor(max_iter=260,learning_rate=.05,max_leaf_nodes=15,l2_regularization=2.0,random_state=43)

    clf.fit(Xtri,tr.home_win)
    margin.fit(Xtri,tr.margin)
    total.fit(Xtri,tr.total_points)

    ph=clf.predict_proba(Xtei)[:,1]
    pm=margin.predict(Xtei)
    pt=total.predict(Xtei)
    margin_resid=te.margin.to_numpy()-pm
    total_resid=te.total_points.to_numpy()-pt

    payload={
      "sport":sport,"features":FEATURES,"imputer":imputer,
      "home_win_model":clf,"margin_model":margin,"total_model":total,
      "margin_sigma":float(max(np.std(margin_resid),1.0)),
      "total_sigma":float(max(np.std(total_resid),1.0)),
      "train_rows":len(tr),"test_rows":len(te),
      "accuracy":float(accuracy_score(te.home_win,(ph>=.5).astype(int))),
      "brier":float(brier_score_loss(te.home_win,ph)),
      "margin_mae":float(mean_absolute_error(te.margin,pm)),
      "total_mae":float(mean_absolute_error(te.total_points,pt))
    }
    joblib.dump(payload,MODEL_DIR/f"bootstrap_{sport.lower()}.joblib")
    return payload,{"status":"trained","rows":len(d),"accuracy":payload["accuracy"],
                    "brier":payload["brier"],"margin_mae":payload["margin_mae"],
                    "total_mae":payload["total_mae"]}

def load_bootstrap(sport):
    p=MODEL_DIR/f"bootstrap_{sport.lower()}.joblib"
    return joblib.load(p) if p.exists() else None

def latest_team_features(sport,home,away):
    g=read_sql("SELECT * FROM historical_games WHERE sport=? ORDER BY game_date",(sport,))
    if g.empty:return None
    g["game_date"]=pd.to_datetime(g.game_date,errors="coerce")
    date=pd.Timestamp.max
    hh=_team_history_games(g,home,date);ah=_team_history_games(g,away,date)
    h5=_team_stats(hh,home,5);a5=_team_stats(ah,away,5)
    h10=_team_stats(hh,home,10);a10=_team_stats(ah,away,10)
    return pd.DataFrame([{
      "home_l5_win":h5["win"],"away_l5_win":a5["win"],
      "home_l10_win":h10["win"],"away_l10_win":a10["win"],
      "home_l5_pf":h5["pf"],"away_l5_pf":a5["pf"],
      "home_l10_pf":h10["pf"],"away_l10_pf":a10["pf"],
      "home_l5_pa":h5["pa"],"away_l5_pa":a5["pa"],
      "home_l10_pa":h10["pa"],"away_l10_pa":a10["pa"],
      "home_l5_margin":h5["margin"],"away_l5_margin":a5["margin"],
      "home_l10_margin":h10["margin"],"away_l10_margin":a10["margin"],
    }])

def normal_cdf(x):
    return .5*(1+math.erf(x/math.sqrt(2)))

def predict_matchup(sport,home,away):
    model=load_bootstrap(sport)
    if model is None:return None
    X=latest_team_features(sport,home,away)
    if X is None:return None
    Xi=model["imputer"].transform(X[model["features"]])
    p_home=float(model["home_win_model"].predict_proba(Xi)[0,1])
    margin=float(model["margin_model"].predict(Xi)[0])
    total=float(model["total_model"].predict(Xi)[0])
    return {"home_win_prob":p_home,"pred_margin":margin,"pred_total":total,
            "margin_sigma":model["margin_sigma"],"total_sigma":model["total_sigma"],
            "source":"Historical form model"}

def line_probability(pred,market,selection,line,home,away):
    if pred is None:return None
    if market=="Moneyline":
        return pred["home_win_prob"] if selection==home else 1-pred["home_win_prob"]
    if market=="Spread":
        # Home margin M. Home covers when M + home_line > 0.
        if selection==home:
            threshold=-float(line)
            z=(threshold-pred["pred_margin"])/pred["margin_sigma"]
            return 1-normal_cdf(z)
        # Away line means away covers when -M + away_line > 0 => M < away_line
        threshold=float(line)
        z=(threshold-pred["pred_margin"])/pred["margin_sigma"]
        return normal_cdf(z)
    if market=="Total":
        z=(float(line)-pred["pred_total"])/pred["total_sigma"]
        pov=1-normal_cdf(z)
        return pov if selection=="Over" else 1-pov
    return None
