
from pathlib import Path
from datetime import datetime,timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import brier_score_loss
from database import read_sql,connect
from analytics import implied,novig
from historical_bootstrap import predict_matchup,line_probability

MODEL_DIR=Path(__file__).parent/"data"/"models";MODEL_DIR.mkdir(parents=True,exist_ok=True)

def training(sport,market):
    m=read_sql("""SELECT m.* FROM market_snapshots m JOIN(
      SELECT event_id,market,outcome,MAX(captured_at) mx FROM market_snapshots
      GROUP BY event_id,market,outcome) z
      ON m.event_id=z.event_id AND m.market=z.market AND m.outcome=z.outcome AND m.captured_at=z.mx""")
    r=read_sql("SELECT * FROM game_results WHERE sport=? AND completed=1",(sport,))
    if m.empty or r.empty:return pd.DataFrame()
    x=m[m.sport.eq(sport)].merge(r[["event_id","home_score","away_score"]],on="event_id")
    rows=[]
    for _,g in x.groupby("event_id"):
        b=g.iloc[0];hs=float(b.home_score);aas=float(b.away_score)
        if market=="Moneyline":
            q=g[g.market.eq("h2h")];p={}
            for t,sg in q.groupby("outcome"):p[t]=np.nanmedian([implied(v) for v in sg.price])
            if b.home_team in p and b.away_team in p:
                ph,_=novig(p[b.home_team],p[b.away_team])
                rows.append({"home_team":b.home_team,"away_team":b.away_team,"line":0,
                             "market_prob":ph,"target":int(hs>aas)})
        elif market=="Spread":
            q=g[(g.market.eq("spreads"))&(g.outcome.eq(b.home_team))].dropna(subset=["point"])
            if len(q):
                l=float(q.point.median());p=np.nanmedian([implied(v) for v in q.price])
                rows.append({"home_team":b.home_team,"away_team":b.away_team,"line":l,
                             "market_prob":p,"target":int(hs+l>aas)})
        else:
            q=g[(g.market.eq("totals"))&(g.outcome.eq("Over"))].dropna(subset=["point"])
            if len(q):
                l=float(q.point.median());p=np.nanmedian([implied(v) for v in q.price])
                rows.append({"home_team":b.home_team,"away_team":b.away_team,"line":l,
                             "market_prob":p,"target":int(hs+aas>l)})
    return pd.DataFrame(rows)

def train(sport,market,minrows):
    d=training(sport,market)
    if len(d)<minrows or (len(d) and d.target.nunique()<2):
        return None,{"status":f"Need {minrows} completed rows.","rows":len(d)}
    X=d[["home_team","away_team","line","market_prob"]];y=d.target
    pre=ColumnTransformer([
      ("cat",OneHotEncoder(handle_unknown="ignore"),["home_team","away_team"]),
      ("num",SimpleImputer(strategy="median"),["line","market_prob"])
    ])
    model=Pipeline([("pre",pre),("lr",LogisticRegression(max_iter=1000,C=.5))])
    model.fit(X,y);p=model.predict_proba(X)[:,1];b=float(brier_score_loss(y,p))
    joblib.dump(model,MODEL_DIR/f"{sport.lower()}_{market.lower()}.joblib")
    with connect() as con:
        con.execute("INSERT INTO model_runs VALUES(NULL,?,?,?,?,?,?,?)",
          (datetime.now(timezone.utc).isoformat(),sport,market,len(d),"brier",b,"V3"))
    return model,{"status":"trained","rows":len(d),"brier":b}

def apply(df):
    if df is None or df.empty:return df
    out=df.copy()

    # First use the historical rolling-form models when available.
    pred_cache={}
    for idx,r in out.iterrows():
        key=(r.sport,r.home_team,r.away_team)
        if key not in pred_cache:
            try: pred_cache[key]=predict_matchup(*key)
            except Exception: pred_cache[key]=None
        pred=pred_cache[key]
        if pred is not None:
            try:
                mp=line_probability(pred,r.market,r.selection,r.line,r.home_team,r.away_team)
                if mp is not None and np.isfinite(mp):
                    # Conservative blend: 70% learned model + 30% no-vig market.
                    final=float(np.clip(.70*mp+.30*float(r.market_prob),.03,.97))
                    out.at[idx,"model_prob"]=final
                    out.at[idx,"edge"]=final-float(r.market_prob)
                    out.at[idx,"prob_source"]="Historical form model + market"
            except Exception:
                pass

    # Older local odds-calibration model may refine rows if it exists and no bootstrap model was used.
    for (sport,market),idxs in out.groupby(["sport","market"]).groups.items():
        pth=MODEL_DIR/f"{sport.lower()}_{market.lower()}.joblib"
        if not pth.exists():continue
        use=[i for i in idxs if out.at[i,"prob_source"]=="No-vig market consensus"]
        if not use:continue
        model=joblib.load(pth);sub=out.loc[use]
        X=pd.DataFrame({"home_team":sub.home_team,"away_team":sub.away_team,
                        "line":sub.line.fillna(0),"market_prob":sub.market_prob})
        pb=model.predict_proba(X)[:,1]
        if market in ("Moneyline","Spread"):
            pred=np.where(sub.selection.eq(sub.home_team),pb,1-pb)
        else:pred=np.where(sub.selection.eq("Over"),pb,1-pb)
        out.loc[use,"model_prob"]=pred
        out.loc[use,"edge"]=pred-sub.market_prob.astype(float)
        out.loc[use,"prob_source"]="Local odds calibration model"
    return out
