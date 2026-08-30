
import numpy as np
import pandas as pd
from analytics import decimal,american

SIZES=[2,3,4,5,6,8,10,12,16]

def score(r,mode):
    p=float(r.model_prob);d=decimal(r.best_price);e=float(r.edge);ev=p*d-1
    if mode=="Highest Chance":return 4*np.log(max(p,1e-8))+1.2*e+.15*np.log(d)
    if mode=="Best Value":return 2.5*np.log(max(p,1e-8))+2.2*e+ev+.35*np.log(d)
    return 2.1*np.log(max(p,1e-8))+1.4*e+.85*np.log(d)+.55*ev

def build(df,n,minp,minedge,mode,allow_same):
    if df is None or df.empty:return None
    d=df.copy()
    d=d[pd.to_numeric(d.model_prob,errors="coerce").fillna(0)>=minp]
    d=d[pd.to_numeric(d.edge,errors="coerce").fillna(-99)>=minedge]
    d=d.dropna(subset=["best_price"])
    if d.empty:return None
    d["rank_score"]=d.apply(lambda r:score(r,mode),axis=1)
    d=d.sort_values(["rank_score","model_prob","edge"],ascending=False)
    picks=[];events=set();playerkeys=set()
    for _,r in d.iterrows():
        if not allow_same and r.event_id in events:continue
        if r.get("leg_type")=="Player Prop":
            pk=(r.event_id,r.get("player_name"),r.get("market"))
            if pk in playerkeys:continue
            playerkeys.add(pk)
        picks.append(r);events.add(r.event_id)
        if len(picks)>=n:break
    legs=pd.DataFrame(picks)
    out={"legs":legs,"requested":n,"complete":len(legs)>=n}
    if len(legs)>=2:
        cp=float(np.prod(legs.model_prob.astype(float)))
        cd=float(np.prod([decimal(o) for o in legs.best_price]))
        out.update({"combined_prob":cp,"decimal_odds":cd,"american_odds":american(cd)})
    return out
