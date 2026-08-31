
import math
import numpy as np
import pandas as pd

def implied(o):
    if pd.isna(o):return np.nan
    o=float(o)
    if o==0:return np.nan
    return 100/(o+100) if o>0 else (-o)/((-o)+100)

def decimal(o):
    o=float(o)
    return 1+o/100 if o>0 else 1+100/abs(o)

def american(d):
    d=float(d)
    if d<=1:return np.nan
    return round((d-1)*100) if d>=2 else round(-100/(d-1))

def novig(a,b):
    s=a+b
    return (a/s,b/s) if s else (np.nan,np.nan)

def best(g):
    if g.empty:return np.nan,None
    x=g.dropna(subset=["price"]).sort_values("price",ascending=False).iloc[0]
    return float(x.price),str(x.bookmaker)

def team_consensus(raw):
    if raw is None or raw.empty:return pd.DataFrame()
    d=raw.copy();d["imp"]=d.price.apply(implied);rows=[]
    for eid,g in d.groupby("event_id"):
        b=g.iloc[0];home=b.home_team;away=b.away_team
        h=g[g.market.eq("h2h")]
        if len(h):
            hp=h[h.outcome.eq(home)].imp.median();ap=h[h.outcome.eq(away)].imp.median()
            if pd.notna(hp) and pd.notna(ap):
                ph,pa=novig(hp,ap)
                for sel,p in [(home,ph),(away,pa)]:
                    pr,bk=best(h[h.outcome.eq(sel)])
                    rows.append(_row(b,"Moneyline",sel,np.nan,pr,bk,p))
        s=g[g.market.eq("spreads")]
        if len(s):
            for sel in [home,away]:
                x=s[s.outcome.eq(sel)].dropna(subset=["point"])
                if not len(x):continue
                line=float(x.point.median())
                near=x[(x.point-line).abs()<=.51]
                if not len(near):near=x
                other=away if sel==home else home
                y=s[s.outcome.eq(other)].dropna(subset=["point"])
                y=y[(y.point+line).abs()<=.51]
                px=near.imp.median();py=y.imp.median() if len(y) else 1-px
                p,_=novig(px,py);pr,bk=best(near)
                rows.append(_row(b,"Spread",sel,line,pr,bk,p))
        t=g[g.market.eq("totals")].dropna(subset=["point"])
        if len(t):
            line=float(t.point.median());near=t[(t.point-line).abs()<=.51]
            if not len(near):near=t
            o=near[near.outcome.eq("Over")];u=near[near.outcome.eq("Under")]
            if len(o) and len(u):
                po,pu=novig(o.imp.median(),u.imp.median())
                for sel,p,x in [("Over",po,o),("Under",pu,u)]:
                    pr,bk=best(x);rows.append(_row(b,"Total",sel,line,pr,bk,p))
    return pd.DataFrame(rows)

def _row(b,m,s,l,price,book,p):
    return {"sport":b.sport,"event_id":b.event_id,"commence_time":b.commence_time,
      "away_team":b.away_team,"home_team":b.home_team,"market":m,"selection":s,
      "line":l,"best_price":price,"best_book":book,"market_prob":float(p),
      "model_prob":float(p),"edge":0.0,"prob_source":"No-vig market consensus",
      "leg_type":"Team"}

def prop_consensus(raw):
    if raw is None or raw.empty:return pd.DataFrame()
    d=raw.copy();d["imp"]=d.price.apply(implied);rows=[]
    keys=["sport","event_id","commence_time","home_team","away_team","market","player_name","point"]
    for k,g in d.groupby(keys,dropna=False):
        sd={}
        for side,x in g.groupby("side"):
            pr,bk=best(x);sd[side]=(x.imp.median(),pr,bk)
        if "Over" not in sd or "Under" not in sd:continue
        po,pu=novig(sd["Over"][0],sd["Under"][0])
        for side,p in [("Over",po),("Under",pu)]:
            row=dict(zip(keys,k));_,pr,bk=sd[side]
            row.update({"side":side,"selection":f"{row['player_name']} {side}",
              "line":row["point"],"best_price":pr,"best_book":bk,
              "market_prob":float(p),"model_prob":float(p),"edge":0.0,
              "prob_source":"No-vig prop consensus","leg_type":"Player Prop"})
            rows.append(row)
    return pd.DataFrame(rows)

def perf_prob(values,line,side):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if len(a)<3:return np.nan,{}
    w=np.array([.90**i for i in range(len(a))]);w=w/w.sum()
    mean=float((a*w).sum());sd=max(float(np.sqrt((w*(a-mean)**2).sum())),1.0)
    z=(float(line)-mean)/sd
    pov=1-(.5*(1+math.erf(z/math.sqrt(2))))
    hit=float(np.mean(a>float(line)))
    pov=float(np.clip(.6*pov+.4*hit,.05,.95))
    return (pov if side=="Over" else 1-pov),{"games":len(a),"avg":mean,"hit":hit}

def blend(mp,sp,n):
    if pd.isna(sp):return float(mp),0.0
    w=min(.65,.65*min(max(n,0),20)/20)
    return float(np.clip(w*sp+(1-w)*mp,.03,.97)),w

def conf(p,e):
    if p>=.68 and e>=.04:return "A+"
    if p>=.63 and e>=.025:return "A"
    if p>=.59 and e>=.01:return "B"
    if p>=.55:return "C"
    return "Pass"

def enrich(df):
    if df is None or df.empty:return df
    x=df.copy()
    grades=[]
    for _,r in x.iterrows():
        p=float(r.model_prob); e=float(r.edge)
        g=conf(p,e)
        # MLB/NHL can be used immediately while their local calibration models
        # accumulate completed games. This does NOT invent an edge: edge stays 0.
        # Strong no-vig consensus favorites receive at most a B fallback grade.
        if str(r.get("sport","")) in ("MLB","NHL") and str(r.get("prob_source","")).startswith("No-vig") and e <= 1e-9:
            if p >= .62:
                g="B"
            elif p >= .55:
                g="C"
        grades.append(g)
    x["confidence"]=grades
    x["ev_per_unit"]=[float(p)*decimal(o)-1 if pd.notna(o) else np.nan for p,o in zip(x.model_prob,x.best_price)]
    return x
