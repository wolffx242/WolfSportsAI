
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from database import init_db,append_df,upsert_results,read_sql,count
from odds_api import featured,scores,events,event_props
from analytics import team_consensus,prop_consensus,perf_prob,blend,enrich
from stats_sources import stat_series
from parlay import build,SIZES
from team_models import apply,train
from historical_bootstrap import bootstrap_sport,train_bootstrap_models,load_bootstrap
from game_center import matchup_packet,history_for_team,current_line_hit_rates
from autopilot import start_worker,latest_backtests
from database import get_autopilot_status

ROOT=Path(__file__).parent
init_db()

# Store user settings outside the extracted program folder.
# This avoids Windows "Access is denied" errors when Downloads/antivirus
# prevents atomic replacement of .env files.
APPDATA_ROOT = Path(os.getenv("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
SETTINGS_DIR = APPDATA_ROOT / "WolfSportsAI"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"

def load_settings():
    try:
        if SETTINGS_FILE.exists():
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def save_settings(settings):
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_DIR / "settings.tmp.json"
    tmp.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    tmp.replace(SETTINGS_FILE)

SETTINGS = load_settings()
def _secret_api_key():
    try:
        return str(st.secrets.get("ODDS_API_KEY","")).strip()
    except Exception:
        return ""

KEY = str(_secret_api_key() or SETTINGS.get("ODDS_API_KEY") or os.getenv("ODDS_API_KEY","")).strip()
CLOUD_MODE = bool(os.getenv("STREAMLIT_SHARING_MODE") or os.getenv("STREAMLIT_SERVER_HEADLESS"))


@st.cache_resource
def _start_autopilot_worker(api_key):
    # The worker is a daemon thread and runs only while this Streamlit process is alive.
    return start_worker(
        api_key=api_key,
        sports=("NBA","NFL"),
        refresh_minutes=15,
        retrain_hours=12,
        backtest_hours=6
    )

# Do not start network/background work before Streamlit paints the page.
# The worker is started near the end of the script after the selected page renders.
AUTOPILOT_THREAD = None

st.set_page_config(
    page_title="WolfSportsAI",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- UI THEME ----------
st.markdown("""
<style>
:root{
  --wolf-bg:#070b11;
  --wolf-panel:#0d141d;
  --wolf-panel2:#111a25;
  --wolf-border:#1e2a38;
  --wolf-text:#e8eef6;
  --wolf-muted:#8391a2;
  --wolf-accent:#22d3ee;
  --wolf-green:#31d38a;
  --wolf-red:#ff6b7a;
  --wolf-yellow:#f0c95a;
}
html,body,[class*="css"]{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
[data-testid="stAppViewContainer"]{background:var(--wolf-bg);}
[data-testid="stHeader"]{background:rgba(7,11,17,.75);backdrop-filter:blur(12px);}
[data-testid="stSidebar"]{background:#090e15;border-right:1px solid var(--wolf-border);}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:var(--wolf-muted);}
.block-container{padding-top:1.25rem;max-width:1500px;}
h1,h2,h3{letter-spacing:-.025em;}
.wolf-brand{display:flex;align-items:center;gap:12px;margin:0 0 18px 0;}
.wolf-logo{width:44px;height:44px;border-radius:12px;display:grid;place-items:center;
 background:linear-gradient(145deg,#142535,#0b1017);border:1px solid #26384a;font-size:25px;}
.wolf-title{font-size:1.4rem;font-weight:800;color:var(--wolf-text);line-height:1.05;}
.wolf-sub{font-size:.78rem;color:var(--wolf-muted);margin-top:4px;}
.hero{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;margin-bottom:18px;flex-wrap:wrap;}
.hero h1{font-size:2rem;margin:0;color:var(--wolf-text);}
.hero p{margin:5px 0 0;color:var(--wolf-muted);}
.status-pill{border:1px solid #1f7259;background:#0c2a22;color:#78e7bd;border-radius:999px;
 padding:7px 12px;font-size:.8rem;font-weight:700;}
.status-pill.off{border-color:#66323a;background:#281218;color:#ff9fac;}
.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:6px 0 20px;}
.metric-card{background:linear-gradient(180deg,#101923,#0d141d);border:1px solid var(--wolf-border);
 border-radius:14px;padding:14px 16px;min-width:0;}
.metric-label{font-size:.74rem;color:var(--wolf-muted);text-transform:uppercase;letter-spacing:.08em;font-weight:700;}
.metric-value{font-size:1.45rem;color:var(--wolf-text);font-weight:800;margin-top:4px;}
.metric-note{font-size:.75rem;color:var(--wolf-muted);margin-top:3px;}
.section-card{background:var(--wolf-panel);border:1px solid var(--wolf-border);border-radius:15px;padding:16px;margin-bottom:14px;}
.pick-card{background:linear-gradient(180deg,#111a25,#0d141d);border:1px solid var(--wolf-border);border-radius:14px;
 padding:14px;margin-bottom:10px;}
.pick-head{display:flex;justify-content:space-between;gap:12px;align-items:center;}
.pick-league{font-size:.7rem;font-weight:800;letter-spacing:.08em;color:var(--wolf-muted);text-transform:uppercase;}
.pick-game{font-size:.95rem;font-weight:700;color:var(--wolf-text);margin-top:3px;}
.pick-selection{font-size:1.05rem;font-weight:800;color:#fff;margin-top:9px;}
.pick-meta{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-top:12px;}
.pick-stat{background:#0a1119;border:1px solid #172433;border-radius:10px;padding:8px;}
.pick-stat span{display:block;font-size:.68rem;text-transform:uppercase;color:var(--wolf-muted);font-weight:700;}
.pick-stat strong{display:block;font-size:.92rem;color:var(--wolf-text);margin-top:2px;}
.grade{display:inline-flex;align-items:center;justify-content:center;min-width:40px;padding:5px 9px;border-radius:9px;
 font-weight:900;font-size:.78rem;border:1px solid #23593f;background:#0e2b20;color:#6de4aa;}
.grade.b{border-color:#5b5229;background:#292510;color:#eedf79;}
.grade.c{border-color:#3d4e62;background:#141d28;color:#a9bed5;}
.market-chip{display:inline-block;border:1px solid #243648;background:#101a25;color:#b8c7d7;border-radius:999px;
 padding:4px 8px;font-size:.7rem;margin-right:5px;}
div[data-testid="stButton"] button{border-radius:10px;border:1px solid #294055;font-weight:700;}
div[data-testid="stButton"] button[kind="primary"]{background:#1bceda;color:#061014;border:none;}
div[data-testid="stSelectbox"]>div>div, div[data-testid="stTextInput"]>div>div>input,
div[data-testid="stNumberInput"] input{background:#0d141d;border-color:#243447;}
[data-testid="stDataFrame"]{border:1px solid var(--wolf-border);border-radius:12px;overflow:hidden;}
hr{border-color:var(--wolf-border);}
.small-muted{font-size:.78rem;color:var(--wolf-muted);}
.parlay-summary{background:linear-gradient(145deg,#112129,#0c141b);border:1px solid #244553;border-radius:14px;padding:16px;}
@media(max-width:900px){
 .metric-grid{grid-template-columns:repeat(2,minmax(0,1fr));}
 .pick-meta{grid-template-columns:repeat(2,minmax(0,1fr));}
}
</style>
""", unsafe_allow_html=True)

# ---------- STATE ----------
if "raw" not in st.session_state:
    st.session_state.raw=pd.DataFrame()
if "props" not in st.session_state:
    st.session_state.props=pd.DataFrame()
if "player" not in st.session_state:
    st.session_state.player=pd.DataFrame()
if "parlay" not in st.session_state:
    st.session_state.parlay=None

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("""
    <div class="wolf-brand">
      <div class="wolf-logo">🐺</div>
      <div>
        <div class="wolf-title">WolfSportsAI</div>
        <div class="wolf-sub">Sports Edge Terminal</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if "nav" not in st.session_state:
        st.session_state.nav="🔥 Best Bets"

    # A page change requested by a button is queued in pending_nav.
    # Apply it here, BEFORE the nav widget is instantiated. This avoids
    # StreamlitAPIException: session_state.nav cannot be modified after
    # the widget with key "nav" is instantiated.
    if "pending_nav" in st.session_state:
        st.session_state.nav = st.session_state.pop("pending_nav")

    page=st.radio(
        "Navigation",
        ["🔥 Best Bets","🏀 NBA","🏈 NFL","🎯 Game Center","👤 Player Lab","🔎 Prop Scanner",
         "🧾 Parlay Builder","🧠 Model Lab","⚙️ Settings"],
        label_visibility="collapsed",
        key="nav"
    )
    st.divider()
    st.caption("LIVE ENGINE")
    sports=st.multiselect("Leagues",["NBA","NFL"],default=["NBA","NFL"])
    auto=st.toggle("Auto refresh",True)
    mins=st.select_slider("Refresh interval",options=[5,10,15,20,30,45,60],value=15,format_func=lambda x:f"{x} min")
    if auto:
        st_autorefresh(interval=mins*60*1000,key="wolf_auto")

    if KEY:
        st.markdown('<div class="status-pill">● API Connected</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill off">● API Key Needed</div>',unsafe_allow_html=True)

    st.divider()
    ap=get_autopilot_status()
    st.caption("AUTOPILOT")
    _ap_thread = st.session_state.get("_autopilot_thread")
    if _ap_thread and _ap_thread.is_alive():
        st.markdown('<div class="status-pill">● AI Autopilot Running</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill off">● Autopilot Starting</div>',unsafe_allow_html=True)
    st.caption(f"Cycles: {int(ap.get('cycle_count') or 0)}")

# ---------- DATA ----------
def cache(s):
    return read_sql("""SELECT * FROM market_snapshots WHERE sport=? AND captured_at=
      (SELECT MAX(captured_at) FROM market_snapshots WHERE sport=?)""",(s,s))

def refresh():
    frames=[];errs=[]
    for s in sports:
        try:
            d,_=featured(s,KEY)
            if len(d):
                append_df("market_snapshots",d)
                frames.append(d)
            sc,_=scores(s,KEY)
            if len(sc):
                upsert_results(sc)
        except Exception as e:
            errs.append(f"{s}: {e}")
            d=cache(s)
            if len(d):
                frames.append(d)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(),errs

def score_props(pc,namefilter=None):
    if pc is None or pc.empty:
        return pd.DataFrame()
    x=pc.copy()
    if namefilter:
        x=x[x.player_name.str.contains(namefilter,case=False,na=False)]
    cache_stats={};rows=[]
    for _,r in x.iterrows():
        key=(r.sport,r.player_name,r.market)
        sp=np.nan;meta={}
        try:
            if key not in cache_stats:
                vals,res=stat_series(r.sport,r.player_name,r.market,20)
                cache_stats[key]=vals
            vals=cache_stats[key]
            sp,meta=perf_prob(vals,r.point,r.side)
        except Exception:
            pass
        p,w=blend(r.market_prob,sp,meta.get("games",0))
        rr=r.copy()
        rr["model_prob"]=p
        rr["edge"]=p-r.market_prob
        rr["games_used"]=meta.get("games",0)
        rr["recent_average"]=meta.get("avg",np.nan)
        rr["recent_hit_rate"]=meta.get("hit",np.nan)
        rr["stats_weight"]=w
        rr["prob_source"]="Recent stats + market blend" if w else "No-vig prop consensus"
        rows.append(rr)
    return enrich(pd.DataFrame(rows)) if rows else pd.DataFrame()

if KEY and st.session_state.raw.empty:
    try:
        st.session_state.raw,_=refresh()
    except Exception:
        pass

team=enrich(apply(team_consensus(st.session_state.raw)))

# ---------- SHARED UI ----------
def grade_class(g):
    if str(g).startswith("A"):
        return "grade"
    if str(g)=="B":
        return "grade b"
    return "grade c"

def fmt_odds(x):
    if pd.isna(x):
        return "—"
    return f"{float(x):+.0f}"

def top_header(title,subtitle):
    st.markdown(
        f"""<div class="hero">
          <div><h1>{title}</h1><p>{subtitle}</p></div>
          <div class="status-pill {'off' if not KEY else ''}">{'● LIVE DATA' if KEY else '● DEMO / CACHE'}</div>
        </div>""",
        unsafe_allow_html=True
    )


def betting_filter_bar(df, key_prefix="marketfilter"):
    """User-controlled league/market/direction filtering."""
    if df is None or len(df) == 0:
        return df

    c1,c2,c3=st.columns([1.1,1.5,1.4])

    with c1:
        league_choice=st.segmented_control(
            "League",
            ["All","NBA","NFL"],
            default="All",
            key=f"{key_prefix}_league"
        )

    with c2:
        market_choice=st.segmented_control(
            "Bet type",
            ["All","Moneyline","Spread","Total"],
            default="All",
            key=f"{key_prefix}_market"
        )

    with c3:
        total_side=st.segmented_control(
            "Total side",
            ["Both","Over","Under"],
            default="Both",
            key=f"{key_prefix}_totalside",
            disabled=market_choice not in ("All","Total")
        )

    out=df.copy()

    if league_choice and league_choice!="All" and "sport" in out.columns:
        out=out[out["sport"].astype(str).str.upper()==league_choice]

    market_map={"Moneyline":"h2h","Spread":"spreads","Total":"totals"}
    if market_choice and market_choice!="All" and "market" in out.columns:
        out=out[out["market"].astype(str).str.lower()==market_map[market_choice]]

    # Only apply Over/Under direction to rows that are totals.
    if total_side and total_side!="Both" and "market" in out.columns and "selection" in out.columns:
        is_total=out["market"].astype(str).str.lower().eq("totals")
        side_match=out["selection"].astype(str).str.lower().str.startswith(total_side.lower())
        out=out[(~is_total) | side_match]

    st.caption(f"Showing {len(out):,} of {len(df):,} available betting opportunities.")
    return out

def metric_cards():
    total_rows=count("market_snapshots")
    finals=count("game_results")
    props=count("player_prop_snapshots")
    qualified=int((team.model_prob>=.58).sum()) if len(team) else 0
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card"><div class="metric-label">Markets Tracked</div><div class="metric-value">{total_rows:,}</div><div class="metric-note">Archived sportsbook rows</div></div>
      <div class="metric-card"><div class="metric-label">Results Stored</div><div class="metric-value">{finals:,}</div><div class="metric-note">For model training</div></div>
      <div class="metric-card"><div class="metric-label">Prop Observations</div><div class="metric-value">{props:,}</div><div class="metric-note">Player market history</div></div>
      <div class="metric-card"><div class="metric-label">58%+ Team Spots</div><div class="metric-value">{qualified}</div><div class="metric-note">Current board</div></div>
    </div>
    """,unsafe_allow_html=True)

def pick_cards(df,limit=8):
    if df is None or df.empty:
        st.info("No qualifying live markets yet. Add your API key in Settings and refresh data.")
        return
    q=df.sort_values(["model_prob","edge"],ascending=False).head(limit)
    for _,r in q.iterrows():
        game=f"{r.away_team} @ {r.home_team}"
        line="" if pd.isna(r.line) else f" {float(r.line):+g}" if r.market=="Spread" else f" {float(r.line):g}"
        selection=f"{r.selection}{line}"
        model=float(r.model_prob)*100
        market=float(r.market_prob)*100
        edge=float(r.edge)*100
        ev=float(r.ev_per_unit)*100 if pd.notna(r.ev_per_unit) else 0
        g=str(r.confidence)
        st.markdown(f"""
        <div class="pick-card">
          <div class="pick-head">
            <div>
              <div class="pick-league">{r.sport} • {r.market}</div>
              <div class="pick-game">{game}</div>
            </div>
            <span class="{grade_class(g)}">{g}</span>
          </div>
          <div class="pick-selection">{selection}</div>
          <div><span class="market-chip">{fmt_odds(r.best_price)} @ {r.best_book or 'Best Book'}</span>
               <span class="market-chip">{r.prob_source}</span></div>
          <div class="pick-meta">
            <div class="pick-stat"><span>Model</span><strong>{model:.1f}%</strong></div>
            <div class="pick-stat"><span>Market</span><strong>{market:.1f}%</strong></div>
            <div class="pick-stat"><span>Edge</span><strong>{edge:+.1f}%</strong></div>
            <div class="pick-stat"><span>EV / Unit</span><strong>{ev:+.1f}%</strong></div>
          </div>
        </div>
        """,unsafe_allow_html=True)

def market_table(df):
    if df is None or df.empty:
        st.info("No market data available.")
        return
    q=df.copy()
    q["Model %"]=(q.model_prob*100).round(1)
    q["Market %"]=(q.market_prob*100).round(1)
    q["Edge %"]=(q.edge*100).round(1)
    q["EV %"]=(q.ev_per_unit*100).round(1)
    q["Odds"]=q.best_price.apply(fmt_odds)
    cols=["commence_time","away_team","home_team","market","selection","line","Odds","best_book",
          "Model %","Market %","Edge %","EV %","confidence"]
    st.dataframe(q[cols].sort_values(["Model %","Edge %"],ascending=False),
                 use_container_width=True,hide_index=True,height=520)

def filter_bar(df,key):
    if df is None or df.empty:
        return df
    c1,c2,c3,c4=st.columns([1.2,1.2,1,1])
    markets=c1.multiselect("Market",sorted(df.market.dropna().unique()),default=sorted(df.market.dropna().unique()),key=f"{key}_markets")
    grade_opts=sorted(df.confidence.dropna().astype(str).unique())
    grades=c2.multiselect("Grade",grade_opts,default=grade_opts,key=f"{key}_grades")
    minp=c3.slider("Min probability",50,75,54,key=f"{key}_prob")/100
    mine=c4.slider("Min edge %",-5,15,-5,key=f"{key}_edge")/100
    return df[df.market.isin(markets)&df.confidence.astype(str).isin(grades)&(df.model_prob>=minp)&(df.edge>=mine)]


def open_game(event_id):
    st.session_state.selected_event_id=event_id
    st.session_state.pending_nav="🎯 Game Center"
    st.rerun()

def game_buttons(df,keyprefix):
    if df is None or df.empty:
        return
    evs=df[["event_id","sport","away_team","home_team","commence_time"]].drop_duplicates("event_id")
    st.markdown('<div class="small-muted">CLICK A GAME TO OPEN FULL MATCHUP ANALYSIS</div>',unsafe_allow_html=True)
    cols=st.columns(3)
    for i,(_,r) in enumerate(evs.head(12).iterrows()):
        with cols[i%3]:
            if st.button(f"{r.sport}  •  {r.away_team} @ {r.home_team}",
                         key=f"{keyprefix}_{r.event_id}",use_container_width=True):
                open_game(r.event_id)

def stat_compare_card(home,away,hs,aas,label):
    def v(d,k,fmt):
        x=d.get(k,np.nan) if isinstance(d,dict) else np.nan
        if x is None or (isinstance(x,float) and np.isnan(x)):return "—"
        return fmt(x)
    st.markdown(f"""
    <div class="section-card">
      <div class="pick-league">{label}</div>
      <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:center;margin-top:12px">
        <div>
          <div class="pick-selection">{home}</div>
          <div class="small-muted">Win {v(hs,'win_pct',lambda x:f'{x:.0%}')} • PF {v(hs,'pf',lambda x:f'{x:.1f}')} • PA {v(hs,'pa',lambda x:f'{x:.1f}')}</div>
          <div class="small-muted">Avg margin {v(hs,'margin',lambda x:f'{x:+.1f}')}</div>
        </div>
        <div class="small-muted">VS</div>
        <div style="text-align:right">
          <div class="pick-selection">{away}</div>
          <div class="small-muted">Win {v(aas,'win_pct',lambda x:f'{x:.0%}')} • PF {v(aas,'pf',lambda x:f'{x:.1f}')} • PA {v(aas,'pa',lambda x:f'{x:.1f}')}</div>
          <div class="small-muted">Avg margin {v(aas,'margin',lambda x:f'{x:+.1f}')}</div>
        </div>
      </div>
    </div>
    """,unsafe_allow_html=True)

def line_book_table(raw,event_id,market_key):
    x=raw[(raw.event_id==event_id)&(raw.market==market_key)].copy()
    if x.empty:
        st.info("No sportsbook rows for this market.")
        return
    show=x[["bookmaker","outcome","point","price"]].sort_values(["outcome","price"],ascending=[True,False])
    st.dataframe(show,use_container_width=True,hide_index=True)

# ---------- PAGE CONTENT ----------
if page=="🔥 Best Bets":
    top_header("Best Bets","Your highest-rated NBA and NFL opportunities in one place.")
    st.markdown("### Choose league and bet type")
    if _secret_api_key():
        st.caption("☁️ Cloud mode ready • server-side API key protected • public dashboard")

    metric_cards()
    ap=get_autopilot_status()
    bt=latest_backtests()
    c1,c2,c3,c4=st.columns(4)
    c1.metric("AI Autopilot","RUNNING" if (AUTOPILOT_THREAD and AUTOPILOT_THREAD.is_alive()) else "STOPPED")
    c2.metric("Last NBA train",str(ap.get("last_retrain_nba") or "Waiting")[:19])
    c3.metric("Last NFL train",str(ap.get("last_retrain_nfl") or "Waiting")[:19])
    c4.metric("Last data refresh",str(ap.get("last_data_refresh") or "Waiting")[:19])
    if ap.get("last_error"):
        st.caption("Autopilot note: "+str(ap.get("last_error"))[:240])
    a,b=st.columns([1,1])
    with a:
        st.subheader("Board controls")
        board_mode=st.segmented_control("View",["Cards","Table"],default="Cards")
    with b:
        if st.button("↻ Refresh all live data",type="primary",use_container_width=True,disabled=not bool(KEY)):
            with st.spinner("Refreshing sportsbook lines and recent scores..."):
                st.session_state.raw,errs=refresh()
            for e in errs:st.warning(e)
            st.rerun()
    filtered=filter_bar(team,"best")
    game_buttons(team,"bestgame")
    st.subheader("Top opportunities")
    if board_mode=="Cards":
        pick_cards(filtered,10)
    else:
        market_table(filtered)

elif page in ("🏀 NBA","🏈 NFL"):
    sport="NBA" if "NBA" in page else "NFL"
    top_header(sport,"Live moneyline, spread and total markets with model-vs-market edge.")
    d=team[team.sport.eq(sport)] if len(team) else team
    game_buttons(d,f"{sport.lower()}game")
    filtered=filter_bar(d,sport.lower())
    c1,c2=st.columns([1.4,1])
    with c1:
        st.subheader("Top picks")
        pick_cards(filtered,6)
    with c2:
        st.subheader("Market board")
        market_table(filtered)


elif page=="🎯 Game Center":
    top_header("Game Center","Open one matchup and drill into moneyline, spread, totals and recent-form evidence.")
    if st.button("← Back to Best Bets", key="gamecenter_back"):
        st.session_state.pending_nav="🔥 Best Bets"
        st.rerun()
    eid=st.session_state.get("selected_event_id")
    if not eid:
        st.info("Choose a game from Best Bets, NBA, or NFL first.")
    else:
        evraw=st.session_state.raw[st.session_state.raw.event_id.eq(eid)]
        evteam=team[team.event_id.eq(eid)] if len(team) else pd.DataFrame()
        if evraw.empty:
            st.warning("That game is no longer in the current live feed.")
        else:
            b=evraw.iloc[0]
            sport=str(b.sport);home=str(b.home_team);away=str(b.away_team)
            st.markdown(f"<h2 style='margin-bottom:3px'>{away} <span style='color:#8391a2'>@</span> {home}</h2>",unsafe_allow_html=True)
            st.caption(f"{sport} • {b.commence_time}")
            packet=matchup_packet(sport,home,away)
            model=packet.get("model")

            if model:
                c1,c2,c3,c4=st.columns(4)
                c1.metric(f"{home} win",f"{model['home_win_prob']:.1%}")
                c2.metric("Projected margin",f"{home} {model['pred_margin']:+.1f}")
                c3.metric("Projected total",f"{model['pred_total']:.1f}")
                c4.metric("Model","Historical Form")
            else:
                st.info("Run Historical Bootstrap in Model Lab to unlock the matchup form model.")

            market_tabs=st.tabs(["H2H / Moneyline","Spread","Total O/U","Head-to-Head Games"])
            with market_tabs[0]:
                window=st.segmented_control("History window",["Last 5","Last 10"],default="Last 5",key=f"h2hwin_{eid}")
                n=5 if window=="Last 5" else 10
                hs=packet["home_l5"] if n==5 else packet["home_l10"]
                aas=packet["away_l5"] if n==5 else packet["away_l10"]
                stat_compare_card(home,away,hs,aas,window)
                st.subheader("Current moneyline")
                cur=evteam[evteam.market.eq("Moneyline")]
                if len(cur):
                    q=cur.copy();q["Model %"]=(q.model_prob*100).round(1);q["Market %"]=(q.market_prob*100).round(1);q["Edge %"]=(q.edge*100).round(1)
                    st.dataframe(q[["selection","best_price","best_book","Model %","Market %","Edge %","confidence"]],use_container_width=True,hide_index=True)
                line_book_table(st.session_state.raw,eid,"h2h")

            with market_tabs[1]:
                cur=evteam[evteam.market.eq("Spread")]
                if cur.empty:
                    st.info("No current spread available.")
                else:
                    side=st.selectbox("Analyze spread side",cur.selection.tolist(),key=f"spreadside_{eid}")
                    row=cur[cur.selection.eq(side)].iloc[0]
                    nopt=st.segmented_control("Trend window",["Last 5","Last 10"],default="Last 5",key=f"spreadwin_{eid}")
                    n=5 if nopt=="Last 5" else 10
                    rate=current_line_hit_rates(sport,side,"Spread",row.line,n)
                    hist=history_for_team(sport,side,n)
                    c1,c2,c3,c4=st.columns(4)
                    c1.metric("Current line",f"{float(row.line):+g}")
                    c2.metric("Model probability",f"{float(row.model_prob):.1%}")
                    c3.metric(f"{nopt} vs today's line",f"{rate['hit_rate']:.0%}" if pd.notna(rate["hit_rate"]) else "—")
                    c4.metric("Best price",fmt_odds(row.best_price),delta=str(row.best_book))
                    st.caption("The Last 5/10 hit rate asks whether each past result would have covered TODAY'S posted spread; it is not historical ATS closing-line data.")
                    if len(hist):
                        plot=hist[["game_date","margin"]].sort_values("game_date")
                        st.bar_chart(plot.set_index("game_date"))
                    st.subheader("Book-by-book spread")
                    line_book_table(st.session_state.raw,eid,"spreads")

            with market_tabs[2]:
                cur=evteam[evteam.market.eq("Total")]
                if cur.empty:
                    st.info("No current total available.")
                else:
                    line=float(cur.line.dropna().median())
                    nopt=st.segmented_control("Trend window",["Last 5","Last 10"],default="Last 5",key=f"totalwin_{eid}")
                    n=5 if nopt=="Last 5" else 10
                    hh=history_for_team(sport,home,n);aa=history_for_team(sport,away,n)
                    home_over=float(((hh.team_score+hh.opp_score)>line).mean()) if len(hh) else np.nan
                    away_over=float(((aa.team_score+aa.opp_score)>line).mean()) if len(aa) else np.nan
                    over=cur[cur.selection.eq("Over")]
                    under=cur[cur.selection.eq("Under")]
                    c1,c2,c3,c4=st.columns(4)
                    c1.metric("Current total",f"{line:g}")
                    c2.metric(f"{home} {nopt} > line",f"{home_over:.0%}" if pd.notna(home_over) else "—")
                    c3.metric(f"{away} {nopt} > line",f"{away_over:.0%}" if pd.notna(away_over) else "—")
                    if len(over): c4.metric("Model Over",f"{float(over.iloc[0].model_prob):.1%}")
                    if model:
                        st.info(f"Historical-form projected score total: {model['pred_total']:.1f}")
                    st.subheader("Current Over / Under")
                    q=cur.copy();q["Model %"]=(q.model_prob*100).round(1);q["Edge %"]=(q.edge*100).round(1)
                    st.dataframe(q[["selection","line","best_price","best_book","Model %","Edge %","confidence"]],use_container_width=True,hide_index=True)
                    st.subheader("Book-by-book totals")
                    line_book_table(st.session_state.raw,eid,"totals")

            with market_tabs[3]:
                h2hn=st.segmented_control("Meetings",["Last 5","Last 10"],default="Last 5",key=f"meet_{eid}")
                q=packet["h2h5"] if h2hn=="Last 5" else packet["h2h10"]
                if q.empty:
                    st.info("No head-to-head games found in the local historical database.")
                else:
                    q=q.copy()
                    q["winner"]=np.where(q.home_score>q.away_score,q.home_team,q.away_team)
                    q["total"]=q.home_score+q.away_score
                    st.dataframe(q[["game_date","away_team","away_score","home_team","home_score","winner","total"]],use_container_width=True,hide_index=True)


elif page=="👤 Player Lab":
    top_header("Player Lab","Search one player and compare live prop lines against recent performance.")
    st.markdown('<div class="section-card">',unsafe_allow_html=True)
    c1,c2,c3=st.columns([1,2.4,1])
    sp=c1.selectbox("League",["NBA","NFL"],key="playerleague")
    name=c2.text_input("Search player",placeholder="Try: Jalen Brunson or Josh Allen")
    scan=c3.slider("Games to scan",1,20,8,key="playerscan")
    search=st.button("Analyze player",type="primary",use_container_width=True,disabled=not bool(KEY))
    st.markdown('</div>',unsafe_allow_html=True)

    if search:
        frames=[]
        with st.spinner("Loading current props and recent player performance..."):
            try:
                ev,_=events(sp,KEY)
                for e in ev[:scan]:
                    try:
                        p,_=event_props(sp,e["id"],KEY)
                        if len(p):
                            append_df("player_prop_snapshots",p)
                            frames.append(p)
                    except Exception:
                        pass
                rawp=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
                st.session_state.player=score_props(prop_consensus(rawp),name.strip() or None)
            except Exception as e:
                st.error(str(e))

    p=st.session_state.player
    if len(p):
        q=p.sort_values(["model_prob","edge"],ascending=False)
        st.subheader(f"Player opportunities")
        pick_cards(q.rename(columns={"side":"_side"}),8)
        q2=q.copy()
        q2["Model %"]=(q2.model_prob*100).round(1)
        q2["Edge %"]=(q2.edge*100).round(1)
        q2["Hit %"]=(q2.recent_hit_rate*100).round(1)
        st.subheader("Detailed prop breakdown")
        st.dataframe(q2[["player_name","market","side","point","best_price","best_book",
                         "Model %","Edge %","games_used","recent_average","Hit %",
                         "confidence","away_team","home_team"]],
                     use_container_width=True,hide_index=True)
    else:
        st.info("Search a player to build a live prop board.")

elif page=="🔎 Prop Scanner":
    top_header("Prop Scanner","Scan an entire slate and rank player props by modeled probability and edge.")
    c1,c2,c3=st.columns([1,1,2])
    sp=c1.selectbox("League",["NBA","NFL"],key="propscanleague")
    n=c2.slider("Games",1,20,6,key="propscangames")
    player_filter=c3.text_input("Optional player filter",placeholder="Leave blank for all players")
    if st.button("Scan live props",type="primary",disabled=not bool(KEY)):
        frames=[]
        with st.spinner("Fetching prop markets and evaluating recent player histories..."):
            try:
                ev,_=events(sp,KEY)
                for e in ev[:n]:
                    try:
                        p,_=event_props(sp,e["id"],KEY)
                        if len(p):
                            append_df("player_prop_snapshots",p)
                            frames.append(p)
                    except Exception:
                        pass
                rawp=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
                st.session_state.props=score_props(prop_consensus(rawp),player_filter.strip() or None)
            except Exception as e:
                st.error(str(e))
    q=st.session_state.props
    if len(q):
        c1,c2,c3=st.columns(3)
        minp=c1.slider("Min model %",50,80,55,key="scanminp")/100
        mine=c2.slider("Min edge %",-5,20,-2,key="scanmine")/100
        market_names=sorted(q.market.dropna().unique())
        markets=c3.multiselect("Prop market",market_names,default=market_names)
        z=q[(q.model_prob>=minp)&(q.edge>=mine)&q.market.isin(markets)].copy()
        z["Model %"]=(z.model_prob*100).round(1);z["Edge %"]=(z.edge*100).round(1)
        z["EV %"]=(z.ev_per_unit*100).round(1)
        st.dataframe(z[["player_name","market","side","point","best_price","best_book",
                        "Model %","Edge %","EV %","games_used","recent_average",
                        "confidence","away_team","home_team"]].sort_values(["Model %","Edge %"],ascending=False),
                     use_container_width=True,hide_index=True,height=600)
        st.download_button("Download filtered prop board",z.to_csv(index=False).encode(),
                           "wolfsports_prop_board.csv","text/csv")

elif page=="🧾 Parlay Builder":
    top_header("Parlay Builder","Build compact value sheets or go all the way to a 16-leg longshot.")
    st.markdown('<div class="section-card">',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    legs=c1.selectbox("Legs",SIZES,index=2)
    mode=c2.selectbox("Strategy",["Highest Chance","Best Value","Longshot"])
    minp=c3.slider("Minimum leg probability",50,75,55,key="parlayp")/100
    mine=c4.slider("Minimum model edge %",-5,15,-1,key="parlaye")/100
    c5,c6=st.columns(2)
    include_props=c5.toggle("Include player props",True)
    same=c6.toggle("Allow multiple legs from same game",False)
    build_it=st.button(f"Build {legs}-leg {mode} parlay",type="primary",use_container_width=True)
    st.markdown('</div>',unsafe_allow_html=True)

    alllegs=team.copy()
    if include_props and len(st.session_state.props):
        alllegs=pd.concat([alllegs,st.session_state.props],ignore_index=True,sort=False) if len(alllegs) else st.session_state.props

    if build_it:
        st.session_state.parlay=build(alllegs,legs,minp,mine,mode,same)

    r=st.session_state.parlay
    if r:
        p=r["legs"]
        if not r["complete"]:
            st.warning(f"Only {len(p)} qualifying legs were available. WolfSportsAI did not weaken your filters just to fill the ticket.")
        if len(p)>=2:
            st.markdown(f"""
            <div class="parlay-summary">
              <div class="metric-grid" style="margin:0">
                <div class="metric-card"><div class="metric-label">Legs</div><div class="metric-value">{len(p)}</div></div>
                <div class="metric-card"><div class="metric-label">Est. Combined Chance</div><div class="metric-value">{r['combined_prob']:.5%}</div></div>
                <div class="metric-card"><div class="metric-label">Approx. Odds</div><div class="metric-value">{r['american_odds']:+.0f}</div></div>
                <div class="metric-card"><div class="metric-label">Strategy</div><div class="metric-value" style="font-size:1.05rem">{mode}</div></div>
              </div>
            </div>
            """,unsafe_allow_html=True)
        if len(p):
            q=p.copy();q["Model %"]=(q.model_prob*100).round(1);q["Edge %"]=(q.edge*100).round(1)
            cols=[c for c in ["sport","leg_type","player_name","market","selection","line",
                              "best_price","best_book","Model %","Edge %","confidence",
                              "away_team","home_team"] if c in q.columns]
            st.dataframe(q[cols],use_container_width=True,hide_index=True)
            st.download_button("Download parlay sheet",q.to_csv(index=False).encode(),
                               f"wolfsports_{legs}_leg_parlay.csv","text/csv")
    else:
        st.info("Choose your settings and build a parlay sheet.")

elif page=="🧠 Model Lab":
    top_header("Model Lab","Autopilot continuously retrains and backtests while WolfSportsAI is running.")
    st.success("AI Autopilot is automatic. You do not need to press Train or Backtest. Manual controls below are only for forcing an immediate rebuild.")
    ap=get_autopilot_status()
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Autopilot","RUNNING" if (AUTOPILOT_THREAD and AUTOPILOT_THREAD.is_alive()) else "STOPPED")
    c2.metric("Cycles",int(ap.get("cycle_count") or 0))
    c3.metric("NBA last backtest",str(ap.get("last_backtest_nba") or "Waiting")[:19])
    c4.metric("NFL last backtest",str(ap.get("last_backtest_nfl") or "Waiting")[:19])

    bt=latest_backtests()
    st.subheader("Automatic backtest history")
    if len(bt):
        show=bt.copy()
        for col in ["accuracy","brier","margin_mae","total_mae"]:
            if col in show.columns:
                show[col]=pd.to_numeric(show[col],errors="coerce").round(4)
        st.dataframe(show,use_container_width=True,hide_index=True)
    else:
        st.info("Autopilot has not completed its first historical backtest yet.")
    st.subheader("Historical Bootstrap")
    st.write("This fills the model from past real game results instead of waiting for 150 future games. NBA uses NBA Stats game logs; NFL uses nflverse schedules.")
    b1,b2,b3=st.columns([1,1,2])
    boot_sport=b1.selectbox("Bootstrap league",["NBA","NFL"])
    boot_seasons=b2.slider("Seasons",2,8,4 if boot_sport=="NBA" else 6)
    with b3:
        st.write("")
        st.write("")
        if st.button(f"Download {boot_seasons} seasons + train {boot_sport}",type="primary",use_container_width=True):
            with st.spinner("Downloading historical games, calculating rolling Last 5/10 features, and training models..."):
                try:
                    games,feats=bootstrap_sport(boot_sport,boot_seasons)
                    model,info=train_bootstrap_models(boot_sport,150 if boot_sport=="NBA" else 100)
                    if model is None:
                        st.warning(info["status"])
                    else:
                        st.success(f"{boot_sport} bootstrap complete: {len(games):,} games, {info['rows']:,} model rows.")
                        st.write(f"Validation accuracy: {info['accuracy']:.1%} • Brier: {info['brier']:.3f} • Margin MAE: {info['margin_mae']:.2f} • Total MAE: {info['total_mae']:.2f}")
                except Exception as e:
                    st.error(str(e))
    nba_boot=load_bootstrap("NBA");nfl_boot=load_bootstrap("NFL")
    c1,c2=st.columns(2)
    c1.info("NBA historical form model: " + ("READY" if nba_boot else "Not trained"))
    c2.info("NFL historical form model: " + ("READY" if nfl_boot else "Not trained"))
    st.divider()
    st.subheader("Legacy odds-calibration models")
    st.caption("These still learn from odds snapshots + final scores collected by WolfSportsAI, but the historical form model above no longer requires you to wait for 150 future games.")
    cols=st.columns(3)
    for i,m in enumerate(["Moneyline","Spread","Total"]):
        with cols[i]:
            st.markdown(f'<div class="section-card"><h3>{m}</h3>',unsafe_allow_html=True)
            for sp in ["NBA","NFL"]:
                if st.button(f"Train {sp} {m}",key=f"train_{sp}_{m}",use_container_width=True):
                    _,info=train(sp,m,150 if sp=="NBA" else 100)
                    if info["status"]=="trained":
                        st.success(f"{info['rows']} rows • Brier {info['brier']:.3f}")
                    else:
                        st.warning(f"{info['status']} Current: {info['rows']}")
            st.markdown('</div>',unsafe_allow_html=True)
    h=read_sql("SELECT * FROM model_runs ORDER BY trained_at DESC LIMIT 25")
    st.subheader("Training history")
    if len(h):
        st.dataframe(h,use_container_width=True,hide_index=True)
    else:
        st.info("No local model training runs yet.")

elif page=="⚙️ Settings":
    top_header("Settings","Connect live data and manage local WolfSportsAI behavior.")
    st.subheader("Live odds connection")
    if _secret_api_key():
        st.success("Server API key is configured securely through Streamlit Secrets.")
        st.caption("Visitors cannot see the API key.")
    else:
        k=st.text_input("The Odds API key",value=KEY,type="password")
        if st.button("Save API key",type="primary"):
            try:
                current=load_settings()
                current["ODDS_API_KEY"]=k.strip()
                save_settings(current)
                st.success("API key saved to your Windows user settings. Restart WolfSportsAI once.")
                st.caption(f"Saved at: {SETTINGS_FILE}")
            except Exception as e:
                st.error(f"Could not save the API key: {e}")
                st.info("For cloud deployment, add ODDS_API_KEY in Streamlit Secrets.")
    st.divider()
    st.subheader("Autopilot")
    st.write("Automatic schedule while WolfSportsAI is open:")
    st.write("• Odds/results refresh: every 15 minutes")
    st.write("• Model retraining: every 12 hours")
    st.write("• Backtesting: every 6 hours")
    st.caption("On a free cloud host, Autopilot runs while the app instance is awake. Free hosts may sleep/restart inactive apps; WolfSportsAI rebuilds its historical model state when needed.")
    st.subheader("Data sources")
    st.write("Sportsbook odds: The Odds API")
    st.write("NBA player history: nba_api")
    st.write("NFL player history: nflreadpy / nflverse")
    st.subheader("Local database")
    st.code(str((ROOT/"data"/"wolfsports.db").resolve()))
    st.caption("All local learning history remains on your computer.")


# Start Autopilot only after the page UI has been built.
# This keeps slow APIs/history bootstrap from blocking the first cloud render.
if KEY and "_autopilot_thread" not in st.session_state:
    try:
        st.session_state["_autopilot_thread"] = _start_autopilot_worker(KEY)
    except Exception as _ap_start_error:
        st.session_state["_autopilot_start_error"] = str(_ap_start_error)
