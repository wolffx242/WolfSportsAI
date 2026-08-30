
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from database import init_db,append_df,upsert_results,upsert_upcoming_games,read_sql,count
from odds_api import featured,scores,events,event_props,upcoming_events
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
        refresh_minutes=5,
        retrain_hours=4,
        backtest_hours=12
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

st.markdown("""
<style>
/* V6.2 trading-terminal / sportsbook-inspired interaction layer */
:root{
  --wolf-aqua:#62e0cf;
  --wolf-aqua-soft:#173d38;
  --wolf-orange:#e76f34;
  --wolf-card:#0b1015;
  --wolf-card2:#10161c;
  --wolf-line:#202a32;
}
.block-container{max-width:1700px;padding-left:1.35rem;padding-right:1.35rem}
[data-testid="stSidebar"]{min-width:190px;max-width:230px}
[data-testid="stSidebar"] .stRadio > div{gap:.15rem}
[data-testid="stSidebar"] .stRadio label{
  border-radius:9px;padding:.28rem .45rem;transition:.15s ease;
}
[data-testid="stSidebar"] .stRadio label:hover{background:#121920}
.wolf-toolbar{
  display:flex;gap:8px;align-items:center;flex-wrap:wrap;
  border-bottom:1px solid var(--wolf-line);padding:0 0 12px;margin-bottom:12px
}
.wolf-tab{
  border:1px solid #27333d;background:#0b1117;color:#aeb8c2;
  border-radius:10px;padding:7px 12px;font-size:.78rem;font-weight:800
}
.game-row{
  background:linear-gradient(180deg,#101419,#0b0f13);
  border:1px solid #1e272f;border-radius:15px;padding:12px 14px;margin:10px 0;
}
.game-row-top{display:grid;grid-template-columns:minmax(190px,1.2fr) repeat(4,minmax(135px,1fr));gap:10px;align-items:stretch}
.team-stack{display:flex;flex-direction:column;justify-content:center;gap:8px;padding:4px 8px}
.team-line{display:flex;justify-content:space-between;gap:10px;align-items:center}
.team-name{font-size:.92rem;font-weight:800;color:#f5f7fa}
.team-meta{font-size:.69rem;color:#73818f;text-transform:uppercase;letter-spacing:.06em}
.market-box{
  position:relative;background:#0d1318;border:1px solid #202a32;border-radius:11px;
  padding:10px;min-height:68px;overflow:hidden
}
.market-box:before{
  content:"";position:absolute;left:0;top:0;right:0;height:3px;
  background:linear-gradient(90deg,#50d6bd 0 55%,#d75d37 55% 72%,#52606a 72%)
}
.market-label{font-size:.67rem;text-transform:uppercase;letter-spacing:.08em;color:#71808d;font-weight:800;margin-bottom:8px}
.market-pick{font-size:.91rem;font-weight:850;color:#f4f7f9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.market-odds{font-size:.77rem;color:#b9c5ce;margin-top:4px}
.market-edge{font-size:.69rem;color:#54d9bd;margin-top:3px;font-weight:800}
.game-row-foot{
  display:flex;justify-content:space-between;gap:8px;align-items:center;
  padding-top:10px;margin-top:10px;border-top:1px solid #182028
}
.terminal-label{font-size:.69rem;color:#73818f;text-transform:uppercase;letter-spacing:.08em;font-weight:800}
.hit-strip{display:flex;gap:4px;margin-top:8px}
.hit-dot{height:5px;flex:1;border-radius:999px;background:#26313a}
.hit-dot.hit{background:#5ce0c3}.hit-dot.miss{background:#d65a3b}
.player-prop-row{
  display:grid;grid-template-columns:minmax(210px,1.6fr) .7fr .7fr .7fr .7fr;
  gap:10px;align-items:center;background:#0d1217;border:1px solid #1d262e;
  border-radius:12px;padding:11px 13px;margin:7px 0
}
.player-main{font-weight:850;color:#f3f6f8}
.player-sub{font-size:.73rem;color:#81909d;margin-top:2px}
.kpi-mini{font-size:.73rem;color:#81909d}
.kpi-mini strong{display:block;color:#edf2f5;font-size:.86rem;margin-top:2px}
.hero-match{
  background:
   radial-gradient(circle at 18% 5%,rgba(36,180,154,.26),transparent 28%),
   radial-gradient(circle at 82% 0%,rgba(214,93,49,.25),transparent 32%),
   #0d1217;
  border:1px solid #273039;border-radius:18px;padding:20px;margin-bottom:14px
}
.hero-match-title{font-size:1.7rem;font-weight:900;text-align:center;color:#f6f8fa}
.hero-match-sub{text-align:center;color:#82909c;font-size:.82rem;margin-top:4px}
.hero-match-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:16px}
.hero-stat{background:rgba(6,10,13,.62);border:1px solid #202a32;border-radius:12px;padding:12px;text-align:center}
.hero-stat span{display:block;color:#768590;font-size:.67rem;text-transform:uppercase;font-weight:800}
.hero-stat strong{display:block;color:#f4f7f9;font-size:1.08rem;margin-top:4px}
.detail-card{background:#0c1116;border:1px solid #1d2730;border-radius:14px;padding:15px;margin:8px 0}
.detail-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}
.detail-stat{background:#10161c;border-radius:10px;padding:10px}
.detail-stat span{display:block;color:#71808c;font-size:.67rem;text-transform:uppercase;font-weight:800}
.detail-stat strong{display:block;color:#f3f6f8;font-size:.94rem;margin-top:3px}
@media(max-width:1100px){
 .game-row-top{grid-template-columns:1fr 1fr}
 .team-stack{grid-column:1/-1}
 .player-prop-row{grid-template-columns:1fr 1fr}
 .hero-match-stats,.detail-grid{grid-template-columns:1fr 1fr}
}
@media(max-width:650px){
 .game-row-top,.player-prop-row,.hero-match-stats,.detail-grid{grid-template-columns:1fr}
 .block-container{padding-left:.7rem;padding-right:.7rem}
}
</style>
""", unsafe_allow_html=True)


st.markdown(r"""<style>
/* V7 MATCHUPS — pick-first table layout */
.matchup-wrap{background:#0b1118;border:1px solid #1c2835;border-radius:14px;margin:10px 0 14px;overflow:hidden}
.matchup-head{display:grid;grid-template-columns:minmax(240px,2fr) minmax(130px,1fr) minmax(130px,1fr) minmax(120px,.8fr);gap:10px;padding:9px 14px;background:#0e1620;border-bottom:1px solid #1c2835;color:#748398;font-size:.65rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase}
.matchup-body{display:grid;grid-template-columns:minmax(240px,2fr) minmax(130px,1fr) minmax(130px,1fr) minmax(120px,.8fr);gap:10px;align-items:center;padding:12px 14px}
.matchup-teams{display:grid;gap:10px}
.matchup-team{display:flex;align-items:center;justify-content:space-between;gap:10px;min-height:32px}
.matchup-team-name{font-weight:800;color:#f4f7fb;font-size:.88rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.matchup-side{font-size:.63rem;color:#6f8095;font-weight:700}
.matchup-market{display:grid;gap:8px}
.matchup-line{background:#111b26;border:1px solid #213142;border-radius:8px;padding:7px 9px;display:flex;align-items:center;justify-content:space-between;gap:7px;min-height:31px}
.matchup-line strong{font-size:.78rem;color:#f8fafc}
.matchup-line span{font-size:.7rem;color:#91a1b5}
.win-cell{display:grid;gap:8px}
.win-row{min-height:31px;display:flex;align-items:center;gap:8px}
.win-pct{font-size:.78rem;font-weight:900;color:#e8eef7;min-width:42px;text-align:right}
.win-track{height:5px;flex:1;background:#182432;border-radius:99px;overflow:hidden}
.win-fill{height:100%;background:#22d3ee;border-radius:99px}
.win-fill.alt{background:#5ce0c3}
.matchup-ai{display:grid;gap:7px}
.ai-chip{border:1px solid #234052;background:#0d1b24;border-radius:8px;padding:7px 9px;text-align:center}
.ai-chip b{display:block;color:#22d3ee;font-size:.78rem}
.ai-chip span{font-size:.61rem;color:#748398}
.matchup-foot{display:flex;justify-content:space-between;align-items:center;padding:8px 14px;border-top:1px solid #182431;background:#0c131b;color:#718096;font-size:.66rem}
@media(max-width:900px){
 .matchup-head{display:none}
 .matchup-body{grid-template-columns:1fr 1fr}
 .matchup-teams{grid-column:1/-1}
 .matchup-foot{gap:10px;flex-wrap:wrap}
}
</style>""", unsafe_allow_html=True)

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
        ["🔥 Best Bets","🏀 NBA","🏈 NFL","📅 Upcoming Games","🎯 Game Center","👤 Player Lab","🔎 Prop Scanner",
         "🧾 Parlay Builder","🧠 Model Lab","⚙️ Settings"],
        label_visibility="collapsed",
        key="nav"
    )
    st.divider()
    st.caption("LIVE ENGINE")
    sports=st.multiselect("Leagues",["NBA","NFL"],default=["NBA","NFL"])
    auto=st.toggle(
        "Background refresh",
        True,
        help="Autopilot updates data without repeatedly reloading the whole page."
    )
    mins=st.select_slider(
        "Refresh target",
        options=[5,10,15,20,30,45,60],
        value=5,
        format_func=lambda x:f"{x} min"
    )
    st.caption("Fast mode: normal clicks use cached/local data.")
    st.caption("Scheduled mode: heavy AI work runs during low-traffic maintenance windows.")

    if KEY:
        st.markdown('<div class="status-pill">● API Connected</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill off">● API Key Needed</div>',unsafe_allow_html=True)

    st.divider()
    ap=get_autopilot_status()
    st.caption("AUTOPILOT")
    _ap_thread = st.session_state.get("_autopilot_thread")
    if _ap_thread and _ap_thread.is_alive():
        st.markdown('<div class="status-pill">● Scheduled AI Maintenance</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill off">● Scheduler Starting</div>',unsafe_allow_html=True)
    st.caption(f"Cycles: {int(ap.get('cycle_count') or 0)}")

# ---------- FAST DATA CACHE ----------
@st.cache_data(ttl=900, show_spinner=False)
def cached_featured(sport, api_key):
    return featured(sport, api_key)

@st.cache_data(ttl=900, show_spinner=False)
def cached_scores(sport, api_key):
    return scores(sport, api_key)

@st.cache_data(ttl=900, show_spinner=False)
def cached_events(sport, api_key):
    return events(sport, api_key)

@st.cache_data(ttl=900, show_spinner=False)
def cached_event_props(sport, event_id, api_key):
    return event_props(sport, event_id, api_key)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_stat_series(sport, player_name, market, n=20):
    vals, resolved = stat_series(sport, player_name, market, n)
    return [float(v) for v in list(vals)], resolved

@st.cache_data(ttl=30, show_spinner=False)
def cached_snapshot(sport):
    return read_sql("""SELECT * FROM market_snapshots WHERE sport=? AND captured_at=
      (SELECT MAX(captured_at) FROM market_snapshots WHERE sport=?)""",(sport,sport))


@st.cache_data(ttl=60, show_spinner=False)
def cached_upcoming_schedule():
    return read_sql("""
        SELECT event_id,sport,commence_time,home_team,away_team,source,updated_at
        FROM upcoming_games
        WHERE datetime(commence_time) >= datetime('now','-3 hours')
        ORDER BY datetime(commence_time), sport, away_team
    """)



@st.cache_data(ttl=900, show_spinner=False, max_entries=8)
def cached_team_model(raw_json):
    """Expensive consensus + ML enrichment, computed once per odds snapshot."""
    if not raw_json:
        return pd.DataFrame()
    frame=pd.read_json(raw_json,orient="split")
    if frame.empty:
        return pd.DataFrame()
    return enrich(apply(team_consensus(frame)))

@st.cache_data(ttl=900, show_spinner=False, max_entries=16)
def cached_market_view(team_json, sport, market):
    """Prebuilt sport/market slice for instant tab switching."""
    if not team_json:
        return pd.DataFrame()
    frame=pd.read_json(team_json,orient="split")
    if frame.empty:
        return frame
    out=frame
    if sport and sport!="All":
        out=out[out["sport"].astype(str).eq(str(sport))]
    if market and market!="All":
        out=out[out["market"].astype(str).eq(str(market))]
    return out.reset_index(drop=True)

# ---------- DATA ----------
def cache(s):
    return cached_snapshot(s).copy()

def refresh():
    frames=[];errs=[]
    for s in sports:
        try:
            d,_=cached_featured(s,KEY)
            if len(d):
                append_df("market_snapshots",d)
                frames.append(d)
            sc,_=cached_scores(s,KEY)
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
                vals,res=cached_stat_series(r.sport,r.player_name,r.market,20)
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

# IMPORTANT: never call an external API before the selected page renders.
# On first load use the newest locally stored snapshot only.
if st.session_state.raw.empty:
    _frames=[]
    for _sport in sports:
        try:
            _d=cache(_sport)
            if _d is not None and len(_d):
                _frames.append(_d)
        except Exception:
            pass
    if _frames:
        st.session_state.raw=pd.concat(_frames,ignore_index=True)

if len(st.session_state.raw):
    _raw_json=st.session_state.raw.to_json(orient="split",date_format="iso")
    team=cached_team_model(_raw_json).copy()
    _team_json=team.to_json(orient="split",date_format="iso") if len(team) else ""
else:
    team=pd.DataFrame()
    _team_json=""


APP_TIMEZONE="America/Nassau"

def game_datetime_parts(value):
    """Display sportsbook times in Bahamas local time."""
    try:
        ts=pd.to_datetime(value,utc=True,errors="coerce")
        if pd.isna(ts):
            return "Date TBA","Time TBA"
        ts=ts.tz_convert(APP_TIMEZONE)
        return ts.strftime("%a, %b %d, %Y"), ts.strftime("%I:%M %p").lstrip("0")
    except Exception:
        return "Date TBA","Time TBA"

def game_datetime_label(value):
    d,t=game_datetime_parts(value)
    return f"{d} • {t}"

def future_only(df):
    """Parlays must use games that have not started yet."""
    if df is None or df.empty or "commence_time" not in df.columns:
        return df
    x=df.copy()
    ts=pd.to_datetime(x["commence_time"],utc=True,errors="coerce")
    now=pd.Timestamp.now(tz="UTC")
    return x[ts.notna() & (ts>now)].copy()

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
            ["Moneyline","Spread","Total","All"],
            default="Moneyline",
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

    out=df

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
              <div class="small-muted">🗓 {game_datetime_label(r.get("commence_time"))} • Bahamas time</div>
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
    q["AI Probability %"]=(q.model_prob*100).round(1)
    q["Market %"]=(q.market_prob*100).round(1)
    q["Edge %"]=(q.edge*100).round(1)
    q["EV %"]=(q.ev_per_unit*100).round(1)
    q["Odds"]=q.best_price.apply(fmt_odds)
    cols=["commence_time","away_team","home_team","market","selection","line","Odds","best_book",
          "AI Probability %","Market %","Edge %","EV %","confidence"]
    st.dataframe(q[cols].sort_values(["AI Probability %","Edge %"],ascending=False),
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



def _first_market_row(g, market, selection=None):
    if g is None or g.empty:
        return None
    q=g[g["market"].astype(str).str.lower().eq(market.lower())]
    if selection is not None and len(q):
        q=q[q["selection"].astype(str).str.lower().eq(selection.lower())]
    if q.empty:
        return None
    q=q.sort_values(["model_prob","edge"],ascending=False)
    return q.iloc[0]

def _market_box(label,row):
    if row is None:
        return f"""<div class="market-box"><div class="market-label">{label}</div>
        <div class="market-pick">No line</div><div class="market-odds">—</div></div>"""
    line=""
    if pd.notna(row.get("line",np.nan)):
        if str(row.get("market","")).lower()=="spread":
            line=f" {float(row.line):+g}"
        elif str(row.get("market","")).lower()=="total":
            line=f" {float(row.line):g}"
    pick=f"{row.selection}{line}"
    edge=float(row.get("edge",0) or 0)*100
    return f"""<div class="market-box">
      <div class="market-label">{label}</div>
      <div class="market-pick">{pick}</div>
      <div class="market-odds">{fmt_odds(row.best_price)} • {row.best_book or 'Best Book'}</div>
      <div class="market-edge">AI {float(row.model_prob)*100:.0f}% • Edge {edge:+.1f}%</div>
    </div>"""

def _game_win_probs(g, away_team, home_team):
    """Return AI straight-up win probabilities for both teams.
    Prefer current moneyline model rows; fall back to matchup model.
    """
    away_p=home_p=np.nan
    ml=g[g["market"].astype(str).str.lower().eq("moneyline")] if len(g) else pd.DataFrame()
    if len(ml):
        for _,r in ml.iterrows():
            sel=str(r.get("selection",""))
            if sel==str(away_team):
                away_p=float(r.model_prob)
            elif sel==str(home_team):
                home_p=float(r.model_prob)

    # If only one side is present, make the other complementary.
    if pd.notna(home_p) and pd.isna(away_p):
        away_p=1-home_p
    if pd.notna(away_p) and pd.isna(home_p):
        home_p=1-away_p

    # Historical matchup model fallback.
    if pd.isna(home_p) or pd.isna(away_p):
        try:
            pred=predict_matchup(str(g.iloc[0].sport),str(home_team),str(away_team))
            if pred and pred.get("home_win_prob") is not None:
                home_p=float(pred["home_win_prob"])
                away_p=1-home_p
        except Exception:
            pass

    if pd.isna(home_p) or pd.isna(away_p):
        home_p=away_p=.50
    s=max(home_p+away_p,1e-9)
    return away_p/s,home_p/s

def _side_market_row(g, market, team=None, total_side=None):
    q=g[g["market"].astype(str).str.lower().eq(str(market).lower())]
    if team is not None and len(q):
        q=q[q["selection"].astype(str).eq(str(team))]
    if total_side is not None and len(q):
        q=q[q["selection"].astype(str).str.lower().str.startswith(str(total_side).lower())]
    if q.empty:
        return None
    return q.sort_values(["model_prob","edge"],ascending=False).iloc[0]

def _line_text(row, market):
    if row is None:
        return "—", "—"
    point=row.get("line",np.nan)
    if pd.isna(point):
        point=row.get("point",np.nan)
    sel=str(row.get("selection",""))
    if str(market).lower()=="moneyline":
        txt=fmt_odds(row.best_price)
    elif str(market).lower()=="spread":
        txt=(f"{float(point):+g}" if pd.notna(point) else sel)
    else:
        prefix="O" if sel.lower().startswith("over") else "U"
        txt=(f"{prefix} {float(point):g}" if pd.notna(point) else sel)
    return txt,fmt_odds(row.best_price)

def game_market_board(df,keyprefix="board",limit=16,display_market=None):
    """WolfSportsAI matchup board: teams, current line, AI win probability, and best price."""
    if df is None or df.empty:
        st.info("No live games match the selected filters.")
        return

    # Detect the active market from the filtered frame unless caller specifies one.
    available=[str(x).lower() for x in df.market.dropna().unique()]
    market=(display_market or ("Moneyline" if "moneyline" in available else
                              "Spread" if "spread" in available else
                              "Total" if "total" in available else "Moneyline"))

    evs=df[["event_id","sport","away_team","home_team","commence_time"]].drop_duplicates("event_id").head(limit)
    st.markdown(
        f'<div class="pick-ribbon"><strong>{market}</strong> • AI WIN % is WolfSportsAI straight-up win probability, not public betting percentage.</div>',
        unsafe_allow_html=True
    )

    for _,ev in evs.iterrows():
        # Use all current team rows for the event when possible, not just a single displayed market.
        allg=team[team.event_id.eq(ev.event_id)] if len(team) and "event_id" in team.columns else df[df.event_id.eq(ev.event_id)]
        g=df[df.event_id.eq(ev.event_id)]
        away_p,home_p=_game_win_probs(allg,ev.away_team,ev.home_team)

        if market=="Spread":
            ar=_side_market_row(g,"Spread",ev.away_team)
            hr=_side_market_row(g,"Spread",ev.home_team)
        elif market=="Total":
            ar=_side_market_row(g,"Total",total_side="Over")
            hr=_side_market_row(g,"Total",total_side="Under")
        else:
            ar=_side_market_row(g,"Moneyline",ev.away_team)
            hr=_side_market_row(g,"Moneyline",ev.home_team)

        a_line,a_odds=_line_text(ar,market)
        h_line,h_odds=_line_text(hr,market)
        a_model=float(ar.model_prob)*100 if ar is not None else away_p*100
        h_model=float(hr.model_prob)*100 if hr is not None else home_p*100
        leader=ev.away_team if away_p>=home_p else ev.home_team
        leadpct=max(away_p,home_p)*100

        st.markdown(f"""
        <div class="matchup-wrap">
          <div class="matchup-head">
            <div>Matchup</div><div>{market} / Best Price</div><div>AI Win %</div><div>AI Lean</div>
          </div>
          <div class="matchup-body">
            <div class="matchup-teams">
              <div class="matchup-team"><span class="matchup-team-name">{ev.away_team}</span><span class="matchup-side">AWAY</span></div>
              <div class="matchup-team"><span class="matchup-team-name">{ev.home_team}</span><span class="matchup-side">HOME</span></div>
            </div>
            <div class="matchup-market">
              <div class="matchup-line"><strong>{a_line}</strong><span>{a_odds}</span></div>
              <div class="matchup-line"><strong>{h_line}</strong><span>{h_odds}</span></div>
            </div>
            <div class="win-cell">
              <div class="win-row"><span class="win-pct">{away_p*100:.1f}%</span><div class="win-track"><div class="win-fill" style="width:{away_p*100:.1f}%"></div></div></div>
              <div class="win-row"><span class="win-pct">{home_p*100:.1f}%</span><div class="win-track"><div class="win-fill alt" style="width:{home_p*100:.1f}%"></div></div></div>
            </div>
            <div class="matchup-ai">
              <div class="ai-chip"><b>{leader}</b><span>{leadpct:.1f}% win probability</span></div>
              <div class="ai-chip"><b>{max(a_model,h_model):.1f}%</b><span>selected-market model</span></div>
            </div>
          </div>
          <div class="matchup-foot">
            <span>{ev.sport} • 🗓 {game_datetime_label(ev.commence_time)} • Bahamas time • Best available sportsbook price</span>
            <span>Matchup analysis →</span>
          </div>
        </div>
        """,unsafe_allow_html=True)

        if st.button(
            f"Open matchup: {ev.away_team} @ {ev.home_team}",
            key=f"{keyprefix}_{ev.event_id}",
            use_container_width=True
        ):
            open_game(ev.event_id)

def prop_list_cards(df,limit=18):
    if df is None or df.empty:
        st.info("No player props match the selected filters.")
        return
    q=df.sort_values(["model_prob","edge"],ascending=False).head(limit)
    for _,r in q.iterrows():
        hit = float(r.get("recent_hit_rate",np.nan))
        hit_txt = "—" if pd.isna(hit) else f"{hit:.0%}"
        avg = r.get("recent_average",np.nan)
        avg_txt="—" if pd.isna(avg) else f"{float(avg):.1f}"
        point = r.get("point",np.nan)
        point_txt="—" if pd.isna(point) else f"{float(point):g}"
        st.markdown(f"""<div class="player-prop-row">
          <div>
            <div class="player-main">{r.player_name}</div>
            <div class="player-sub">{r.away_team} @ {r.home_team} • {r.side} {point_txt} {r.market}</div>
          </div>
          <div class="kpi-mini">Best Price<strong>{fmt_odds(r.best_price)}</strong></div>
          <div class="kpi-mini">L5/L10 Hit<strong>{hit_txt}</strong></div>
          <div class="kpi-mini">Recent Avg<strong>{avg_txt}</strong></div>
          <div class="kpi-mini">AI Probability<strong>{float(r.model_prob):.0%}</strong></div>
        </div>""",unsafe_allow_html=True)

def prop_history_chart(sport,player,market,line,side,games=10):
    try:
        vals,_=cached_stat_series(sport,player,market,max(5,games))
    except Exception:
        vals=[]
    vals=list(vals)[-games:] if vals is not None else []
    if not vals:
        st.info("Recent game-by-game data is not available for this prop.")
        return
    d=pd.DataFrame({"Game":[f"G{i+1}" for i in range(len(vals))],"Value":[float(v) for v in vals]})
    if str(side).lower()=="under":
        d["Hit"]=d["Value"]<float(line)
    else:
        d["Hit"]=d["Value"]>float(line)
    base=alt.Chart(d).encode(x=alt.X("Game:N",sort=None,title=None))
    bars=base.mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
        y=alt.Y("Value:Q",title=None),
        color=alt.condition("datum.Hit",alt.value("#5ce0c3"),alt.value("#d65a3b")),
        tooltip=["Game","Value","Hit"]
    )
    rule=alt.Chart(pd.DataFrame({"line":[float(line)]})).mark_rule(strokeDash=[4,4],color="#d8e0e5").encode(y="line:Q")
    st.altair_chart((bars+rule).properties(height=310),use_container_width=True)

def open_game(event_id):
    st.session_state.selected_event_id=event_id
    st.session_state.pending_nav="🎯 Game Center"
    try:
        st.rerun(scope="app")
    except Exception:
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


@st.fragment
def league_market_fragment(sport, team_json):
    """Only this section reruns when the user switches Moneyline/Spread/Total."""
    m1,m2=st.columns([1.4,1])
    with m1:
        league_market=st.segmented_control(
            "Market",
            ["Moneyline","Spread","Total","All"],
            default="Moneyline",
            key=f"{sport}_manual_market"
        )
    with m2:
        total_side=st.segmented_control(
            "Totals",
            ["Both","Over","Under"],
            default="Both",
            key=f"{sport}_manual_total",
            disabled=league_market not in ("All","Total")
        )

    d=cached_market_view(team_json,sport,league_market).copy()

    if total_side!="Both" and len(d):
        is_total=d.market.astype(str).eq("Total")
        d=d[(~is_total)|d.selection.astype(str).str.lower().eq(total_side.lower())]

    game_market_board(
        d,
        f"{sport.lower()}board",
        18,
        league_market if league_market!="All" else None
    )


def friendly_pick_label(row):
    """Turn raw market data into a sportsbook-style pick description."""
    market=str(row.get("market","")).strip().lower()
    selection=str(row.get("selection","")).strip()
    away=str(row.get("away_team","")).strip()
    home=str(row.get("home_team","")).strip()

    point=row.get("line",np.nan)
    if pd.isna(point):
        point=row.get("point",np.nan)

    matchup=f"{away}/{home}" if away and home else (away or home or "Game")

    if market in ("moneyline","h2h"):
        team=selection or "Team"
        return f"{team} Moneyline"

    if market in ("spread","spreads"):
        team=selection or "Team"
        if pd.notna(point):
            try:
                return f"{team} {float(point):+g}"
            except Exception:
                pass
        return f"{team} Spread"

    if market in ("total","totals"):
        side="Over" if selection.lower().startswith("over") else (
            "Under" if selection.lower().startswith("under") else selection.title()
        )
        if pd.notna(point):
            try:
                return f"{matchup} {side} {float(point):g}"
            except Exception:
                pass
        return f"{matchup} {side}"

    # Player props or other markets: preserve useful selection and line.
    if pd.notna(point):
        try:
            return f"{selection} {float(point):g}".strip()
        except Exception:
            pass
    return selection or str(row.get("market","Pick")).title()

def friendly_market_name(value):
    m=str(value).strip().lower()
    return {
        "h2h":"Moneyline",
        "moneyline":"Moneyline",
        "spreads":"Spread",
        "spread":"Spread",
        "totals":"Total",
        "total":"Total",
    }.get(m,str(value).title())

# ---------- PAGE CONTENT ----------
if page=="🔥 Best Bets":
    top_header("Games","Live NBA & NFL board with AI probability, best price and fast matchup drilldowns.")
    st.markdown("### Game filters")
    filtered_board=betting_filter_bar(team,"bestbets_filter")
    a,b=st.columns([1.4,.7])
    with a:
        search_game=st.text_input("Search games",placeholder="Search team name...")
    with b:
        if st.button("↻ Refresh live board",type="primary",use_container_width=True,disabled=not bool(KEY)):
            cached_featured.clear()
            cached_scores.clear()
            cached_snapshot.clear()
            cached_team_model.clear()
            cached_market_view.clear()
            with st.spinner("Getting fresh sportsbook lines..."):
                st.session_state.raw,errs=refresh()
            for e in errs:
                st.warning(e)
            st.rerun()
    if search_game and len(filtered_board):
        s=search_game.strip().lower()
        filtered_board=filtered_board[
          filtered_board.away_team.astype(str).str.lower().str.contains(s,na=False) |
          filtered_board.home_team.astype(str).str.lower().str.contains(s,na=False)
        ]
    metric_cards()
    st.markdown('<div class="terminal-label">TODAY / LIVE BOARD</div>',unsafe_allow_html=True)
    if filtered_board is None or filtered_board.empty:
        st.markdown("""
        <div class="hero-match">
          <div class="hero-match-title">WolfSportsAI is ready</div>
          <div class="hero-match-sub">The interface loads immediately. Live odds are refreshed by Autopilot in the background.</div>
          <div class="hero-match-stats">
            <div class="hero-stat"><span>NBA</span><strong>Enabled</strong></div>
            <div class="hero-stat"><span>NFL</span><strong>Enabled</strong></div>
            <div class="hero-stat"><span>Live Engine</span><strong>Starting</strong></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.warning("⏳ WolfSportsAI is loading its AI models and preparing the live betting board. The first load can take a little longer while data is cached. Please give it a moment — once initialization finishes, Moneyline, Spread and Over/Under switching should be much faster.")
        st.caption("You can leave this page open while initialization completes. If the board still has not appeared after a short wait, use Refresh live board once.")
    else:
        game_market_board(filtered_board,"bestboard",16,st.session_state.get("bestbets_filter_market"))
    st.divider()
    with st.expander("Show ranked AI opportunities"):
        pick_cards(filtered_board,10)

elif page in ("🏀 NBA","🏈 NFL"):
    sport="NBA" if "NBA" in page else "NFL"
    top_header(
        f"{sport} Games",
        "Instant market switching — Moneyline, Spread and Total are precomputed from the same loaded odds snapshot."
    )
    league_market_fragment(sport,_team_json)


elif page=="📅 Upcoming Games":
    top_header(
        "Upcoming Games",
        "Upcoming NBA and NFL matchups are cached from the internet so the AI and Parlay Builder can prepare before game time."
    )

    c1,c2=st.columns([2,1])
    with c1:
        upcoming_sport=st.segmented_control(
            "League",["All","NBA","NFL"],default="All",key="upcoming_sport"
        )
    with c2:
        if st.button("↻ Refresh upcoming schedule",use_container_width=True,disabled=not bool(KEY)):
            rows=[];errs=[]
            with st.spinner("Checking upcoming NBA and NFL schedules..."):
                for _sport in ("NBA","NFL"):
                    try:
                        _up,_=upcoming_events(_sport,KEY)
                        if len(_up):
                            upsert_upcoming_games(_up)
                            rows.append(_up)
                    except Exception as e:
                        errs.append(f"{_sport}: {e}")
            cached_upcoming_schedule.clear()
            for e in errs:
                st.warning(e)

    upcoming=cached_upcoming_schedule().copy()

    # First visit after a new deployment: populate only this page, never block other pages.
    if upcoming.empty and KEY:
        with st.spinner("Loading the upcoming schedule for the first time..."):
            for _sport in ("NBA","NFL"):
                try:
                    _up,_=upcoming_events(_sport,KEY)
                    if len(_up):
                        upsert_upcoming_games(_up)
                except Exception:
                    pass
        cached_upcoming_schedule.clear()
        upcoming=cached_upcoming_schedule().copy()

    if upcoming_sport!="All" and len(upcoming):
        upcoming=upcoming[upcoming.sport.eq(upcoming_sport)]

    if upcoming.empty:
        st.info("No upcoming games are cached yet. Use Refresh upcoming schedule when the API is connected.")
    else:
        upcoming["Game Date"]=upcoming["commence_time"].apply(lambda x:game_datetime_parts(x)[0])
        upcoming["Game Time"]=upcoming["commence_time"].apply(lambda x:game_datetime_parts(x)[1])
        upcoming["Matchup"]=upcoming["away_team"].astype(str)+" @ "+upcoming["home_team"].astype(str)
        # Mark whether current sportsbook markets are already loaded for AI/parlays.
        market_ids=set(team.event_id.astype(str)) if len(team) and "event_id" in team.columns else set()
        upcoming["Markets Ready"]=upcoming["event_id"].astype(str).apply(
            lambda x:"✅ Odds + AI ready" if x in market_ids else "🕒 Waiting for sportsbook lines"
        )
        show=upcoming[["sport","Game Date","Game Time","Matchup","Markets Ready"]].rename(
            columns={"sport":"League"}
        )
        st.dataframe(show,use_container_width=True,hide_index=True,height=560)
        ready=(upcoming["Markets Ready"]=="✅ Odds + AI ready").sum()
        st.caption(
            f"{len(upcoming)} upcoming games shown • {ready} currently have sportsbook markets available for WolfSportsAI picks. "
            "Games without real odds are shown on the schedule but are not invented or forced into parlays."
        )

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
            packet=matchup_packet(sport,home,away)
            model=packet.get("model")
            if model:
                st.markdown(f"""<div class="hero-match">
                  <div class="hero-match-title">{away} @ {home}</div>
                  <div class="hero-match-sub">{sport} • {b.commence_time}</div>
                  <div class="hero-match-stats">
                    <div class="hero-stat"><span>{home} win probability</span><strong>{model['home_win_prob']:.0%}</strong></div>
                    <div class="hero-stat"><span>Projected margin</span><strong>{home} {model['pred_margin']:+.1f}</strong></div>
                    <div class="hero-stat"><span>Projected total</span><strong>{model['pred_total']:.1f}</strong></div>
                  </div>
                </div>""",unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="hero-match">
                  <div class="hero-match-title">{away} @ {home}</div>
                  <div class="hero-match-sub">{sport} • {b.commence_time}</div>
                </div>""",unsafe_allow_html=True)
                st.info("Historical matchup model is still building.")

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
                    q=cur.copy();q["AI Probability %"]=(q.model_prob*100).round(1);q["Market %"]=(q.market_prob*100).round(1);q["Edge %"]=(q.edge*100).round(1)
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
                    q=cur.copy();q["AI Probability %"]=(q.model_prob*100).round(1);q["Edge %"]=(q.edge*100).round(1)
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
                ev,_=cached_events(sp,KEY)
                for e in ev[:scan]:
                    try:
                        p,_=cached_event_props(sp,e["id"],KEY)
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
        st.subheader("Player opportunities")
        prop_list_cards(q,14)
        labels=[]
        for i,r in q.head(25).iterrows():
            labels.append(f"{r.player_name} • {r.side} {float(r.point):g} {r.market}")
        choice=st.selectbox("Open detailed prop view",labels,key="player_detail_pick")
        idx=labels.index(choice)
        r=q.head(25).iloc[idx]
        st.markdown(f"""<div class="hero-match">
          <div class="hero-match-title">{r.player_name}</div>
          <div class="hero-match-sub">{r.away_team} @ {r.home_team} • {r.side} {float(r.point):g} {r.market}</div>
          <div class="hero-match-stats">
            <div class="hero-stat"><span>AI probability</span><strong>{float(r.model_prob):.0%}</strong></div>
            <div class="hero-stat"><span>Recent hit rate</span><strong>{float(r.recent_hit_rate):.0%}</strong></div>
            <div class="hero-stat"><span>Best price</span><strong>{fmt_odds(r.best_price)}</strong></div>
          </div>
        </div>""",unsafe_allow_html=True)
        win=st.segmented_control("History",["L5","L10","L20"],default="L10",key="player_chart_window")
        ng={"L5":5,"L10":10,"L20":20}[win]
        prop_history_chart(r.sport,r.player_name,r.market,r.point,r.side,ng)
        st.caption("Green = prop would have hit the selected line. Red = miss. Dashed line = current sportsbook line.")
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
                ev,_=cached_events(sp,KEY)
                for e in ev[:n]:
                    try:
                        p,_=cached_event_props(sp,e["id"],KEY)
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
        st.markdown('<div class="terminal-label">RANKED PROP BOARD</div>',unsafe_allow_html=True)
        prop_list_cards(z,24)
        with st.expander("Open full prop table"):
            st.dataframe(z[["player_name","market","side","point","best_price","best_book",
                            "Model %","Edge %","EV %","games_used","recent_average",
                            "confidence","away_team","home_team"]].sort_values(["AI Probability %","Edge %"],ascending=False),
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

    alllegs=future_only(team)
    if include_props and len(st.session_state.props):
        _future_props=future_only(st.session_state.props)
        alllegs=pd.concat([alllegs,_future_props],ignore_index=True,sort=False) if len(alllegs) else _future_props

    # Parlay quality gate: never use C or Pass selections.
    # Only A+, A and B graded opportunities are eligible for generated parlays.
    allowed_parlay_grades={"A+","A","B"}
    if len(alllegs) and "confidence" in alllegs.columns:
        alllegs=alllegs[
            alllegs["confidence"].astype(str).str.upper().isin(allowed_parlay_grades)
        ].copy()

    st.caption("🛡️ Parlay quality filter: only A+, A and B confidence picks are eligible. C and Pass picks are automatically excluded.")

    if build_it:
        st.session_state.parlay=build(alllegs,legs,minp,mine,mode,same)

    r=st.session_state.parlay
    if r:
        p=r["legs"]
        if not r["complete"]:
            st.warning(f"Only {len(p)} A+/A/B qualifying legs were available. WolfSportsAI will not add C-grade picks just to fill the ticket.")
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
            q=p.copy();q["AI Probability %"]=(q.model_prob*100).round(1);q["Edge %"]=(q.edge*100).round(1)
            cols=[c for c in ["sport","leg_type","player_name","market","selection","line",
                              "best_price","best_book","Model %","Edge %","confidence",
                              "away_team","home_team"] if c in q.columns]
            st.caption("Pick names are formatted like a bet slip: team + market/line, or matchup + Over/Under total.")
            _parlay_view=q[cols].copy()
            if len(_parlay_view):
                _parlay_view["Pick"]=_parlay_view.apply(friendly_pick_label,axis=1)
                if "market" in _parlay_view.columns:
                    _parlay_view["Bet Type"]=_parlay_view["market"].apply(friendly_market_name)
                _rename={
                    "best_price":"Odds",
                    "best_book":"Sportsbook",
                    "edge":"Edge %",
                    "confidence":"Confidence",
                    "away_team":"Away",
                    "home_team":"Home"
                }
                _parlay_view=_parlay_view.rename(columns=_rename)
                if "commence_time" in q.columns:
                    _parlay_view["Game Date"]=q["commence_time"].apply(lambda x:game_datetime_parts(x)[0]).values
                    _parlay_view["Game Time"]=q["commence_time"].apply(lambda x:game_datetime_parts(x)[1]).values
                _preferred=["sport","Pick","Bet Type","Game Date","Game Time","Odds","Sportsbook","Edge %","Confidence","Away","Home"]
                _preferred=[c for c in _preferred if c in _parlay_view.columns]
                _parlay_view=_parlay_view[_preferred]
            st.dataframe(_parlay_view,use_container_width=True,hide_index=True)
            st.download_button("Download parlay sheet",_parlay_view.to_csv(index=False).encode(),
                               f"wolfsports_{legs}_leg_parlay.csv","text/csv")
    else:
        st.info("Choose your settings and build a parlay sheet.")
elif page=="🧠 Model Lab":
    top_header("Model Lab","Autopilot continuously retrains and backtests while WolfSportsAI is running.")
    st.success("AI Autopilot is automatic. You do not need to press Train or Backtest. Manual controls below are only for forcing an immediate rebuild.")
    ap=get_autopilot_status()
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Autopilot","RUNNING" if (st.session_state.get("_autopilot_thread") and st.session_state["_autopilot_thread"].is_alive()) else "STARTING")
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

    st.info("Speed mode: model learning, historical updates and maintenance are scheduled for low-traffic windows. Outside those windows the dashboard primarily serves cached data and predictions.")
    st.write("Automatic schedule while WolfSportsAI is open:")
    st.write("• Odds/results refresh: every 15 minutes")
    st.write("• Afternoon maintenance: 4:00–4:30 PM (Bahamas time)")
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
