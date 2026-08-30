
from pathlib import Path
import sqlite3
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "wolfsports.db"

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.execute("PRAGMA journal_mode=WAL;")
    return con

def init_db():
    with connect() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS market_snapshots(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          captured_at TEXT,sport TEXT,event_id TEXT,commence_time TEXT,
          home_team TEXT,away_team TEXT,bookmaker TEXT,market TEXT,
          outcome TEXT,point REAL,price REAL
        );
        CREATE INDEX IF NOT EXISTS idx_market_event ON market_snapshots(event_id);
        CREATE INDEX IF NOT EXISTS idx_market_time ON market_snapshots(captured_at);

        CREATE TABLE IF NOT EXISTS game_results(
          event_id TEXT PRIMARY KEY,sport TEXT,commence_time TEXT,
          home_team TEXT,away_team TEXT,home_score REAL,away_score REAL,
          completed INTEGER,updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS upcoming_games(
          event_id TEXT PRIMARY KEY,
          sport TEXT NOT NULL,
          commence_time TEXT NOT NULL,
          home_team TEXT,
          away_team TEXT,
          source TEXT,
          updated_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_upcoming_sport_time
          ON upcoming_games(sport,commence_time);

        CREATE TABLE IF NOT EXISTS player_prop_snapshots(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          captured_at TEXT,sport TEXT,event_id TEXT,commence_time TEXT,
          home_team TEXT,away_team TEXT,bookmaker TEXT,market TEXT,
          player_name TEXT,side TEXT,point REAL,price REAL
        );
        CREATE INDEX IF NOT EXISTS idx_prop_player ON player_prop_snapshots(player_name);

        CREATE TABLE IF NOT EXISTS model_runs(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          trained_at TEXT,sport TEXT,market TEXT,rows_used INTEGER,
          metric_name TEXT,metric_value REAL,notes TEXT
        );

        CREATE TABLE IF NOT EXISTS historical_games(
          game_key TEXT PRIMARY KEY,
          sport TEXT NOT NULL,
          season TEXT,
          game_date TEXT,
          home_team TEXT,
          away_team TEXT,
          home_score REAL,
          away_score REAL,
          source TEXT,
          spread_line REAL,
          total_line REAL
        );
        CREATE INDEX IF NOT EXISTS idx_hist_sport_date ON historical_games(sport,game_date);
        CREATE INDEX IF NOT EXISTS idx_hist_home ON historical_games(home_team);
        CREATE INDEX IF NOT EXISTS idx_hist_away ON historical_games(away_team);


        CREATE TABLE IF NOT EXISTS autopilot_status(
          id INTEGER PRIMARY KEY CHECK(id=1),
          last_cycle TEXT,
          last_data_refresh TEXT,
          last_retrain_nba TEXT,
          last_retrain_nfl TEXT,
          last_backtest_nba TEXT,
          last_backtest_nfl TEXT,
          last_error TEXT,
          cycle_count INTEGER DEFAULT 0
        );
        INSERT OR IGNORE INTO autopilot_status(id,cycle_count) VALUES(1,0);

        CREATE TABLE IF NOT EXISTS backtest_runs(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          tested_at TEXT NOT NULL,
          sport TEXT NOT NULL,
          model_name TEXT NOT NULL,
          rows_tested INTEGER,
          accuracy REAL,
          brier REAL,
          margin_mae REAL,
          total_mae REAL,
          notes TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_backtest_sport_time ON backtest_runs(sport,tested_at);

        CREATE TABLE IF NOT EXISTS historical_features(
          game_key TEXT PRIMARY KEY,
          sport TEXT NOT NULL,
          season TEXT,
          game_date TEXT,
          home_team TEXT,
          away_team TEXT,
          home_score REAL,
          away_score REAL,
          home_win INTEGER,
          margin REAL,
          total_points REAL,
          home_l5_win REAL,
          away_l5_win REAL,
          home_l10_win REAL,
          away_l10_win REAL,
          home_l5_pf REAL,
          away_l5_pf REAL,
          home_l10_pf REAL,
          away_l10_pf REAL,
          home_l5_pa REAL,
          away_l5_pa REAL,
          home_l10_pa REAL,
          away_l10_pa REAL,
          home_l5_margin REAL,
          away_l5_margin REAL,
          home_l10_margin REAL,
          away_l10_margin REAL,
          spread_line REAL,
          total_line REAL
        );
        CREATE INDEX IF NOT EXISTS idx_hf_sport_date ON historical_features(sport,game_date);
        """)

def append_df(table, df):
    if df is None or df.empty:
        return
    with connect() as con:
        df.to_sql(table, con, if_exists="append", index=False)


def upsert_upcoming_games(df):
    if df is None or df.empty:
        return
    cols=["event_id","sport","commence_time","home_team","away_team","source","updated_at"]
    x=df.copy()
    for c in cols:
        if c not in x.columns:
            x[c]=None
    with connect() as con:
        for row in x[cols].itertuples(index=False,name=None):
            con.execute("""
            INSERT INTO upcoming_games
            (event_id,sport,commence_time,home_team,away_team,source,updated_at)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(event_id) DO UPDATE SET
              sport=excluded.sport,
              commence_time=excluded.commence_time,
              home_team=excluded.home_team,
              away_team=excluded.away_team,
              source=excluded.source,
              updated_at=excluded.updated_at
            """,row)
        # Remove games that are safely in the past; completed results remain elsewhere.
        con.execute("DELETE FROM upcoming_games WHERE commence_time < datetime('now','-12 hours')")


def upsert_results(df):
    if df is None or df.empty:
        return
    cols=["event_id","sport","commence_time","home_team","away_team",
          "home_score","away_score","completed","updated_at"]
    with connect() as con:
        for row in df[cols].itertuples(index=False, name=None):
            con.execute("""
            INSERT INTO game_results VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(event_id) DO UPDATE SET
              sport=excluded.sport,commence_time=excluded.commence_time,
              home_team=excluded.home_team,away_team=excluded.away_team,
              home_score=excluded.home_score,away_score=excluded.away_score,
              completed=excluded.completed,updated_at=excluded.updated_at
            """, row)

def read_sql(q, params=()):
    with connect() as con:
        return pd.read_sql_query(q, con, params=params)

def count(table):
    try:
        with connect() as con:
            return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except Exception:
        return 0


def upsert_historical_games(df):
    if df is None or df.empty:
        return
    cols=["game_key","sport","season","game_date","home_team","away_team",
          "home_score","away_score","source","spread_line","total_line"]
    x=df.copy()
    for c in cols:
        if c not in x.columns:
            x[c]=None
    with connect() as con:
        for row in x[cols].itertuples(index=False,name=None):
            con.execute("""
            INSERT INTO historical_games
            (game_key,sport,season,game_date,home_team,away_team,home_score,away_score,source,spread_line,total_line)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(game_key) DO UPDATE SET
              sport=excluded.sport,season=excluded.season,game_date=excluded.game_date,
              home_team=excluded.home_team,away_team=excluded.away_team,
              home_score=excluded.home_score,away_score=excluded.away_score,
              source=excluded.source,spread_line=COALESCE(excluded.spread_line,historical_games.spread_line),
              total_line=COALESCE(excluded.total_line,historical_games.total_line)
            """,row)

def replace_historical_features(df, sport):
    if df is None:
        return
    with connect() as con:
        con.execute("DELETE FROM historical_features WHERE sport=?",(sport,))
        if not df.empty:
            df.to_sql("historical_features",con,if_exists="append",index=False)


def get_autopilot_status():
    q=read_sql("SELECT * FROM autopilot_status WHERE id=1")
    return q.iloc[0].to_dict() if len(q) else {}

def update_autopilot_status(**kwargs):
    if not kwargs:
        return
    allowed={
      "last_cycle","last_data_refresh","last_retrain_nba","last_retrain_nfl",
      "last_backtest_nba","last_backtest_nfl","last_error","cycle_count"
    }
    clean={k:v for k,v in kwargs.items() if k in allowed}
    if not clean:
        return
    sets=", ".join([f"{k}=?" for k in clean])
    vals=list(clean.values())+[1]
    with connect() as con:
        con.execute(f"UPDATE autopilot_status SET {sets} WHERE id=?",vals)

def insert_backtest_run(row):
    cols=["tested_at","sport","model_name","rows_tested","accuracy","brier","margin_mae","total_mae","notes"]
    vals=[row.get(c) for c in cols]
    with connect() as con:
        con.execute("""INSERT INTO backtest_runs
          (tested_at,sport,model_name,rows_tested,accuracy,brier,margin_mae,total_mae,notes)
          VALUES(?,?,?,?,?,?,?,?,?)""",vals)
