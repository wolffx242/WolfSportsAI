
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import threading
import time
import traceback

import pandas as pd

from database import (
    append_df, upsert_results, read_sql, get_autopilot_status,
    update_autopilot_status, insert_backtest_run
)
from odds_api import featured, scores
from historical_bootstrap import (
    bootstrap_sport, train_bootstrap_models, load_bootstrap
)

_LOCK=threading.Lock()
_THREAD=None
_STOP=threading.Event()
_BOOTSTRAP_ATTEMPT={}

def utcnow():
    return datetime.now(timezone.utc)

def iso(dt=None):
    return (dt or utcnow()).isoformat()

def parse_dt(x):
    if not x:
        return None
    try:
        return datetime.fromisoformat(str(x).replace("Z","+00:00"))
    except Exception:
        return None

def due(last, minutes):
    d=parse_dt(last)
    return d is None or (utcnow()-d)>=timedelta(minutes=minutes)

def _completed_count(sport):
    q=read_sql("SELECT COUNT(*) n FROM game_results WHERE sport=? AND completed=1",(sport,))
    return int(q.iloc[0].n) if len(q) else 0

def _historical_count(sport):
    q=read_sql("SELECT COUNT(*) n FROM historical_features WHERE sport=?",(sport,))
    return int(q.iloc[0].n) if len(q) else 0

def _record_backtest(sport, payload):
    row={
      "tested_at":iso(),
      "sport":sport,
      "model_name":"Historical form model",
      "rows_tested":payload.get("test_rows"),
      "accuracy":payload.get("accuracy"),
      "brier":payload.get("brier"),
      "margin_mae":payload.get("margin_mae"),
      "total_mae":payload.get("total_mae"),
      "notes":"Automatic rolling holdout backtest"
    }
    insert_backtest_run(row)

def run_cycle(api_key, sports=("NBA","NFL"), refresh_minutes=15,
              retrain_hours=12, backtest_hours=6,
              bootstrap_if_missing=True):
    if not _LOCK.acquire(blocking=False):
        return {"status":"busy"}
    try:
        st=get_autopilot_status()
        cycle_count=int(st.get("cycle_count") or 0)+1
        update_autopilot_status(last_cycle=iso(),cycle_count=cycle_count,last_error=None)

        # 1) Refresh live odds + recent scores automatically
        if api_key and due(st.get("last_data_refresh"),refresh_minutes):
            errs=[]
            for sport in sports:
                try:
                    m,_=featured(sport,api_key)
                    if len(m): append_df("market_snapshots",m)
                    sc,_=scores(sport,api_key)
                    if len(sc): upsert_results(sc)
                except Exception as e:
                    errs.append(f"{sport}: {e}")
            update_autopilot_status(last_data_refresh=iso())
            if errs:
                update_autopilot_status(last_error=" | ".join(errs))

        # 2) Bootstrap historical data automatically if absent
        for sport in sports:
            hist_count=_historical_count(sport)
            # Public history endpoints can be slow/rate-limited. Retry at most every 6 hours
            # instead of hammering them every 5-minute worker wake-up.
            _last_boot=_BOOTSTRAP_ATTEMPT.get(sport)
            _boot_due=_last_boot is None or (utcnow()-_last_boot)>=timedelta(hours=6)
            if bootstrap_if_missing and hist_count < (150 if sport=="NBA" else 100) and _boot_due:
                _BOOTSTRAP_ATTEMPT[sport]=utcnow()
                try:
                    seasons=4 if sport=="NBA" else 6
                    bootstrap_sport(sport,seasons)
                except Exception as e:
                    update_autopilot_status(last_error=f"{sport} bootstrap: {e}")

        # 3) Retrain automatically on a cadence
        for sport in sports:
            key=f"last_retrain_{sport.lower()}"
            last=st.get(key)
            if due(last,retrain_hours*60):
                try:
                    model,info=train_bootstrap_models(sport,150 if sport=="NBA" else 100)
                    if model is not None:
                        update_autopilot_status(**{key:iso()})
                        # train_bootstrap_models already evaluates chronologically;
                        # store that holdout as an automatic backtest too.
                        _record_backtest(sport,model)
                        update_autopilot_status(**{f"last_backtest_{sport.lower()}":iso()})
                except Exception as e:
                    update_autopilot_status(last_error=f"{sport} retrain: {e}")

        # 4) Backtest current model again on cadence by retraining/holdout evaluation.
        for sport in sports:
            key=f"last_backtest_{sport.lower()}"
            last=get_autopilot_status().get(key)
            if due(last,backtest_hours*60):
                try:
                    model,info=train_bootstrap_models(sport,150 if sport=="NBA" else 100)
                    if model is not None:
                        _record_backtest(sport,model)
                        update_autopilot_status(**{key:iso()})
                except Exception as e:
                    update_autopilot_status(last_error=f"{sport} backtest: {e}")

        return {"status":"ok","cycle_count":cycle_count}
    except Exception:
        update_autopilot_status(last_error=traceback.format_exc()[-1200:])
        return {"status":"error"}
    finally:
        _LOCK.release()

def _loop(api_key,sports,refresh_minutes,retrain_hours,backtest_hours):
    # Run once shortly after startup, then periodically.
    while not _STOP.is_set():
        try:
            run_cycle(
                api_key=api_key,
                sports=sports,
                refresh_minutes=refresh_minutes,
                retrain_hours=retrain_hours,
                backtest_hours=backtest_hours,
                bootstrap_if_missing=True
            )
        except Exception:
            pass
        # Wake every 5 minutes; due() decides what work actually needs doing.
        _STOP.wait(300)

def start_worker(api_key,sports=("NBA","NFL"),refresh_minutes=15,retrain_hours=12,backtest_hours=12):
    global _THREAD
    if _THREAD and _THREAD.is_alive():
        return _THREAD
    _STOP.clear()
    _THREAD=threading.Thread(
        target=_loop,
        args=(api_key,tuple(sports),int(refresh_minutes),int(retrain_hours),int(backtest_hours)),
        daemon=True,
        name="WolfSportsAI-Autopilot"
    )
    _THREAD.start()
    return _THREAD

def stop_worker():
    _STOP.set()

def latest_backtests():
    return read_sql("""SELECT * FROM backtest_runs
                       ORDER BY tested_at DESC LIMIT 20""")
