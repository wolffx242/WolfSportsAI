
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

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
_LAST_COMPLETED_COUNTS={}

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

def _fetch_one_sport(sport, api_key):
    """Fetch odds + scores for one sport. Designed to run in a worker thread."""
    result={"sport":sport,"market":None,"scores":None,"errors":[]}
    try:
        m,_=featured(sport,api_key)
        result["market"]=m
    except Exception as e:
        result["errors"].append(f"odds: {e}")
    try:
        sc,_=scores(sport,api_key)
        result["scores"]=sc
    except Exception as e:
        result["errors"].append(f"scores: {e}")
    return result

def _completed_count_safe(sport):
    try:
        return _completed_count(sport)
    except Exception:
        return 0

def run_cycle(api_key, sports=("NBA","NFL"), refresh_minutes=5,
              retrain_hours=4, backtest_hours=12,
              bootstrap_if_missing=True):
    if not _LOCK.acquire(blocking=False):
        return {"status":"busy"}

    try:
        st=get_autopilot_status()
        cycle_count=int(st.get("cycle_count") or 0)+1
        update_autopilot_status(last_cycle=iso(),cycle_count=cycle_count,last_error=None)

        changed_sports=set()
        errors=[]

        # 1) Fast lane: NBA and NFL refresh in parallel.
        if api_key and due(st.get("last_data_refresh"),refresh_minutes):
            before={s:_completed_count_safe(s) for s in sports}

            with ThreadPoolExecutor(max_workers=min(4,max(1,len(sports)*2))) as ex:
                futs={ex.submit(_fetch_one_sport,s,api_key):s for s in sports}
                for fut in as_completed(futs):
                    sport=futs[fut]
                    try:
                        r=fut.result()
                        m=r.get("market")
                        sc=r.get("scores")
                        if m is not None and len(m):
                            append_df("market_snapshots",m)
                        if sc is not None and len(sc):
                            upsert_results(sc)
                        if r.get("errors"):
                            errors.extend([f"{sport} {x}" for x in r["errors"]])
                    except Exception as e:
                        errors.append(f"{sport}: {e}")

            after={s:_completed_count_safe(s) for s in sports}
            for s in sports:
                prev=_LAST_COMPLETED_COUNTS.get(s,before.get(s,0))
                now=after.get(s,0)
                if now > prev or now > before.get(s,0):
                    changed_sports.add(s)
                _LAST_COMPLETED_COUNTS[s]=now

            update_autopilot_status(last_data_refresh=iso())

        # 2) Historical bootstrap is one-time / low-frequency only.
        for sport in sports:
            hist_count=_historical_count(sport)
            min_rows=150 if sport=="NBA" else 100
            _last_boot=_BOOTSTRAP_ATTEMPT.get(sport)
            _boot_due=_last_boot is None or (utcnow()-_last_boot)>=timedelta(hours=24)

            if bootstrap_if_missing and hist_count < min_rows and _boot_due:
                _BOOTSTRAP_ATTEMPT[sport]=utcnow()
                try:
                    seasons=4 if sport=="NBA" else 6
                    bootstrap_sport(sport,seasons)
                    # New historical rows justify training.
                    changed_sports.add(sport)
                except Exception as e:
                    errors.append(f"{sport} bootstrap: {e}")

        # 3) Retrain only when new results arrived, model is missing,
        #    or the safety cadence has elapsed.
        for sport in sports:
            key=f"last_retrain_{sport.lower()}"
            payload=load_bootstrap(sport)
            model_missing=payload is None
            cadence_due=due(st.get(key),retrain_hours*60)

            if sport in changed_sports or model_missing or cadence_due:
                try:
                    model,info=train_bootstrap_models(
                        sport,150 if sport=="NBA" else 100
                    )
                    if model is not None:
                        update_autopilot_status(**{key:iso()})

                        # Retraining already includes chronological holdout evaluation,
                        # so save that evaluation as the backtest instead of training again.
                        _record_backtest(sport,model)
                        update_autopilot_status(
                            **{f"last_backtest_{sport.lower()}":iso()}
                        )
                except Exception as e:
                    errors.append(f"{sport} retrain: {e}")

        # 4) Full backtest safety check only if no recent training evaluation exists.
        for sport in sports:
            key=f"last_backtest_{sport.lower()}"
            latest=get_autopilot_status().get(key)
            if due(latest,backtest_hours*60):
                try:
                    model,info=train_bootstrap_models(
                        sport,150 if sport=="NBA" else 100
                    )
                    if model is not None:
                        _record_backtest(sport,model)
                        update_autopilot_status(**{key:iso()})
                except Exception as e:
                    errors.append(f"{sport} backtest: {e}")

        if errors:
            update_autopilot_status(last_error=" | ".join(errors)[:1200])

        return {
            "status":"ok",
            "cycle_count":cycle_count,
            "changed_sports":sorted(changed_sports),
            "errors":errors
        }

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
        _STOP.wait(60)

def start_worker(api_key,sports=("NBA","NFL"),refresh_minutes=5,retrain_hours=4,backtest_hours=12):
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
