
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from database import (
    append_df, upsert_results, upsert_upcoming_games, read_sql, get_autopilot_status,
    update_autopilot_status, insert_backtest_run
)
from odds_api import featured, scores, upcoming_events
from historical_bootstrap import (
    bootstrap_sport, train_bootstrap_models, load_bootstrap
)
from team_models import train as train_calibration

_LOCK=threading.Lock()
_THREAD=None
_STOP=threading.Event()
_BOOTSTRAP_ATTEMPT={}
_LAST_COMPLETED_COUNTS={}
APP_TZ=ZoneInfo("America/Nassau")
_LAST_WINDOW_RUN={}
MODEL_SPORTS={"NBA","NFL"}

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
    result={"sport":sport,"market":None,"scores":None,"upcoming":None,"errors":[]}
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
    try:
        up,_=upcoming_events(sport,api_key)
        result["upcoming"]=up
    except Exception as e:
        result["errors"].append(f"schedule: {e}")
    return result

def _completed_count_safe(sport):
    try:
        return _completed_count(sport)
    except Exception:
        return 0

def run_cycle(api_key, sports=("NBA","NFL","MLB","NHL"), refresh_minutes=5,
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

        # 1) Fast lane: NBA, NFL, MLB and NHL refresh in parallel.
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
                        up=r.get("upcoming")
                        if m is not None and len(m):
                            append_df("market_snapshots",m)
                        if sc is not None and len(sc):
                            upsert_results(sc)
                        if up is not None and len(up):
                            upsert_upcoming_games(up)
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

        # 2) Historical bootstrap is currently available for NBA/NFL.
        # MLB/NHL still collect odds/results and can use local odds-calibration models
        # after enough completed games have accumulated.
        for sport in [s for s in sports if s in MODEL_SPORTS]:
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

        # 3) Retrain historical-form models for supported leagues only.
        for sport in [s for s in sports if s in MODEL_SPORTS]:
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

        # 4) Full backtest safety check for supported historical-form leagues.
        for sport in [s for s in sports if s in MODEL_SPORTS]:
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

        # 5) MLB/NHL learn from the odds/results WolfSportsAI collects.
        # Keep this lightweight: train only when new completed games arrive or overnight.
        for sport in [s for s in sports if s in ("MLB","NHL")]:
            if sport in changed_sports or retrain_hours == 0:
                for market in ("Moneyline","Spread","Total"):
                    try:
                        train_calibration(sport,market,60)
                    except Exception as e:
                        errors.append(f"{sport} {market} calibration: {e}")

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


def local_now():
    return datetime.now(APP_TZ)

def maintenance_window(now=None):
    """Return active maintenance window in America/Nassau, or None."""
    now = now or local_now()
    minutes = now.hour * 60 + now.minute

    # Morning quick maintenance: 07:00–07:30.
    if 7*60 <= minutes < 7*60+30:
        return "morning"

    # Afternoon quick maintenance: 16:00–16:30.
    if 16*60 <= minutes < 16*60+30:
        return "afternoon"

    # Overnight heavy maintenance: 01:00–04:00.
    if 1*60 <= minutes < 4*60:
        return "overnight"

    return None

def _window_key(window, now=None):
    now = now or local_now()
    return f"{now.date().isoformat()}:{window}"


def refresh_personnel_context(sports=("NBA","NFL","MLB","NHL")):
    """
    Scheduled personnel refresh hook.

    WolfSportsAI's current core providers reliably supply games, results,
    historical stats and odds. Injury/starting-lineup coverage varies by
    league/source. This hook records the maintenance attempt and is the
    single place to attach a dedicated injuries/lineups provider later
    without touching the user-facing UI.
    """
    # Keep this non-blocking and provider-safe for now.
    return {
        "sports": list(sports),
        "checked_at": iso(),
        "injuries": "provider-dependent",
        "rosters": "provider-dependent",
        "lineups": "provider-dependent",
    }

def scheduled_cycle(api_key, sports=("NBA","NFL","MLB","NHL")):
    """Run only the work assigned to the active maintenance window."""
    now = local_now()
    window = maintenance_window(now)
    if not window:
        return {"status":"idle","window":None}

    key = _window_key(window, now)
    if _LAST_WINDOW_RUN.get(window) == key:
        return {"status":"already_ran","window":window}

    # Mark first so Streamlit reruns cannot start duplicate maintenance.
    _LAST_WINDOW_RUN[window] = key

    if window in ("morning","afternoon"):
        personnel = refresh_personnel_context(sports)
        # Quick windows: fresh lines/results and incremental model update.
        # No historical bootstrap or duplicate full backtest.
        result = run_cycle(
            api_key=api_key,
            sports=sports,
            refresh_minutes=0,
            retrain_hours=9999,
            backtest_hours=9999,
            bootstrap_if_missing=False
        )
        result["window"]=window
        result["personnel"]=personnel
        return result

    personnel = refresh_personnel_context(sports)
    # Overnight: full maintenance. Historical bootstrap if missing,
    # retraining, validation/backtesting and all database/model housekeeping.
    result = run_cycle(
        api_key=api_key,
        sports=sports,
        refresh_minutes=0,
        retrain_hours=0,
        backtest_hours=0,
        bootstrap_if_missing=True
    )
    result["window"]="overnight"
    result["personnel"]=personnel
    return result

def _loop(api_key,sports):
    # Check schedule once per minute. Heavy work only starts inside maintenance windows.
    while not _STOP.is_set():
        try:
            scheduled_cycle(api_key, sports)
        except Exception:
            pass
        _STOP.wait(60)

def start_worker(api_key,sports=("NBA","NFL","MLB","NHL"),refresh_minutes=5,retrain_hours=4,backtest_hours=12):
    """Start lightweight scheduler. Parameters retained for backward compatibility."""
    global _THREAD
    if _THREAD and _THREAD.is_alive():
        return _THREAD
    _STOP.clear()
    _THREAD=threading.Thread(
        target=_loop,
        args=(api_key,tuple(sports)),
        daemon=True,
        name="WolfSportsAI-Scheduled-Maintenance"
    )
    _THREAD.start()
    return _THREAD

def stop_worker():
    _STOP.set()

def latest_backtests():
    return read_sql("""SELECT * FROM backtest_runs
                       ORDER BY tested_at DESC LIMIT 20""")
