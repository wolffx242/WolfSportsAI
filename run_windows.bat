@echo off
cd /d "%~dp0"
title WolfSportsAI V3
echo ============================================
echo           WolfSportsAI V3
echo ============================================
if not exist .venv (
    echo Creating Python environment...
    python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Starting dashboard...
python -m streamlit run app.py
pause
