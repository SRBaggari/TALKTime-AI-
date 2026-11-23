@echo off
REM Start TalkTime AI locally (works offline once models are downloaded).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Virtual environment not found. Run setup_windows.bat first.
    exit /b 1
)
set HF_HUB_DISABLE_TELEMETRY=1
set PYANNOTE_METRICS_ENABLED=0
.venv\Scripts\python -m streamlit run app.py --server.address localhost --browser.gatherUsageStats false
