@echo off
REM TalkTime AI - one-time Windows setup (needs internet once).
setlocal
cd /d "%~dp0"

echo == TalkTime AI setup ==
where uv >nul 2>nul
if %errorlevel%==0 (
    echo Creating Python 3.12 virtual environment with uv...
    uv venv --python 3.12 .venv || goto :error
    set "PIP=uv pip install --python .venv"
) else (
    echo Creating virtual environment with py -3.12 ...
    py -3.12 -m venv .venv || py -3.11 -m venv .venv || goto :error
    .venv\Scripts\python -m pip install --upgrade pip
    set "PIP=.venv\Scripts\python -m pip install"
)

echo Installing CPU-only PyTorch...
%PIP% torch torchaudio --index-url https://download.pytorch.org/whl/cpu || goto :error
echo Installing requirements...
%PIP% -r requirements.txt || goto :error
echo Installing spaCy English model...
.venv\Scripts\python -m spacy download en_core_web_sm || goto :error

if not exist .env copy .env.example .env >nul

where ffmpeg >nul 2>nul
if not %errorlevel%==0 (
    echo.
    echo WARNING: FFmpeg not found. Install it for MP3/M4A support:  winget install Gyan.FFmpeg
)

echo Downloading the default transcription model (Whisper small, ~480 MB)...
.venv\Scripts\python scripts\download_models.py --whisper small

echo.
echo Setup complete.
echo For speaker diarization add your Hugging Face token to .env and run:
echo     .venv\Scripts\python scripts\download_models.py --diarization
echo Start the app with run_windows.bat
exit /b 0

:error
echo Setup failed. See the messages above.
exit /b 1
