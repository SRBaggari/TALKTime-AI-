"""Model status, one-time downloads and the offline-readiness startup check."""
from __future__ import annotations

import importlib.util
import logging
import shutil
from dataclasses import dataclass

import config
from core.diarization_engine import DiarizationEngine, is_diarization_available
from core.transcription_engine import TranscriptionEngine, is_whisper_available, whisper_repo

logger = logging.getLogger(__name__)

REQUIRED_PACKAGES = {
    "streamlit": "streamlit", "plotly": "plotly", "numpy": "numpy", "soundfile": "soundfile",
    "sqlalchemy": "SQLAlchemy", "reportlab": "reportlab", "pandas": "pandas",
    "faster_whisper": "faster-whisper", "pyannote.audio": "pyannote.audio", "torch": "torch",
}
OPTIONAL_PACKAGES = {"spacy": "spaCy (topics)", "vaderSentiment": "VADER (sentiment)", "librosa": "librosa"}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fix: str = ""
    required: bool = True


def _has_package(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def package_checks() -> list[CheckResult]:
    out = []
    for mod, name in REQUIRED_PACKAGES.items():
        ok = _has_package(mod)
        out.append(CheckResult(f"Package: {name}", ok, "Installed" if ok else "Missing",
                               "Run setup_windows.bat (or pip install -r requirements.txt)"))
    for mod, name in OPTIONAL_PACKAGES.items():
        ok = _has_package(mod)
        out.append(CheckResult(f"Optional: {name}", ok, "Installed" if ok else "Missing — feature disabled",
                               f"pip install {mod}", required=False))
    return out


def model_status(whisper_size: str) -> dict:
    from core.nlp_utils import is_spacy_available

    return {
        "diarization": is_diarization_available(),
        "diarization_loaded": DiarizationEngine.is_loaded(),
        "whisper": {size: is_whisper_available(size) for size in config.WHISPER_SIZES},
        "whisper_selected": is_whisper_available(whisper_size),
        "whisper_loaded": TranscriptionEngine.is_loaded(),
        "spacy": is_spacy_available(),
        "hf_token": config.get_hf_token() is not None,
    }


def startup_checks(settings: config.AppSettings) -> list[CheckResult]:
    from database.db import check_database
    from services.storage_service import check_writable

    checks = package_checks()
    db_ok, db_msg = check_database()
    checks.append(CheckResult("SQLite database", db_ok, db_msg, "Check permissions on data/database"))
    st_ok, st_msg = check_writable()
    checks.append(CheckResult("Storage permissions", st_ok, st_msg, "Make the data/ folder writable"))
    ff = shutil.which("ffmpeg")
    checks.append(CheckResult("FFmpeg", bool(ff), "Found on PATH" if ff else "Not found — only WAV/FLAC/OGG can be decoded",
                              "winget install Gyan.FFmpeg  (then restart the terminal)"))
    w_ok = is_whisper_available(settings.whisper_model)
    checks.append(CheckResult(f"Transcription model (Whisper {settings.whisper_model})", w_ok,
                              "Cached locally" if w_ok else "Not downloaded",
                              f"Settings → Models → Download, or scripts\\download_models.py --whisper {settings.whisper_model}"))
    d_ok = is_diarization_available()
    checks.append(CheckResult("Diarization model (pyannote Community-1)", d_ok,
                              "Cached locally" if d_ok else "Not downloaded (Hugging Face token needed once)",
                              "See README → 'Hugging Face model access', then scripts\\download_models.py --diarization"))
    from core.nlp_utils import is_spacy_available

    sp = is_spacy_available()
    checks.append(CheckResult("spaCy English model", sp, "Installed" if sp else "Missing — topic extraction uses keyword fallback",
                              "python -m spacy download en_core_web_sm", required=False))
    return checks


def offline_ready(checks: list[CheckResult]) -> bool:
    return all(c.ok for c in checks if c.required)


# --------------------------------------------------------------------------- downloads (one-time, needs internet)

def download_whisper(size: str) -> str:
    """Download only the files faster-whisper needs (same as its own downloader)."""
    from faster_whisper.utils import download_model

    if size not in config.WHISPER_SIZES:
        raise ValueError(size)
    return download_model(size)


def download_diarization() -> str:
    from huggingface_hub import snapshot_download

    token = config.get_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Add it to the .env file (see README).")
    return snapshot_download(config.DIARIZATION_MODEL_ID, token=token)


def release_models() -> None:
    TranscriptionEngine.unload()
    DiarizationEngine.unload()
