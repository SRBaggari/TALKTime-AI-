"""Central configuration for TalkTime AI.

All paths are resolved relative to the project root so the app works from any
working directory. Secrets (Hugging Face token) are read from the environment
or a local, git-ignored ``.env`` file and are never written to logs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at import time (tests may run without it)
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent

if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env", override=False)

# ---------------------------------------------------------------------------
# Privacy: disable every telemetry channel we know about *before* any ML
# library is imported anywhere in the process.
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

DATA_DIR = Path(os.environ.get("TALKTIME_DATA_DIR", BASE_DIR / "data")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
DATABASE_DIR = DATA_DIR / "database"
LOG_DIR = DATA_DIR / "logs"
TEMP_DIR = DATA_DIR / "tmp"
ASSETS_DIR = BASE_DIR / "assets"

DATABASE_URL = os.environ.get("TALKTIME_DATABASE_URL", f"sqlite:///{(DATABASE_DIR / 'talktime.db').as_posix()}")

APP_NAME = "TalkTime AI"
APP_TAGLINE = "Speaker diarization & conversation analytics — fully local"
APP_VERSION = "1.0.0"

# Audio
TARGET_SAMPLE_RATE = 16_000
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".aac", ".wma", ".mp4"}
DEFAULT_MAX_UPLOAD_MB = 500
MIN_AUDIO_SECONDS = 0.5
MIN_FREE_DISK_MB = 500

# Models
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"
WHISPER_REPO_TEMPLATE = "Systran/faster-whisper-{size}"
WHISPER_SIZES = ["tiny", "base", "small", "medium"]
SPACY_MODEL = "en_core_web_sm"

# Speaker colours — validated categorical order for the dark surface (#111827):
# lightness band, chroma floor, colour-vision-deficiency separation and contrast
# all pass. Blue and violet cannot be adjacent (ΔE 1.9 for protanopia), so blue
# is slot 4. Speakers are always also labelled by name (never colour alone).
SPEAKER_COLORS = [
    "#199E70",  # green
    "#9085E9",  # violet
    "#D95926",  # orange
    "#3987E5",  # blue
    "#C98500",  # amber
    "#D55181",  # magenta
    "#0CA30C",  # leaf green
    "#E66767",  # red
]


def speaker_color(index: int) -> str:
    return SPEAKER_COLORS[index % len(SPEAKER_COLORS)]


def get_hf_token() -> str | None:
    """Return the Hugging Face token from env / .env, or None. Never logged."""
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value and not value.startswith("hf_your"):
            return value
    return None


def mask_token(token: str | None) -> str:
    """Show only enough of a secret to recognise it: 'hf_…EqsV'."""
    if not token:
        return "not set"
    return f"{token[:3]}…{token[-4:]}" if len(token) > 10 else "set"


def remove_hf_token_from_env_file(env_path: Path | None = None) -> bool:
    """Remove HF_TOKEN lines from the local .env (the token is only needed for the one-time download)."""
    env_path = env_path or (BASE_DIR / ".env")
    if not env_path.is_file():
        return False
    lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
    keys = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    kept = [ln for ln in lines if not ln.strip().split("=", 1)[0].strip() in keys]
    if len(kept) == len(lines):
        return False
    env_path.write_text("".join(kept), encoding="utf-8")
    for k in keys:
        os.environ.pop(k, None)
    return True


@dataclass
class AppSettings:
    """User-adjustable settings (persisted in SQLite ``app_settings``)."""

    whisper_model: str = "small"
    compute_type: str = "int8"
    cpu_threads: int = max(1, min(8, (os.cpu_count() or 4) // 2))
    beam_size: int = 5
    vad_filter: bool = True
    default_language: str = "auto"
    word_timestamps: bool = True
    min_speakers: int = 0  # 0 = automatic
    max_speakers: int = 0  # 0 = automatic
    keep_models_loaded: bool = True
    # Analytics thresholds (seconds)
    min_silence: float = 1.0
    long_silence: float = 5.0
    min_overlap: float = 0.2
    turn_merge_gap: float = 0.5
    uncertainty_threshold: float = 0.6
    question_answer_window: float = 30.0
    # Storage / retention
    keep_original_audio: bool = True
    retention_days: int = 0  # 0 = keep forever
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB
    # Optional local LLM
    ollama_enabled: bool = False
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    extra: dict = field(default_factory=dict)

    @classmethod
    def field_names(cls) -> list[str]:
        return [f for f in cls.__dataclass_fields__ if f != "extra"]


def ensure_directories() -> None:
    for d in (UPLOAD_DIR, PROCESSED_DIR, REPORTS_DIR, DATABASE_DIR, LOG_DIR, TEMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
