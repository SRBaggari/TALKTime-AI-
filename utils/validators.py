"""Generic validators (language codes, speaker counts, Ollama URL)."""
from __future__ import annotations

from urllib.parse import urlparse

# Subset of Whisper languages offered in the UI (code -> name).
SUPPORTED_LANGUAGES: dict[str, str] = {
    "auto": "Auto-detect",
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "ur": "Urdu",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "ar": "Arabic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}


def validate_language(code: str | None) -> str | None:
    """Return a Whisper language code, or None for auto-detect."""
    if not code or code == "auto":
        return None
    code = code.strip().lower()
    if code not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language code: {code}")
    return code


def validate_speaker_bounds(min_speakers: int | None, max_speakers: int | None) -> tuple[int | None, int | None]:
    """Normalise speaker bounds; 0/None means automatic."""
    lo = int(min_speakers) if min_speakers else None
    hi = int(max_speakers) if max_speakers else None
    if lo is not None and lo < 1:
        raise ValueError("Minimum speakers must be at least 1")
    if hi is not None and hi < 1:
        raise ValueError("Maximum speakers must be at least 1")
    if lo is not None and hi is not None and lo > hi:
        raise ValueError("Minimum speakers cannot exceed maximum speakers")
    if hi is not None and hi > 20:
        raise ValueError("Maximum speakers is limited to 20")
    return lo, hi


def is_local_url(url: str) -> bool:
    """Ollama must run locally — refuse remote hosts so audio-derived text never leaves the machine."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
