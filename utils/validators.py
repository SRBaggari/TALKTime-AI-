"""Generic validators (language codes, speaker counts, Ollama URL)."""
from __future__ import annotations

from urllib.parse import urlparse

from core.language_utils import WHISPER_LANGUAGES, ordered_language_options

# All Whisper languages, priority languages first (code -> name).
SUPPORTED_LANGUAGES: dict[str, str] = {c: ("Auto-detect" if c == "auto" else WHISPER_LANGUAGES[c])
                                       for c in ordered_language_options()}



def validate_language(code: str | None) -> str | None:
    """Return a Whisper language code, or None for auto-detect."""
    if not code or code == "auto":
        return None
    code = code.strip().lower()
    if code not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Language '{code}' is not supported by Whisper. Choose Auto-detect or a listed language.")
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
