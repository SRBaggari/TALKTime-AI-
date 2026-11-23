"""Voice Activity Detection (speech vs non-speech) on the original timeline.

Primary method: Silero VAD (ONNX model bundled inside faster-whisper — local,
no download). Fallback: adaptive energy VAD. Timestamps always refer to the
full, untrimmed recording, so they line up with diarization and playback.

Non-speech regions are further split into *quiet* (near the noise floor) and
*background sound* (energetic but not speech: noise, music, typing…), so loud
noise is never mistaken for speech and low volume is never mistaken for silence.
"""
from __future__ import annotations

import logging

import numpy as np

from core import intervals as iv
from core.audio_preprocessor import rms_envelope

logger = logging.getLogger(__name__)


def is_silero_available() -> bool:
    try:
        from faster_whisper.vad import get_speech_timestamps  # noqa: F401

        return True
    except Exception:
        return False


def _silero(audio: np.ndarray, sr: int, threshold: float, min_speech_ms: int, min_silence_ms: int,
            speech_pad_ms: int) -> list[iv.Interval]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(threshold=threshold, min_speech_duration_ms=min_speech_ms,
                      min_silence_duration_ms=min_silence_ms, speech_pad_ms=speech_pad_ms)
    stamps = get_speech_timestamps(np.ascontiguousarray(audio, dtype=np.float32), opts, sampling_rate=sr)
    return [(s["start"] / sr, s["end"] / sr) for s in stamps]


def _energy(audio: np.ndarray, sr: int, min_speech_ms: int, min_silence_ms: int, margin_db: float = 10.0,
            frame_s: float = 0.03) -> list[iv.Interval]:
    db, hop = rms_envelope(audio, sr, frame_s)
    if db.size == 0:
        return []
    floor = float(np.percentile(db, 10))
    loud = float(np.percentile(db, 95))
    # Threshold sits above the noise floor, but never so high that speech fails it when the
    # recording is almost all speech (then the 10th percentile is itself speech level).
    thr = max(min(floor + margin_db, loud - 15.0), -60.0)
    active = db > thr
    regions, start = [], None
    for i, a in enumerate(np.append(active, False)):
        if a and start is None:
            start = i
        elif not a and start is not None:
            regions.append((start * hop, i * hop))
            start = None
    merged = iv.merge(regions, gap=min_silence_ms / 1000)
    return [r for r in merged if r[1] - r[0] >= min_speech_ms / 1000]


def detect_speech(audio: np.ndarray, sr: int, threshold: float = 0.5, min_speech_ms: int = 250,
                  min_silence_ms: int = 300, speech_pad_ms: int = 100, method: str = "auto") -> dict:
    """Return speech / non-speech intervals (seconds) over the whole recording.

    ``min_silence_ms``: pauses shorter than this stay inside a speech region, so
    natural pauses between words are not reported as non-speech.
    """
    duration = audio.size / sr if sr else 0.0
    used = method
    speech: list[iv.Interval] = []
    if method in ("auto", "silero") and is_silero_available():
        try:
            speech = _silero(audio, sr, threshold, min_speech_ms, min_silence_ms, speech_pad_ms)
            used = "silero"
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Silero VAD failed (%s); using energy VAD", exc.__class__.__name__)
            used = "energy"
    else:
        used = "energy"
    if used == "energy":
        speech = _energy(audio, sr, min_speech_ms, min_silence_ms)
    speech = [(round(max(0.0, s), 3), round(min(duration, e), 3)) for s, e in iv.merge(speech) if e > s]
    non_speech = [(round(s, 3), round(e, 3)) for s, e in iv.complement(speech, 0.0, duration)]
    total_speech = iv.total(speech)
    return {
        "method": used,
        "speech": speech,
        "non_speech": non_speech,
        "speech_duration": round(total_speech, 3),
        "speech_ratio": total_speech / duration if duration else 0.0,
        "params": {"threshold": threshold, "min_speech_ms": min_speech_ms, "min_silence_ms": min_silence_ms,
                   "speech_pad_ms": speech_pad_ms},
    }


def region_level_db(audio: np.ndarray, sr: int, start: float, end: float) -> float:
    seg = audio[int(start * sr): int(end * sr)]
    if seg.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(seg, dtype=np.float64))))
    return 20 * np.log10(rms + 1e-10)


def classify_non_speech(audio: np.ndarray, sr: int, non_speech: list[iv.Interval], noise_floor_db: float,
                        margin_db: float = 10.0, min_duration: float = 0.3) -> list[dict]:
    """Label each non-speech region as ``quiet`` or ``background_sound``."""
    out = []
    for s, e in non_speech:
        if e - s < min_duration:
            continue
        level = region_level_db(audio, sr, s, e)
        out.append({"start": s, "end": e, "duration": round(e - s, 3), "level_db": round(level, 1),
                    "kind": "background_sound" if level > noise_floor_db + margin_db else "quiet"})
    return out


def compare_with_diarization(vad_speech: list[iv.Interval], diar_speech: list[iv.Interval],
                             min_duration: float = 0.5) -> dict:
    """Agreement between VAD and diarization speech regions (both are measurements).

    * ``unattributed_speech`` — VAD hears speech but no speaker was assigned
      (possible diarization miss, laughter, distant voice).
    * ``diarized_without_vad`` — diarization speech the VAD considers non-speech.
    """
    unattributed = [r for r in iv.subtract(vad_speech, diar_speech) if r[1] - r[0] >= min_duration]
    extra = [r for r in iv.subtract(diar_speech, vad_speech) if r[1] - r[0] >= min_duration]
    union = iv.total(iv.merge(list(vad_speech) + list(diar_speech)))
    agree = iv.total(iv.intersect(vad_speech, diar_speech))
    return {
        "agreement": (agree / union) if union else 1.0,
        "unattributed_speech": [list(r) for r in unattributed],
        "unattributed_total": round(iv.total(unattributed), 3),
        "diarized_without_vad": [list(r) for r in extra],
        "diarized_without_vad_total": round(iv.total(extra), 3),
    }
