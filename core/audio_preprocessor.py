"""Audio preprocessing: DC removal, peak normalisation, energy envelope and
writing the canonical 16 kHz mono WAV that both models and the player use
(so on-screen timestamps match playback exactly)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def preprocess(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Remove DC offset and peak-normalise. Returns a new float32 array."""
    if audio.size == 0:
        return audio.astype(np.float32)
    out = audio.astype(np.float32) - float(np.mean(audio))
    peak = float(np.max(np.abs(out)))
    if peak > 1e-6:
        out *= target_peak / peak
    return out


def is_effectively_silent(audio: np.ndarray, threshold_dbfs: float = -60.0) -> bool:
    """True when the whole recording is below ``threshold_dbfs`` RMS."""
    if audio.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return rms <= 0 or 20 * np.log10(rms + 1e-12) < threshold_dbfs


def rms_envelope(audio: np.ndarray, sr: int, frame_s: float = 0.05) -> tuple[np.ndarray, float]:
    """Frame-wise RMS in dBFS. Returns ``(db_values, hop_seconds)``."""
    hop = max(1, int(sr * frame_s))
    n = audio.size // hop
    if n == 0:
        return np.array([], dtype=np.float32), frame_s
    frames = audio[: n * hop].reshape(n, hop).astype(np.float64)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return (20 * np.log10(rms + 1e-10)).astype(np.float32), hop / sr


def write_wav(audio: np.ndarray, sr: int, path: Path) -> Path:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path
