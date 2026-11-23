"""Decode any supported audio file to 16 kHz mono float32.

FFmpeg is used when available (handles MP3/M4A/AAC/OGG/WEBM/...). libsndfile
(via soundfile) is the fallback for WAV/FLAC/OGG when FFmpeg is missing.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

import config

logger = logging.getLogger(__name__)


class AudioLoadError(Exception):
    """Raised for corrupted, empty or undecodable audio."""


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def _load_with_ffmpeg(path: Path, sr: int) -> np.ndarray:
    cmd = [
        ffmpeg_path() or "ffmpeg",
        "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(sr),
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=1800, check=False)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "ignore").strip().splitlines()
        raise AudioLoadError("Could not decode audio (file may be corrupted or not audio). "
                             + (msg[-1][:200] if msg else ""))
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _load_with_soundfile(path: Path, sr: int) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise AudioLoadError("Neither FFmpeg nor soundfile is available to decode audio.") from exc
    try:
        data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises various errors for bad input
        raise AudioLoadError("Could not decode audio without FFmpeg. Install FFmpeg for MP3/M4A support.") from exc
    mono = data.mean(axis=1)
    if file_sr != sr:
        try:
            import librosa
            mono = librosa.resample(mono, orig_sr=file_sr, target_sr=sr)
        except ImportError:  # simple linear resample fallback
            n = int(round(len(mono) * sr / file_sr))
            mono = np.interp(np.linspace(0, len(mono) - 1, n), np.arange(len(mono)), mono).astype(np.float32)
    return mono.astype(np.float32)


def load_audio(path: str | Path, sr: int = config.TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Return ``(waveform, sample_rate)`` — mono float32 in [-1, 1]."""
    path = Path(path)
    if not path.is_file():
        raise AudioLoadError("Audio file not found.")
    if path.stat().st_size == 0:
        raise AudioLoadError("The audio file is empty.")

    if ffmpeg_path():
        audio = _load_with_ffmpeg(path, sr)
    else:
        logger.warning("FFmpeg not found; falling back to soundfile")
        audio = _load_with_soundfile(path, sr)

    if audio.size == 0:
        raise AudioLoadError("The file contains no decodable audio samples.")
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    duration = audio.size / sr
    if duration < config.MIN_AUDIO_SECONDS:
        raise AudioLoadError(f"Recording is too short ({duration:.2f}s). Minimum is {config.MIN_AUDIO_SECONDS}s.")
    return audio, sr


def probe_audio(path: str | Path) -> dict:
    """Original stream metadata (sample rate, channels, codec, bitrate). Never raises."""
    path = Path(path)
    info: dict = {}
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            import json

            proc = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                                   "stream=codec_name,sample_rate,channels,bit_rate:format=format_name,duration,bit_rate",
                                   "-of", "json", str(path)], capture_output=True, timeout=60, check=False)
            data = json.loads(proc.stdout or b"{}")
            st = (data.get("streams") or [{}])[0]
            fmt = data.get("format") or {}
            info = {
                "codec": st.get("codec_name"),
                "sample_rate": int(st["sample_rate"]) if st.get("sample_rate") else None,
                "channels": int(st["channels"]) if st.get("channels") else None,
                "bit_rate": int(st.get("bit_rate") or fmt.get("bit_rate") or 0) or None,
                "container": fmt.get("format_name"),
                "duration": float(fmt["duration"]) if fmt.get("duration") else None,
            }
        except Exception:
            info = {}
    if not info.get("sample_rate"):
        try:
            import soundfile as sf

            i = sf.info(str(path))
            info.update(sample_rate=i.samplerate, channels=i.channels, codec=i.subtype, container=i.format,
                        duration=i.duration)
        except Exception:
            pass
    return info


def audio_duration(audio: np.ndarray, sr: int) -> float:
    return float(audio.size) / float(sr)
