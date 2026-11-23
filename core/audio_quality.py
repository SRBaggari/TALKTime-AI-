"""Recording quality analysis — measured from the *un-normalised* decoded audio.

Metrics (all in dBFS unless noted):
* ``speech_level`` — RMS over VAD speech regions (so long silences don't drag
  it down: low volume is judged on speech, never on silence).
* ``noise_floor`` — RMS over non-speech regions (or the 10th-percentile frame
  level when there is too little non-speech).
* ``snr_db`` — speech_level − noise_floor (an estimate; speech regions contain noise too).
  The floor is clamped at −90 dBFS: lower values mean digitally silent gaps
  (synthetic or heavily gated audio), reported via ``digital_silence``.
* ``clipping_ratio`` — fraction of samples at |x| ≥ 0.99 full scale; clipping
  *events* are runs of ≥3 consecutive such samples (the signature of a flattened peak).
* ``bandwidth`` — frequency below which 99% of speech energy lies (telephone
  audio is ≈3.4 kHz). Limited to 8 kHz because analysis runs at 16 kHz.

Score: starts at 100; documented penalties are subtracted per issue (see PENALTIES).
"""
from __future__ import annotations

import numpy as np

from core import intervals as iv
from core.audio_preprocessor import rms_envelope

DIGITAL_SILENCE_DB = -90.0

PENALTIES = {
    "low_snr_severe": 30, "low_snr": 15, "very_low_volume": 25, "low_volume": 12, "clipping_severe": 25,
    "clipping": 10, "noisy_background": 10, "narrowband": 5, "little_speech": 10, "long_silence": 5,
    "low_sample_rate": 5, "dc_offset": 3,
}

RECOMMENDATIONS = {
    "low_snr_severe": "Background noise is close to the speech level. Record in a quieter room, move the microphone closer to speakers, or use a directional/headset microphone.",
    "low_snr": "Some background noise is present. Reducing fans, traffic or echo will improve transcription and speaker separation.",
    "very_low_volume": "Speech is very quiet. Increase input gain or move closer to the microphone; transcription accuracy drops sharply at this level.",
    "low_volume": "Speech level is low. A little more input gain would help.",
    "clipping_severe": "The signal is heavily clipped (distorted). Lower the recording gain — clipped audio cannot be repaired afterwards.",
    "clipping": "Some peaks are clipped. Reduce input gain slightly.",
    "noisy_background": "The noise floor is high even when nobody is speaking.",
    "narrowband": "Audio is band-limited (telephone quality). Recording directly (not over a phone line) improves accuracy.",
    "little_speech": "Very little speech was detected. Check that the correct file/channel was recorded.",
    "long_silence": "The recording contains long silent stretches; trimming them shortens processing time.",
    "low_sample_rate": "The original sample rate is low. Record at 16 kHz or higher.",
    "dc_offset": "A DC offset is present (often a hardware issue). It is corrected automatically during processing.",
}


def _frames_in(intervals: list[iv.Interval], n_frames: int, hop: float) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for s, e in intervals:
        mask[int(s / hop): int(np.ceil(e / hop))] = True
    return mask


def _bandwidth(audio: np.ndarray, sr: int, speech: list[iv.Interval], max_seconds: float = 60.0) -> float | None:
    chunks, total = [], 0.0
    for s, e in speech:
        take = min(e - s, max_seconds - total)
        if take <= 0:
            break
        chunks.append(audio[int(s * sr): int((s + take) * sr)])
        total += take
    if not chunks:
        return None
    x = np.concatenate(chunks).astype(np.float64)
    n = 1024
    if x.size < n:
        return None
    frames = x[: (x.size // n) * n].reshape(-1, n) * np.hanning(n)
    power = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).sum(axis=0)
    cum = np.cumsum(power) / max(power.sum(), 1e-20)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    return float(freqs[np.searchsorted(cum, 0.99)])


def analyze_quality(audio: np.ndarray, sr: int, speech: list[iv.Interval], probe: dict | None = None,
                    long_silence: float = 10.0) -> dict:
    """``audio``: decoded, NOT normalised, mono float32. ``speech``: VAD speech intervals."""
    probe = probe or {}
    duration = audio.size / sr if sr else 0.0
    db, hop = rms_envelope(audio, sr, 0.05)
    x = audio.astype(np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    overall_rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    to_db = lambda v: float(20 * np.log10(v + 1e-10))  # noqa: E731

    speech_mask = _frames_in(speech, db.size, hop) if db.size else np.zeros(0, bool)
    speech_frames = db[speech_mask] if db.size else db
    non_speech_frames = db[~speech_mask] if db.size else db
    speech_level = float(10 * np.log10(np.mean(10 ** (speech_frames / 10)))) if speech_frames.size else None
    if non_speech_frames.size * hop >= 1.0:
        noise_floor = float(np.percentile(non_speech_frames, 50))
    else:
        noise_floor = float(np.percentile(db, 10)) if db.size else -120.0
    digital_silence = noise_floor < DIGITAL_SILENCE_DB
    noise_floor = max(noise_floor, DIGITAL_SILENCE_DB)  # below this is digital silence, not a real noise floor
    snr = (speech_level - noise_floor) if speech_level is not None else None

    clipped = np.abs(x) >= 0.99
    clip_ratio = float(clipped.mean()) if x.size else 0.0
    runs = 0
    if clipped.any():
        edges = np.diff(np.concatenate(([0], clipped.astype(np.int8), [0])))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        runs = int(np.sum((ends - starts) >= 3))
    dc = float(np.mean(x)) if x.size else 0.0
    speech_dur = iv.total(speech)
    silences = [(s, e) for s, e in iv.complement(speech, 0.0, duration) if e - s >= long_silence]
    bandwidth = _bandwidth(audio, sr, speech)
    orig_sr = probe.get("sample_rate") or sr

    issues: list[dict] = []

    def add(key, severity, message):
        issues.append({"key": key, "severity": severity, "message": message, "recommendation": RECOMMENDATIONS[key],
                       "penalty": PENALTIES[key]})

    if speech_level is None or speech_dur < 0.5:
        add("little_speech", "serious", "No clear speech detected — the recording may be silent, extremely quiet, or non-speech audio.")
    else:
        if snr is not None and snr < 10:
            add("low_snr_severe", "serious", f"Estimated signal-to-noise ratio is {snr:.0f} dB (speech barely above noise).")
        elif snr is not None and snr < 20:
            add("low_snr", "warning", f"Estimated signal-to-noise ratio is {snr:.0f} dB.")
        if speech_level < -45:
            add("very_low_volume", "serious", f"Speech level is {speech_level:.0f} dBFS (very quiet).")
        elif speech_level < -35:
            add("low_volume", "warning", f"Speech level is {speech_level:.0f} dBFS (quiet).")
        if duration and speech_dur / duration < 0.1:
            add("little_speech", "warning", f"Only {speech_dur / duration * 100:.0f}% of the recording contains speech.")
    if clip_ratio > 0.001 or runs > 50:
        add("clipping_severe", "serious", f"{clip_ratio * 100:.2f}% of samples are clipped ({runs} clipping events).")
    elif clip_ratio > 0.0001 or runs > 5:
        add("clipping", "warning", f"{runs} clipping events detected.")
    if noise_floor > -40 and (snr is None or snr < 25):
        add("noisy_background", "warning", f"Noise floor is {noise_floor:.0f} dBFS.")
    if bandwidth is not None and bandwidth < 3800:
        add("narrowband", "info", f"99% of speech energy is below {bandwidth / 1000:.1f} kHz.")
    if orig_sr and orig_sr < 16000:
        add("low_sample_rate", "info", f"Original sample rate is {orig_sr} Hz.")
    if silences:
        add("long_silence", "info", f"{len(silences)} silent section(s) of {long_silence:.0f}s or longer "
                                    f"({iv.total(silences):.0f}s total).")
    if abs(dc) > 0.01:
        add("dc_offset", "info", f"DC offset of {dc:.3f}.")

    score = max(0, 100 - sum(i["penalty"] for i in issues))
    label = "Excellent" if score >= 85 else "Good" if score >= 70 else "Fair" if score >= 50 else "Poor"
    return {
        "score": score,
        "label": label,
        "duration": round(duration, 3),
        "format": {"codec": probe.get("codec"), "container": probe.get("container"), "sample_rate": orig_sr,
                   "channels": probe.get("channels"), "bit_rate": probe.get("bit_rate")},
        "peak_dbfs": round(to_db(peak), 1),
        "rms_dbfs": round(to_db(overall_rms), 1),
        "speech_level_dbfs": round(speech_level, 1) if speech_level is not None else None,
        "noise_floor_dbfs": round(noise_floor, 1),
        "digital_silence": digital_silence,
        "snr_db": round(snr, 1) if snr is not None else None,
        "clipping_ratio": clip_ratio,
        "clipping_events": runs,
        "dc_offset": round(dc, 4),
        "bandwidth_hz": round(bandwidth) if bandwidth else None,
        "speech_duration": round(speech_dur, 3),
        "long_silences": [list(s) for s in silences],
        "issues": issues,
        "recommendations": [i["recommendation"] for i in issues if i["severity"] != "info"] or
                           ["No significant problems detected."],
        "method": "Levels from 50 ms RMS frames; speech/non-speech from VAD; score = 100 − documented penalties.",
    }
