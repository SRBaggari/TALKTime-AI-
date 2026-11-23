"""Silence, pause and response-delay analysis.

Silence here means *no diarized speech from anyone* (speech-activity based).
Gaps shorter than ``min_silence`` are treated as normal pauses and ignored, so
brief pauses inside sentences are not counted as meaningful silence.
An optional acoustic (low-energy) measure is computed from the waveform.
"""
from __future__ import annotations

import statistics
from collections import defaultdict

import numpy as np

from core import intervals as iv
from core.types import SpeakerSegment


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "total": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "total": round(sum(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "max": round(max(values), 3),
    }


def analyze_silence(
    segments: list[SpeakerSegment],
    recording_duration: float,
    min_silence: float = 1.0,
    long_silence: float = 5.0,
    turns: list[SpeakerSegment] | None = None,
) -> dict:
    duration = float(recording_duration)
    speech = iv.merge((s.start, s.end) for s in segments)
    gaps = iv.complement(speech, 0.0, duration)
    seg_sorted = sorted(segments, key=lambda s: s.end)

    def speaker_ending_at(t: float) -> str | None:
        cands = [s.speaker for s in seg_sorted if abs(s.end - t) < 1e-6]
        return cands[-1] if cands else None

    def speaker_starting_at(t: float) -> str | None:
        cands = [s.speaker for s in segments if abs(s.start - t) < 1e-6]
        return cands[0] if cands else None

    silences = []
    for s, e in gaps:
        d = e - s
        if d < min_silence:
            continue
        before = speaker_ending_at(s) if s > 0 else None
        after = speaker_starting_at(e) if e < duration else None
        if before is None and s <= 1e-6:
            kind = "leading"
        elif after is None and e >= duration - 1e-6:
            kind = "trailing"
        elif before == after:
            kind = "within_speaker"
        else:
            kind = "between_speakers"
        silences.append({
            "start": round(s, 3), "end": round(e, 3), "duration": round(d, 3),
            "kind": kind, "before": before, "after": after, "long": d >= long_silence,
        })

    # Response delays at speaker changes (using turns). Negative = overlap.
    response_delays = []
    per_responder: dict[str, list[float]] = defaultdict(list)
    if turns:
        ordered = sorted(turns, key=lambda t: t.start)
        latest_end, latest_spk = None, None
        for t in ordered:
            if latest_spk is not None and t.speaker != latest_spk:
                delay = t.start - latest_end
                response_delays.append({"from": latest_spk, "to": t.speaker, "at": round(t.start, 3), "delay": round(delay, 3)})
                if delay >= 0:
                    per_responder[t.speaker].append(delay)
            if latest_end is None or t.end >= latest_end:
                latest_end, latest_spk = t.end, t.speaker

    internal = [x["duration"] for x in silences if x["kind"] in ("within_speaker", "between_speakers")]
    within = defaultdict(list)
    for x in silences:
        if x["kind"] == "within_speaker":
            within[x["before"]].append(x["duration"])

    positive_delays = [r["delay"] for r in response_delays if r["delay"] >= 0]
    return {
        "thresholds": {"min_silence": min_silence, "long_silence": long_silence},
        "silences": silences,
        "total_non_speech": round(iv.total(gaps), 3),
        "total_silence": round(sum(x["duration"] for x in silences), 3),
        "conversation_silence": _stats(internal),
        "longest_silence": max((x["duration"] for x in silences), default=0.0),
        "average_silence": round(statistics.fmean([x["duration"] for x in silences]), 3) if silences else 0.0,
        "num_silences": len(silences),
        "num_long_silences": sum(1 for x in silences if x["long"]),
        "between_speakers": _stats([x["duration"] for x in silences if x["kind"] == "between_speakers"]),
        "within_speaker_pauses": {spk: _stats(v) for spk, v in within.items()},
        "leading_silence": next((x["duration"] for x in silences if x["kind"] == "leading"), 0.0),
        "trailing_silence": next((x["duration"] for x in silences if x["kind"] == "trailing"), 0.0),
        "response_delays": response_delays,
        "response_delay_stats": _stats(positive_delays),
        "response_delay_by_speaker": {spk: _stats(v) for spk, v in per_responder.items()},
    }


def acoustic_silence(audio: np.ndarray, sr: int, min_silence: float = 1.0, margin_db: float = 6.0,
                     frame_s: float = 0.05) -> dict:
    """Low-energy regions measured directly from the waveform.

    Threshold is adaptive: (10th-percentile frame energy) + ``margin_db``, never
    above -35 dBFS, so it follows the recording's own noise floor.
    """
    from core.audio_preprocessor import rms_envelope

    db, hop = rms_envelope(audio, sr, frame_s)
    if db.size == 0:
        return {"threshold_db": None, "total": 0.0, "count": 0, "regions": []}
    threshold = float(min(np.percentile(db, 10) + margin_db, -35.0))
    quiet = db < threshold
    regions = []
    start = None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None:
            start = i
        elif not q and start is not None:
            s, e = start * hop, i * hop
            if e - s >= min_silence:
                regions.append((round(s, 3), round(e, 3)))
            start = None
    return {
        "threshold_db": round(threshold, 1),
        "total": round(sum(e - s for s, e in regions), 3),
        "count": len(regions),
        "regions": regions,
    }
