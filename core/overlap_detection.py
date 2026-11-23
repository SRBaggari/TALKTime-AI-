"""Detect overlapping-speech events between pairs of speakers."""
from __future__ import annotations

from core.speaking_time import speaker_intervals
from core.types import SpeakerSegment

SIMULTANEOUS_START_TOLERANCE = 0.15  # seconds


def detect_overlaps(segments: list[SpeakerSegment], min_overlap: float = 0.2) -> list[dict]:
    """Return one event per pairwise intersection of different speakers' speech.

    ``ongoing_speaker`` is whoever was already talking; ``incoming_speaker`` is
    whoever started during their speech. If both start within
    ``SIMULTANEOUS_START_TOLERANCE`` neither is ongoing (``simultaneous=True``).
    """
    groups = speaker_intervals(segments)
    items = sorted(((s, e, spk) for spk, ivs in groups.items() for s, e in ivs), key=lambda x: (x[0], x[1]))
    events: list[dict] = []
    for i, (a_s, a_e, a_spk) in enumerate(items):
        for b_s, b_e, b_spk in items[i + 1:]:
            if b_s >= a_e:
                break
            if b_spk == a_spk:
                continue
            start, end = max(a_s, b_s), min(a_e, b_e)
            if end - start < min_overlap:
                continue
            simultaneous = abs(b_s - a_s) <= SIMULTANEOUS_START_TOLERANCE
            events.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "speakers": sorted([a_spk, b_spk]),
                "ongoing_speaker": None if simultaneous else a_spk,
                "incoming_speaker": None if simultaneous else b_spk,
                "simultaneous": simultaneous,
                "ongoing_interval": [a_s, a_e],
                "incoming_interval": [b_s, b_e],
            })
    events.sort(key=lambda e: e["start"])
    return events
