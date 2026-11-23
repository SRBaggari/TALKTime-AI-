"""Classify overlap events conservatively.

Categories (heuristic — never presented as intent):
* ``possible_interruption`` — someone started while another speaker had been
  talking for a while, and the original speaker stopped during the overlap while
  the newcomer kept going (the floor changed hands under overlap).
* ``brief_overlap`` — a short overlap where the original speaker kept the floor
  (often backchannels such as "yeah", "right").
* ``overlapping_speech`` — longer simultaneous talk where nobody clearly yielded.
* ``simultaneous_start`` — both started at (almost) the same time.
"""
from __future__ import annotations

from collections import Counter

LABELS = {
    "possible_interruption": "Possible interruption",
    "brief_overlap": "Brief overlap / backchannel",
    "overlapping_speech": "Overlapping speech",
    "simultaneous_start": "Simultaneous start",
}


def classify_overlap(event: dict, min_prior_speech: float = 0.5, brief_max: float = 1.0) -> str:
    if event["simultaneous"]:
        return "simultaneous_start"
    a_s, a_e = event["ongoing_interval"]
    b_s, b_e = event["incoming_interval"]
    prior = b_s - a_s
    ongoing_yields = a_e <= b_e
    if ongoing_yields and prior >= min_prior_speech:
        return "possible_interruption"
    if not ongoing_yields and (b_e - b_s) <= brief_max:
        return "brief_overlap"
    return "overlapping_speech"


def analyze_interruptions(events: list[dict], speakers: list[str], recording_duration: float) -> dict:
    classified = []
    initiated: Counter = Counter()
    received: Counter = Counter()
    overlaps_by_speaker: Counter = Counter()
    for ev in events:
        kind = classify_overlap(ev)
        item = {k: v for k, v in ev.items() if k not in ("ongoing_interval", "incoming_interval")}
        item["kind"] = kind
        item["label"] = LABELS[kind]
        classified.append(item)
        for spk in ev["speakers"]:
            overlaps_by_speaker[spk] += 1
        if kind == "possible_interruption":
            initiated[ev["incoming_speaker"]] += 1
            received[ev["ongoing_speaker"]] += 1

    minutes = recording_duration / 60 if recording_duration else 0
    n_int = sum(1 for c in classified if c["kind"] == "possible_interruption")
    return {
        "events": classified,
        "counts": dict(Counter(c["kind"] for c in classified)),
        "total_overlap_events": len(classified),
        "possible_interruptions": n_int,
        "interruptions_per_minute": (n_int / minutes) if minutes else 0.0,
        "overlap_events_per_minute": (len(classified) / minutes) if minutes else 0.0,
        "per_speaker": {
            spk: {
                "initiated": initiated.get(spk, 0),
                "received": received.get(spk, 0),
                "overlap_events": overlaps_by_speaker.get(spk, 0),
            }
            for spk in speakers
        },
        "note": "Overlaps are detected from diarization timing. A 'possible interruption' means the floor changed "
                "hands during overlapping speech; it does not imply intent or rudeness.",
    }
