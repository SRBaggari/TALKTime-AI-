"""Speaking-time analytics computed exactly from diarized intervals.

Definitions
-----------
* **Individual speaking time** — length of the union of one speaker's intervals.
  Overlapping speech is credited to *every* speaker who is talking.
* **Combined speaker time** — sum of individual times (counts overlap once per
  speaker, so it can exceed the union).
* **Speech union** — time during which at least one person speaks.
* **Overlap duration** — time during which two or more people speak at once.
* **Non-speech time** — recording duration minus the speech union.

Percentages (denominators stated explicitly)
---------------------------------------------
* ``share_of_speaking_time`` = individual / combined speaker time. Sums to 100%
  across speakers. This is the primary "talk-time share".
* ``share_of_recording`` = individual / recording duration. Does not sum to 100%
  (silence is excluded, overlap is counted for each speaker).
"""
from __future__ import annotations

from collections import defaultdict

from core import intervals as iv
from core.types import SpeakerSegment

DENOMINATOR_NOTES = {
    "share_of_speaking_time": "Speaker time ÷ combined speaker time (sum over all speakers; overlap counted per speaker). Sums to 100%.",
    "share_of_recording": "Speaker time ÷ total recording duration. Excludes silence, so values do not sum to 100%.",
}


def speaker_intervals(segments: list[SpeakerSegment]) -> dict[str, list[iv.Interval]]:
    groups: dict[str, list[iv.Interval]] = defaultdict(list)
    for s in segments:
        if s.end > s.start:
            groups[s.speaker].append((s.start, s.end))
    return {k: iv.merge(v) for k, v in groups.items()}


def build_turns(segments: list[SpeakerSegment], merge_gap: float = 0.5) -> list[SpeakerSegment]:
    """Speaking turns: a speaker's consecutive intervals joined when the pause is
    <= ``merge_gap`` and no other speaker *starts* talking inside that pause."""
    groups = speaker_intervals(segments)
    all_starts = sorted((s, spk) for spk, ivs in groups.items() for s, _ in ivs)
    turns: list[SpeakerSegment] = []
    for spk, ivs in groups.items():
        cur_s, cur_e = ivs[0]
        for s, e in ivs[1:]:
            gap = s - cur_e
            interrupted = any(cur_e < st < s and other != spk for st, other in all_starts)
            if gap <= merge_gap and not interrupted:
                cur_e = max(cur_e, e)
            else:
                turns.append(SpeakerSegment(cur_s, cur_e, spk))
                cur_s, cur_e = s, e
        turns.append(SpeakerSegment(cur_s, cur_e, spk))
    return sorted(turns, key=lambda t: (t.start, t.end))


def compute_speaking_time(
    segments: list[SpeakerSegment], recording_duration: float, turn_merge_gap: float = 0.5
) -> dict:
    groups = speaker_intervals(segments)
    turns = build_turns(segments, turn_merge_gap)
    individual = {spk: iv.total(ivs) for spk, ivs in groups.items()}
    combined = sum(individual.values())
    union = iv.total(x for ivs in groups.values() for x in ivs)
    overlap_regions = iv.multi_coverage(groups, 2)
    overlap = iv.total(overlap_regions)
    duration = max(float(recording_duration), union)

    speakers: dict[str, dict] = {}
    for spk in sorted(groups, key=lambda k: int(k.split()[-1]) if k.split()[-1].isdigit() else 1_000):
        spk_turns = [t.duration for t in turns if t.speaker == spk]
        ivs = groups[spk]
        others = [x for k, v in groups.items() if k != spk for x in v]
        speakers[spk] = {
            "total_time": round(individual[spk], 3),
            "share_of_speaking_time": (individual[spk] / combined) if combined else 0.0,
            "share_of_recording": (individual[spk] / duration) if duration else 0.0,
            "solo_time": round(iv.total(iv.subtract(ivs, others)), 3),
            "overlap_time": round(iv.total(iv.intersect(ivs, others)), 3),
            "num_turns": len(spk_turns),
            "avg_turn": (sum(spk_turns) / len(spk_turns)) if spk_turns else 0.0,
            "longest_turn": max(spk_turns) if spk_turns else 0.0,
            "shortest_turn": min(spk_turns) if spk_turns else 0.0,
            "first_spoke": ivs[0][0],
            "last_spoke": ivs[-1][1],
            "interruptions_initiated": 0,
            "interruptions_received": 0,
        }

    return {
        "recording_duration": round(duration, 3),
        "num_speakers": len(groups),
        "speakers": speakers,
        "summary": {
            "combined_speaker_time": round(combined, 3),
            "speech_union": round(union, 3),
            "overlap_duration": round(overlap, 3),
            "non_speech_time": round(max(0.0, duration - union), 3),
            "speech_ratio": (union / duration) if duration else 0.0,
            "overlap_ratio_of_speech": (overlap / union) if union else 0.0,
            "total_turns": len(turns),
        },
        "turns": [t.to_dict() for t in turns],
        "overlap_regions": [list(r) for r in overlap_regions],
        "denominators": DENOMINATOR_NOTES,
    }
