"""Combine diarization (who/when) with transcription (what) by temporal overlap.

Method
------
* Word level (preferred, when Whisper word timestamps exist): each word goes to
  the speaker whose diarized intervals overlap it the most. Consecutive words of
  the same speaker are grouped into transcript segments, so one Whisper segment
  can be split at a speaker change.
* Segment level (fallback): the whole segment goes to the speaker with the
  largest overlap.
* A word/segment with *no* diarized overlap is snapped to the nearest speaker
  only if one is within ``max_snap_gap`` seconds, and is always marked uncertain.

Confidence = assigned speaker's overlap / total diarized overlap within the
segment (1.0 means no competing speaker). Segments are flagged ``uncertain``
when confidence < threshold, when little diarized speech lies under the text,
or when the assignment was snapped.
"""
from __future__ import annotations

from collections import defaultdict

from core import intervals as iv
from core.types import SpeakerSegment, TranscriptSegment, Word

UNKNOWN = "Unknown"


def _speaker_intervals(diar: list[SpeakerSegment]) -> dict[str, list[iv.Interval]]:
    groups: dict[str, list[iv.Interval]] = defaultdict(list)
    for s in diar:
        groups[s.speaker].append((s.start, s.end))
    return {k: iv.merge(v) for k, v in groups.items()}


def _overlaps(start: float, end: float, groups: dict[str, list[iv.Interval]]) -> dict[str, float]:
    out = {}
    for spk, ivs in groups.items():
        o = sum(iv.overlap_length((start, end), x) for x in ivs if x[1] > start and x[0] < end)
        if o > 0:
            out[spk] = o
    return out


def _nearest(start: float, end: float, groups: dict[str, list[iv.Interval]], max_gap: float) -> str | None:
    best, best_gap = None, max_gap
    for spk, ivs in groups.items():
        for s, e in ivs:
            gap = max(s - end, start - e, 0.0)
            if gap <= best_gap:
                best, best_gap = spk, gap
    return best


def _assign(start: float, end: float, groups, max_snap_gap: float) -> tuple[str, dict[str, float], bool]:
    ov = _overlaps(start, end, groups)
    if ov:
        return max(ov.items(), key=lambda kv: kv[1])[0], ov, False
    near = _nearest(start, end, groups, max_snap_gap)
    return (near or UNKNOWN), {}, True


def _finalize(seg: TranscriptSegment, groups, threshold: float, snapped: bool) -> TranscriptSegment:
    dur = max(seg.end - seg.start, 1e-6)
    ov = _overlaps(seg.start, seg.end, groups)
    total_ov = sum(ov.values())
    if seg.speaker in ov and total_ov > 0:
        seg.speaker_confidence = round(ov[seg.speaker] / total_ov, 3)
    else:
        seg.speaker_confidence = 0.0
    seg.overlapping_speakers = sorted(
        s for s, o in ov.items() if s != seg.speaker and o >= min(0.2, 0.2 * dur)
    )
    coverage = total_ov / dur
    seg.uncertain = bool(snapped or seg.speaker == UNKNOWN or seg.speaker_confidence < threshold or coverage < 0.3)
    return seg


def _smooth_word_runs(words: list[Word], min_run: float = 0.3) -> None:
    """Re-label very short single-speaker blips sandwiched by the same speaker (boundary jitter)."""
    if len(words) < 3:
        return
    runs: list[list[int]] = []
    for i, w in enumerate(words):
        if runs and words[runs[-1][0]].speaker == w.speaker:
            runs[-1].append(i)
        else:
            runs.append([i])
    for k in range(1, len(runs) - 1):
        prev_spk = words[runs[k - 1][0]].speaker
        next_spk = words[runs[k + 1][0]].speaker
        run = runs[k]
        run_dur = words[run[-1]].end - words[run[0]].start
        if prev_spk == next_spk and run_dur < min_run and len(run) <= 2:
            for i in run:
                words[i].speaker = prev_spk


def align(
    transcript: list[TranscriptSegment],
    diarization: list[SpeakerSegment],
    uncertainty_threshold: float = 0.6,
    word_level: bool = True,
    max_snap_gap: float = 1.0,
) -> list[TranscriptSegment]:
    """Return new transcript segments, each labelled with a speaker."""
    groups = _speaker_intervals(diarization)
    out: list[TranscriptSegment] = []
    for seg in transcript:
        if word_level and seg.words:
            words = [Word(w.start, w.end, w.word, w.probability) for w in seg.words]
            snapped_flags = {}
            for i, w in enumerate(words):
                spk, _, snapped = _assign(w.start, max(w.end, w.start + 0.01), groups, max_snap_gap)
                w.speaker = spk
                snapped_flags[i] = snapped
            _smooth_word_runs(words)
            run: list[int] = []
            for i, w in enumerate(words + [None]):
                if w is not None and (not run or words[run[-1]].speaker == w.speaker):
                    run.append(i)
                    continue
                if run:
                    rw = [words[j] for j in run]
                    text = "".join(x.word for x in rw).strip()
                    if text:
                        new = TranscriptSegment(
                            start=rw[0].start, end=rw[-1].end, text=text, words=rw,
                            avg_logprob=seg.avg_logprob, no_speech_prob=seg.no_speech_prob,
                            speaker=rw[0].speaker, language=seg.language,
                            language_probability=seg.language_probability,
                            original_text=seg.original_text if len(run) == len(words) else None,
                        )
                        out.append(_finalize(new, groups, uncertainty_threshold, all(snapped_flags[j] for j in run)))
                run = [i] if w is not None else []
        else:
            spk, _, snapped = _assign(seg.start, seg.end, groups, max_snap_gap)
            new = TranscriptSegment(
                start=seg.start, end=seg.end, text=seg.text, words=list(seg.words),
                avg_logprob=seg.avg_logprob, no_speech_prob=seg.no_speech_prob, speaker=spk,
                language=seg.language, language_probability=seg.language_probability,
                original_text=seg.original_text,
            )
            out.append(_finalize(new, groups, uncertainty_threshold, snapped))
    out.sort(key=lambda s: (s.start, s.end))
    return out
