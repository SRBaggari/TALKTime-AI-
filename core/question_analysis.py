"""Heuristic question detection and question→response matching.

This is *not* semantic QA matching. A question is considered "likely answered"
when a different speaker takes a turn soon after it. Every result carries a
confidence value and the method used.
"""
from __future__ import annotations

import re
from collections import defaultdict

from core.nlp_utils import split_sentences
from core.types import TranscriptSegment

WH_WORDS = r"(what|why|how|when|where|who|whom|whose|which)"
AUX = r"(do|does|did|is|are|was|were|am|can|could|would|will|should|shall|may|might|have|has|had|isn't|aren't|don't|doesn't|didn't|won't|wouldn't|can't|couldn't|shouldn't)"
_WH_START = re.compile(rf"^\s*(so|and|but|okay|ok|well|then)?[\s,]*{WH_WORDS}\b", re.I)
_AUX_START = re.compile(rf"^\s*(so|and|but|okay|ok|well|then)?[\s,]*{AUX}\s+(i|you|we|they|he|she|it|there|this|that|the|anyone|someone|everyone)\b", re.I)
_TAG = re.compile(r",\s*(right|isn't it|aren't you|don't you|correct|yes|no)\s*[?.]?$", re.I)
_QMARK = re.compile(r"[?？]\s*$")


def detect_question(sentence: str) -> tuple[bool, float, str]:
    """Return ``(is_question, confidence, method)`` for one sentence."""
    s = sentence.strip()
    words = s.split()
    if not s:
        return False, 0.0, ""
    if _QMARK.search(s):
        if len(words) <= 1:  # "Right?", "Okay?" — usually a check-in, not a real question
            return True, 0.35, "question mark (single word)"
        return True, 0.9, "question mark"
    if _WH_START.search(s) and len(words) >= 3:
        return True, 0.6, "interrogative word"
    if _AUX_START.search(s):
        return True, 0.55, "auxiliary inversion"
    if _TAG.search(s):
        return True, 0.45, "tag question"
    return False, 0.0, ""


def analyze_questions(segments: list[TranscriptSegment], window: float = 30.0, min_confidence: float = 0.4) -> dict:
    segs = sorted(segments, key=lambda s: s.start)
    questions: list[dict] = []
    for idx, seg in enumerate(segs):
        for sent in split_sentences(seg.text):
            is_q, conf, method = detect_question(sent)
            if not is_q or conf < min_confidence:
                continue
            q = {
                "speaker": seg.speaker, "text": sent, "start": round(seg.start, 3), "end": round(seg.end, 3),
                "confidence": conf, "method": method, "answered": False, "answered_by": None,
                "answer_text": None, "answer_start": None, "response_delay": None, "answer_confidence": 0.0,
            }
            same_speaker_between = False
            for nxt in segs[idx + 1:]:
                if nxt.start - seg.end > window:
                    break
                if nxt.speaker == seg.speaker:
                    same_speaker_between = True
                    continue
                if nxt.speaker in (None, "Unknown"):
                    continue
                delay = max(0.0, nxt.start - seg.end)
                a_conf = 0.8
                if same_speaker_between:
                    a_conf -= 0.2
                if delay > 10:
                    a_conf -= 0.3
                elif delay > 5:
                    a_conf -= 0.1
                if any(detect_question(x)[0] for x in split_sentences(nxt.text)):
                    a_conf -= 0.2  # responded with a question
                q.update(answered=True, answered_by=nxt.speaker, answer_text=nxt.text[:300],
                         answer_start=round(nxt.start, 3), response_delay=round(delay, 3),
                         answer_confidence=round(max(0.1, a_conf), 2))
                break
            questions.append(q)

    per_speaker: dict[str, dict] = defaultdict(lambda: {"asked": 0, "got_response": 0, "possibly_unanswered": 0, "responded_to_others": 0})
    for q in questions:
        ps = per_speaker[q["speaker"]]
        ps["asked"] += 1
        if q["answered"]:
            ps["got_response"] += 1
            per_speaker[q["answered_by"]]["responded_to_others"] += 1
        else:
            ps["possibly_unanswered"] += 1

    return {
        "questions": questions,
        "total_questions": len(questions),
        "likely_answered": sum(1 for q in questions if q["answered"]),
        "possibly_unanswered": sum(1 for q in questions if not q["answered"]),
        "per_speaker": dict(per_speaker),
        "method": "Questions: punctuation, interrogative words and auxiliary inversion. Responses: the next turn by a "
                  f"different speaker within {window:.0f}s. Heuristic — not semantic answer verification.",
    }
