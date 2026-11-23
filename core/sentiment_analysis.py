"""Text-based sentiment using VADER (local, lexicon + rules, English only).

This classifies the *wording* of each transcript segment as positive, negative
or neutral. It does not infer emotions, personality or mental state from voice.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache

from core.nlp_utils import is_english
from core.types import TranscriptSegment

POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05

DISCLAIMER = ("Sentiment is an approximate classification of the transcript text (VADER lexicon). It reflects word "
              "choice only — not tone of voice, emotion, intent or mental state — and can misread sarcasm or domain terms.")


@lru_cache(maxsize=1)
def _analyzer():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        return SentimentIntensityAnalyzer()
    except ImportError:
        return None


def is_sentiment_available() -> bool:
    return _analyzer() is not None


def label_for(score: float) -> str:
    if score >= POS_THRESHOLD:
        return "positive"
    if score <= NEG_THRESHOLD:
        return "negative"
    return "neutral"


def score_text(text: str) -> float:
    an = _analyzer()
    if an is None:
        raise RuntimeError("vaderSentiment is not installed")
    return float(an.polarity_scores(text.replace("[unclear]", ""))["compound"])


def analyze_sentiment(segments: list[TranscriptSegment], language: str | None = "en", window: int = 5) -> dict:
    """Annotate segments in place (``sentiment``, ``sentiment_score``) and aggregate."""
    if not is_english(language):
        return {"available": False, "reason": f"Sentiment model supports English only (detected: {language}).",
                "disclaimer": DISCLAIMER}
    if not is_sentiment_available():
        return {"available": False, "reason": "vaderSentiment is not installed.", "disclaimer": DISCLAIMER}

    by_speaker_scores: dict[str, list[tuple[float, float]]] = defaultdict(list)
    by_speaker_labels: dict[str, Counter] = defaultdict(Counter)
    overall = Counter()
    timeline = []
    for seg in sorted(segments, key=lambda s: s.start):
        score = score_text(seg.text)
        seg.sentiment_score = round(score, 4)
        seg.sentiment = label_for(score)
        spk = seg.speaker or "Unknown"
        by_speaker_scores[spk].append((score, max(seg.duration, 0.1)))
        by_speaker_labels[spk][seg.sentiment] += 1
        overall[seg.sentiment] += 1
        timeline.append({"time": round(seg.start, 3), "score": round(score, 4), "speaker": spk, "label": seg.sentiment})

    # Rolling mean over the last ``window`` segments to show trend.
    for i, point in enumerate(timeline):
        chunk = timeline[max(0, i - window + 1): i + 1]
        point["rolling"] = round(sum(p["score"] for p in chunk) / len(chunk), 4)

    per_speaker = {}
    for spk, vals in by_speaker_scores.items():
        wsum = sum(w for _, w in vals)
        mean = sum(s * w for s, w in vals) / wsum if wsum else 0.0
        per_speaker[spk] = {"mean_score": round(mean, 4), "label": label_for(mean), "distribution": dict(by_speaker_labels[spk])}

    n = sum(overall.values())
    third = max(1, len(timeline) // 3)
    trend = None
    if len(timeline) >= 6:
        first = sum(p["score"] for p in timeline[:third]) / third
        last = sum(p["score"] for p in timeline[-third:]) / third
        trend = {"first_third": round(first, 4), "last_third": round(last, 4), "change": round(last - first, 4)}
    return {
        "available": True,
        "overall_distribution": dict(overall),
        "overall_share": {k: v / n for k, v in overall.items()} if n else {},
        "per_speaker": per_speaker,
        "timeline": timeline,
        "trend": trend,
        "disclaimer": DISCLAIMER,
    }
