"""Shared lightweight NLP helpers (sentence splitting, optional spaCy)."""
from __future__ import annotations

import logging
import re
from functools import lru_cache

import config

logger = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?？。！])\s+")


@lru_cache(maxsize=1)
def get_spacy():
    """Load the small English spaCy model once; return None if not installed."""
    try:
        import spacy

        return spacy.load(config.SPACY_MODEL)
    except Exception as exc:  # OSError when model missing, ImportError when spaCy missing
        logger.warning("spaCy model unavailable (%s); using rule-based fallbacks", exc.__class__.__name__)
        return None


def is_spacy_available() -> bool:
    return get_spacy() is not None


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def is_english(language: str | None) -> bool:
    return language in (None, "", "en")


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to", "in", "on", "at", "for", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "our", "their", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "do", "does",
    "did", "have", "has", "had", "will", "would", "can", "could", "should", "shall", "may", "might", "must",
    "not", "no", "yes", "yeah", "okay", "ok", "um", "uh", "like", "just", "really", "very", "also", "there",
    "here", "about", "all", "some", "any", "more", "most", "other", "into", "out", "up", "down", "over",
    "again", "too", "than", "only", "own", "same", "now", "well", "know", "think", "going", "get", "got",
    "go", "thing", "things", "lot", "kind", "sort", "mean", "right", "actually", "something", "anything",
    "everything", "one", "way", "time", "say", "said", "see", "let", "make", "want", "need", "oh", "hmm",
    "yep", "sure", "thank", "thanks", "hello", "hi", "bye", "gonna", "wanna", "don't", "it's", "i'm", "that's",
}
