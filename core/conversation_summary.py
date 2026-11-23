"""Conversation intelligence: topics, key points, decisions, action items,
commitments and important statements — local and rule-based.

An optional local Ollama model can produce an abstractive summary. It is
disabled by default and restricted to localhost.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict

from core.nlp_utils import STOPWORDS, get_spacy, is_english, split_sentences
from core.types import TranscriptSegment
from utils.time_formatter import format_duration
from utils.validators import is_local_url

logger = logging.getLogger(__name__)

_DAYS = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|tonight|today|next week|next month|end of (the )?(day|week|month)|eod|eow)"
PATTERNS = {
    "decision": re.compile(
        r"\b(we('ve| have)? decided|(i|we)('ve| have)? agreed|we agree|let's go with|we('ll| will) go with|"
        r"(the )?final decision|decision is|settled on|approved|we're going with|going forward we|it's decided|"
        r"we('ll| will) (use|adopt|proceed|move forward))\b", re.I),
    "commitment": re.compile(
        r"\b(i('ll| will)|i'm going to|i am going to|i can (do|take|handle|send|share)|i promise|"
        r"i('ll| will) make sure|leave it with me|i('ll| will) take care|we('ll| will) make sure)\b", re.I),
    "action_item": re.compile(
        r"\b(action item|to-?do|follow[- ]up|need(s)? to|have to|has to|must|should (send|share|review|update|"
        r"check|prepare|schedule|call|email|finish|complete|fix)|can you|could you|please|assign(ed)?|"
        r"by " + _DAYS + r"|deadline|due)\b", re.I),
    "important": re.compile(
        r"\b(important|critical|crucial|key (point|issue|thing)|priority|urgent|concern(ed)?|risk|"
        r"problem|issue|blocker|main (point|goal|thing)|bottom line|the point is|keep in mind|remember)\b", re.I),
}
_DEADLINE = re.compile(r"\b(by |before |on |until )?" + _DAYS + r"\b", re.I)
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")


def _sentences(segments: list[TranscriptSegment]) -> list[dict]:
    out = []
    for seg in sorted(segments, key=lambda s: s.start):
        for sent in split_sentences(seg.text.replace("[unclear]", "").strip()):
            if len(sent.split()) >= 3:
                out.append({"text": sent, "speaker": seg.speaker, "start": round(seg.start, 3)})
    return out


def extract_topics(segments: list[TranscriptSegment], top_n: int = 8, language: str | None = "en") -> list[dict]:
    """Rank candidate topic phrases (spaCy noun chunks + entities, else keyword frequency)."""
    counts: Counter = Counter()
    first_seen: dict[str, float] = {}
    speakers: dict[str, set] = defaultdict(set)
    nlp = get_spacy() if is_english(language) else None
    segs = sorted(segments, key=lambda s: s.start)

    def add(term: str, seg: TranscriptSegment, weight: float = 1.0):
        term = term.strip().lower()
        if len(term) < 3 or term in STOPWORDS:
            return
        counts[term] += weight
        first_seen.setdefault(term, seg.start)
        speakers[term].add(seg.speaker)

    if nlp is not None:
        for seg, doc in zip(segs, nlp.pipe((s.text for s in segs), batch_size=64)):
            for chunk in doc.noun_chunks:
                toks = [t for t in chunk if not t.is_stop and not t.is_punct and t.is_alpha and t.pos_ != "PRON"]
                if toks:
                    add(" ".join(t.lemma_ for t in toks), seg, 1.0 + 0.5 * (len(toks) - 1))
            for ent in doc.ents:
                if ent.label_ in {"ORG", "PRODUCT", "GPE", "PERSON", "EVENT", "WORK_OF_ART", "LAW", "NORP", "FAC"}:
                    add(ent.text, seg, 1.5)
    else:
        for seg in segs:
            for tok in _TOKEN.findall(seg.text):
                if tok.lower() not in STOPWORDS:
                    add(tok, seg)

    # Merge single words into multi-word phrases that contain them.
    ranked = []
    for term, c in counts.most_common(top_n * 4):
        if c < 2:
            continue
        if any(term != t and term in t.split() and counts[t] >= c * 0.6 for t, _ in ranked):
            continue
        ranked.append((term, c))
        if len(ranked) >= top_n:
            break
    return [{"topic": t, "mentions": round(c, 1), "first_mention": first_seen[t], "speakers": sorted(s for s in speakers[t] if s)}
            for t, c in ranked]


def extract_insights(segments: list[TranscriptSegment], language: str | None = "en") -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {k: [] for k in PATTERNS}
    if not is_english(language):
        return out
    for s in _sentences(segments):
        for kind, pat in PATTERNS.items():
            m = pat.search(s["text"])
            if not m:
                continue
            if kind == "action_item" and "?" in s["text"] and not re.search(r"\b(can|could) you\b", s["text"], re.I):
                continue
            deadline = _DEADLINE.search(s["text"])
            conf = 0.7 if kind in ("decision", "commitment") else 0.6
            if deadline and kind in ("action_item", "commitment"):
                conf += 0.15
            out[kind].append({**s, "match": m.group(0), "deadline": deadline.group(0) if deadline else None,
                              "confidence": round(min(conf, 0.9), 2), "method": "rule-based pattern"})
    return out


def key_points(segments: list[TranscriptSegment], topics: list[dict], max_points: int = 6) -> list[dict]:
    """Extractive key points: sentences scored by topic-term density and cue phrases."""
    sents = _sentences(segments)
    if not sents:
        return []
    weights = {t["topic"]: t["mentions"] for t in topics}
    scored = []
    for i, s in enumerate(sents):
        low = s["text"].lower()
        score = sum(w for term, w in weights.items() if term in low)
        if PATTERNS["important"].search(s["text"]) or PATTERNS["decision"].search(s["text"]):
            score += 3
        if re.search(r"\d", s["text"]):
            score += 0.5
        n_words = len(s["text"].split())
        if n_words < 6:
            score *= 0.5
        elif n_words > 45:
            score *= 0.7
        if score > 0:
            scored.append((score, i, s))
    top = sorted(scored, key=lambda x: -x[0])[:max_points]
    return [dict(s, score=round(sc, 2)) for sc, _, s in sorted(top, key=lambda x: x[1])]


_LABEL = re.compile(r"\bSpeaker \d+\b")


def with_display_names(text: str, names: dict[str, str]) -> str:
    """Replace stable labels ("Speaker 2") in stored text with user-chosen display names."""
    return _LABEL.sub(lambda m: names.get(m.group(0), m.group(0)), text or "")


def build_summary(stats: dict, topics: list[dict], questions: dict, insights: dict, participation: dict) -> str:
    """Deterministic, fact-based overview built only from measured/extracted results."""
    st = stats["speaking"]
    n = st["num_speakers"]
    parts = [f"This {format_duration(st['recording_duration'])} recording contains {n} detected speaker{'s' if n != 1 else ''}"]
    if n >= 2 and participation.get("dominant_speaker"):
        dom = participation["dominant_speaker"]
        parts[-1] += f"; {dom} spoke the most ({participation['dominant_share'] * 100:.0f}% of speaking time)."
    else:
        parts[-1] += "."
    s = st["summary"]
    parts.append(f"Speech covers {s['speech_ratio'] * 100:.0f}% of the recording with {format_duration(s['overlap_duration'])} of overlapping speech.")
    if topics:
        parts.append("Frequently discussed: " + ", ".join(t["topic"] for t in topics[:5]) + ".")
    if questions.get("total_questions"):
        parts.append(f"{questions['total_questions']} question(s) were detected; {questions['possibly_unanswered']} may have gone unanswered.")
    dec, act = len(insights.get("decision", [])), len(insights.get("action_item", []))
    if dec or act:
        parts.append(f"Rule-based extraction found {dec} possible decision(s) and {act} possible action item(s).")
    return " ".join(parts)


# --------------------------------------------------------------------------- Ollama (optional)

def ollama_status(url: str, timeout: float = 2.0) -> dict:
    if not is_local_url(url):
        return {"available": False, "reason": "Only a local Ollama URL (localhost) is permitted."}
    try:
        import requests

        r = requests.get(url.rstrip("/") + "/api/tags", timeout=timeout)
        r.raise_for_status()
        return {"available": True, "models": [m.get("name") for m in r.json().get("models", [])]}
    except Exception as exc:
        return {"available": False, "reason": f"Ollama not reachable at {url} ({exc.__class__.__name__})."}


def ollama_summarize(segments: list[TranscriptSegment], url: str, model: str, max_chars: int = 24_000,
                     timeout: float = 600.0) -> str:
    if not is_local_url(url):
        raise ValueError("Ollama URL must be local (localhost/127.0.0.1).")
    import requests

    lines = [f"{s.speaker}: {s.text}" for s in sorted(segments, key=lambda s: s.start)]
    transcript = "\n".join(lines)[:max_chars]
    prompt = ("Summarise this conversation transcript. Give: 1) a 3-5 sentence summary, 2) main topics, "
              "3) decisions, 4) action items with owners. Only use information in the transcript.\n\n" + transcript)
    r = requests.post(url.rstrip("/") + "/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
                      timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "").strip()
