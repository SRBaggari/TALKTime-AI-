"""Deterministic Devanagari → native Indic script transliteration.

Whisper frequently writes Telugu, Tamil, Kannada, Malayalam, Bengali, Gujarati
and Punjabi speech in *Devanagari* (a training-data artefact), even when it
correctly identifies the language. The Unicode Brahmic blocks are ISCII-aligned
(same offset for corresponding letters), so the script can be restored with a
fixed mapping. Characters with no counterpart in the target block (e.g. Tamil
has no aspirated consonants) fall back to the closest base letter.

The original Whisper text is always kept alongside, and the line is marked as
transliterated — this is a script conversion, not a correction of the words.
"""
from __future__ import annotations

import unicodedata

from core.language_utils import script_profile

DEVANAGARI = 0x0900
TARGET_BLOCKS = {"te": 0x0C00, "ta": 0x0B80, "kn": 0x0C80, "ml": 0x0D00, "bn": 0x0980, "as": 0x0980,
                 "gu": 0x0A80, "pa": 0x0A00, "or": 0x0B00}
SCRIPT_NAME = {"te": "Telugu", "ta": "Tamil", "kn": "Kannada", "ml": "Malayalam", "bn": "Bengali", "as": "Bengali",
               "gu": "Gujarati", "pa": "Gurmukhi", "or": "Oriya"}
# Languages whose native script *is* Devanagari — never transliterate these.
DEVANAGARI_LANGS = {"hi", "mr", "ne", "sa"}
# Devanagari letters absent from a target script → closest base letter (offset from block start).
_FALLBACK = {
    0x16: 0x15, 0x18: 0x17, 0x1B: 0x1A, 0x1D: 0x1C, 0x20: 0x1F, 0x22: 0x21, 0x25: 0x24, 0x27: 0x26,
    0x2B: 0x2A, 0x2D: 0x2C, 0x36: 0x37, 0x39: 0x38,
}
# Second fallback: voiced → unvoiced (Tamil has no separate voiced stops).
_VOICED = {0x17: 0x15, 0x1C: 0x1A, 0x21: 0x1F, 0x26: 0x24, 0x2C: 0x2A}
_SHARED = {0x0964, 0x0965}  # danda / double danda are used by all Indic scripts


def _assigned(cp: int) -> bool:
    return bool(unicodedata.name(chr(cp), ""))


def devanagari_to(text: str, lang: str) -> str:
    base = TARGET_BLOCKS[lang]
    out = []
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F and cp not in _SHARED:
            off = cp - DEVANAGARI
            if lang == "ta" and off == 0x02:  # anusvara is written as ம் in Tamil
                out.append("ம்")
                continue
            target = base + off
            for table in (_FALLBACK, _VOICED):  # e.g. Tamil gha → ga (absent) → ka
                if not _assigned(target) and off in table:
                    off = table[off]
                    target = base + off
            if _assigned(target):
                out.append(chr(target))
                continue
            if unicodedata.category(ch).startswith("M"):  # unmappable sign (e.g. nukta) → drop
                continue
            out.append(ch)  # keep original letter rather than invent one
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def needs_transliteration(text: str, lang: str | None) -> bool:
    if not lang or lang in DEVANAGARI_LANGS or lang not in TARGET_BLOCKS:
        return False
    prof = script_profile(text)
    return prof.get("Devanagari", 0) >= 0.5


def normalize_script(text: str, lang: str | None) -> tuple[str, str | None]:
    """Return ``(text_in_native_script, original_or_None)``."""
    if needs_transliteration(text, lang):
        return devanagari_to(text, lang), text
    return text, None
