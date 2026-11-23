"""Languages, writing-script detection and per-language feature support.

Code-switching (e.g. Telugu–English) is detected from the *script* of the words
Whisper outputs: Telugu words appear in Telugu script, English words in Latin.
This is a measured property of the transcript text, not a model guess. Its
limit: languages that share a script (Hindi/Marathi, English/Spanish) cannot be
told apart this way.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Whisper language codes → names (the languages Whisper was trained on).
WHISPER_LANGUAGES: dict[str, str] = {
    "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish", "ru": "Russian", "ko": "Korean",
    "fr": "French", "ja": "Japanese", "pt": "Portuguese", "tr": "Turkish", "pl": "Polish", "ca": "Catalan",
    "nl": "Dutch", "ar": "Arabic", "sv": "Swedish", "it": "Italian", "id": "Indonesian", "hi": "Hindi",
    "fi": "Finnish", "vi": "Vietnamese", "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay",
    "cs": "Czech", "ro": "Romanian", "da": "Danish", "hu": "Hungarian", "ta": "Tamil", "no": "Norwegian",
    "th": "Thai", "ur": "Urdu", "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian", "la": "Latin",
    "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak", "te": "Telugu", "fa": "Persian",
    "lv": "Latvian", "bn": "Bengali", "sr": "Serbian", "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada",
    "et": "Estonian", "mk": "Macedonian", "br": "Breton", "eu": "Basque", "is": "Icelandic", "hy": "Armenian",
    "ne": "Nepali", "mn": "Mongolian", "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian", "sw": "Swahili",
    "gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala", "km": "Khmer", "sn": "Shona",
    "yo": "Yoruba", "so": "Somali", "af": "Afrikaans", "oc": "Occitan", "ka": "Georgian", "be": "Belarusian",
    "tg": "Tajik", "sd": "Sindhi", "gu": "Gujarati", "am": "Amharic", "yi": "Yiddish", "lo": "Lao",
    "uz": "Uzbek", "fo": "Faroese", "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen", "nn": "Nynorsk",
    "mt": "Maltese", "sa": "Sanskrit", "lb": "Luxembourgish", "my": "Myanmar", "bo": "Tibetan", "tl": "Tagalog",
    "mg": "Malagasy", "as": "Assamese", "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala", "ha": "Hausa",
    "ba": "Bashkir", "jw": "Javanese", "su": "Sundanese", "yue": "Cantonese",
}

# Languages listed first in the UI.
PRIORITY = ["en", "te", "hi", "ta", "kn", "ml", "mr", "bn", "gu", "pa", "ur"]

SCRIPTS = [
    ("Devanagari", 0x0900, 0x097F), ("Bengali", 0x0980, 0x09FF), ("Gurmukhi", 0x0A00, 0x0A7F),
    ("Gujarati", 0x0A80, 0x0AFF), ("Oriya", 0x0B00, 0x0B7F), ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F), ("Kannada", 0x0C80, 0x0CFF), ("Malayalam", 0x0D00, 0x0D7F),
    ("Sinhala", 0x0D80, 0x0DFF), ("Thai", 0x0E00, 0x0E7F), ("Arabic", 0x0600, 0x06FF),
    ("Hebrew", 0x0590, 0x05FF), ("Cyrillic", 0x0400, 0x04FF), ("Greek", 0x0370, 0x03FF),
    ("Hangul", 0xAC00, 0xD7AF), ("Hiragana/Katakana", 0x3040, 0x30FF), ("CJK", 0x4E00, 0x9FFF),
    ("Georgian", 0x10A0, 0x10FF), ("Armenian", 0x0530, 0x058F),
]
SCRIPT_LANGS = {
    "Latin": "Latin-script language", "Devanagari": "Hindi/Marathi/Nepali", "Bengali": "Bengali/Assamese",
    "Gurmukhi": "Punjabi", "Gujarati": "Gujarati", "Tamil": "Tamil", "Telugu": "Telugu", "Kannada": "Kannada",
    "Malayalam": "Malayalam", "Arabic": "Arabic/Urdu/Persian", "Cyrillic": "Cyrillic-script language",
    "CJK": "Chinese/Japanese", "Hangul": "Korean", "Hiragana/Katakana": "Japanese",
}
LANG_SCRIPT = {"en": "Latin", "te": "Telugu", "hi": "Devanagari", "mr": "Devanagari", "ne": "Devanagari",
               "ta": "Tamil", "kn": "Kannada", "ml": "Malayalam", "bn": "Bengali", "gu": "Gujarati",
               "pa": "Gurmukhi", "ur": "Arabic", "ar": "Arabic", "fa": "Arabic", "ru": "Cyrillic", "uk": "Cyrillic",
               "zh": "CJK", "ja": "CJK", "ko": "Hangul", "el": "Greek", "he": "Hebrew", "th": "Thai"}


def language_name(code: str | None) -> str:
    if not code:
        return "Unknown"
    return WHISPER_LANGUAGES.get(code, code)


def is_supported(code: str | None) -> bool:
    return code is None or code == "auto" or code in WHISPER_LANGUAGES


def ordered_language_options() -> list[str]:
    rest = sorted((c for c in WHISPER_LANGUAGES if c not in PRIORITY), key=lambda c: WHISPER_LANGUAGES[c])
    return ["auto"] + PRIORITY + rest


def char_script(ch: str) -> str | None:
    if not ch.isalpha():
        return None
    cp = ord(ch)
    for name, lo, hi in SCRIPTS:
        if lo <= cp <= hi:
            return name
    if "LATIN" in unicodedata.name(ch, ""):
        return "Latin"
    return "Other"


def script_profile(text: str) -> dict[str, float]:
    """Share of letters per script, e.g. {'Telugu': 0.7, 'Latin': 0.3}."""
    counts = Counter(s for s in (char_script(c) for c in text) if s)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.most_common()} if total else {}


_WORD = re.compile(r"\w+", re.UNICODE)


def word_scripts(text: str) -> Counter:
    out = Counter()
    for w in _WORD.findall(text):
        prof = script_profile(w)
        if prof:
            out[next(iter(prof))] += 1
    return out


def tag_segment(text: str, min_share: float = 0.15, min_words: int = 2) -> dict:
    """Script-based language tag for one transcript segment (code-switch aware)."""
    ws = word_scripts(text)
    total = sum(ws.values())
    if not total:
        return {"scripts": {}, "primary_script": None, "code_switch": False}
    shares = {k: v / total for k, v in ws.most_common()}
    present = [k for k, v in ws.items() if v / total >= min_share and v >= min_words]
    return {"scripts": {k: round(v, 3) for k, v in shares.items()}, "primary_script": next(iter(shares)),
            "code_switch": len(present) >= 2}


def summarize_languages(segments, detected: str | None, probs: list | None = None) -> dict:
    """Recording-level language report.

    * ``languages_spoken`` — seconds of transcribed speech per language, from each line's own
      language (set per speech region in code-switching mode). This is the primary signal.
    * ``script_mix`` — share of words per writing script (secondary, text-based).
    """
    spoken: Counter = Counter()
    transliterated = 0
    for seg in segments:
        lang = getattr(seg, "language", None) or detected
        if lang:
            spoken[lang] += max(seg.end - seg.start, 0.0)
        if getattr(seg, "original_text", None):
            transliterated += 1
    total_spoken = sum(spoken.values())
    languages_spoken = {k: {"name": language_name(k), "seconds": round(v, 2),
                            "share": round(v / total_spoken, 4) if total_spoken else 0.0}
                        for k, v in spoken.most_common()}
    multi_lang = len([v for v in languages_spoken.values() if v["share"] >= 0.1]) >= 2
    words_by_script: Counter = Counter()
    switch_segments = 0
    for seg in segments:
        tag = tag_segment(seg.text)
        seg_words = word_scripts(seg.text)
        words_by_script.update(seg_words)
        if tag["code_switch"]:
            switch_segments += 1
    total = sum(words_by_script.values())
    mix = {k: round(v / total, 3) for k, v in words_by_script.most_common()} if total else {}
    expected = LANG_SCRIPT.get(detected or "")
    secondary = [k for k, v in mix.items() if k != expected and v >= 0.05]
    return {
        "detected": detected,
        "detected_name": language_name(detected),
        "top_probabilities": [(c, round(p, 4)) for c, p in (probs or [])[:5]],
        "script_mix": mix,
        "script_languages": {k: SCRIPT_LANGS.get(k, k) for k in mix},
        "code_switch_segments": switch_segments,
        "code_switching": multi_lang or bool(switch_segments) or len([v for v in mix.values() if v >= 0.1]) >= 2,
        "languages_spoken": languages_spoken,
        "transliterated_segments": transliterated,
        "secondary_scripts": secondary,
        "method": "Language: Whisper language identification — once for the recording, or per speech region in "
                  "code-switching mode. Script mix: Unicode script of each transcribed word.",
    }


def feature_support(code: str | None) -> dict[str, bool]:
    """Which analyses work for a language (honest capability matrix)."""
    english = code in (None, "", "en")
    return {
        "transcription": is_supported(code),
        "speaker_analytics": True,       # audio-based — language independent
        "questions_punctuation": True,   # '?' from Whisper output
        "questions_grammar": english,    # English interrogative patterns
        "sentiment": english,            # VADER is English-only
        "topics_spacy": english,         # en_core_web_sm
        "insights_rules": english,       # English phrase patterns
    }
