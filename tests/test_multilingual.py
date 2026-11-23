"""Multilingual support: language lists, script tagging, code-switching, engine options."""
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from core import language_utils as lu
from core import transcription_engine as te
from core.conversation_summary import extract_topics
from core.question_analysis import detect_question
from core.sentiment_analysis import analyze_sentiment
from tests.conftest import tseg
from utils.validators import SUPPORTED_LANGUAGES, validate_language

TELUGU = "మీటింగ్ రేపు ఉంది"
HINDI = "क्या आप कल आएंगे"
TAMIL = "நாளை கூட்டம் உள்ளது"


def test_priority_languages_listed_first():
    codes = list(SUPPORTED_LANGUAGES)
    assert codes[:5] == ["auto", "en", "te", "hi", "ta"]
    assert len(codes) == len(lu.WHISPER_LANGUAGES) + 1


def test_unsupported_language_rejected_clearly():
    with pytest.raises(ValueError, match="not supported by Whisper"):
        validate_language("xx")
    assert validate_language("te") == "te"


@pytest.mark.parametrize("text,script", [(TELUGU, "Telugu"), (HINDI, "Devanagari"), (TAMIL, "Tamil"),
                                          ("The meeting is tomorrow", "Latin")])
def test_script_detection(text, script):
    assert next(iter(lu.script_profile(text))) == script


def test_code_switch_detected_in_telugu_english_segment():
    tag = lu.tag_segment("మీటింగ్ రేపు ఉంది, please send the report today")
    assert tag["code_switch"] is True
    assert set(tag["scripts"]) == {"Telugu", "Latin"}
    assert lu.tag_segment("please send the report today")["code_switch"] is False


def test_summarize_languages_reports_mix():
    segs = [tseg(0, 2, TELUGU + " ok ok"), tseg(2, 4, "please send the final budget report"), tseg(4, 6, TELUGU)]
    rep = lu.summarize_languages(segs, "te", [("te", 0.81), ("en", 0.12), ("hi", 0.03)])
    assert rep["detected_name"] == "Telugu"
    assert rep["script_mix"]["Telugu"] > 0 and rep["script_mix"]["Latin"] > 0
    assert rep["code_switching"] is True
    assert "Latin" in rep["secondary_scripts"]
    assert rep["top_probabilities"][0] == ("te", 0.81)


def test_feature_support_matrix_is_honest():
    te_support = lu.feature_support("te")
    assert te_support["transcription"] and te_support["speaker_analytics"] and te_support["questions_punctuation"]
    assert not te_support["sentiment"] and not te_support["topics_spacy"]
    assert lu.feature_support("en")["sentiment"]
    assert not lu.feature_support("xx")["transcription"]


def test_nlp_on_non_english_text():
    assert detect_question("మీరు రేపు వస్తారా?")[0]  # punctuation-based detection works for any language
    assert analyze_sentiment([tseg(0, 1, TELUGU, "Speaker 1")], "te")["available"] is False
    topics = extract_topics([tseg(0, 1, "బడ్జెట్ రిపోర్ట్ బడ్జెట్", "Speaker 1"),
                             tseg(1, 2, "బడ్జెట్ చర్చ", "Speaker 2")], language="te")
    assert topics and topics[0]["topic"] == "బడ్జెట్"


def test_engine_passes_multilingual_and_never_translates():
    model = mock.Mock()
    info = SimpleNamespace(language="te", language_probability=0.8, duration_after_vad=4.0,
                           all_language_probs=[("en", 0.1), ("te", 0.8), ("hi", 0.05)])
    model.transcribe.return_value = (iter([]), info)
    eng = te.TranscriptionEngine("small", "int8", 2)
    te.TranscriptionEngine._models[eng.key] = model
    try:
        r = eng.transcribe(np.zeros(16000, dtype=np.float32), language=None, multilingual=True)
    finally:
        te.TranscriptionEngine._models.clear()
    kw = model.transcribe.call_args.kwargs
    assert kw["multilingual"] is True and kw["task"] == "transcribe"
    assert r.language_probs[0] == ("te", 0.8) and r.multilingual
    assert r.load_seconds == 0.0  # model was already in memory


def test_fixed_language_disables_per_segment_detection():
    model = mock.Mock()
    model.transcribe.return_value = (iter([]), SimpleNamespace(language="hi", language_probability=1.0))
    eng = te.TranscriptionEngine("small", "int8", 2)
    te.TranscriptionEngine._models[eng.key] = model
    try:
        eng.transcribe(np.zeros(16000, dtype=np.float32), language="hi", multilingual=True)
    finally:
        te.TranscriptionEngine._models.clear()
    assert model.transcribe.call_args.kwargs["multilingual"] is False


# ---------------------------------------------------------------- script normalisation

def test_devanagari_telugu_is_restored_to_telugu_script():
    from core.transliteration import normalize_script

    text, original = normalize_script("मार्केटिंग खर्चु चाला एक्कुवगा उंदी।", "te")
    assert original == "मार्केटिंग खर्चु चाला एक्कुवगा उंदी।"
    assert next(iter(lu.script_profile(text))) == "Telugu"
    assert text.startswith("మార్కే") and text.endswith("।")


def test_hindi_and_english_never_transliterated():
    from core.transliteration import normalize_script

    assert normalize_script("नमस्ते दोस्तों", "hi") == ("नमस्ते दोस्तों", None)
    assert normalize_script("hello there", "te") == ("hello there", None)


def test_tamil_fallbacks_never_leave_devanagari():
    from core.transliteration import devanagari_to

    out = devanagari_to("गांधी बजट घर", "ta")
    assert not any(0x0900 <= ord(c) <= 0x097F for c in out)
    assert lu.script_profile(out) == {"Tamil": 1.0}


# ---------------------------------------------------------------- per-region code-switching

def test_build_chunks_pads_splits_and_keeps_timeline():
    chunks = te.build_chunks([(1.0, 2.0), (2.3, 3.0), (10.0, 70.0)], duration=80.0)
    assert chunks[0] == (0.8, 2.2) and chunks[1][0] == 2.2  # no overlap after padding
    assert all(e - s <= te.MAX_CHUNK_SECONDS + 1e-9 for s, e in chunks)
    assert chunks[-1][1] == pytest.approx(70.2)


def test_region_mode_detects_language_per_region_and_offsets_time():
    model = mock.Mock()
    model.detect_language.side_effect = [
        ("te", 0.9, [("te", 0.9), ("en", 0.08)]),   # whole recording
        ("te", 0.95, []), ("en", 0.97, []),          # region 1, region 2
    ]

    def fake_transcribe(piece, language=None, **kw):
        text = "मनं एमी चेयाली" if language == "te" else "Please send the report."
        seg = SimpleNamespace(start=0.1, end=1.5, text=text, avg_logprob=-0.2, no_speech_prob=0.01, words=[])
        return iter([seg]), SimpleNamespace(language=language)

    model.transcribe.side_effect = fake_transcribe
    eng = te.TranscriptionEngine("small", "int8", 2)
    te.TranscriptionEngine._models[eng.key] = model
    try:
        r = eng.transcribe(np.zeros(16000 * 10, dtype=np.float32), multilingual=True, regions=[(1.0, 3.0), (5.0, 8.0)])
    finally:
        te.TranscriptionEngine._models.clear()
    assert [s.language for s in r.segments] == ["te", "en"]
    assert r.segments[0].start == pytest.approx(0.8 + 0.1) and r.segments[1].start == pytest.approx(4.8 + 0.1)
    assert next(iter(lu.script_profile(r.segments[0].text))) == "Telugu"  # Devanagari output restored
    assert r.segments[0].original_text == "मनं एमी चेयाली" and r.transliterated_segments == 1
    assert set(r.language_distribution) == {"te", "en"} and r.language == "en"  # en region is longer
    rep = lu.summarize_languages(r.segments, r.language)
    assert set(rep["languages_spoken"]) == {"te", "en"} and rep["code_switching"]
    assert rep["transliterated_segments"] == 1


def test_short_region_uses_recording_language():
    model = mock.Mock()
    model.detect_language.return_value = ("hi", 0.8, [("hi", 0.8)])
    model.transcribe.return_value = (iter([]), SimpleNamespace(language="hi"))
    eng = te.TranscriptionEngine("small", "int8", 2)
    te.TranscriptionEngine._models[eng.key] = model
    try:
        eng.transcribe(np.zeros(16000 * 5, dtype=np.float32), multilingual=True, regions=[(1.0, 1.4)])
    finally:
        te.TranscriptionEngine._models.clear()
    assert model.detect_language.call_count == 1  # only the recording-level detection
    assert model.transcribe.call_args.kwargs["language"] == "hi"
