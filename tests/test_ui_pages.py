"""Smoke-test every Streamlit page with Streamlit's AppTest (no browser)."""
import pytest
from streamlit.testing.v1 import AppTest

import config
from database import repository
from services.analysis_service import compute_metrics
from tests.conftest import ROOT, seg, tseg, words_for

PAGES = ["views/dashboard.py", "views/upload_audio.py", "views/speaker_analysis.py", "views/conversation_timeline.py",
         "views/conversation_intelligence.py", "views/audio_language.py", "views/transcript_search.py",
         "views/call_comparison.py", "views/reports.py", "views/processing_monitor.py", "views/settings.py"]


def _seed(tone_wav, title):
    rid = repository.create_recording(title=title, original_filename="call.wav", status="completed", duration=6.0,
                                      processed_path=str(tone_wav), language="en", processing_seconds=1.2)
    diar = [seg(0, 2.2, "Speaker 1"), seg(2.0, 4.5, "Speaker 2"), seg(4.6, 6, "Speaker 3")]
    text = ["Can we review the budget today?", "Yes, I will send the budget report by Friday.", "Great, that sounds good."]
    tr = [tseg(s.start, s.end, t, s.speaker, words=words_for(s.start, s.end, t)) for s, t in zip(diar, text)]
    tr[1].uncertain = True
    import soundfile as sf

    from core.audio_quality import analyze_quality
    from core.vad import detect_speech

    audio, sr = sf.read(tone_wav, dtype="float32")
    vad = detect_speech(audio, sr, method="energy")
    quality = analyze_quality(audio, sr, [tuple(x) for x in vad["speech"]], {"sample_rate": 44100, "channels": 2})
    m = compute_metrics(diar, tr, 6.0, config.AppSettings(), "en", previous={"quality": quality}, vad=vad,
                        audio=audio, sr=sr, language_probs=[("en", 0.95), ("te", 0.03)])
    m["processing"] = {"total_seconds": 3.0, "stages": {"decode": 0.1, "transcription": 2.0}, "audio_duration": 6.0,
                       "realtime_factor": 0.5, "model_load_seconds": {"transcription": 1.0},
                       "resources": {"available": True, "peak_rss_mb": 800.0, "mean_cpu_percent": 35.0}}
    repository.save_analysis(rid, diar, tr, m)
    return rid


def _run(page):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    if page != "views/dashboard.py":
        at.switch_page(page)
        at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_empty(clean_db, page):
    at = _run(page)
    assert not at.exception, [e.message for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_data(clean_db, tone_wav, page):
    _seed(tone_wav, "Budget review")
    _seed(tone_wav, "Second call")
    at = _run(page)
    assert not at.exception, [e.message for e in at.exception]


def test_dashboard_shows_database_stats(clean_db, tone_wav):
    _seed(tone_wav, "Budget review")
    at = _run("views/dashboard.py")
    html = " ".join(m.value for m in at.markdown)
    assert "Recordings analysed" in html and "Budget review" in html


def test_delete_requires_typed_confirmation(clean_db, tone_wav):
    """Clicking delete opens a confirmation dialog; nothing is deleted and the confirm button stays
    disabled until the phrase is typed. (Typing inside a dialog is a fragment rerun in the browser,
    which AppTest cannot simulate; deletion itself is covered by storage-service tests.)"""
    rid = _seed(tone_wav, "To delete")
    at = _run("views/settings.py")
    next(b for b in at.button if b.label == "Delete recording").click().run()
    assert repository.get_recording(rid) is not None
    confirm = next(b for b in at.button if b.label == "Delete permanently")
    assert confirm.disabled
    assert any(t.label.startswith("Type DELETE") for t in at.text_input)
