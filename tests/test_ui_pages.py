"""Smoke-test every Streamlit page with Streamlit's AppTest (no browser)."""
import pytest
from streamlit.testing.v1 import AppTest

import config
from database import repository
from services.analysis_service import compute_metrics
from tests.conftest import ROOT, seg, tseg, words_for

PAGES = ["pages/dashboard.py", "pages/upload_audio.py", "pages/speaker_analysis.py", "pages/conversation_timeline.py",
         "pages/conversation_intelligence.py", "pages/call_comparison.py", "pages/reports.py", "pages/settings.py"]


def _seed(tone_wav, title):
    rid = repository.create_recording(title=title, original_filename="call.wav", status="completed", duration=6.0,
                                      processed_path=str(tone_wav), language="en", processing_seconds=1.2)
    diar = [seg(0, 2.2, "Speaker 1"), seg(2.0, 4.5, "Speaker 2"), seg(4.6, 6, "Speaker 3")]
    text = ["Can we review the budget today?", "Yes, I will send the budget report by Friday.", "Great, that sounds good."]
    tr = [tseg(s.start, s.end, t, s.speaker, words=words_for(s.start, s.end, t)) for s, t in zip(diar, text)]
    tr[1].uncertain = True
    repository.save_analysis(rid, diar, tr, compute_metrics(diar, tr, 6.0, config.AppSettings(), "en"))
    return rid


def _run(page):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    if page != "pages/dashboard.py":
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
    at = _run("pages/dashboard.py")
    html = " ".join(m.value for m in at.markdown)
    assert "Recordings analysed" in html and "Budget review" in html
