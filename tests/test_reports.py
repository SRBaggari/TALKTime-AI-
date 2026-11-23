import io

import pandas as pd

import config
from core.report_generator import generate_pdf
from database import repository
from services import export_service
from services.analysis_service import compute_metrics
from tests.conftest import seg, tseg


def _data(clean=True):
    rid = repository.create_recording(title="Weekly sync", original_filename="sync.m4a", status="completed",
                                      duration=30.0, language="en", language_probability=0.98)
    diar = [seg(0, 10, "Speaker 1"), seg(9, 20, "Speaker 2"), seg(21, 28, "Speaker 3")]
    tr = [tseg(0, 10, "What is the status of the release? It is important.", "Speaker 1"),
          tseg(10, 20, "We decided to ship on Friday. I will update the release notes.", "Speaker 2"),
          tseg(21, 28, "Great, thanks everyone & see you <soon>.", "Speaker 3")]
    m = compute_metrics(diar, tr, 30.0, config.AppSettings(), "en")
    repository.save_analysis(rid, diar, tr, m)
    repository.rename_speaker(rid, "Speaker 1", "Priya")
    return rid, repository.load_analysis(rid)


def test_pdf_generation(clean_db, tmp_path):
    _, data = _data()
    out = generate_pdf(data, tmp_path / "r.pdf")
    raw = out.read_bytes()
    assert raw.startswith(b"%PDF") and len(raw) > 3000


def test_pdf_without_transcript_and_empty_analysis(clean_db, tmp_path):
    rid = repository.create_recording(title="Empty", original_filename="e.wav", status="completed", duration=5.0)
    m = compute_metrics([seg(0, 1, "Speaker 1")], [], 5.0, config.AppSettings(), "en")
    repository.save_analysis(rid, [seg(0, 1, "Speaker 1")], [], m)
    out = generate_pdf(repository.load_analysis(rid), tmp_path / "e.pdf", include_transcript=False)
    assert out.read_bytes().startswith(b"%PDF")


def test_csv_exports(clean_db):
    _, data = _data()
    tdf = pd.read_csv(io.BytesIO(export_service.to_csv_bytes(export_service.transcript_dataframe(data))), encoding="utf-8-sig")
    assert list(tdf["speaker_name"]) == ["Priya", "Speaker 2", "Speaker 3"]
    sdf = export_service.speakers_dataframe(data)
    assert sdf["share_of_speaking_time_pct"].sum() == __import__("pytest").approx(100, abs=0.05)
    assert len(export_service.segments_dataframe(data)) == 3
    ev = export_service.events_dataframe(data)
    assert "start" in ev.columns
    assert b'"metrics"' in export_service.metrics_json_bytes(data)
