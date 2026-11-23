from pathlib import Path

import config
from database import repository
from services import storage_service
from services.analysis_service import compute_metrics
from tests.conftest import seg, tseg


def _make_recording(**kw):
    fields = dict(title="Test call", original_filename="call.wav", status="completed", duration=20.0)
    fields.update(kw)
    return repository.create_recording(**fields)


def _analysed_recording():
    rid = _make_recording()
    diar = [seg(0, 8, "Speaker 1"), seg(7, 15, "Speaker 2")]
    tr = [tseg(0, 8, "Hello, how are you today?", "Speaker 1"), tseg(8, 15, "I am fine, thanks.", "Speaker 2")]
    metrics = compute_metrics(diar, tr, 20.0, config.AppSettings(), "en")
    repository.save_analysis(rid, diar, tr, metrics)
    return rid


def test_create_get_update_list(clean_db):
    rid = _make_recording(status="pending")
    assert repository.get_recording(rid)["title"] == "Test call"
    repository.update_recording(rid, status="completed", duration=42.0)
    rec = repository.get_recording(rid)
    assert rec["status"] == "completed" and rec["duration"] == 42.0
    assert [r["id"] for r in repository.list_recordings()] == [rid]


def test_save_and_load_analysis(clean_db):
    rid = _analysed_recording()
    data = repository.load_analysis(rid)
    assert [s["label"] for s in data["speakers"]] == ["Speaker 1", "Speaker 2"]
    assert len(data["diarization"]) == 2 and len(data["transcript"]) == 2
    assert data["metrics"]["speaking"]["speakers"]["Speaker 1"]["total_time"] == 8
    assert data["recording"]["num_speakers"] == 2
    assert data["speakers"][0]["color"] == config.SPEAKER_COLORS[0]


def test_rename_speaker_survives_reanalysis(clean_db):
    rid = _analysed_recording()
    repository.rename_speaker(rid, "Speaker 1", "Alice")
    data = repository.load_analysis(rid)
    repository.save_analysis(rid, data["diarization"], data["transcript"], data["metrics"])
    assert repository.speaker_names(rid)["Speaker 1"] == "Alice"


def test_delete_recording_cascades_and_removes_files(clean_db):
    rid = _analysed_recording()
    audio = config.UPLOAD_DIR / f"{rid}.wav"
    audio.write_bytes(b"RIFF")
    report = config.REPORTS_DIR / f"{rid}.pdf"
    report.write_bytes(b"%PDF")
    repository.update_recording(rid, stored_path=str(audio))
    repository.add_report(rid, "pdf", str(report))
    removed = storage_service.delete_recording(rid)
    assert removed == 2 and not audio.exists() and not report.exists()
    assert repository.get_recording(rid) is None
    from sqlalchemy import func, select

    from database.db import session_scope
    from database.models import Speaker, TranscriptSegmentRow

    with session_scope() as s:
        assert s.scalar(select(func.count()).select_from(Speaker).where(Speaker.recording_id == rid)) == 0
        assert s.scalar(select(func.count()).select_from(TranscriptSegmentRow)
                        .where(TranscriptSegmentRow.recording_id == rid)) == 0


def test_delete_all(clean_db):
    _analysed_recording()
    _analysed_recording()
    storage_service.delete_all_data()
    assert repository.list_recordings() == []


def test_dashboard_stats(clean_db):
    _analysed_recording()
    _make_recording(status="failed")
    st = repository.dashboard_stats()
    assert st["recordings"] == 1 and st["total_speakers"] == 2 and st["failed"] == 1
    assert st["total_duration"] == 20.0


def test_settings_roundtrip(clean_db):
    s = config.AppSettings()
    s.whisper_model = "base"
    s.min_silence = 2.5
    s.ollama_enabled = True
    repository.save_app_settings(s)
    loaded = repository.load_app_settings()
    assert loaded.whisper_model == "base" and loaded.min_silence == 2.5 and loaded.ollama_enabled is True
    repository.save_app_settings(config.AppSettings())


def test_retention_deletes_old(clean_db):
    from datetime import datetime, timedelta, timezone

    rid = _make_recording(created_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=40))
    keep = _make_recording()
    assert storage_service.apply_retention(30) == 1
    assert repository.get_recording(rid) is None and repository.get_recording(keep)


def test_save_upload_rejects_invalid(clean_db):
    path, result = storage_service.save_upload("x.exe", b"MZ" + b"\x00" * 100, 10)
    assert not result.ok and path == Path()
