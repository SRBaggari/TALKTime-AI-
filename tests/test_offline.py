"""Offline readiness / missing-model handling."""
import os
from unittest import mock

import config
from services import model_manager as mm
from utils.validators import is_local_url, validate_language, validate_speaker_bounds


def test_telemetry_disabled_by_config():
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["PYANNOTE_METRICS_ENABLED"] in ("0", "false")


def test_startup_checks_report_missing_models(clean_db):
    with mock.patch.object(mm, "is_diarization_available", return_value=False), \
         mock.patch.object(mm, "is_whisper_available", return_value=False):
        checks = mm.startup_checks(config.AppSettings())
    by_name = {c.name: c for c in checks}
    diar = by_name["Diarization model (pyannote Community-1)"]
    assert not diar.ok and "README" in diar.fix
    assert not mm.offline_ready(checks)


def test_startup_checks_ready_when_models_present(clean_db):
    with mock.patch.object(mm, "is_diarization_available", return_value=True), \
         mock.patch.object(mm, "is_whisper_available", return_value=True), \
         mock.patch.object(mm.shutil, "which", return_value="ffmpeg"):
        checks = mm.startup_checks(config.AppSettings())
    assert by_ok(checks, "SQLite database") and by_ok(checks, "Storage permissions")
    assert mm.offline_ready(checks)


def by_ok(checks, name):
    return next(c for c in checks if c.name == name).ok


def test_download_diarization_requires_token(monkeypatch):
    for k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    try:
        mm.download_diarization()
        assert False, "should raise"
    except RuntimeError as exc:
        assert "HF_TOKEN" in str(exc)


def test_placeholder_token_ignored(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_your_token_here")
    assert config.get_hf_token() is None


def test_local_url_only():
    assert is_local_url("http://localhost:11434")
    assert is_local_url("http://127.0.0.1:11434")
    assert not is_local_url("http://192.168.1.5:11434")
    assert not is_local_url("https://api.example.com")


def test_validators():
    assert validate_language("auto") is None and validate_language("en") == "en"
    assert validate_speaker_bounds(0, 0) == (None, None)
    assert validate_speaker_bounds(2, 4) == (2, 4)
    for bad in ((3, 2), (-1, None)):
        try:
            validate_speaker_bounds(*bad)
            assert False
        except ValueError:
            pass


def test_cached_snapshot_accepts_partial_repo_with_required_files(tmp_path):
    """faster-whisper never downloads README/.gitattributes — the model must still count as installed."""
    from core.model_cache import cached_snapshot

    repo = tmp_path / "models--Systran--faster-whisper-small"
    snap = repo / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123")
    for f in ("model.bin", "config.json"):
        (snap / f).write_text("x")
    assert cached_snapshot("Systran/faster-whisper-small", ["model.bin", "config.json"], tmp_path) == snap
    assert cached_snapshot("Systran/faster-whisper-small", ["model.bin", "tokenizer.json"], tmp_path) is None
    assert cached_snapshot("Systran/faster-whisper-tiny", ["model.bin"], tmp_path) is None
