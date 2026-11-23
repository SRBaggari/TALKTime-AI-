"""End-to-end pipeline with mocked ML engines, plus manual corrections."""
from unittest import mock

import pytest

import config
from core.diarization_engine import ModelUnavailableError
from core.types import SpeakerSegment, TranscriptionResult
from database import repository
from services import analysis_service as svc
from tests.conftest import tseg, words_for


def _recording(path):
    return repository.create_recording(title="Tones", original_filename="tones.wav", stored_path=str(path))


def _fake_transcription(*a, **k):
    return TranscriptionResult(
        segments=[tseg(0, 2, "Can we start the review now?", words=words_for(0, 2, "Can we start the review now?")),
                  tseg(3, 5, "Yes, the review looks great.", words=words_for(3, 5, "Yes, the review looks great."))],
        language="en", language_probability=0.99, duration=6.0, model_name="mock-whisper")


def _fake_diarization(*a, **k):
    return [SpeakerSegment(0, 2, "Speaker 1"), SpeakerSegment(3, 5, "Speaker 2")]


_real_detect_speech = svc.detect_speech


@pytest.fixture(autouse=True)
def energy_vad():
    """Synthetic tones are (correctly) not speech to Silero; use the deterministic energy VAD
    so pipeline plumbing can be tested. Real Silero behaviour is tested separately."""
    def fake(audio, sr, **kw):
        kw.pop("method", None)
        return _real_detect_speech(audio, sr, method="energy", **kw)
    with mock.patch.object(svc, "detect_speech", fake):
        yield


@pytest.fixture
def mocked_engines():
    with mock.patch.object(svc.TranscriptionEngine, "transcribe", _fake_transcription), \
         mock.patch.object(svc.DiarizationEngine, "diarize", _fake_diarization):
        yield


def test_full_pipeline(clean_db, tone_wav, mocked_engines):
    rid = _recording(tone_wav)
    progress = []
    metrics = svc.run_analysis(rid, svc.AnalysisOptions(), lambda f, m: progress.append(f))
    rec = repository.get_recording(rid)
    assert rec["status"] == "completed" and rec["num_speakers"] == 2
    assert rec["duration"] == pytest.approx(6.0, abs=0.01)
    assert rec["processing_seconds"] > 0
    assert metrics["speaking"]["speakers"]["Speaker 1"]["total_time"] == 2
    assert metrics["questions"]["questions"][0]["answered_by"] == "Speaker 2"
    assert metrics["silence"]["num_silences"] == 2  # 2-3 s and trailing 5-6 s
    assert progress[-1] == 1.0
    data = repository.load_analysis(rid)
    assert [t.speaker for t in data["transcript"]] == ["Speaker 1", "Speaker 2"]
    assert (config.PROCESSED_DIR / f"{rid}.wav").exists()


def test_single_speaker_mode(clean_db, tone_wav, mocked_engines):
    rid = _recording(tone_wav)
    metrics = svc.run_analysis(rid, svc.AnalysisOptions(diarization_mode="single"))
    assert list(metrics["speaking"]["speakers"]) == ["Speaker 1"]
    assert repository.get_recording(rid)["diarization_mode"] == "single"


def test_failed_transcription_marks_failed(clean_db, tone_wav):
    rid = _recording(tone_wav)
    with mock.patch.object(svc.TranscriptionEngine, "transcribe", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            svc.run_analysis(rid, svc.AnalysisOptions())
    rec = repository.get_recording(rid)
    assert rec["status"] == "failed" and "boom" in rec["error"]


def test_missing_model_marks_failed(clean_db, tone_wav):
    rid = _recording(tone_wav)
    with mock.patch.object(svc.TranscriptionEngine, "transcribe", _fake_transcription), \
         mock.patch("core.diarization_engine.local_diarization_path", return_value=None):
        svc.DiarizationEngine._pipeline = None
        with pytest.raises(ModelUnavailableError):
            svc.run_analysis(rid, svc.AnalysisOptions())
    assert "huggingface" in repository.get_recording(rid)["error"].lower()


def test_silent_recording_rejected(clean_db, tmp_path):
    import numpy as np
    import soundfile as sf

    p = tmp_path / "silent.wav"
    sf.write(p, np.zeros(32000, dtype=np.float32), 16000)
    rid = _recording(p)
    with pytest.raises(svc.AnalysisError, match="silent"):
        svc.run_analysis(rid, svc.AnalysisOptions())


def test_no_speech_detected(clean_db, tone_wav):
    rid = _recording(tone_wav)
    empty = TranscriptionResult([], "en", 0.5, 6.0)
    with mock.patch.object(svc.TranscriptionEngine, "transcribe", return_value=empty), \
         mock.patch.object(svc.DiarizationEngine, "diarize", return_value=[]):
        with pytest.raises(svc.AnalysisError, match="No speech"):
            svc.run_analysis(rid, svc.AnalysisOptions())


def test_reassign_and_merge_speakers(clean_db, tone_wav, mocked_engines):
    rid = _recording(tone_wav)
    svc.run_analysis(rid, svc.AnalysisOptions())
    m = svc.reassign_segment(rid, 1, "Speaker 1")
    assert m["speaking"]["speakers"]["Speaker 1"]["total_time"] == pytest.approx(4)
    assert "Speaker 2" not in m["speaking"]["speakers"]
    data = repository.load_analysis(rid)
    assert data["transcript_corrected"] == [False, True]

    rid2 = _recording(tone_wav)
    svc.run_analysis(rid2, svc.AnalysisOptions())
    m2 = svc.merge_speakers(rid2, "Speaker 2", "Speaker 1")
    assert list(m2["speaking"]["speakers"]) == ["Speaker 1"]
    assert m2["speaking"]["speakers"]["Speaker 1"]["total_time"] == pytest.approx(4)


def test_job_manager_runs_in_background(clean_db, tone_wav, mocked_engines):
    import time

    rid = _recording(tone_wav)
    svc.JobManager.start(rid, svc.AnalysisOptions())
    for _ in range(200):
        st = svc.JobManager.status(rid)
        if st["state"] != "running":
            break
        time.sleep(0.05)
    assert svc.JobManager.status(rid)["state"] == "completed"


def test_vad_skips_models_when_no_speech(clean_db, tone_wav):
    """Real Silero VAD: pure tones contain no speech, so expensive models must not run."""
    rid = _recording(tone_wav)
    transcribe = mock.Mock()
    with mock.patch.object(svc, "detect_speech", _real_detect_speech),          mock.patch.object(svc.TranscriptionEngine, "transcribe", transcribe):
        with pytest.raises(svc.AnalysisError, match="No speech detected by voice activity detection"):
            svc.run_analysis(rid, svc.AnalysisOptions())
    transcribe.assert_not_called()
    rec = repository.get_recording(rid)
    assert rec["model_info"]["failed_stage"] == "vad" and "Audio Quality" in rec["model_info"]["hint"]


def test_pipeline_records_quality_vad_language_and_performance(clean_db, tone_wav, mocked_engines):
    rid = _recording(tone_wav)
    m = svc.run_analysis(rid, svc.AnalysisOptions(multilingual=True))
    assert m["quality"]["score"] is not None and m["quality"]["format"]["sample_rate"] == 16000
    assert m["vad"]["method"] == "energy" and "comparison" in m["vad"]
    assert m["language"]["detected"] == "en" and m["language"]["feature_support"]["sentiment"]
    perf = m["processing"]
    assert set(perf["stages"]) >= {"decode", "vad", "quality", "transcription", "diarization", "analytics"}
    assert perf["resources"]["available"] and perf["resources"]["peak_rss_mb"] > 0
    assert perf["audio_duration"] == pytest.approx(6.0, abs=0.01)
    assert all("acoustic" in x for x in m["silence"]["silences"])
    assert "per_speaker" in m["alignment"] and "Speaker 1" in m["alignment"]["per_speaker"]
    assert m["participation"]["gini"] is not None and m["participation"]["dominance_indicators"]


def test_failure_records_stage_and_recovery_hint(clean_db, tone_wav):
    rid = _recording(tone_wav)
    with mock.patch.object(svc.TranscriptionEngine, "transcribe", side_effect=MemoryError()):
        with pytest.raises(MemoryError):
            svc.run_analysis(rid, svc.AnalysisOptions())
    info = repository.get_recording(rid)["model_info"]
    assert info["failed_stage"] == "transcription" and "memory" in info["hint"].lower()
    assert "decode" in info["stages"]


def test_recompute_keeps_quality_language_and_reclassifies_silence(clean_db, tone_wav, mocked_engines):
    rid = _recording(tone_wav)
    first = svc.run_analysis(rid, svc.AnalysisOptions())
    m = svc.reassign_segment(rid, 1, "Speaker 1")
    assert m["quality"] == first["quality"]
    assert m["language"]["top_probabilities"] == first["language"]["top_probabilities"]
    assert m["processing"]["total_seconds"] == first["processing"]["total_seconds"]
    assert all("acoustic" in x for x in m["silence"]["silences"])
