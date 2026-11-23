"""Optional integration test with REAL models and a REAL recording.

Skipped unless enabled:
    set TALKTIME_INTEGRATION=1
    set TALKTIME_TEST_AUDIO=C:\\path\\to\\conversation.mp3
    set TALKTIME_TEST_SPEAKERS=3          (optional: expected speaker count)
    .venv\\Scripts\\python -m pytest tests\\test_integration.py -s
"""
import os

import pytest

import config
from core.diarization_engine import is_diarization_available
from core.transcription_engine import is_whisper_available
from database import repository
from services import storage_service
from services.analysis_service import AnalysisOptions, run_analysis

AUDIO = os.environ.get("TALKTIME_TEST_AUDIO")
pytestmark = pytest.mark.skipif(os.environ.get("TALKTIME_INTEGRATION") != "1" or not AUDIO,
                                reason="integration test disabled (set TALKTIME_INTEGRATION=1 and TALKTIME_TEST_AUDIO)")


@pytest.mark.parametrize("mode", ["single", "pyannote"])
def test_real_pipeline(clean_db, mode):
    if not is_whisper_available("small"):
        pytest.skip("Whisper small not downloaded")
    if mode == "pyannote" and not is_diarization_available():
        pytest.skip("Diarization model not downloaded")
    data = open(AUDIO, "rb").read()
    path, res = storage_service.save_upload(os.path.basename(AUDIO), data, 2000)
    assert res.ok
    rid = repository.create_recording(title="integration", original_filename=os.path.basename(AUDIO), stored_path=str(path))
    s = config.AppSettings()
    m = run_analysis(rid, AnalysisOptions(diarization_mode=mode, settings=s))
    rec = repository.get_recording(rid)
    assert rec["status"] == "completed"
    tr = repository.load_analysis(rid)["transcript"]
    assert tr and all(t.text for t in tr)
    expected = os.environ.get("TALKTIME_TEST_SPEAKERS")
    print(f"\n[{mode}] {rec['duration']:.1f}s audio in {rec['processing_seconds']:.1f}s, "
          f"{m['speaking']['num_speakers']} speakers: "
          + ", ".join(f"{k}={v['total_time']:.1f}s" for k, v in m["speaking"]["speakers"].items()))
    if mode == "pyannote" and expected:
        assert m["speaking"]["num_speakers"] == int(expected)
