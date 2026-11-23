"""Test configuration: isolate all data (DB, uploads, reports) in a temp dir."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_TMP = tempfile.mkdtemp(prefix="talktime_test_")
os.environ["TALKTIME_DATA_DIR"] = _TMP
os.environ.pop("TALKTIME_DATABASE_URL", None)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import config  # noqa: E402
from core.types import SpeakerSegment, TranscriptSegment, Word  # noqa: E402

config.ensure_directories()


def seg(start, end, speaker):
    return SpeakerSegment(float(start), float(end), speaker)


def tseg(start, end, text, speaker=None, words=None):
    return TranscriptSegment(start=float(start), end=float(end), text=text, speaker=speaker, words=words or [])


def words_for(start, end, text):
    toks = text.split()
    step = (end - start) / len(toks)
    return [Word(start + i * step, start + (i + 1) * step, " " + t, 0.9) for i, t in enumerate(toks)]


@pytest.fixture
def clean_db():
    from database import repository
    from database.db import init_db

    init_db()
    for r in repository.list_recordings():
        repository.delete_recording(r["id"])
    yield
    for r in repository.list_recordings():
        repository.delete_recording(r["id"])


@pytest.fixture
def tone_wav(tmp_path):
    """A 6-second WAV: tone bursts 0-2s and 3-5s, silence elsewhere."""
    import soundfile as sf

    sr = 16000
    t = np.arange(int(6 * sr)) / sr
    audio = np.zeros_like(t, dtype=np.float32)
    for a, b, f in ((0.0, 2.0, 220), (3.0, 5.0, 330)):
        m = (t >= a) & (t < b)
        audio[m] = 0.5 * np.sin(2 * np.pi * f * t[m])
    path = tmp_path / "tones.wav"
    sf.write(path, audio, sr)
    return path
