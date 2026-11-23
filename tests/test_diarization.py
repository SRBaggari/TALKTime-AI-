"""Diarization engine tests — pyannote inference is mocked."""
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from core import diarization_engine as de
from core.types import SpeakerSegment


class FakeAnnotation:
    def __init__(self, tracks):
        self.tracks = tracks

    def itertracks(self, yield_label=False):
        for s, e, label in self.tracks:
            yield SimpleNamespace(start=s, end=e), "_", label


@pytest.fixture(autouse=True)
def reset_pipeline():
    de.DiarizationEngine._pipeline = None
    yield
    de.DiarizationEngine._pipeline = None


def test_relabel_by_first_appearance():
    raw = [SpeakerSegment(5, 6, "SPEAKER_00"), SpeakerSegment(0, 2, "SPEAKER_01"), SpeakerSegment(3, 4, "SPEAKER_02")]
    out = de.relabel_by_first_appearance(raw)
    assert [(s.start, s.speaker) for s in out] == [(0, "Speaker 1"), (3, "Speaker 2"), (5, "Speaker 3")]


def test_missing_model_raises_with_instructions():
    with mock.patch.object(de, "local_diarization_path", return_value=None):
        with pytest.raises(de.ModelUnavailableError) as exc:
            de.DiarizationEngine().load()
    assert "huggingface.co" in str(exc.value) and "HF_TOKEN" in str(exc.value)


def test_diarize_preserves_overlap_and_passes_speaker_bounds():
    calls = {}

    def fake_pipeline(file, hook=None, **kwargs):
        calls.update(kwargs)
        calls["sr"] = file["sample_rate"]
        hook("segmentation", None, completed=1, total=2)
        return SimpleNamespace(speaker_diarization=FakeAnnotation(
            [(0.0, 3.0, "SPEAKER_00"), (2.5, 5.0, "SPEAKER_01"), (5.0, 5.0, "SPEAKER_00")]))

    de.DiarizationEngine._pipeline = fake_pipeline
    progress = []
    segs = de.DiarizationEngine().diarize(np.zeros(16000 * 5, dtype=np.float32), 16000, min_speakers=2,
                                          max_speakers=4, progress=lambda f, m: progress.append(f))
    assert calls == {"min_speakers": 2, "max_speakers": 4, "sr": 16000}
    assert [(s.start, s.end, s.speaker) for s in segs] == [(0.0, 3.0, "Speaker 1"), (2.5, 5.0, "Speaker 2")]
    assert progress and 0 <= progress[0] <= 1


def test_num_speakers_overrides_bounds():
    calls = {}

    def fake_pipeline(file, hook=None, **kwargs):
        calls.update(kwargs)
        return FakeAnnotation([(0, 1, "A")])  # pyannote 3.x style output

    de.DiarizationEngine._pipeline = fake_pipeline
    segs = de.DiarizationEngine().diarize(np.zeros(16000, dtype=np.float32), 16000, num_speakers=3, min_speakers=2)
    assert calls == {"num_speakers": 3}
    assert segs[0].speaker == "Speaker 1"


def test_availability_uses_override_path(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("pipeline: {}")
    monkeypatch.setenv("TALKTIME_DIARIZATION_PATH", str(tmp_path))
    assert de.local_diarization_path() == tmp_path
