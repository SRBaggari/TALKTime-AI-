"""Transcription engine tests — faster-whisper is mocked."""
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from core import transcription_engine as te
from core.diarization_engine import ModelUnavailableError


def fake_segment(start, end, text, words=None, logprob=-0.2, nsp=0.01):
    return SimpleNamespace(start=start, end=end, text=text, avg_logprob=logprob, no_speech_prob=nsp,
                           words=[SimpleNamespace(start=s, end=e, word=w, probability=0.9) for s, e, w in (words or [])])


@pytest.fixture(autouse=True)
def reset_models():
    te.TranscriptionEngine._models.clear()
    yield
    te.TranscriptionEngine._models.clear()


def _engine_with(segments, language="en"):
    model = mock.Mock()
    model.transcribe.return_value = (iter(segments), SimpleNamespace(language=language, language_probability=0.97))
    eng = te.TranscriptionEngine("small", "int8", 2)
    te.TranscriptionEngine._models[eng.key] = model
    return eng, model


def test_transcribe_returns_segments_with_words():
    eng, model = _engine_with([
        fake_segment(0.0, 1.5, " Hello there.", [(0.0, 0.5, " Hello"), (0.6, 1.5, " there.")]),
        fake_segment(1.6, 2.0, "   "),  # empty → dropped
    ])
    progress = []
    r = eng.transcribe(np.zeros(32000, dtype=np.float32), language="en", progress=lambda f, m: progress.append(f))
    assert r.language == "en" and r.language_probability == 0.97
    assert len(r.segments) == 1
    assert r.segments[0].text == "Hello there."
    assert [w.word for w in r.segments[0].words] == [" Hello", " there."]
    assert progress[-1] == 1.0
    kwargs = model.transcribe.call_args.kwargs
    assert kwargs["language"] == "en" and kwargs["word_timestamps"] is True


def test_unintelligible_speech_is_flagged():
    eng, _ = _engine_with([fake_segment(0, 1, "mmm hmm", logprob=-1.5, nsp=0.8)])
    r = eng.transcribe(np.zeros(16000, dtype=np.float32))
    assert r.segments[0].text.startswith("[unclear]")
    assert te.is_low_confidence(r.segments[0])


def test_rejects_wrong_sample_rate():
    eng, _ = _engine_with([])
    with pytest.raises(ValueError):
        eng.transcribe(np.zeros(8000, dtype=np.float32), sr=8000)


def test_missing_model_raises():
    with mock.patch.object(te, "local_whisper_path", return_value=None):
        with pytest.raises(ModelUnavailableError, match="download_models"):
            te.TranscriptionEngine("tiny").load()


def test_unknown_model_size():
    with pytest.raises(ValueError):
        te.TranscriptionEngine("gigantic")
