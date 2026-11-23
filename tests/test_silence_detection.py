import numpy as np
import pytest

from core.silence_detection import acoustic_silence, analyze_silence
from core.speaking_time import build_turns
from tests.conftest import seg


def test_silence_classification_and_threshold():
    segs = [seg(2, 5, "A"), seg(5.5, 8, "A"), seg(10, 12, "B"), seg(12.3, 14, "A")]
    r = analyze_silence(segs, 20, min_silence=1.0, long_silence=5.0, turns=build_turns(segs))
    kinds = {(s["start"], s["kind"]) for s in r["silences"]}
    assert (0, "leading") in kinds
    assert (8, "between_speakers") in kinds
    assert (14, "trailing") in kinds
    # 0.5s and 0.3s gaps are normal pauses, not silence
    assert all(s["duration"] >= 1.0 for s in r["silences"])
    assert r["longest_silence"] == pytest.approx(6)
    assert r["num_long_silences"] == 1
    assert r["total_non_speech"] == pytest.approx(2 + 0.5 + 2 + 0.3 + 6)


def test_within_speaker_pause():
    segs = [seg(0, 3, "A"), seg(5, 8, "A")]
    r = analyze_silence(segs, 8, min_silence=1.0)
    assert r["silences"][0]["kind"] == "within_speaker"
    assert r["within_speaker_pauses"]["A"]["count"] == 1


def test_response_delays():
    segs = [seg(0, 3, "A"), seg(4, 6, "B"), seg(5.5, 9, "A")]
    r = analyze_silence(segs, 10, turns=build_turns(segs))
    delays = [(d["from"], d["to"], d["delay"]) for d in r["response_delays"]]
    assert ("A", "B", 1.0) in delays
    assert ("B", "A", -0.5) in delays  # overlap → negative delay
    assert r["response_delay_by_speaker"]["B"]["mean"] == 1.0


def test_long_silence_only_recording():
    r = analyze_silence([], 30, min_silence=1)
    assert r["total_silence"] == 30 and r["num_silences"] == 1


def test_acoustic_silence(tone_wav):
    import soundfile as sf

    audio, sr = sf.read(tone_wav, dtype="float32")
    noise = np.random.default_rng(0).normal(0, 1e-4, audio.size).astype(np.float32)
    r = acoustic_silence(audio + noise, sr, min_silence=0.5)
    # silent regions: 2-3s and 5-6s
    assert r["count"] == 2
    assert r["total"] == pytest.approx(2.0, abs=0.15)
