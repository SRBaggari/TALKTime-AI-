"""VAD and audio-quality analysis on synthetic signals with known properties."""
import numpy as np
import pytest

from core.audio_loader import probe_audio
from core.audio_quality import analyze_quality
from core.vad import classify_non_speech, compare_with_diarization, detect_speech

SR = 16000


def voiced(seconds, f0=140.0, amp=0.3, seed=0):
    """Speech-like signal: harmonic, amplitude-modulated at syllable rate (~4 Hz)."""
    t = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(seed)
    x = sum((0.6 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) for k in range(1, 12))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t) ** 2
    return (amp * x * env / np.max(np.abs(x))).astype(np.float32)


def silence(seconds, noise=1e-4, seed=1):
    return np.random.default_rng(seed).normal(0, noise, int(seconds * SR)).astype(np.float32)


def test_energy_vad_preserves_original_timestamps():
    audio = np.concatenate([silence(1.0), voiced(2.0), silence(2.0), voiced(1.5), silence(1.0)])
    r = detect_speech(audio, SR, method="energy")
    assert r["method"] == "energy"
    assert len(r["speech"]) == 2
    (s1, e1), (s2, e2) = r["speech"]
    assert s1 == pytest.approx(1.0, abs=0.1) and e1 == pytest.approx(3.0, abs=0.1)
    assert s2 == pytest.approx(5.0, abs=0.1) and e2 == pytest.approx(6.5, abs=0.1)
    assert r["speech_duration"] + sum(e - s for s, e in r["non_speech"]) == pytest.approx(7.5, abs=0.01)


def test_short_pause_stays_inside_speech():
    audio = np.concatenate([voiced(1.0), silence(0.15), voiced(1.0)])
    r = detect_speech(audio, SR, method="energy", min_silence_ms=300)
    assert len(r["speech"]) == 1


def test_silero_vad_runs_locally_and_rejects_pure_noise():
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 0.05, SR * 3).astype(np.float32)
    r = detect_speech(noise, SR, method="silero")
    assert r["method"] == "silero"
    assert r["speech_ratio"] < 0.2  # white noise is not speech


def test_classify_non_speech_quiet_vs_background():
    rng = np.random.default_rng(4)
    audio = np.concatenate([silence(1.0), rng.normal(0, 0.05, SR).astype(np.float32)])
    out = classify_non_speech(audio, SR, [(0.0, 1.0), (1.0, 2.0)], noise_floor_db=-80)
    assert [o["kind"] for o in out] == ["quiet", "background_sound"]


def test_vad_diarization_comparison():
    r = compare_with_diarization([(0, 5), (10, 12)], [(0, 5)])
    assert r["unattributed_total"] == 2 and r["diarized_without_vad_total"] == 0
    assert r["agreement"] == pytest.approx(5 / 7)


# ---------------------------------------------------------------- quality

def test_clean_recording_scores_well():
    audio = np.concatenate([silence(0.5), voiced(4.0, amp=0.4), silence(0.5)])
    q = analyze_quality(audio, SR, [(0.5, 4.5)], {"sample_rate": 44100, "channels": 2})
    assert q["score"] >= 85 and q["label"] == "Excellent"
    assert q["snr_db"] > 40
    assert q["format"]["channels"] == 2


def test_low_volume_is_not_silence():
    audio = np.concatenate([silence(1.0, noise=1e-5), voiced(4.0, amp=0.003)])
    q = analyze_quality(audio, SR, [(1.0, 5.0)])
    keys = {i["key"] for i in q["issues"]}
    assert "very_low_volume" in keys
    assert "little_speech" not in keys
    assert q["speech_level_dbfs"] < -45


def test_noise_detected():
    rng = np.random.default_rng(5)
    audio = voiced(5.0, amp=0.2) + rng.normal(0, 0.05, SR * 5).astype(np.float32)
    q = analyze_quality(audio, SR, [(0.0, 5.0)])
    keys = {i["key"] for i in q["issues"]}
    # all speech, so the noise floor is estimated from the quietest frames
    assert keys & {"low_snr", "low_snr_severe", "noisy_background"}


def test_clipping_detected():
    audio = np.clip(voiced(3.0, amp=1.0) * 3.0, -1.0, 1.0)
    q = analyze_quality(audio, SR, [(0.0, 3.0)])
    assert q["clipping_events"] > 50
    assert any(i["key"] == "clipping_severe" for i in q["issues"])
    assert q["score"] < 85


def test_silent_recording():
    q = analyze_quality(silence(5.0, noise=1e-6), SR, [])
    assert q["speech_level_dbfs"] is None
    assert any(i["key"] == "little_speech" for i in q["issues"])


def test_long_silence_and_narrowband():
    t = np.arange(SR * 3) / SR
    phone = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)
    audio = np.concatenate([phone, silence(12.0)])
    q = analyze_quality(audio, SR, [(0.0, 3.0)], {"sample_rate": 8000}, long_silence=10.0)
    keys = {i["key"] for i in q["issues"]}
    assert {"narrowband", "long_silence", "low_sample_rate"} <= keys
    assert q["bandwidth_hz"] < 3800


def test_probe_reads_original_format(tmp_path):
    import soundfile as sf

    p = tmp_path / "stereo.wav"
    sf.write(p, np.zeros((22050, 2), dtype=np.float32), 22050)
    info = probe_audio(p)
    assert info["sample_rate"] == 22050 and info["channels"] == 2
    assert info["duration"] == pytest.approx(1.0, abs=0.01)
