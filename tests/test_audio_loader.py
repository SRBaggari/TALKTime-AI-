import numpy as np
import pytest

from core.audio_loader import AudioLoadError, load_audio
from core.audio_preprocessor import is_effectively_silent, preprocess
from core.audio_validator import sniff_format, validate_upload
from utils.file_utils import is_within, safe_delete, safe_join, sanitize_filename, unique_filename

WAV_HEADER = b"RIFF\x00\x00\x00\x00WAVEfmt "


# ---------------------------------------------------------------- validation

def test_valid_wav_upload():
    r = validate_upload("call.wav", 1000, WAV_HEADER)
    assert r.ok and r.detected_format == "wav"


@pytest.mark.parametrize("header,fmt", [
    (b"fLaC\x00\x00", "flac"), (b"ID3\x04\x00", "mp3"), (b"\xff\xfb\x90\x00", "mp3"),
    (b"\x00\x00\x00\x20ftypM4A ", "mp4"), (b"OggS\x00", "ogg"),
])
def test_sniff_formats(header, fmt):
    assert sniff_format(header) == fmt


def test_rejects_bad_extension():
    r = validate_upload("payload.exe", 1000, b"MZ\x90\x00")
    assert not r.ok
    assert any("Unsupported" in e for e in r.errors)


def test_rejects_executable_disguised_as_audio():
    r = validate_upload("song.mp3", 1000, b"MZ\x90\x00\x03\x00\x00\x00")
    assert not r.ok and any("not audio" in e for e in r.errors)


def test_rejects_empty_and_oversized():
    assert not validate_upload("a.wav", 0, b"").ok
    assert not validate_upload("a.wav", 11 * 1024 * 1024, WAV_HEADER, max_mb=10).ok


def test_mismatched_extension_warns_only():
    r = validate_upload("a.mp3", 100, WAV_HEADER)
    assert r.ok and r.warnings


# ---------------------------------------------------------------- file safety

def test_sanitize_and_unique_filename():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\win.ini") == "win.ini"
    assert sanitize_filename("my call (1).mp3") == "my_call_1_.mp3"
    a, b = unique_filename("x.wav"), unique_filename("x.wav")
    assert a != b and a.endswith("_x.wav")


def test_path_traversal_blocked(tmp_path):
    with pytest.raises(ValueError):
        safe_join(tmp_path, "..", "evil.txt")
    assert is_within(tmp_path, tmp_path / "a" / "b")


def test_safe_delete_refuses_outside_roots(tmp_path):
    inside = tmp_path / "in" / "f.txt"
    inside.parent.mkdir()
    inside.write_text("x")
    outside = tmp_path / "out.txt"
    outside.write_text("x")
    assert safe_delete(outside, [tmp_path / "in"]) is False and outside.exists()
    assert safe_delete(inside, [tmp_path / "in"]) is True and not inside.exists()


# ---------------------------------------------------------------- loading

def test_load_wav_resamples_to_16k_mono(tmp_path):
    import soundfile as sf

    sr = 44100
    stereo = np.stack([np.sin(np.linspace(0, 100, sr * 2)), np.zeros(sr * 2)], axis=1).astype(np.float32)
    p = tmp_path / "s.wav"
    sf.write(p, stereo, sr)
    audio, out_sr = load_audio(p)
    assert out_sr == 16000
    assert audio.ndim == 1
    assert audio.size / out_sr == pytest.approx(2.0, abs=0.01)


def test_empty_file(tmp_path):
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    with pytest.raises(AudioLoadError, match="empty"):
        load_audio(p)


def test_corrupted_file(tmp_path):
    p = tmp_path / "bad.mp3"
    p.write_bytes(b"ID3" + b"\x00garbage" * 50)
    with pytest.raises(AudioLoadError):
        load_audio(p)


def test_very_short_recording(tmp_path):
    import soundfile as sf

    p = tmp_path / "short.wav"
    sf.write(p, np.zeros(1600, dtype=np.float32), 16000)  # 0.1 s
    with pytest.raises(AudioLoadError, match="too short"):
        load_audio(p)


def test_missing_file(tmp_path):
    with pytest.raises(AudioLoadError, match="not found"):
        load_audio(tmp_path / "nope.wav")


def test_preprocess_normalises_and_detects_silence():
    x = np.full(16000, 0.1, dtype=np.float32) + np.sin(np.linspace(0, 50, 16000)).astype(np.float32) * 0.1
    y = preprocess(x)
    assert abs(float(y.mean())) < 1e-3
    assert float(np.abs(y).max()) == pytest.approx(0.95, abs=1e-3)
    assert is_effectively_silent(np.zeros(16000, dtype=np.float32))
    assert not is_effectively_silent(y)
