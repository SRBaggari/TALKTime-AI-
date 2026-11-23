import math

import pytest

from core.participation_analysis import analyze_participation, normalized_entropy
from core.speaking_time import compute_speaking_time
from tests.conftest import seg


def test_equal_split_is_perfectly_balanced():
    assert normalized_entropy([10, 10, 10]) == pytest.approx(1.0)


def test_single_speaker_undefined():
    assert normalized_entropy([10]) is None
    r = analyze_participation(compute_speaking_time([seg(0, 5, "Speaker 1")], 5))
    assert r["balance_score"] is None
    assert "single speaker" in r["balance_label"].lower()


def test_entropy_known_value():
    p = [0.75, 0.25]
    expected = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)) / math.log(2)
    assert normalized_entropy([75, 25]) == pytest.approx(expected)


def test_silent_participant_lowers_score():
    assert normalized_entropy([10, 10, 0]) < 1.0


def test_dominant_speaker_and_warnings():
    sp = compute_speaking_time([seg(0, 80, "Speaker 1"), seg(80, 100, "Speaker 2")], 100)
    r = analyze_participation(sp)
    assert r["dominant_speaker"] == "Speaker 1"
    assert r["dominant_share"] == pytest.approx(0.8)
    assert any("80%" in w for w in r["warnings"])
    assert "does not by itself indicate poor communication" in r["caveat"]


def test_balanced_two_speakers_no_warning():
    sp = compute_speaking_time([seg(0, 50, "Speaker 1"), seg(50, 100, "Speaker 2")], 100)
    r = analyze_participation(sp)
    assert r["warnings"] == []
    assert r["balance_label"] == "Highly balanced"
    assert r["turn_balance_score"] == pytest.approx(1.0)


def test_gini_known_values():
    from core.participation_analysis import gini

    assert gini([10, 10]) == 0
    assert gini([0, 10]) == pytest.approx(0.5)  # max for N=2 is (N-1)/N
    assert gini([5]) is None


def test_advanced_participation_metrics():
    from core.interruption_analysis import analyze_interruptions
    from core.overlap_detection import detect_overlaps
    from core.silence_detection import analyze_silence
    from core.speaking_time import build_turns

    segs = [seg(0, 30, "Speaker 1"), seg(29, 40, "Speaker 2"), seg(42, 50, "Speaker 1"), seg(51, 60, "Speaker 2")]
    sp = compute_speaking_time(segs, 60)
    sil = analyze_silence(segs, 60, turns=build_turns(segs))
    ints = analyze_interruptions(detect_overlaps(segs), list(sp["speakers"]), 60)
    r = analyze_participation(sp, sil, ints)
    assert r["dominance_ratio"] == pytest.approx(2 * 38 / 58)
    assert r["turns_per_minute"] == pytest.approx(4.0)
    assert r["longest_turn"] == {"speaker": "Speaker 1", "duration": 30}
    s1, s2 = r["per_speaker"]["Speaker 1"], r["per_speaker"]["Speaker 2"]
    assert s1["avg_response_time"] == pytest.approx(2.0)  # 40 → 42
    assert s2["avg_response_time"] == pytest.approx(1.0)  # 50 → 51 (the 29 s start was an overlap)
    assert s2["interruption_balance"] == 1 and s1["interruption_balance"] == -1
    labels = {i["indicator"]: i["speaker"] for i in r["dominance_indicators"]}
    assert labels["Most speaking time"] == "Speaker 1"
    assert labels["Most net possible interruptions"] == "Speaker 2"
    assert "gini" in r["formulas"]
