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
