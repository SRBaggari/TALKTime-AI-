import pytest

from core import intervals as iv
from core.speaking_time import build_turns, compute_speaking_time
from tests.conftest import seg


# ---------------------------------------------------------------- interval math

def test_merge_and_total_union():
    assert iv.merge([(0, 2), (1, 3), (5, 6)]) == [(0, 3), (5, 6)]
    assert iv.total([(0, 2), (1, 3), (5, 6)]) == pytest.approx(4.0)


def test_merge_with_gap():
    assert iv.merge([(0, 1), (1.4, 2)], gap=0.5) == [(0, 2)]
    assert iv.merge([(0, 1), (1.6, 2)], gap=0.5) == [(0, 1), (1.6, 2)]


def test_intersect_subtract_complement():
    assert iv.intersect([(0, 5)], [(2, 3), (4, 8)]) == [(2, 3), (4, 5)]
    assert iv.subtract([(0, 10)], [(2, 3), (5, 6)]) == [(0, 2), (3, 5), (6, 10)]
    assert iv.complement([(1, 2), (4, 5)], 0, 6) == [(0, 1), (2, 4), (5, 6)]


def test_multi_coverage_three_groups():
    groups = {"A": [(0, 10)], "B": [(2, 4)], "C": [(3, 6)]}
    assert iv.multi_coverage(groups, 2) == [(2, 6)]
    assert iv.multi_coverage(groups, 3) == [(3, 4)]


def test_touching_intervals_do_not_overlap():
    assert iv.multi_coverage({"A": [(0, 2)], "B": [(2, 4)]}, 2) == []


# ---------------------------------------------------------------- speaking time

def test_two_speakers_no_overlap():
    r = compute_speaking_time([seg(0, 4, "Speaker 1"), seg(5, 7, "Speaker 2")], recording_duration=10)
    s1, s2 = r["speakers"]["Speaker 1"], r["speakers"]["Speaker 2"]
    assert s1["total_time"] == pytest.approx(4)
    assert s2["total_time"] == pytest.approx(2)
    assert s1["share_of_speaking_time"] == pytest.approx(4 / 6)
    assert s1["share_of_recording"] == pytest.approx(0.4)
    assert r["summary"]["non_speech_time"] == pytest.approx(4)
    assert r["summary"]["overlap_duration"] == 0


def test_overlap_counted_for_both_speakers():
    r = compute_speaking_time([seg(0, 6, "Speaker 1"), seg(4, 8, "Speaker 2")], recording_duration=10)
    assert r["speakers"]["Speaker 1"]["total_time"] == pytest.approx(6)
    assert r["speakers"]["Speaker 2"]["total_time"] == pytest.approx(4)
    assert r["summary"]["combined_speaker_time"] == pytest.approx(10)
    assert r["summary"]["speech_union"] == pytest.approx(8)
    assert r["summary"]["overlap_duration"] == pytest.approx(2)
    assert r["speakers"]["Speaker 1"]["overlap_time"] == pytest.approx(2)
    assert r["speakers"]["Speaker 1"]["solo_time"] == pytest.approx(4)
    # Shares of speaking time always sum to 1
    assert sum(s["share_of_speaking_time"] for s in r["speakers"].values()) == pytest.approx(1.0)
    # Overlap is not removed: combined time exceeds the time anyone was speaking
    assert r["summary"]["combined_speaker_time"] > r["summary"]["speech_union"]


def test_three_speakers_and_turn_stats():
    segs = [seg(0, 2, "Speaker 1"), seg(2.5, 3, "Speaker 2"), seg(3.2, 6, "Speaker 1"),
            seg(6, 9, "Speaker 3"), seg(9.5, 10, "Speaker 1")]
    r = compute_speaking_time(segs, 10)
    s1 = r["speakers"]["Speaker 1"]
    assert r["num_speakers"] == 3
    assert s1["total_time"] == pytest.approx(5.3)
    assert s1["num_turns"] == 3
    assert s1["longest_turn"] == pytest.approx(2.8)
    assert s1["shortest_turn"] == pytest.approx(0.5)
    assert s1["first_spoke"] == 0 and s1["last_spoke"] == 10
    assert s1["avg_turn"] == pytest.approx(5.3 / 3)


def test_turns_merge_short_pauses_unless_other_speaker_starts():
    turns = build_turns([seg(0, 2, "A"), seg(2.3, 4, "A")], merge_gap=0.5)
    assert len(turns) == 1 and turns[0].end == 4
    turns = build_turns([seg(0, 2, "A"), seg(2.1, 2.2, "B"), seg(2.3, 4, "A")], merge_gap=0.5)
    assert len([t for t in turns if t.speaker == "A"]) == 2


def test_single_speaker():
    r = compute_speaking_time([seg(1, 3, "Speaker 1")], 5)
    assert r["speakers"]["Speaker 1"]["share_of_speaking_time"] == 1.0
    assert r["speakers"]["Speaker 1"]["share_of_recording"] == pytest.approx(0.4)


def test_no_speech():
    r = compute_speaking_time([], 5)
    assert r["num_speakers"] == 0
    assert r["summary"]["non_speech_time"] == 5


def test_duplicate_segments_not_double_counted():
    r = compute_speaking_time([seg(0, 3, "Speaker 1"), seg(1, 2, "Speaker 1")], 5)
    assert r["speakers"]["Speaker 1"]["total_time"] == pytest.approx(3)


def test_duration_never_shorter_than_speech():
    r = compute_speaking_time([seg(0, 6, "Speaker 1")], 5)
    assert r["recording_duration"] == 6
