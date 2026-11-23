from core.interruption_analysis import analyze_interruptions, classify_overlap
from core.overlap_detection import detect_overlaps
from tests.conftest import seg


def test_detect_overlap_ongoing_and_incoming():
    ev = detect_overlaps([seg(0, 5, "Speaker 1"), seg(4, 8, "Speaker 2")])
    assert len(ev) == 1
    e = ev[0]
    assert (e["start"], e["end"], e["duration"]) == (4, 5, 1)
    assert e["ongoing_speaker"] == "Speaker 1" and e["incoming_speaker"] == "Speaker 2"


def test_short_overlap_below_threshold_ignored():
    assert detect_overlaps([seg(0, 5, "A"), seg(4.9, 8, "B")], min_overlap=0.2) == []


def test_same_speaker_never_overlaps_itself():
    assert detect_overlaps([seg(0, 5, "A"), seg(2, 6, "A")]) == []


def test_simultaneous_start():
    ev = detect_overlaps([seg(1.0, 3, "A"), seg(1.05, 2, "B")])
    assert ev[0]["simultaneous"] and ev[0]["ongoing_speaker"] is None
    assert classify_overlap(ev[0]) == "simultaneous_start"


def test_possible_interruption_when_floor_changes():
    ev = detect_overlaps([seg(0, 5, "Speaker 1"), seg(4, 9, "Speaker 2")])[0]
    assert classify_overlap(ev) == "possible_interruption"


def test_backchannel_is_brief_overlap():
    ev = detect_overlaps([seg(0, 10, "Speaker 1"), seg(4, 4.6, "Speaker 2")])[0]
    assert classify_overlap(ev) == "brief_overlap"


def test_long_overlap_without_yield():
    ev = detect_overlaps([seg(0, 10, "Speaker 1"), seg(4, 7, "Speaker 2")])[0]
    assert classify_overlap(ev) == "overlapping_speech"


def test_interruption_counts_per_speaker():
    segs = [seg(0, 5, "Speaker 1"), seg(4, 9, "Speaker 2"), seg(8, 12, "Speaker 1")]
    r = analyze_interruptions(detect_overlaps(segs), ["Speaker 1", "Speaker 2"], 60)
    assert r["possible_interruptions"] == 2
    assert r["per_speaker"]["Speaker 2"] == {"initiated": 1, "received": 1, "overlap_events": 2}
    assert r["per_speaker"]["Speaker 1"]["initiated"] == 1
    assert r["interruptions_per_minute"] == 2
    assert "does not imply intent" in r["note"]


def test_three_way_overlap_produces_pairwise_events():
    ev = detect_overlaps([seg(0, 10, "A"), seg(2, 6, "B"), seg(3, 5, "C")])
    pairs = {tuple(e["speakers"]) for e in ev}
    assert pairs == {("A", "B"), ("A", "C"), ("B", "C")}
