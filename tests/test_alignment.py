from core.speaker_alignment import UNKNOWN, align
from tests.conftest import seg, tseg, words_for


def test_segment_level_assignment_by_max_overlap():
    diar = [seg(0, 5, "Speaker 1"), seg(5, 10, "Speaker 2")]
    out = align([tseg(4, 9, "hello there friend")], diar, word_level=False)
    assert out[0].speaker == "Speaker 2"
    assert out[0].speaker_confidence == 0.8
    assert "Speaker 1" in out[0].overlapping_speakers


def test_word_level_splits_segment_at_speaker_change():
    diar = [seg(0, 2, "Speaker 1"), seg(2, 4, "Speaker 2")]
    t = tseg(0, 4, "one two three four", words=words_for(0, 4, "one two three four"))
    out = align([t], diar)
    assert [(o.speaker, o.text) for o in out] == [("Speaker 1", "one two"), ("Speaker 2", "three four")]
    assert all(not o.uncertain for o in out)


def test_overlap_marks_uncertain():
    diar = [seg(0, 4, "Speaker 1"), seg(0, 4, "Speaker 2")]
    out = align([tseg(0, 4, "both talking")], diar, word_level=False)
    assert out[0].uncertain
    assert out[0].speaker_confidence == 0.5


def test_no_overlap_snaps_to_nearby_speaker_but_uncertain():
    out = align([tseg(5.2, 6, "late words")], [seg(0, 5, "Speaker 1")], word_level=False)
    assert out[0].speaker == "Speaker 1" and out[0].uncertain


def test_far_from_any_speaker_is_unknown():
    out = align([tseg(20, 21, "far")], [seg(0, 5, "Speaker 1")], word_level=False)
    assert out[0].speaker == UNKNOWN and out[0].uncertain


def test_word_jitter_smoothing():
    diar = [seg(0, 1.95, "Speaker 1"), seg(1.95, 2.05, "Speaker 2"), seg(2.05, 4, "Speaker 1")]
    text = "a b c d e f g h"
    out = align([tseg(0, 4, text, words=words_for(0, 4, text))], diar)
    assert len(out) == 1 and out[0].speaker == "Speaker 1"
