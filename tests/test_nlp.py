import pytest

from core.conversation_summary import extract_insights, extract_topics, key_points, ollama_status
from core.question_analysis import analyze_questions, detect_question
from core.sentiment_analysis import analyze_sentiment, label_for
from tests.conftest import tseg


# ---------------------------------------------------------------- questions

@pytest.mark.parametrize("text,expected", [
    ("What time does the meeting start?", True),
    ("How do we handle the refund", True),
    ("Can you send me the report", True),
    ("We shipped it yesterday.", False),
    ("The budget is fine.", False),
    ("It works, right?", True),
])
def test_detect_question(text, expected):
    assert detect_question(text)[0] is expected


def test_question_mark_has_higher_confidence_than_heuristic():
    assert detect_question("Is it ready?")[1] > detect_question("Is it ready")[1]


def test_answered_and_unanswered_questions():
    segs = [
        tseg(0, 2, "What is the deadline for the project?", "Speaker 1"),
        tseg(2.5, 5, "It is due next Friday.", "Speaker 2"),
        tseg(6, 8, "Did anyone review the contract?", "Speaker 2"),
        tseg(50, 52, "Okay, moving on.", "Speaker 1"),  # outside 30 s window
    ]
    r = analyze_questions(segs)
    assert r["total_questions"] == 2
    q1, q2 = r["questions"]
    assert q1["answered"] and q1["answered_by"] == "Speaker 2" and q1["response_delay"] == 0.5
    assert q1["answer_confidence"] >= 0.7
    assert not q2["answered"]
    assert r["per_speaker"]["Speaker 2"]["possibly_unanswered"] == 1
    assert "not semantic" in r["method"]


# ---------------------------------------------------------------- sentiment

def test_sentiment_labels():
    assert label_for(0.6) == "positive" and label_for(-0.6) == "negative" and label_for(0.0) == "neutral"


def test_sentiment_classification_per_speaker():
    segs = [
        tseg(0, 2, "This is great, I love the new design!", "Speaker 1"),
        tseg(2, 4, "This is terrible and I hate the delays.", "Speaker 2"),
        tseg(4, 6, "The meeting is on Tuesday.", "Speaker 1"),
    ]
    r = analyze_sentiment(segs, "en")
    assert r["available"]
    assert segs[0].sentiment == "positive" and segs[1].sentiment == "negative" and segs[2].sentiment == "neutral"
    assert r["per_speaker"]["Speaker 2"]["label"] == "negative"
    assert r["overall_distribution"] == {"positive": 1, "negative": 1, "neutral": 1}
    assert "not tone of voice" in r["disclaimer"]


def test_sentiment_non_english_disabled():
    r = analyze_sentiment([tseg(0, 1, "hola", "Speaker 1")], "es")
    assert r["available"] is False and "English only" in r["reason"]


# ---------------------------------------------------------------- intelligence

SEGS = [
    tseg(0, 5, "We need to finalise the marketing budget for the product launch.", "Speaker 1"),
    tseg(5, 10, "The marketing budget is the most important issue this quarter.", "Speaker 2"),
    tseg(10, 15, "We decided to go with the smaller agency for the product launch.", "Speaker 1"),
    tseg(15, 20, "I will send the revised budget by Friday.", "Speaker 2"),
    tseg(20, 25, "Can you schedule a call with the agency next week?", "Speaker 1"),
]


def test_insight_extraction():
    ins = extract_insights(SEGS, "en")
    assert any("decided" in d["text"] for d in ins["decision"])
    assert any("I will send" in c["text"] for c in ins["commitment"])
    assert any(c["deadline"] and "friday" in c["deadline"].lower() for c in ins["commitment"])
    assert any("schedule a call" in a["text"] for a in ins["action_item"])
    assert any("important" in i["text"] for i in ins["important"])


def test_topics_and_key_points():
    topics = extract_topics(SEGS)
    names = [t["topic"] for t in topics]
    assert any("budget" in n for n in names)
    assert any("launch" in n or "agency" in n for n in names)
    pts = key_points(SEGS, topics)
    assert pts and all(p["start"] <= q["start"] for p, q in zip(pts, pts[1:]))  # chronological


def test_ollama_refuses_remote_url():
    r = ollama_status("http://example.com:11434")
    assert r["available"] is False and "local" in r["reason"].lower()
