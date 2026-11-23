"""Participation analytics with documented formulas.

Balance — normalised Shannon entropy of speaking-time shares p_1..p_N (N >= 2):
    balance = H(p) / ln(N),   H(p) = -Σ p_i ln p_i
    1.0 = everyone speaks equally, → 0 as one person does all the talking.

Gini coefficient of speaking time (0 = equal, → 1 = concentrated):
    G = Σ_i Σ_j |x_i − x_j| / (2 N² · mean(x))

Dominance ratio — top speaker's share relative to an equal share:
    D = max(p_i) / (1/N)          (1.0 = equal; N = one person did everything)

Turn rate — speaking turns per minute of recording (per speaker and overall).
Response time — mean/median gap before a speaker answers someone else (from silence analysis).
Interruption balance — possible interruptions initiated − received.

None of these measures says whether a conversation was "good": interviews,
lectures and support calls are legitimately unequal.
"""
from __future__ import annotations

import math


def normalized_entropy(values: list[float]) -> float | None:
    vals = [v for v in values if v > 0]
    n = len(values)
    total = sum(vals)
    if n < 2 or total <= 0:
        return None
    probs = [v / total for v in vals]
    h = -sum(p * math.log(p) for p in probs)
    return max(0.0, min(1.0, h / math.log(n)))


def balance_label(score: float | None) -> str:
    if score is None:
        return "Not applicable (single speaker)"
    if score >= 0.9:
        return "Highly balanced"
    if score >= 0.75:
        return "Balanced"
    if score >= 0.5:
        return "Moderately unbalanced"
    return "Highly unbalanced"


def gini(values: list[float]) -> float | None:
    n = len(values)
    total = sum(values)
    if n < 2 or total <= 0:
        return None
    mean = total / n
    return sum(abs(a - b) for a in values for b in values) / (2 * n * n * mean)


def analyze_participation(speaking: dict, silence: dict | None = None, interruptions: dict | None = None,
                          questions: dict | None = None) -> dict:
    speakers = speaking.get("speakers", {})
    names = list(speakers)
    n = len(names)
    times = [speakers[s]["total_time"] for s in names]
    turns = [speakers[s]["num_turns"] for s in names]
    total_time = sum(times)
    total_turns = sum(turns)
    time_share = {s: (t / total_time if total_time else 0.0) for s, t in zip(names, times)}
    turn_share = {s: (t / total_turns if total_turns else 0.0) for s, t in zip(names, turns)}

    balance = normalized_entropy(times)
    turn_balance = normalized_entropy(turns)
    dominant = max(names, key=lambda s: time_share[s]) if names else None

    warnings: list[str] = []
    if n >= 2 and dominant:
        equal = 1.0 / n
        if time_share[dominant] >= max(0.5, 1.5 * equal):
            warnings.append(
                f"{dominant} accounted for {time_share[dominant] * 100:.0f}% of speaking time "
                f"(an equal split would be {equal * 100:.0f}%)."
            )
        for s in names:
            if time_share[s] < 0.5 * equal:
                warnings.append(f"{s} contributed {time_share[s] * 100:.0f}% of speaking time — lower than other participants.")

    minutes = (speaking.get("recording_duration") or 0) / 60
    delays = (silence or {}).get("response_delay_by_speaker", {})
    ints = (interruptions or {}).get("per_speaker", {})
    qs = (questions or {}).get("per_speaker", {})
    per_speaker = {}
    for s in names:
        st = speakers[s]
        d = delays.get(s, {})
        i = ints.get(s, {})
        per_speaker[s] = {
            "time_share": time_share[s],
            "turn_share": turn_share[s],
            "turns_per_minute": (st["num_turns"] / minutes) if minutes else 0.0,
            "longest_turn": st["longest_turn"],
            "avg_turn": st["avg_turn"],
            "avg_response_time": d.get("mean") if d.get("count") else None,
            "median_response_time": d.get("median") if d.get("count") else None,
            "responses": d.get("count", 0),
            "interruption_balance": i.get("initiated", 0) - i.get("received", 0),
            "questions_asked": qs.get(s, {}).get("asked", 0),
        }

    # Dominance indicators: several independent signals, reported side by side.
    indicators = []
    if n >= 2:
        def top(key, label, fmt, explain):
            vals = {s: per_speaker[s][key] for s in names if per_speaker[s][key] is not None}
            if vals and max(vals.values()) > 0:
                who = max(vals, key=vals.get)
                indicators.append({"indicator": label, "speaker": who, "value": fmt(vals[who]), "explanation": explain})
        top("time_share", "Most speaking time", lambda v: f"{v * 100:.0f}%", "Share of combined speaking time.")
        top("longest_turn", "Longest uninterrupted turn", lambda v: f"{v:.1f}s", "Single longest stretch holding the floor.")
        top("turn_share", "Most turns", lambda v: f"{v * 100:.0f}%", "Share of all speaking turns.")
        top("interruption_balance", "Most net possible interruptions", lambda v: f"+{v}",
            "Possible interruptions initiated minus received.")
        top("questions_asked", "Most questions asked", lambda v: str(v), "Often indicates who is steering the conversation.")
    longest = max(((s, speakers[s]["longest_turn"]) for s in names), key=lambda x: x[1], default=(None, 0.0))
    all_delays = [per_speaker[s]["avg_response_time"] for s in names if per_speaker[s]["avg_response_time"] is not None]

    return {
        "num_speakers": n,
        "gini": gini(times),
        "dominance_ratio": (max(time_share.values()) * n) if n >= 2 else None,
        "turns_per_minute": (total_turns / minutes) if minutes else 0.0,
        "longest_turn": {"speaker": longest[0], "duration": longest[1]},
        "avg_response_time": (sum(all_delays) / len(all_delays)) if all_delays else None,
        "per_speaker": per_speaker,
        "dominance_indicators": indicators,
        "formulas": {
            "balance": "H(p)/ln(N), H(p) = −Σ pᵢ ln pᵢ over speaking-time shares",
            "gini": "Σᵢ Σⱼ |xᵢ − xⱼ| / (2N²·mean(x)) over speaking times",
            "dominance_ratio": "max(pᵢ) ÷ (1/N)",
            "turns_per_minute": "speaking turns ÷ recording minutes",
            "response_time": "gap between the end of one speaker's turn and the next speaker's start (overlaps excluded)",
        },
        "balance_score": balance,
        "balance_label": balance_label(balance),
        "turn_balance_score": turn_balance,
        "time_share": time_share,
        "turn_share": turn_share,
        "dominant_speaker": dominant,
        "dominant_share": time_share.get(dominant, 0.0) if dominant else 0.0,
        "warnings": warnings,
        "method": "Normalised Shannon entropy of speaking-time shares: H(p)/ln(N). 1.0 = perfectly equal, 0 = one speaker only.",
        "caveat": "Unequal participation is expected in many conversations (interviews, presentations, support calls) "
                  "and does not by itself indicate poor communication.",
    }
