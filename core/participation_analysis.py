"""Participation balance via normalised Shannon entropy.

For speaking-time shares p_1..p_N (N >= 2 speakers):

    balance = H(p) / ln(N),   H(p) = -Σ p_i ln p_i

balance = 1.0 when everyone speaks equally and approaches 0 when one person
does all the talking. It is scale-free and comparable across different
numbers of speakers. Undefined (None) for a single speaker.
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


def analyze_participation(speaking: dict) -> dict:
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

    return {
        "num_speakers": n,
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
