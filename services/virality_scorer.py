"""
Virality scorer: predicts engagement likelihood (0–100) and returns improvement suggestions.
Used after video assembly to show users a score and optional tips.
"""

from dataclasses import dataclass
from typing import List, Optional

from models.schemas import VideoScript


@dataclass
class ViralityMetrics:
    """Input metrics for scoring."""
    hook_type: str  # e.g. shock_stat, belief_attack, specific_promise, contrarian, threat
    script_word_count: int
    reading_level_grade: int  # 3–4 = 3rd grade, etc.
    pattern_interrupt_count: int
    has_proof: bool
    cta_type: str  # dm_keyword, comment_keyword, save_this, generic
    video_duration_seconds: int


# Hook quality (0–25)
HOOK_SCORES = {
    "shock_stat": 25,
    "belief_attack": 23,
    "specific_promise": 22,
    "contrarian": 20,
    "threat": 18,
    "question": 17,
    "default": 15,
}

# CTA quality (0–15)
CTA_SCORES = {
    "dm_keyword": 15,
    "comment_keyword": 14,
    "save_this": 12,
    "generic": 8,
}


def _infer_hook_type(script: Optional[VideoScript]) -> str:
    """Infer hook type from first scene dialogue."""
    if not script or not script.scenes:
        return "default"
    first = (script.scenes[0].dialogue or "").lower()
    if any(w in first for w in ("percent", "%", "stat", "number", "million")):
        return "shock_stat"
    if any(w in first for w in ("wrong", "lie", "myth", "don't believe")):
        return "belief_attack"
    if any(w in first for w in ("here's how", "exactly how", "step by step")):
        return "specific_promise"
    if any(w in first for w in ("actually", "truth is", "nobody tells you")):
        return "contrarian"
    if any(w in first for w in ("stop", "quit", "danger", "losing")):
        return "threat"
    if "?" in first:
        return "question"
    return "default"


def _infer_cta_type(script: Optional[VideoScript]) -> str:
    """Infer CTA from last scene dialogue."""
    if not script or not script.scenes:
        return "generic"
    last = (script.scenes[-1].dialogue or "").lower()
    if any(w in last for w in ("dm", "message me", "link in bio")):
        return "dm_keyword"
    if any(w in last for w in ("comment", "drop a", "reply")):
        return "comment_keyword"
    if any(w in last for w in ("save this", "screenshot", "bookmark")):
        return "save_this"
    return "generic"


def calculate_virality_score(metrics: ViralityMetrics) -> int:
    """Return 0–100 virality score."""
    score = 0

    # 1. Hook (0–25)
    score += HOOK_SCORES.get(metrics.hook_type, HOOK_SCORES["default"])

    # 2. Pattern interrupts (0–20) – aim ~every 3s
    dur = max(1, metrics.video_duration_seconds)
    per_sec = metrics.pattern_interrupt_count / dur
    if per_sec >= 0.33:
        score += 20
    elif per_sec >= 0.2:
        score += 15
    else:
        score += 10

    # 3. Proof (0–15)
    score += 15 if metrics.has_proof else 0

    # 4. Reading level (0–15) – 3rd–4th grade best
    if metrics.reading_level_grade <= 4:
        score += 15
    elif metrics.reading_level_grade <= 6:
        score += 10
    else:
        score += 5

    # 5. CTA (0–15)
    score += CTA_SCORES.get(metrics.cta_type, CTA_SCORES["generic"])

    # 6. Duration (0–10) – 30–45s sweet spot
    if 30 <= metrics.video_duration_seconds <= 45:
        score += 10
    elif metrics.video_duration_seconds <= 60:
        score += 7
    else:
        score += 3

    return min(score, 100)


def get_virality_suggestions(metrics: ViralityMetrics, score: int) -> List[str]:
    """Return improvement suggestions when score < 70."""
    suggestions = []
    if score >= 70:
        return suggestions
    if not metrics.has_proof:
        suggestions.append("Add proof-of-work overlays to boost credibility")
    if metrics.reading_level_grade > 6:
        suggestions.append("Simplify language to 3rd-grade reading level")
    dur = max(1, metrics.video_duration_seconds)
    if metrics.pattern_interrupt_count / dur < 0.3:
        suggestions.append("Add more pattern interrupts (aim for every 3 seconds)")
    if metrics.hook_type == "default":
        suggestions.append("Consider a stronger hook (e.g. shock stat or contrarian opening)")
    return suggestions


def score_from_script(
    script: Optional[VideoScript],
    video_duration_seconds: int,
    pattern_interrupt_count: Optional[int] = None,
    has_proof: bool = False,
) -> tuple:
    """
    Convenience: compute ViralityMetrics from script + duration, return (score, suggestions).
    reading_level_grade defaults to 4 (3rd–4th grade viral style).
    """
    word_count = 0
    if script and script.scenes:
        for s in script.scenes:
            word_count += len((s.dialogue or "").split())
    interrupts = pattern_interrupt_count if pattern_interrupt_count is not None else max(1, video_duration_seconds // 3)
    metrics = ViralityMetrics(
        hook_type=_infer_hook_type(script),
        script_word_count=word_count,
        reading_level_grade=4,
        pattern_interrupt_count=interrupts,
        has_proof=has_proof,
        cta_type=_infer_cta_type(script),
        video_duration_seconds=video_duration_seconds,
    )
    score = calculate_virality_score(metrics)
    suggestions = get_virality_suggestions(metrics, score)
    return (score, suggestions)
