"""
Expands short user prompts into a richer creative brief for script generation.
Keeps production complexity on the backend so a simple prompt still yields a polished video.
"""

import logging
import httpx

from config import get_settings, sanitize_header_token

logger = logging.getLogger(__name__)

# Only expand when prompt is short (e.g. one line / a few words)
SHORT_PROMPT_MAX_CHARS = 150


def _auth_header() -> str:
    key = sanitize_header_token((get_settings().openai_api_key or ""))
    if not key:
        raise ValueError("OPENAI_API_KEY is not set or invalid.")
    return "Bearer " + key


async def expand_prompt(prompt: str) -> str:
    """
    Turn a short idea into a 1–2 sentence creative brief. No hardcoding: every
    prompt is expanded the same way. The script generator will apply the user's
    chosen tone (literal, viral, etc.) when generating the script.
    If expansion fails or prompt is already long, returns the original prompt.
    """
    prompt = (prompt or "").strip()
    if len(prompt) > SHORT_PROMPT_MAX_CHARS:
        return prompt
    if not prompt:
        return prompt

    system = """You expand short video ideas into a one- or two-sentence creative brief for AI video (Kling 3.0). The brief will drive a multi-shot script: one clear step per shot.

RULES:
- One main character (who), one clear activity, setting/mood. Keep the user's exact intent.
- Spell out the full real-world sequence in order. Each step will become one video shot—so use simple, concrete verbs: grabs, peels, eats; opens, pours, drinks; unwraps, reads.
- Food/drink: object → prepare → consume (e.g. "grabs a banana, peels it, then eats it"). Never skip the middle step (e.g. peel before eat).
- Opening something: show it → open/unwrap → use. Making: ingredients → steps → result.
- Answer: what is there at the start? First action? Next? End. Keep wording simple so each step maps to one shot.

Output only the brief, no labels, no "Brief:" prefix."""

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": _auth_header(),
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 200,
                },
            )
            r.raise_for_status()
            data = r.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
            if content and len(content) > len(prompt):
                logger.info(f"Expanded short prompt ({len(prompt)} chars) to brief ({len(content)} chars)")
                return content
    except Exception as e:
        logger.warning(f"Prompt expansion skipped: {e}")

    return prompt


# Kling / pipeline: max scenes we allow (matches validators and API)
SCENE_COUNT_MIN = 1
SCENE_COUNT_MAX = 20
AUTO_SCENE_MAX = 6


def infer_scene_count_from_brief(brief: str) -> int:
    """
    Infer how many scenes (one per distinct action) from an expanded creative brief.
    Uses then/semicolon/clauses. Ensures peel-before-eat (and similar) sequences get at least 3 scenes
    so we don't skip the middle step (e.g. grab → peel → eat).
    """
    import re
    brief = (brief or "").strip()
    if not brief:
        return 3

    p_lower = brief.lower()
    # Heuristic: count sequence markers ("then", ";") + 1
    then_count = len(re.findall(r"\bthen\b", p_lower))
    semicolon_count = brief.count(";")
    by_then = min(SCENE_COUNT_MAX, max(SCENE_COUNT_MIN, then_count + 1))
    by_semicolon = min(SCENE_COUNT_MAX, max(SCENE_COUNT_MIN, semicolon_count + 1))
    if then_count > 0 or semicolon_count > 0:
        heuristic = max(by_then, by_semicolon)
    else:
        # Fallback: sentence-like clauses (period or semicolon split)
        clauses = [s.strip() for s in re.split(r"[.;]", brief) if s.strip()]
        heuristic = min(SCENE_COUNT_MAX, max(SCENE_COUNT_MIN, len(clauses))) if clauses else 3

    # Peel-before-eat (or similar) needs at least 3 scenes: grab → peel → eat. Otherwise the video
    # can hallucinate eating without showing the peel.
    if "peel" in p_lower and ("eat" in p_lower or "grab" in p_lower or "take" in p_lower):
        heuristic = max(heuristic, 3)
        logger.info(f"Auto scene count: brief has peel + eat/grab → at least 3 scenes → {heuristic}")
    elif then_count > 0 or semicolon_count > 0:
        logger.info(f"Auto scene count (heuristic): {heuristic} scenes")
    else:
        logger.info(f"Auto scene count (clauses): {heuristic} scenes")

    capped = min(heuristic, AUTO_SCENE_MAX)
    if capped != heuristic:
        logger.info(f"Auto scene count capped to {AUTO_SCENE_MAX} (was {heuristic})")
    return capped
