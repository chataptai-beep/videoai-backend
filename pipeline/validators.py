"""
Pipeline validators: request, script, and KIE payload validation.
"""

import re
from typing import List, Optional, Tuple

from config import get_settings
from models.schemas import VideoScript, AspectRatio
from pipeline.style_presets import list_style_ids

# Kling / KIE constraints (align with video_generator and kling_prompt_builder)
MAX_MULTI_PROMPT_TOTAL_SECONDS = 15
MIN_MULTI_PROMPT_TOTAL_SECONDS = 3
KLING_PROMPT_MAX_CHARS = 500
SCENE_COUNT_MIN = 1
SCENE_COUNT_MAX = 20
PROMPT_MIN_LEN = 10
PROMPT_MAX_LEN = 20000

# SSRF: block webhook to private/localhost
_PRIVATE_IP_PATTERN = re.compile(
    r"^(?:https?://)?(?:localhost|127\.\d+\.\d+\.\d+|10\.\d+\.\d+\.\d+|"
    r"172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+)(?::\d+)?",
    re.IGNORECASE,
)
_METADATA_PATTERN = re.compile(r"169\.254\.\d+\.\d+|metadata\.google\.internal", re.IGNORECASE)


def _is_blocked_webhook_url(url: str) -> bool:
    """True if URL targets private/localhost (SSRF risk)."""
    if not url or not url.strip():
        return False
    u = url.strip().lower()
    if _PRIVATE_IP_PATTERN.search(u):
        return True
    if _METADATA_PATTERN.search(u):
        return True
    return False


def validate_request(
    prompt: str,
    scene_count: int,
    aspect_ratio: AspectRatio,
    style_id: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Validate POST /generate request before creating a job.
    Returns (errors, warnings). errors=blocking; warnings=non-blocking hints.
    """
    errors: List[str] = []
    warnings: List[str] = []
    s = get_settings()
    p = (prompt or "").strip()
    if len(p) < PROMPT_MIN_LEN:
        errors.append(f"Prompt must be at least {PROMPT_MIN_LEN} characters")
    if len(p) > PROMPT_MAX_LEN:
        errors.append(f"Prompt must be at most {PROMPT_MAX_LEN} characters")
    blocklist = (getattr(s, "prompt_blocklist", None) or "").strip()
    if blocklist:
        prompt_lower = p.lower()
        for phrase in blocklist.split(","):
            phrase = phrase.strip().lower()
            if phrase and phrase in prompt_lower:
                errors.append(f"Prompt contains blocked phrase")
                break
    if webhook_url and getattr(s, "block_webhook_private_ips", True):
        if _is_blocked_webhook_url(webhook_url):
            errors.append("webhook_url cannot target localhost or private IPs")
    if scene_count < SCENE_COUNT_MIN or scene_count > SCENE_COUNT_MAX:
        errors.append(f"scene_count must be between {SCENE_COUNT_MIN} and {SCENE_COUNT_MAX}")
    elif scene_count > 6:
        warnings.append("More than 6 scenes use single-shot mode (longer generation time)")
    if aspect_ratio not in (AspectRatio.PORTRAIT, AspectRatio.LANDSCAPE, AspectRatio.SQUARE):
        errors.append("Invalid aspect_ratio")
    if style_id and (style_id or "").strip():
        valid_ids = set(list_style_ids())
        if style_id.strip().lower() not in valid_ids:
            errors.append(f"style_id must be one of: {', '.join(sorted(valid_ids))}")
    return errors, warnings


def validate_script(script: VideoScript) -> List[str]:
    """
    Validate script before KIE call.
    Returns list of error strings; empty means valid.
    """
    errors: List[str] = []
    if not script:
        return ["Script is missing"]
    if not (getattr(script, "core_concept", None) or "").strip():
        errors.append("Script missing core_concept")
    if not (getattr(script, "character_description", None) or "").strip():
        errors.append("Script missing character_description")
    scenes = getattr(script, "scenes", None) or []
    if not scenes:
        errors.append("Script has no scenes")
    for i, s in enumerate(scenes):
        vd = (getattr(s, "visual_description", None) or "").strip()
        if not vd:
            errors.append(f"Scene {i + 1} missing visual_description")
    return errors


def validate_before_kie(
    script: VideoScript,
    aspect_ratio: AspectRatio,
    multi_prompt: List[dict],
) -> List[str]:
    """
    Validate KIE payload before calling Kling (avoid burning credits on invalid data).
    Returns list of error strings; empty means valid.
    """
    errors: List[str] = []
    script_errs = validate_script(script)
    if script_errs:
        return script_errs

    if aspect_ratio not in (AspectRatio.PORTRAIT, AspectRatio.LANDSCAPE, AspectRatio.SQUARE):
        errors.append("Invalid aspect_ratio for KIE")

    if not multi_prompt:
        errors.append("multi_prompt is empty")
        return errors

    total_duration = 0
    for idx, item in enumerate(multi_prompt):
        prompt = (item.get("prompt") or "").strip()
        duration = item.get("duration", 3)
        try:
            d = int(duration)
        except (TypeError, ValueError):
            d = 3
        if d < 1 or d > 12:
            errors.append(f"Prompt {idx + 1}: duration must be 1-12 seconds")
        total_duration += d
        if len(prompt) > KLING_PROMPT_MAX_CHARS:
            errors.append(f"Prompt {idx + 1}: exceeds {KLING_PROMPT_MAX_CHARS} chars")
        if not prompt:
            errors.append(f"Prompt {idx + 1}: empty prompt")

    if total_duration < MIN_MULTI_PROMPT_TOTAL_SECONDS:
        errors.append(f"Total duration {total_duration}s below KIE minimum {MIN_MULTI_PROMPT_TOTAL_SECONDS}s")
    if total_duration > MAX_MULTI_PROMPT_TOTAL_SECONDS:
        errors.append(f"Total duration {total_duration}s exceeds KIE maximum {MAX_MULTI_PROMPT_TOTAL_SECONDS}s")

    return errors
