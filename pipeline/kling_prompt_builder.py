"""
Kling 3.0 prompt builder — aligned with Kling best practices for perfect video.

Kling 3.0 expects: Subject + Action + Environment + Camera (optional).
- Anchor subject in shot 1; same phrase every shot (character consistency).
- One clear action per shot; describe motion explicitly (camera + subject).
- Include setting/environment so Kling grounds the scene.
- Max 500 chars per prompt; we keep each shot clear and shootable.
"""

import logging
from typing import List, Optional

from models.schemas import Scene

logger = logging.getLogger(__name__)

KLING_PROMPT_MAX_CHARS = 500
DEFAULT_NEGATIVE_PHRASES = "No text, no subtitles, no logos, no watermarks."


def _trim_to_kling_max(text: str, max_chars: int = KLING_PROMPT_MAX_CHARS) -> str:
    """Trim to max_chars at a word boundary; Kling expects prompts under 500 chars."""
    if not text or len(text) <= max_chars:
        return (text or "").strip()
    s = text[: max_chars - 3].rstrip()
    last_space = s.rfind(" ")
    if last_space > max_chars // 2:
        s = s[: last_space]
    return s.rstrip(".,") + "..."


# Kling understands these; use consistently (fal blog: "profile shots, macro close-ups, tracking, POV, shot-reverse-shot")
SHOT_LABELS = {
    "drone_wide": "Wide aerial shot",
    "pov": "POV shot",
    "close_up": "Close-up",
    "tracking": "Tracking shot",
    "low_angle": "Low angle shot",
    "over_shoulder": "Over-the-shoulder shot",
    "dolly_in": "Dolly in",
    "high_angle": "High angle shot",
    "dutch_angle": "Dutch angle shot",
    "medium_static": "Medium shot",
}

# Explicit camera behavior per shot type (fal: "describe how the camera behaves: tracking, following, freezing, panning")
CAMERA_BEHAVIOR = {
    "drone_wide": "Camera sweeps high above, establishing the scene.",
    "pov": "First-person view; camera is the character's eyes.",
    "close_up": "Camera stays tight on the subject.",
    "tracking": "Camera tracks with the subject, following the action.",
    "low_angle": "Camera low, looking up at the subject.",
    "over_shoulder": "Camera behind subject's shoulder.",
    "dolly_in": "Camera pushes in toward the subject.",
    "high_angle": "Camera above, looking down.",
    "dutch_angle": "Camera tilted for tension.",
    "medium_static": "Medium shot, camera stays with the action.",
}


def _anchor_subject(character_description: str) -> str:
    """One short, consistent subject phrase for all shots (fal: 'define core subjects at the beginning')."""
    raw = (character_description or "").strip()
    if not raw or len(raw) <= 2:
        logger.warning("Kling: character_description empty or too short; using fallback.")
        return "The person"
    # First phrase or first 5 words max — keep it anchorable
    subject = raw.split(",")[0].strip() if "," in raw else raw
    subject = " ".join(subject.split()[:5]) if len(subject) > 30 else subject
    return subject or "The person"


def _setting_phrase(background_theme: str, visual_style: str) -> str:
    """One short setting phrase (fal: layer environment context)."""
    theme = (background_theme or "").strip()
    style = (visual_style or "").strip()
    if theme and style:
        return f"{theme}, {style}."
    if theme:
        return f"{theme}."
    if style:
        return f"{style}."
    return ""


def _shot_and_camera(camera_angle: str) -> tuple:
    """Shot type label + camera behavior (fal: 'describe motion explicitly')."""
    key = (camera_angle or "").strip().lower().replace("-", "_")
    label = SHOT_LABELS.get(key, "Medium shot")
    behavior = CAMERA_BEHAVIOR.get(key, "Camera stays with the action.")
    return label, behavior


def build_kling_prompt_for_scene(
    scene: Scene,
    scene_index: int,
    total_scenes: int,
    character_description: str = "",
    background_theme: str = "",
    visual_style: str = "",
    core_concept: str = "",
    previous_scenes_descriptions: Optional[List[str]] = None,
    simple_style: bool = True,
    avoid_phrases: Optional[str] = None,
) -> str:
    """
    Build one Kling O3 prompt per shot. Cinematic structure: subject → setting → shot type → camera → action.
    simple_style=True: compact but still anchored (subject + action + optional style).
    simple_style=False: full cinematic (subject, setting, shot type, camera behavior, action, optional dialogue).
    """
    action = (scene.visual_description or "").strip()
    if not action:
        action = "Same character continues the moment."

    subject = _anchor_subject(character_description)
    angle_key = (getattr(scene, "camera_angle", None) or "").strip().lower().replace("-", "_")
    shot_label, camera_behavior = _shot_and_camera(angle_key)

    if simple_style:
        action = action.rstrip(".")
        if action and not action[0].isupper():
            action = action[0].upper() + action[1:]
        setting = _setting_phrase(background_theme, visual_style)
        if setting:
            line = f"{subject}. {setting.rstrip('.')} {action}."
        else:
            line = f"{subject} {action}."
        if (scene.dialogue or "").strip():
            spoken = (scene.dialogue or "").strip()
            line = f"{line.rstrip('.')} {subject} says: \"{spoken}\" (spoken dialogue, no on-screen text)."
        if (visual_style or "").strip() and not setting:
            style = visual_style.strip().rstrip(".")
            line = f"{line.rstrip('.')}, {style}."
        if avoid_phrases:
            line = f"{line.rstrip('.')} {avoid_phrases.strip()}"
        return _trim_to_kling_max(line)

    # Full cinematic: anchor subject, setting, shot type, camera, action (fal blog structure)
    action = action.rstrip(".")
    if action and not action[0].isupper():
        action = action[0].upper() + action[1:]
    parts = []
    parts.append(f"{subject}.")
    setting = _setting_phrase(background_theme, visual_style)
    if setting:
        parts.append(setting)
    parts.append(f"{shot_label}. {camera_behavior} {action}.")
    dialogue = (scene.dialogue or "").strip()
    if dialogue:
        tone = "attention-grabbing" if scene_index == 0 else ("conclusive" if scene_index >= total_scenes - 1 else "natural")
        parts.append(f"{subject} says (spoken, {tone}): \"{dialogue}\". No on-screen text.")
    if previous_scenes_descriptions:
        parts.append("Continuing from previous moment.")
    line = " ".join(parts)
    if avoid_phrases:
        line = f"{line.rstrip('.')} {avoid_phrases.strip()}"
    return _trim_to_kling_max(line)


def build_kling_multi_shot_prompt(
    scenes: List[Scene],
    character_description: str = "",
    background_theme: str = "",
    visual_style: str = "",
    core_concept: str = "",
    simple_style: bool = True,
    shot_duration_seconds: int = 3,
    avoid_phrases: Optional[str] = None,
) -> List[dict]:
    """
    Build multi_prompt list for fal Kling O3. Shot 1 establishes subject + setting; shots 2+ use
    'Same character, same setting' for continuity (fal: 'keep descriptions consistent across shots').
    shot_duration_seconds: 1–12 per Kling; default 3 (one short scene per action).
    Returns [{"prompt": str, "duration": "3"}, ...].
    """
    duration = min(12, max(1, shot_duration_seconds))
    result = []
    prev_descriptions = []
    subject = _anchor_subject(character_description)
    setting_phrase = _setting_phrase(background_theme, visual_style)

    for i, scene in enumerate(scenes):
        prompt = build_kling_prompt_for_scene(
            scene=scene,
            scene_index=i,
            total_scenes=len(scenes),
            character_description=character_description,
            background_theme=background_theme,
            visual_style=visual_style or "",
            core_concept=core_concept or "",
            previous_scenes_descriptions=prev_descriptions[:] if prev_descriptions else None,
            simple_style=simple_style,
            avoid_phrases=avoid_phrases,
        )
        if simple_style and i >= 1:
            if not prompt.strip().lower().startswith("same character"):
                prompt = f"Same character, same setting. {prompt.strip()}"
            prompt = _trim_to_kling_max(prompt)
        result.append({"prompt": prompt, "duration": str(duration)})
        prev_descriptions.append((scene.visual_description or "").strip())
    return result
