"""
Veo 3–optimized prompt enhancer.
Builds per-scene prompts using the 8-element formula: subject, context, action,
style, camera, lighting, audio (dialogue in quotes, SFX, ambient), and exclusions (no subtitles).
Designed for kie.ai Veo 3 / Google Veo–style APIs.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from models.schemas import Scene

logger = logging.getLogger(__name__)

# Shot progression: start wide, end close for emotion (map scene index to shot type)
SHOT_TYPES = ["Wide shot", "Medium shot", "Medium shot", "Medium shot", "Close-up"]

# Camera movement variety for visual interest (cycle by scene index)
CAMERA_MOVEMENTS = [
    {"movement": "Static on tripod", "angle": "Eye level"},
    {"movement": "Slow dolly-in", "angle": "Slightly below eye level"},
    {"movement": "Gentle pan right", "angle": "Eye level"},
    {"movement": "Slow zoom in", "angle": "Eye level"},
    {"movement": "Static", "angle": "Over the shoulder"},
]

# Camera angle from script -> Veo-friendly description
CAMERA_ANGLE_TO_DESCRIPTION = {
    "drone_wide": ("Drone or wide establishing shot", "aerial view, high above"),
    "pov": ("POV shot", "first-person view"),
    "close_up": ("Close-up", "face or hands fill the frame"),
    "tracking": ("Tracking shot", "camera moves with the subject"),
    "low_angle": ("Low angle", "camera low, looking up"),
    "over_shoulder": ("Over the shoulder", "behind character's shoulder"),
    "dolly_in": ("Dolly in", "camera pushes toward subject"),
    "high_angle": ("High angle", "above subject looking down"),
    "dutch_angle": ("Dutch angle", "tilted frame"),
    "medium_static": ("Medium shot", "waist-up, locked off"),
}


@dataclass
class VeoPromptConfig:
    """8-element config for one Veo prompt."""
    shot_type: str
    subject_description: str
    subject_action: str
    location: str
    time_of_day: str
    mood: str
    camera_movement: str
    camera_angle: str
    lighting_type: str
    lighting_mood: str
    dialogue_lines: List[str]
    sfx: List[str]
    ambient: str
    music: Optional[str]
    style: str
    exclude: List[str]


def _extract_topic(prompt: str) -> str:
    """Remove filler and return core topic."""
    if not prompt:
        return ""
    lower = prompt.lower().strip()
    for phrase in ("make me a video about", "create a video about", "video about", "make a video about"):
        if phrase in lower:
            lower = lower.replace(phrase, "").strip()
    return lower or prompt.strip()[:100]


def _select_shot_type(beat_index: int, total_beats: int) -> str:
    if beat_index == 0:
        return "Wide shot"
    if beat_index >= total_beats - 1:
        return "Close-up"
    return "Medium shot"


def _get_camera(beat_index: int, scene_angle: Optional[str]) -> tuple:
    """Return (movement, angle) – use script camera_angle if mapped, else cycle."""
    angle_key = (scene_angle or "").strip().lower().replace("-", "_")
    if angle_key in CAMERA_ANGLE_TO_DESCRIPTION:
        desc, detail = CAMERA_ANGLE_TO_DESCRIPTION[angle_key]
        return (desc, detail)
    idx = beat_index % len(CAMERA_MOVEMENTS)
    c = CAMERA_MOVEMENTS[idx]
    return (c["movement"], c["angle"])


def _generate_subject(topic: str, character_description: str, beat_index: int) -> tuple:
    """(subject_description, subject_action)."""
    if character_description and len(character_description.strip()) > 10:
        desc = character_description.strip()[:200]
    else:
        is_product = any(w in topic for w in ("product", "item", "tool", "app"))
        is_person = any(w in topic for w in ("selling", "teaching", "explaining", "talking", "coffee", "routine", "story"))
        if is_person:
            desc = "confident person in casual outfit"
        elif is_product:
            desc = f"{topic[:50]} product on clean surface"
        else:
            desc = "person in modern minimalist setting"
    if beat_index == 0:
        action = "looking directly at camera with authentic presence"
    else:
        action = "gesturing naturally while speaking"
    return (desc, action)


def _generate_context(topic: str, background_theme: str) -> tuple:
    """(location, time_of_day, mood)."""
    if background_theme and len(background_theme.strip()) > 5:
        loc = background_theme.strip()[:120]
        return (loc, "natural lighting", "consistent with scene")
    lower = topic.lower()
    if any(w in lower for w in ("business", "selling", "money", "profit")):
        return ("modern office with clean desk and laptop", "bright morning light", "professional yet approachable")
    if any(w in lower for w in ("ai", "tech", "automation", "software")):
        return ("minimalist home office with monitors", "natural window light", "innovative and focused")
    if any(w in lower for w in ("coffee", "fitness", "health", "lifestyle", "kitchen", "morning")):
        return ("cozy space with natural materials", "warm morning light", "authentic and relatable")
    return ("neutral indoor space", "soft natural lighting", "calm and confident")


def _generate_lighting(topic: str, visual_style: Optional[str]) -> tuple:
    """(lighting_type, lighting_mood)."""
    if visual_style and len(visual_style.strip()) > 5:
        return (visual_style.strip()[:80], "consistent mood")
    lower = (topic or "").lower()
    if any(w in lower for w in ("business", "professional", "corporate")):
        return ("Clean studio lighting", "even illumination, minimal shadows")
    if any(w in lower for w in ("creative", "art", "design")):
        return ("Dramatic side lighting", "artistic shadows, high contrast")
    return ("Soft natural window light", "warm and inviting atmosphere")


def _generate_sfx(topic: str) -> List[str]:
    lower = topic.lower()
    if "coffee" in lower:
        return ["coffee brewing", "cup clinking"]
    if any(w in lower for w in ("tech", "ai", "software", "computer")):
        return ["keyboard typing", "mouse clicks"]
    if any(w in lower for w in ("business", "office")):
        return ["paper rustling", "pen writing"]
    if any(w in lower for w in ("fitness", "gym", "workout")):
        return ["weights clinking", "footsteps"]
    if any(w in lower for w in ("cooking", "chef", "kitchen")):
        return ["sizzling pan", "chopping sounds"]
    return ["ambient room tone"]


def _generate_ambient(topic: str) -> str:
    lower = topic.lower()
    if "office" in lower or "business" in lower:
        return "quiet office atmosphere, distant activity"
    if "home" in lower or "kitchen" in lower or "coffee" in lower:
        return "peaceful home ambience, subtle background"
    if "outdoor" in lower or "nature" in lower:
        return "gentle outdoor sounds, birds chirping"
    if "cafe" in lower or "coffee" in lower:
        return "soft cafe ambience, espresso machine"
    return "clean indoor atmosphere"


def _select_style(topic: str) -> str:
    lower = topic.lower()
    if any(w in lower for w in ("how to", "tutorial", "learn", "explain")):
        return "Professional educational style"
    if any(w in lower for w in ("story", "journey", "transformation")):
        return "Cinematic documentary style"
    return "Cinematic realism"


def build_veo_prompt(
    *,
    scene: Scene,
    scene_index: int,
    total_scenes: int,
    topic: str,
    character_description: str = "",
    background_theme: str = "",
    visual_style: Optional[str] = None,
) -> str:
    """
    Build a single Veo 3–optimized prompt for one scene. Per Veo guidelines: clear and specific,
    single-scene focus, dialogue with colons (no quotation marks), no on-screen text.
    """
    topic_clean = _extract_topic(topic)
    shot_type = _select_shot_type(scene_index, total_scenes)
    subj_desc, subj_action = _generate_subject(topic_clean, character_description, scene_index)
    loc, time_of_day, mood = _generate_context(topic_clean, background_theme)
    cam_mov, cam_angle = _get_camera(scene_index, getattr(scene, "camera_angle", None))
    light_type, light_mood = _generate_lighting(topic_clean, visual_style)
    dialogue_raw = (scene.dialogue or "").strip()
    dialogue_lines = [line.strip() for line in dialogue_raw.split("\n") if line.strip()][:2]
    sfx = _generate_sfx(topic_clean)
    ambient = _generate_ambient(topic_clean)
    style = _select_style(topic_clean)

    config = VeoPromptConfig(
        shot_type=shot_type,
        subject_description=subj_desc,
        subject_action=subj_action,
        location=loc,
        time_of_day=time_of_day,
        mood=mood,
        camera_movement=cam_mov,
        camera_angle=cam_angle,
        lighting_type=light_type,
        lighting_mood=light_mood,
        dialogue_lines=dialogue_lines,
        sfx=sfx,
        ambient=ambient,
        music="subtle upbeat background music, low volume",
        style=style,
        exclude=["subtitles", "text overlays"],
    )

    return _format_veo_prompt(config, subj_desc)


def _format_veo_prompt(config: VeoPromptConfig, subject_label: str) -> str:
    """Turn config into natural-language Veo prompt. Per Veo guidelines: use colons for speech, no quotation marks."""
    parts = []

    # 1. Style + shot + subject + action (single-scene, clear and specific)
    parts.append(
        f"{config.style} {config.shot_type} of {config.subject_description} "
        f"{config.subject_action}"
    )
    # 2. Context
    parts.append(f"in {config.location}, {config.time_of_day}.")
    # 3. Camera
    parts.append(f"{config.camera_movement} camera. {config.camera_angle}.")
    # 4. Lighting
    parts.append(f"{config.lighting_type} with {config.lighting_mood}.")
    # 5. Dialogue: Veo guideline – use colon after speaker, no quotation marks (avoids text rendered in frame)
    if config.dialogue_lines:
        first_word = (config.subject_description.split() or ["Subject"])[0]
        for line in config.dialogue_lines:
            parts.append(f"{first_word} says: {line}")
    # 6. Audio layers (2–3 max: dialogue already above; SFX + ambient + music)
    audio_parts = []
    if config.sfx:
        audio_parts.append(f"SFX: {', '.join(config.sfx)}")
    if config.ambient:
        audio_parts.append(f"Ambient: {config.ambient}")
    if config.music:
        audio_parts.append(f"Music: {config.music}")
    if audio_parts:
        parts.append(f"Audio: {'; '.join(audio_parts)}.")
    # 7. Exclusions – prevent on-screen text
    if config.exclude:
        parts.append(f"No {', '.join(config.exclude)}.")

    return " ".join(parts)
