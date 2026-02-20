"""
Deterministic style presets for Kling 3.0 prompting.
These presets keep visual style consistent and reduce LLM drift.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

@dataclass(frozen=True)
class StylePreset:
    style_id: str
    label: str
    visual_style: str
    background_hint: str
    pacing: str
    camera_notes: str
    audio_notes: str
    negative_phrases: str

STYLE_PRESETS: Dict[str, StylePreset] = {
    "real_life": StylePreset(
        style_id="real_life",
        label="Real Life",
        visual_style="Natural light, realistic textures, handheld realism, subtle depth of field.",
        background_hint="A real, lived-in location with practical lighting.",
        pacing="Grounded pacing, one clear action per beat.",
        camera_notes="Handheld or lightly stabilized, intimate framing.",
        audio_notes="Natural room tone, subtle ambience, clear dialogue if present.",
        negative_phrases="No text, no subtitles, no logos, no watermarks, no UI overlays.",
    ),
    "ugc": StylePreset(
        style_id="ugc",
        label="UGC",
        visual_style="Smartphone UGC look, soft window light, authentic casual realism.",
        background_hint="Home or office environment, informal and believable.",
        pacing="Snappy pacing, short beats, high energy.",
        camera_notes="Handheld, eye-level, quick reframes.",
        audio_notes="Natural voice, minimal music, light ambient sound.",
        negative_phrases="No text, no captions, no logos, no watermarks.",
    ),
    "cinematic": StylePreset(
        style_id="cinematic",
        label="Cinematic",
        visual_style="Cinematic lighting, controlled contrast, filmic color grade, shallow depth of field.",
        background_hint="Cinematic location with rich textures and motivated lighting.",
        pacing="Measured pacing, dramatic beats, strong visual payoff.",
        camera_notes="Dolly, tracking, or intentional handheld movement.",
        audio_notes="Subtle cinematic ambience, clear dialogue, no overpowering music.",
        negative_phrases="No text, no subtitles, no logos, no watermarks.",
    ),
    "documentary": StylePreset(
        style_id="documentary",
        label="Documentary",
        visual_style="Documentary realism, neutral color, clear visibility, credible lighting.",
        background_hint="Real-world location that supports the subject matter.",
        pacing="Steady pacing, informative beats, natural transitions.",
        camera_notes="Stable handheld or tripod, observational framing.",
        audio_notes="Clean dialogue, low ambient bed, subtle environment sound.",
        negative_phrases="No text, no subtitles, no logos, no watermarks.",
    ),
    "product": StylePreset(
        style_id="product",
        label="Product",
        visual_style="Premium product ad, crisp lighting, clean highlights, sharp focus.",
        background_hint="Clean setting or studio environment with controlled light.",
        pacing="Punchy beats, clear demo steps, satisfying reveal.",
        camera_notes="Macro close-ups, controlled moves, clean framing.",
        audio_notes="Subtle sound design, soft swishes, clear dialogue if present.",
        negative_phrases="No text, no subtitles, no logos, no watermarks.",
    ),
}

_KEYWORD_STYLE_MAP = [
    ("ugc", ["ugc", "selfie", "vlog", "tiktok", "creator", "influencer", "talking head", "phone"]),
    ("documentary", ["documentary", "interview", "case study", "behind the scenes", "bts", "reportage"]),
    ("product", ["product", "unboxing", "demo", "review", "launch", "feature", "app", "saas", "tool"]),
    ("cinematic", ["cinematic", "film", "movie", "hollywood", "epic", "dramatic", "moody"]),
]

_VIBE_STYLE_MAP = {
    "literal": "real_life",
    "realistic": "real_life",
    "viral": "ugc",
    "documentary": "documentary",
    "ad": "product",
    "tutorial": "product",
    "cinematic": "cinematic",
}

def list_style_ids() -> Iterable[str]:
    return STYLE_PRESETS.keys()

def get_style_preset(style_id: str) -> Optional[StylePreset]:
    if not style_id:
        return None
    return STYLE_PRESETS.get(style_id.strip().lower())

def select_style_preset(prompt: str, vibe: Optional[str], style_id: Optional[str]) -> StylePreset:
    explicit = get_style_preset(style_id or "")
    if explicit:
        return explicit
    if vibe:
        mapped = _VIBE_STYLE_MAP.get(vibe.strip().lower())
        if mapped and mapped in STYLE_PRESETS:
            return STYLE_PRESETS[mapped]
    prompt_lower = (prompt or "").lower()
    for style_key, keywords in _KEYWORD_STYLE_MAP:
        if any(keyword in prompt_lower for keyword in keywords):
            return STYLE_PRESETS[style_key]
    return STYLE_PRESETS["real_life"]
