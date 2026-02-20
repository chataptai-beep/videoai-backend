"""
Storyboard Generator: turns a prompt into a structured scene-by-scene plan.

Creates a storyboard immediately after prompt expansion, before script generation.
Each scene has: scene_number, action, setting, characters, camera_suggestion, transition_from_previous.
"""

import json
import logging
from typing import List, Optional

import httpx

from config import get_settings, sanitize_header_token
from models.schemas import Storyboard, StoryboardScene

logger = logging.getLogger(__name__)

VALID_CAMERA_ANGLES = [
    "drone_wide", "pov", "close_up", "tracking", "low_angle",
    "over_shoulder", "dolly_in", "high_angle", "dutch_angle", "medium_static"
]


def _pick_valid_camera(angle: str, used: set) -> str:
    """Return a valid camera angle, preferring the suggested one if unused."""
    s = (angle or "").strip().lower().replace("-", "_")
    if s in VALID_CAMERA_ANGLES and s not in used:
        return s
    for a in VALID_CAMERA_ANGLES:
        if a not in used:
            return a
    return VALID_CAMERA_ANGLES[len(used) % len(VALID_CAMERA_ANGLES)]


class StoryboardGenerator:
    """
    Generates a storyboard from a text prompt using OpenAI GPT-4o-mini.
    Output is a list of scenes with action, setting, characters, camera, and transition notes.
    """

    def _auth_header(self) -> str:
        key = sanitize_header_token((get_settings().openai_api_key or ""))
        if not key:
            raise ValueError("OPENAI_API_KEY is not set. Add OPENAI_API_KEY to .env and restart.")
        return "Bearer " + key

    async def generate(
        self,
        prompt: str,
        scene_count: int = 5,
        vibe: Optional[str] = None,
    ) -> Storyboard:
        """
        Generate a storyboard from the prompt.
        Returns a Storyboard with scene_count scenes.
        """
        vibe_hint = (vibe or "literal").strip()
        system = f"""You are a video director. Given a prompt, output a storyboard: a shot-by-shot plan.

RULES:
- Output EXACTLY {scene_count} scenes, in narrative order.
- Each scene must have: scene_number, action, setting, characters, camera_suggestion, transition_from_previous.
- action: one clear visual beat (what we see happen).
- setting: where it takes place (same or evolving).
- characters: who appears (keep consistent).
- camera_suggestion: one of: drone_wide, pov, close_up, tracking, low_angle, over_shoulder, dolly_in, high_angle, dutch_angle, medium_static. Use different angles per scene.
- transition_from_previous: how this scene flows from the last (e.g. "cut to hands", "same room, character moves").

Vibe: {vibe_hint}. Structure scenes accordingly.

Output ONLY valid JSON in this format (no markdown, no code blocks):
{{
  "scenes": [
    {{
      "scene_number": 1,
      "action": "Brief visual action",
      "setting": "Where",
      "characters": "Who",
      "camera_suggestion": "close_up",
      "transition_from_previous": "Opens on..."
    }}
  ]
}}"""

        user = f"""Create a storyboard for this video:

"{prompt}"

Output exactly {scene_count} scenes. Each scene = one clear visual beat. Same characters and setting unless the story demands a change. Use different camera angles per scene."""

        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": self._auth_header(), "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.6,
            "max_tokens": 2000,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        content = content.strip()
        if content.startswith("```"):
            for marker in ("```json", "```"):
                if content.startswith(marker):
                    content = content[len(marker):].strip()
                    break
            if content.endswith("```"):
                content = content[:-3].strip()

        parsed = json.loads(content)
        raw_scenes = parsed.get("scenes") or []

        # Build StoryboardScene list and normalize camera angles
        used_angles = set()
        scenes: List[StoryboardScene] = []
        for i, raw in enumerate(raw_scenes[:scene_count]):
            if i >= scene_count:
                break
            num = raw.get("scene_number") or (i + 1)
            action = (raw.get("action") or "").strip() or "Visual beat"
            setting = (raw.get("setting") or "").strip() or "Scene"
            chars = (raw.get("characters") or "").strip() or "Character"
            cam = _pick_valid_camera(raw.get("camera_suggestion") or "", used_angles)
            used_angles.add(cam)
            trans = (raw.get("transition_from_previous") or "").strip() or "Cut"
            scenes.append(
                StoryboardScene(
                    scene_number=int(num) if isinstance(num, (int, float)) else (i + 1),
                    action=action,
                    setting=setting,
                    characters=chars,
                    camera_suggestion=cam,
                    transition_from_previous=trans,
                )
            )

        if len(scenes) < scene_count:
            raise ValueError(f"Storyboard returned {len(scenes)} scenes, expected {scene_count}")

        return Storyboard(scenes=scenes)
