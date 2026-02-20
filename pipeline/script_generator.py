"""
Script Generator using OpenAI GPT-4o-mini.
Generates structured 5-scene video scripts from text prompts.
"""

import json
import logging
import re
from typing import Optional

import httpx

from config import get_settings, sanitize_header_token
from models.schemas import Scene, VideoScript, Storyboard
from pipeline.style_presets import StylePreset

logger = logging.getLogger(__name__)

VALID_CAMERA_ANGLES = [
    "drone_wide", "pov", "close_up", "tracking", "low_angle",
    "over_shoulder", "dolly_in", "high_angle", "dutch_angle", "medium_static"
]

# Pairs of (prompt_keywords, script_keywords): if prompt has left and script has right, override style
STYLE_CONTRADICTION_PROMPT = ("cozy", "warm", "soft", "sunset", "day", "light", "bright", "peaceful")
STYLE_CONTRADICTION_SCRIPT = ("neon", "cyberpunk", "dark", "noir", "night", "cold", "harsh", "grim")


def _sanitize_script_style(script_data: dict, prompt: str) -> None:
    """Override visual_style when it clearly contradicts the user prompt (e.g. cozy vs neon)."""
    prompt_lower = (prompt or "").lower()
    style = (script_data.get("visual_style") or "").lower()
    if not style:
        return
    prompt_has_warm = any(w in prompt_lower for w in STYLE_CONTRADICTION_PROMPT)
    script_has_cold = any(w in style for w in STYLE_CONTRADICTION_SCRIPT)
    if prompt_has_warm and script_has_cold:
        script_data["visual_style"] = "Warm, soft lighting, inviting mood. Professional."
        logger.info("Overrode visual_style (script contradicted prompt: warm vs cold/neon)")
        return
    prompt_has_cold = any(w in prompt_lower for w in STYLE_CONTRADICTION_SCRIPT)
    script_has_warm = any(w in style for w in STYLE_CONTRADICTION_PROMPT)
    if prompt_has_cold and script_has_warm:
        script_data["visual_style"] = "Cool, high-contrast lighting, cinematic. Professional."
        logger.info("Overrode visual_style (script contradicted prompt: dark/neon vs warm)")


def _normalize_script_scenes(scenes: list, scene_count: int) -> None:
    """Ensure exactly scene_count scenes and unique camera_angle per scene."""
    if len(scenes) > scene_count:
        del scenes[scene_count:]
    if len(scenes) < scene_count:
        raise Exception(f"Script returned {len(scenes)} scenes but {scene_count} were requested")
    used = set()
    for i, scene in enumerate(scenes):
        angle = (getattr(scene, "camera_angle", None) or "").strip().lower().replace("-", "_")
        if angle in VALID_CAMERA_ANGLES and angle not in used:
            used.add(angle)
            if not scene.camera_angle:
                scene.camera_angle = angle
        else:
            for a in VALID_CAMERA_ANGLES:
                if a not in used:
                    scene.camera_angle = a
                    used.add(a)
                    break
            else:
                scene.camera_angle = VALID_CAMERA_ANGLES[i % len(VALID_CAMERA_ANGLES)]


class ScriptGenerator:
    """
    Generates video scripts using OpenAI's GPT-4o-mini.
    Produces structured JSON output with character description and scenes.
    """
    
    def _get_system_prompt(self, scene_count: int, style_context: str, tone_context: str, brand_context: str = "", value_context: str = "") -> str:
        """Generate a script that matches what the user imagines when they read their prompt."""
        extra_context = "\n".join(s for s in (brand_context, value_context) if s)
        return f"""You are a producer and director for viral short-form video (Reels, TikTok, Shorts).

{style_context}
{tone_context}
{extra_context}

PRIMARY RULE: Deliver the video the user imagined—polished and complete. Match their prompt; same characters, setting, mood. When the idea involves an object (food, gift, etc.), show the full sequence: scene 1 = object + first action; then prepare; then use/consume. core_concept = one sentence of what this video shows.
If the prompt implies more beats than scenes, compress without losing the ending: include a clear beginning, middle, and final payoff.

VIRAL REEL STRUCTURE (act as producer):
- Scene 1 = HOOK: Grab attention in the first second. One clear visual or action. No slow intro.
- Scenes 2 to N-1 = BUILD: Develop the idea step by step. Same character/setting as the user asked for.
- Scene N = PAYOFF: Clear ending—revelation, result, or takeaway. Satisfying close.

CAMERA ANGLES & QUICK CUTS (keep story consistent, vary the shot):
- Assign a DIFFERENT camera_angle to each scene so the edit feels like quick cuts between angles. Same character and setting throughout—only the camera changes.
- Use exactly one of these per scene (no duplicates in one video): drone_wide, pov, close_up, tracking, low_angle, over_shoulder, dolly_in, high_angle, dutch_angle, medium_static.
- drone_wide = aerial/drone shot, high above, sweeping. pov = first-person view, viewer is the character. close_up = face or hands, tight. tracking = camera moves with subject. low_angle = camera low, looking up. over_shoulder = behind character's shoulder. dolly_in = camera pushes toward subject. high_angle = above subject, not drone. dutch_angle = tilted frame. medium_static = waist-up, locked off.
- Pick angles that fit the beat (e.g. hook = punchy close_up or drone_wide; payoff = dolly_in or close_up).

TECHNICAL (optimized for Kling video model):
- Format: Vertical 9:16. Fast pacing.
- character_description: ONE short phrase only (e.g. "Grandma, elderly woman in blue cardigan"). Same exact phrase in every scene—required for Kling consistency.
- visual_description: ONE clear subject motion per scene (what we SEE—e.g. picks up the cup, takes a sip). 10–25 words, concrete and shootable. No vague or abstract actions.
- dialogue: Short, punchy caption that fits this moment.
- dialogue: Spoken line for the character (audio), which we can also use as captions.
- visual_style and background_theme: Taken directly FROM the user's prompt (lighting, mood, place).

Output ONLY valid JSON in this exact format (no markdown, no code blocks):
{{
  "core_concept": "One sentence: what this video MUST show, from the user's prompt.",
  "character_description": "One short phrase for the protagonist (e.g. 'Woman with red hair in white shirt'). Same in every scene.",
  "visual_style": "Style derived from user prompt (lighting, mood, era, look).",
  "background_theme": "Setting and environment from user prompt.",
  "scenes": [
    {{
      "scene_number": 1,
      "visual_description": "Concrete action for this shot; character clothing.",
      "dialogue": "Punchy caption for this scene.",
      "camera_angle": "drone_wide"
    }}
  ]
}}

You MUST output EXACTLY {scene_count} scenes. Output ONLY valid JSON."""

    def _get_system_prompt_viral(self, scene_count: int, style_context: str, tone_context: str, brand_context: str = "", value_context: str = "") -> str:
        """Viral short-form structure: HOOK → PROBLEM → INSIGHT → SOLUTION → CTA. Output is JSON for pipeline."""
        extra_context = "\n".join(s for s in (brand_context, value_context) if s)
        return f"""You are a viral short-form video script generator. When given a topic or idea, you will:

{style_context}
{tone_context}
{extra_context}

1. ANALYZE the input to extract: core topic, implied audience, best hook type, appropriate CTA keyword.
2. GENERATE a script that follows this structure (map to exactly {scene_count} scenes):
   - HOOK (0-2s): Scroll-stopping opening → Scene 1
   - PROBLEM (2-6s): Relatable pain point → Scene 2
   - INSIGHT (6-15s): The shift or revelation → Scene 3
   - SOLUTION (15-80%): The better way → Scenes 4 to N-1
   - CTA (last 2-3s): Clear action → Scene N

3. RULES:
   - 3rd-grade reading level. Short punchy lines. No fluff.
   - Natural spoken dialogue. Every line earns its place.
   - Infer tone from topic (business = authoritative, lifestyle = relatable). Generate a relevant CTA keyword.

4. OUTPUT: Valid JSON only. Kling expects one anchor character phrase (same every scene) and one clear motion per scene.
   If the prompt implies more beats than scenes, compress without losing the ending: include a clear beginning, middle, and final payoff.
   Use this exact format (no markdown, no code blocks):
{{
  "core_concept": "One sentence: the main takeaway or story of this video.",
  "character_description": "One short phrase for who we see (e.g. 'Host in casual blazer'). Same exact phrase every scene—required for Kling.",
  "visual_style": "Look and mood (e.g. professional, casual, energetic).",
  "background_theme": "Setting (e.g. office, kitchen, minimal).",
  "scenes": [
    {{
      "scene_number": 1,
      "visual_description": "One clear subject motion we SEE (action + framing). Hook moment.",
      "dialogue": "Exact spoken line for this scene. Short and punchy.",
      "camera_angle": "close_up"
    }},
    {{
      "scene_number": 2,
      "visual_description": "Visual for problem / pain point.",
      "dialogue": "Spoken line.",
      "camera_angle": "medium_static"
    }}
    // ... exactly {scene_count} scenes total
  ]
}}

CAMERA ANGLES (use one per scene, all different): drone_wide, pov, close_up, tracking, low_angle, over_shoulder, dolly_in, high_angle, dutch_angle, medium_static.
You MUST output EXACTLY {scene_count} scenes. Output ONLY valid JSON."""

    def _get_system_prompt_literal(self, scene_count: int, style_context: str, tone_context: str, brand_context: str = "", value_context: str = "") -> str:
        """Real-life sequence: one activity in N steps, in order."""
        extra_context = "\n".join(s for s in (brand_context, value_context) if s)
        return f"""You are a scriptwriter for a short video. Break the activity into exactly {scene_count} steps that really happen, in order.

{style_context}
{tone_context}
{extra_context}

SEQUENCE:
- Scene 1 = first moment: object + character's first action (e.g. banana in basket, she grabs it). Keep it simple.
- Scenes 2 to N-1 = middle steps in order (peel then eat; unwrap then open; etc.). One step per scene.
- Scene N = last moment (finishing, satisfied).
- Food: show item → prepare → eat. Drinks: container → make/pour → drink. Opening: show it → unwrap → use.
If the prompt implies more beats than scenes, compress without losing the ending: include a clear beginning, middle, and final payoff.

RULES (Kling 3.0: one action per shot = best results):
- SAME character and setting every scene. One continuous sequence.
- character_description: ONE short phrase (e.g. "Grandma, elderly woman in blue cardigan"). Use the EXACT same phrase every scene—Kling needs this for consistency.
- visual_description: Exactly ONE physical action per scene. Never combine two actions (e.g. use "She peels the banana" in one scene and "She takes a bite" in the next—not "She peels and eats"). Scene 1: object + first action (e.g. banana in basket, she grabs it). 10–25 words, concrete and shootable.
- dialogue: Spoken line for the character (audio), which we can also use as captions.
- visual_style and background_theme: from the prompt (e.g. cozy kitchen, morning light).
- Use a DIFFERENT camera_angle per scene: drone_wide, pov, close_up, tracking, low_angle, over_shoulder, dolly_in, high_angle, dutch_angle, medium_static.

OUTPUT: Valid JSON only (no markdown):
{{
  "core_concept": "One sentence: [character] [doing the exact activity from the prompt].",
  "character_description": "One short phrase for the person (e.g. 'Grandma, elderly woman in blue cardigan'). Same in every scene.",
  "visual_style": "Mood and lighting from prompt.",
  "background_theme": "Where this happens.",
  "scenes": [
    {{ "scene_number": 1, "visual_description": "Concrete first moment we see.", "dialogue": "Optional short line.", "camera_angle": "medium_static" }},
    ... exactly {scene_count} scenes in real-life order
  ]
}}

You MUST output EXACTLY {scene_count} scenes in logical real-life order. Output ONLY valid JSON."""

    def __init__(self):
        self.api_key = get_settings().openai_api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def _extract_dialogue_lines(self, prompt: str) -> list[str]:
        text = (prompt or "").strip()
        if not text:
            return []
        pattern = re.compile(
            r"\[(?P<speaker>[^\]]+)\]\s*:?\s*[\"\u201c](?P<line>[^\"\u201d]+)[\"\u201d]"
        )
        matches = pattern.findall(text)
        if not matches:
            return []
        lines = []
        for speaker, line in matches:
            speaker_clean = " ".join((speaker or "").split())
            line_clean = " ".join((line or "").split())
            if speaker_clean and line_clean:
                lines.append(f"{speaker_clean}: {line_clean}")
        return lines

    def _extract_dialogue_lock(self, prompt: str) -> tuple[str, str, list[str]]:
        """Extract explicit dialogue lines from the prompt for verbatim reuse."""
        lines = self._extract_dialogue_lines(prompt)
        if not lines:
            return "", "", []
        speakers = []
        for line in lines:
            speaker = line.split(":", 1)[0].strip()
            if speaker and speaker not in speakers:
                speakers.append(speaker)
        speaker_hint = "Include these speakers in character_description: " + ", ".join(speakers) + "."
        lock = (
            "Dialogue lock (use verbatim, in order). Do not invent new dialogue. "
            "If fewer scenes than lines, merge multiple lines into the same scene; "
            "if more scenes, leave remaining dialogue empty.\n" + "\n".join(lines)
        )
        return speaker_hint, lock, lines

    def _apply_dialogue_lock(self, script: VideoScript, lines: list[str]) -> None:
        if not lines:
            return
        scenes = script.scenes
        if not scenes:
            return
        if len(lines) <= len(scenes):
            for idx, line in enumerate(lines):
                scenes[idx].dialogue = line
            return
        # More lines than scenes: merge extra lines into last scene
        for idx, scene in enumerate(scenes):
            if idx < len(scenes) - 1:
                scene.dialogue = lines[idx]
            else:
                remainder = lines[idx:]
                scene.dialogue = " ".join(remainder)
    
    def _validate_script(self, script: VideoScript) -> bool:
        """Return True if script meets Kling needs: core_concept, anchor character_description, and every scene has visual_description."""
        if not (script.core_concept or "").strip():
            return False
        if not (getattr(script, "character_description", None) or "").strip():
            return False
        for s in script.scenes:
            if not (getattr(s, "visual_description", None) or "").strip():
                return False
        return True

    def _storyboard_block(self, storyboard: Optional[Storyboard]) -> str:
        """Format storyboard for injection into user prompt."""
        if not storyboard or not getattr(storyboard, "scenes", None):
            return ""
        lines = ["STORYBOARD (follow this structure; same order and beats):"]
        for s in storyboard.scenes:
            lines.append(
                f"  Scene {s.scene_number}: {s.action} | Setting: {s.setting} | "
                f"Characters: {s.characters} | Camera: {s.camera_suggestion} | "
                f"Transition: {s.transition_from_previous}"
            )
        return "\n".join(lines) + "\n\n"

    async def generate(
        self,
        prompt: str,
        scene_count: int = 5,
        vibe: Optional[str] = None,
        tone: float = 0.5,
        brand_slider: float = 0.5,
        value_slider: float = 0.5,
        user_expectation: Optional[str] = None,
        style_preset: Optional[StylePreset] = None,
        storyboard: Optional[Storyboard] = None,
    ) -> VideoScript:
        """Generate a video script from a text prompt. One prompt → one script → one video."""
        extra = f"Deliver this: {user_expectation.strip()}\n\n" if (user_expectation or "").strip() else ""
        tone_label, tone_guidance = self._tone_guidance(tone)
        brand_label, brand_guidance = self._brand_guidance(brand_slider)
        value_label, value_guidance = self._value_guidance(value_slider)
        style_context = self._style_context(style_preset)
        tone_context = f"TONE LOCK: {tone_label}. {tone_guidance}" if tone_label else ""
        brand_context = f"BRAND: {brand_label}. {brand_guidance}" if brand_label else ""
        value_context = f"CONTENT FOCUS: {value_label}. {value_guidance}" if value_label else ""
        # Default literal; viral/documentary/etc. when requested
        effective_vibe = (vibe or "").strip() or "literal"
        vibe_line = ""
        use_viral_system = False
        use_literal_system = False
        speaker_hint, dialogue_lock, dialogue_lines = self._extract_dialogue_lock(prompt)
        dialogue_block = f"\n{speaker_hint}\n{dialogue_lock}\n" if dialogue_lock else ""
        if effective_vibe:
            v = effective_vibe.lower()
            if v == "literal" or v == "realistic":
                use_literal_system = True
            elif v == "viral" or v == "shortform":
                use_viral_system = True
            elif v == "documentary":
                vibe_line = "Tone: documentary, authoritative, voiceover feel. "
            elif v == "ad":
                vibe_line = "Tone: commercial ad, punchy, high impact. "
            elif v == "tutorial":
                vibe_line = "Tone: tutorial, clear step-by-step, educational. "
            elif v == "cinematic":
                vibe_line = "Tone: cinematic, film-like, dramatic. "
            else:
                vibe_line = f"Tone: {effective_vibe}. "
        # No auto-override: the chosen vibe (literal, viral, documentary, ad, etc.) always drives the script.

        storyboard_block = self._storyboard_block(storyboard)
        if use_literal_system:
            user_prompt = f"""{extra}{storyboard_block}USER INPUT: {prompt}
{dialogue_block}

Style preset: {style_preset.label if style_preset else 'Default'}.
Tone slider: {tone_label}. {tone_guidance}

Break this into exactly {scene_count} scenes that show what would REALLY happen, in real-world order. Same character and setting in every scene.

Universal rule: Scene 1 = object and first action. Then show each step in order (prepare, then consume; open, then use; etc.). Scene {scene_count} = last moment (finishing, satisfied). Keep wording simple.
Examples: "eating a banana" → scene 1: banana in basket, she grabs it; then peel; then eat. "Making coffee" → scene 1: kettle/coffee, she starts; then pour/brew; then drinks. "Opening a gift" → scene 1: wrapped box, she picks it up; then unwraps; then opens.

Each visual_description = one concrete thing we SEE. Dialogue minimal. Output ONLY valid JSON with core_concept, character_description, visual_style, background_theme, and exactly {scene_count} scenes (scene_number, visual_description, dialogue, camera_angle)."""
            logger.info(f"Generating literal/real-life script for prompt: {prompt[:50]}...")
            script = await self._generate_once(
                prompt,
                scene_count,
                user_prompt,
                system_prompt_literal=True,
                style_context=style_context,
                tone_context=tone_context,
                brand_context=brand_context,
                value_context=value_context,
            )
            if not self._validate_script(script):
                logger.warning("Literal script validation failed, retrying with stricter prompt...")
                retry_prompt = user_prompt + "\n\nBe more specific: each visual_description must be one clear, shootable moment (what we see on camera). Same character throughout. Output ONLY valid JSON."
                script = await self._generate_once(
                    prompt,
                    scene_count,
                    retry_prompt,
                    system_prompt_literal=True,
                    style_context=style_context,
                    tone_context=tone_context,
                    brand_context=brand_context,
                    value_context=value_context,
                )
            self._apply_dialogue_lock(script, dialogue_lines)
            return script
        if use_viral_system:
            user_prompt = f"""{extra}{storyboard_block}USER INPUT: {prompt}
{dialogue_block}

Generate script now. Return ONLY valid JSON with core_concept, character_description, visual_style, background_theme, and exactly {scene_count} scenes (each with scene_number, visual_description, dialogue, camera_angle). Follow HOOK → PROBLEM → INSIGHT → SOLUTION → CTA structure. Short punchy dialogue, 3rd-grade reading level."""
            user_prompt = (
                f"{user_prompt}\n\nStyle preset: {style_preset.label if style_preset else 'Default'}."
                f"\nTone slider: {tone_label}. {tone_guidance}"
            )
            logger.info(f"Generating viral script for prompt: {prompt[:50]}...")
            script = await self._generate_once(
                prompt,
                scene_count,
                user_prompt,
                system_prompt_viral=True,
                style_context=style_context,
                tone_context=tone_context,
                brand_context=brand_context,
                value_context=value_context,
            )
            if not self._validate_script(script):
                logger.warning("Viral script validation failed, retrying with stricter prompt...")
                retry_prompt = user_prompt + "\n\nBe more specific: every scene needs a concrete visual_description (what we SEE) and a short dialogue line. Output ONLY valid JSON."
                script = await self._generate_once(
                    prompt,
                    scene_count,
                    retry_prompt,
                    system_prompt_viral=True,
                    style_context=style_context,
                    tone_context=tone_context,
                    brand_context=brand_context,
                    value_context=value_context,
                )
            self._apply_dialogue_lock(script, dialogue_lines)
            return script
        else:
            user_prompt = f"""{extra}{storyboard_block}The user imagined a video. Write the script for that video—what they see in their mind. Same characters, setting, mood.

User's prompt (imagine this as the finished video):
"{prompt}"
{vibe_line}
{dialogue_block}
Requirements:
- core_concept: one sentence that is exactly what the user imagined.
- Every scene = one moment from that imagined video. Same characters, setting, and mood as in the prompt.
- Scene 1 = hook, Scenes 2–{scene_count - 1} = build, Scene {scene_count} = payoff.
- Give each scene a different camera_angle. Same character and setting throughout.
- Output EXACTLY {scene_count} scenes. Output ONLY valid JSON, no markdown, no code blocks."""

            user_prompt = (
                f"{user_prompt}\n\nStyle preset: {style_preset.label if style_preset else 'Default'}."
                f"\nTone slider: {tone_label}. {tone_guidance}"
            )
            logger.info(f"Generating script for prompt: {prompt[:50]}...")
            script = await self._generate_once(
                prompt,
                scene_count,
                user_prompt,
                style_context=style_context,
                tone_context=tone_context,
                brand_context=brand_context,
                value_context=value_context,
            )
        if self._validate_script(script):
            self._apply_dialogue_lock(script, dialogue_lines)
            return script
        logger.warning("Script validation failed (empty core_concept or visual_description), retrying with stricter prompt...")
        retry_prompt = user_prompt + "\n\nBe more specific: describe exactly what we SEE in each scene (concrete actions and visuals). Every visual_description must be at least one full sentence."
        script = await self._generate_once(
            prompt,
            scene_count,
            retry_prompt,
            style_context=style_context,
            tone_context=tone_context,
            brand_context=brand_context,
            value_context=value_context,
        )
        self._apply_dialogue_lock(script, dialogue_lines)
        return script

    def _auth_header(self) -> str:
        """Bearer token for API; raises clear error if key is missing (avoids 'Illegal header value b\"Bearer \"')."""
        # Read from config at call time so restart-reloaded keys are used
        key = sanitize_header_token((get_settings().openai_api_key or ""))
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is not set or invalid. Add OPENAI_API_KEY=your_key to the .env file in the project root and restart the server."
            )
        return "Bearer " + key

    def _tone_guidance(self, tone: float) -> tuple[str, str]:
        value = 0.5 if tone is None else float(tone)
        if value <= 0.35:
            return "funny", "Playful, witty, light humor. Keep the beats snappy."
        if value >= 0.7:
            return "serious", "Cinematic, grounded, credible. No jokes."
        return "balanced", "Balanced between credible and entertaining."

    def _brand_guidance(self, brand: float) -> tuple[str, str]:
        """Personal (0) ↔ Business (1). Influences voice and framing."""
        value = 0.5 if brand is None else float(brand)
        if value <= 0.35:
            return "personal", "Personal brand: relatable, human, individual voice. Avoid corporate jargon."
        if value >= 0.7:
            return "business", "Business brand: professional, authoritative, polished. Clear brand voice."
        return "mixed", "Balanced: approachable but credible. Neither purely personal nor corporate."

    def _value_guidance(self, value_slider: float) -> tuple[str, str]:
        """Promotional (0) ↔ Value (1). Promotional = salesy, CTA-heavy; Value = educational, helpful."""
        value = 0.5 if value_slider is None else float(value_slider)
        if value <= 0.35:
            return "promotional", "Promotional: strong CTAs, product-focused, conversion-oriented. Salesy but punchy."
        if value >= 0.7:
            return "value", "Value-first: educational, helpful, no hard sell. Focus on insight and takeaway."
        return "balanced", "Balanced: useful content with a clear but soft CTA."

    def _style_context(self, preset: Optional[StylePreset]) -> str:
        if not preset:
            return ""
        return (
            "STYLE LOCK (do not change):\n"
            f"- visual_style: {preset.visual_style}\n"
            f"- background_theme: {preset.background_hint}\n"
            f"- pacing: {preset.pacing}\n"
            f"- camera_notes: {preset.camera_notes}\n"
            f"- audio_notes: {preset.audio_notes}\n"
            f"- negatives: {preset.negative_phrases}"
        )

    def _apply_style_preset(self, script_data: dict, style_context: str) -> None:
        if not style_context:
            return
        visual_style = (script_data.get("visual_style") or "").strip()
        background_theme = (script_data.get("background_theme") or "").strip()
        if "visual_style:" in style_context:
            style_line = style_context.split("visual_style:", 1)[-1].split("\n", 1)[0].strip()
            if style_line:
                script_data["visual_style"] = style_line
        if "background_theme:" in style_context:
            bg_line = style_context.split("background_theme:", 1)[-1].split("\n", 1)[0].strip()
            if background_theme:
                if bg_line and bg_line.lower() not in background_theme.lower():
                    script_data["background_theme"] = f"{background_theme} {bg_line}".strip()
                else:
                    script_data["background_theme"] = background_theme
            else:
                script_data["background_theme"] = bg_line

    async def _generate_once(
        self,
        prompt: str,
        scene_count: int,
        user_prompt: str,
        system_prompt_viral: bool = False,
        system_prompt_literal: bool = False,
        style_context: str = "",
        tone_context: str = "",
        brand_context: str = "",
        value_context: str = "",
    ) -> VideoScript:
        if system_prompt_literal:
            system_content = self._get_system_prompt_literal(scene_count, style_context, tone_context, brand_context, value_context)
        elif system_prompt_viral:
            system_content = self._get_system_prompt_viral(scene_count, style_context, tone_context, brand_context, value_context)
        else:
            system_content = self._get_system_prompt(scene_count, style_context, tone_context, brand_context, value_context)
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.base_url,
                headers={
                    "Authorization": self._auth_header(),
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.4,
                    "max_tokens": 4000,
                    "response_format": {"type": "json_object"}
                }
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise ValueError("OpenAI response had no choices")
            msg = choices[0].get("message") or {}
            content = (msg.get("content") or "").strip()
            if not content:
                raise ValueError("OpenAI returned empty script content")
            script_data = json.loads(content)
            scenes_raw = script_data.get("scenes") or []
            if not scenes_raw:
                raise ValueError("OpenAI script has no scenes")
            _sanitize_script_style(script_data, prompt)
            self._apply_style_preset(script_data, style_context)
            def _int_scene_num(v, i: int):
                if v is None:
                    return i + 1
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return i + 1
            scenes = [
                Scene(
                    scene_number=_int_scene_num(s.get("scene_number"), i),
                    visual_description=(s.get("visual_description") or "Same character, clear action.").strip(),
                    dialogue=(s.get("dialogue") or "").strip(),
                    camera_angle=s.get("camera_angle")
                )
                for i, s in enumerate(scenes_raw)
            ]
            _normalize_script_scenes(scenes, scene_count)
            core = (script_data.get("core_concept") or "").strip() or prompt[:200].strip()
            if not core:
                core = prompt[:200].strip()
            char_desc = script_data.get("character_description") or "Person, neutral appearance, professional."
            return VideoScript(
                core_concept=core,
                character_description=char_desc,
                scenes=scenes,
                visual_style=script_data.get("visual_style"),
                background_theme=script_data.get("background_theme")
            )
    
    async def generate_image_prompt(
        self,
        character_description: str,
        first_scene_description: Optional[str] = None
    ) -> dict:
        """
        Generate an optimized image prompt for the reference character image.
        If first_scene_description is provided, align reference with first scene (setting/style).
        """
        system_prompt = """You are an elite visual prompt writer. Generate ONE portrait reference image prompt for a single subject.

The goal is character/style consistency for video generation.

Hard requirements:
- Single person only, waist-up, centered, neutral studio pose, friendly expression
- Plain seamless white studio background (#FFFFFF), no gradients, no shadows on backdrop
- No text, no logos, no icons, no UI, no watermark
- No collage/multi-panel/grid/frames
- No devices or screens (no laptop/phone/tablet)
- Natural, soft front key light, minimal fill, clean color

Output JSON only: { "image_prompt": "string", "negative_prompt": "string" }"""

        context = character_description
        if first_scene_description and first_scene_description.strip():
            context = f"{character_description}\n\nFirst scene will show: {first_scene_description.strip()[:300]}. The reference image should match this context (same person, same setting/style) so the first video clip aligns with it."
        user_prompt = f"""Create a reference image prompt for this character:

{context}

Make it hyper-realistic, 4K, with detailed textures and consistent lighting.
Output ONLY valid JSON."""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": self._auth_header(),
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.5,
                        "max_tokens": 500,
                        "response_format": {"type": "json_object"}
                    }
                )
                
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []
                msg = (choices[0].get("message") or {}) if choices else {}
                content = (msg.get("content") or "").strip()
                if content:
                    return json.loads(content)
                raise ValueError("OpenAI returned empty image prompt")
        except Exception as e:
            logger.error(f"Failed to generate image prompt: {e}")
            # Return a sensible default
            return {
                "image_prompt": f"Professional portrait photo, {character_description}, waist-up, centered, white studio background, soft lighting, 4K, hyper-realistic",
                "negative_prompt": "text, watermark, logo, blurry, distorted, multiple people"
            }
