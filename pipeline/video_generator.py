"""
Video Generator using kie.ai Kling 3.0 text-to-video.
One user prompt → backend expands to N scenes → one KIE multi_prompt call → one video.

We use Standard 720p with native audio: mode=std, quality=720p, sound=True (30 credits/s).
KIE API: POST /api/v1/jobs/createTask (model kling-3.0/video), then poll GET /api/v1/jobs/recordInfo.
Multi-shot: input.multi_shots=true, input.multi_prompt=[{prompt, duration (1-12)}, ...].
"""

import asyncio
import json
import logging
import math
from typing import List, Optional

import httpx

from config import get_settings, sanitize_header_token, settings as app_settings
from models.schemas import Scene, AspectRatio, VideoScript
from pipeline.kling_prompt_builder import (
    build_kling_prompt_for_scene,
    build_kling_multi_shot_prompt,
    KLING_PROMPT_MAX_CHARS,
    DEFAULT_NEGATIVE_PHRASES,
)

logger = logging.getLogger(__name__)

KIE_KLING_MODEL = "kling-3.0/video"
KIE_BASE = "https://api.kie.ai/api/v1"
# KIE multi_prompt: each shot duration 1-12 seconds (default from config: 3s per action)
MAX_MULTI_PROMPT_SHOTS = 6
MAX_MULTI_PROMPT_TOTAL_SECONDS = 15
MIN_MULTI_PROMPT_TOTAL_SECONDS = 3


class VideoGenerator:
    """
    Generates video clips using kie.ai Kling 3.0 text-to-video.
    Uses createTask (async) then poll recordInfo for result URL.
    """

    def __init__(self):
        self._kie_key = ""
        self._ensure_key()

    def _ensure_key(self) -> None:
        key = sanitize_header_token((get_settings().kie_api_key or ""))
        if not key:
            raise ValueError(
                "KIE_API_KEY is not set. Add KIE_API_KEY=your_kie_key to .env (get key from https://kie.ai/api-key)."
            )
        self._kie_key = key

    def _headers(self) -> dict:
        self._ensure_key()
        return {
            "Authorization": f"Bearer {self._kie_key}",
            "Content-Type": "application/json",
        }

    async def generate_scene_video(
        self,
        scene: Scene,
        reference_image_url: str,
        aspect_ratio: AspectRatio = AspectRatio.PORTRAIT,
        scene_index: int = 0,
        character_description: str = "",
        background_theme: str = "",
        core_concept: str = "",
        visual_style: str = "",
        original_prompt: str = "",
        total_scenes: int = 1,
        previous_scenes_descriptions: Optional[list] = None,
        narrative_role: str = "",
        use_veo_enhancer: Optional[bool] = None,
    ) -> str:
        """Generate one scene with Kling 3.0 (single-shot) via KIE. reference_image_url unused for text-to-video."""
        prompt = build_kling_prompt_for_scene(
            scene=scene,
            scene_index=scene_index,
            total_scenes=total_scenes,
            character_description=character_description or "",
            background_theme=background_theme or "",
            visual_style=visual_style or "",
            core_concept=core_concept or original_prompt or "",
            previous_scenes_descriptions=previous_scenes_descriptions,
            avoid_phrases=DEFAULT_NEGATIVE_PHRASES,
        )
        if reference_image_url:
            if scene_index == 0:
                prompt = (
                    "Use the reference image ONLY for character identity and wardrobe. "
                    "Ignore the neutral pose; start in action. "
                    + prompt
                )
            else:
                prompt = (
                    "Continue seamlessly from the reference image, which is the exact last frame of the previous scene. "
                    "Start in the same pose, framing, and lighting. "
                    + prompt
                )
        logger.info(f"Generating video for Scene {scene.scene_number} with Kling 3.0 (KIE)...")
        video_url = await self._run_kling_single(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            reference_image_url=reference_image_url,
        )
        logger.info(f"Video generated for Scene {scene.scene_number}: {video_url}")
        return video_url

    def build_multi_prompt_only(
        self,
        script: VideoScript,
        aspect_ratio: AspectRatio,
    ) -> List[dict]:
        """
        Build the multi_prompt list we would send to KIE (no API call).
        Use for validation before calling generate_video_multi_prompt.
        """
        if not script or not getattr(script, "scenes", None) or len(script.scenes) == 0:
            raise ValueError("Script has no scenes")
        scenes = script.scenes[:MAX_MULTI_PROMPT_SHOTS]
        shot_sec = self._multi_prompt_shot_duration(len(scenes))
        multi_prompt = build_kling_multi_shot_prompt(
            scenes=scenes,
            character_description=script.character_description or "",
            background_theme=script.background_theme or "",
            visual_style=script.visual_style or "",
            core_concept=script.core_concept or "",
            simple_style=False,
            shot_duration_seconds=shot_sec,
            avoid_phrases=DEFAULT_NEGATIVE_PHRASES,
        )
        # Apply same trim as in _run_kling_multi_prompt so validation matches reality
        def _trim(p: str) -> str:
            p = (p or "").strip()
            return p[:KLING_PROMPT_MAX_CHARS] if len(p) > KLING_PROMPT_MAX_CHARS else p

        return [
            {
                "prompt": _trim(item.get("prompt") or ""),
                "duration": min(12, max(1, int(item.get("duration", "3")))),
            }
            for item in multi_prompt
        ]

    async def generate_video_multi_prompt(
        self,
        script: VideoScript,
        aspect_ratio: AspectRatio,
    ) -> str:
        """
        One KIE call with multi_prompt: all scenes in one request, one video out.
        """
        if not script or not getattr(script, "scenes", None) or len(script.scenes) == 0:
            raise ValueError("Script has no scenes")
        scenes = script.scenes[:MAX_MULTI_PROMPT_SHOTS]
        if len(script.scenes) > MAX_MULTI_PROMPT_SHOTS:
            logger.warning(f"Capping at {MAX_MULTI_PROMPT_SHOTS} shots; had {len(script.scenes)}")
        shot_sec = self._multi_prompt_shot_duration(len(scenes))
        multi_prompt = build_kling_multi_shot_prompt(
            scenes=scenes,
            character_description=script.character_description or "",
            background_theme=script.background_theme or "",
            visual_style=script.visual_style or "",
            core_concept=script.core_concept or "",
            simple_style=False,
            shot_duration_seconds=shot_sec,
            avoid_phrases=DEFAULT_NEGATIVE_PHRASES,
        )
        for idx, item in enumerate(multi_prompt):
            p = (item.get("prompt") or "")[:80]
            logger.debug(f"Kling prompt {idx + 1}: {p}...")
        logger.info(f"Generating one video with Kling 3.0 multi_prompt ({len(multi_prompt)} shots) via KIE...")
        return await self._run_kling_multi_prompt(multi_prompt=multi_prompt, aspect_ratio=aspect_ratio)

    def _is_audio_error(self, e: Exception) -> bool:
        msg = (str(e) or "").lower()
        return "audio" in msg or "sound" in msg or "unable to generate" in msg

    def _aspect_ratio_str(self, aspect_ratio: AspectRatio) -> str:
        if aspect_ratio == AspectRatio.PORTRAIT:
            return "9:16"
        if aspect_ratio == AspectRatio.LANDSCAPE:
            return "16:9"
        return "1:1"

    async def _run_kling_multi_prompt(
        self,
        multi_prompt: List[dict],
        aspect_ratio: AspectRatio,
    ) -> str:
        """Call KIE createTask with multi_shots=true and multi_prompt; poll until success; return video URL. Retry without sound on sound error."""
        self._ensure_key()
        ar = self._aspect_ratio_str(aspect_ratio)
        # KIE / Kling: each prompt must be <= 500 chars; per-shot duration 1-12
        def _trim(p: str) -> str:
            p = (p or "").strip()
            return p[:KLING_PROMPT_MAX_CHARS] if len(p) > KLING_PROMPT_MAX_CHARS else p

        default_d = self._multi_prompt_shot_duration(len(multi_prompt))
        kie_multi = [
            {
                "prompt": _trim(item.get("prompt") or ""),
                "duration": min(12, max(1, int(item.get("duration", default_d)))),
            }
            for item in multi_prompt
        ]
        # Standard 720p with audio: 30 credits/s. mode=std, quality=720p, sound=True.
        quality = getattr(app_settings, "kling_quality", "720p") or "720p"
        payload = {
            "model": KIE_KLING_MODEL,
            "input": {
                "multi_shots": True,
                "multi_prompt": kie_multi,
                "aspect_ratio": ar,
                "mode": "std",
                "quality": quality,
                "sound": True,
                "duration": str(sum(p.get("duration", default_d) for p in kie_multi)),
                "prompt": _trim(kie_multi[0].get("prompt") or ""),
            },
        }
        try:
            return await self._create_and_poll(payload)
        except Exception as e:
            if self._is_audio_error(e) and getattr(app_settings, "kling_sound_retry", True):
                logger.warning(f"Kling sound error, retrying without sound: {e}")
                payload["input"]["sound"] = False
                return await self._create_and_poll(payload)
            raise

    async def _run_kling_single(
        self,
        prompt: str,
        aspect_ratio: AspectRatio,
        reference_image_url: Optional[str] = None,
    ) -> str:
        """Single-shot Kling 3.0 via KIE. Standard 720p with audio. Retry without sound on sound error."""
        self._ensure_key()
        quality = getattr(app_settings, "kling_quality", "720p") or "720p"
        image_urls = []
        if reference_image_url:
            image_urls = [reference_image_url]
            if getattr(app_settings, "kie_image_urls_as_string", False):
                image_urls = [str(reference_image_url)]
        payload = {
            "model": KIE_KLING_MODEL,
            "input": {
                "prompt": prompt[:5000],
                "duration": "5",
                "aspect_ratio": self._aspect_ratio_str(aspect_ratio),
                "mode": "std",
                "quality": quality,
                "multi_shots": False,
                "sound": True,
                "multi_prompt": [],
            },
        }
        if image_urls:
            payload["input"]["image_urls"] = image_urls
        try:
            return await self._create_and_poll(payload)
        except Exception as e:
            if self._is_audio_error(e) and getattr(app_settings, "kling_sound_retry", True):
                logger.warning(f"Kling sound error, retrying without sound: {e}")
                payload["input"]["sound"] = False
                return await self._create_and_poll(payload)
            raise

    def _multi_prompt_shot_duration(self, shot_count: int) -> int:
        """Resolve per-shot duration so total stays within KIE limits (3–15s)."""
        base = int(getattr(app_settings, "kling_shot_duration_seconds", 3) or 3)
        count = max(1, int(shot_count or 1))
        max_per_shot = max(1, int(MAX_MULTI_PROMPT_TOTAL_SECONDS / count))
        min_per_shot = max(1, int(math.ceil(MIN_MULTI_PROMPT_TOTAL_SECONDS / count)))
        duration = min(base, max_per_shot)
        duration = max(min_per_shot, duration)
        return min(12, max(1, duration))

    async def _create_and_poll(self, payload: dict) -> str:
        """POST createTask, then poll recordInfo until state success/fail; return first result URL."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            r = await client.post(
                f"{KIE_BASE}/jobs/createTask",
                headers=self._headers(),
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
        if data.get("code") != 200:
            raise RuntimeError(f"KIE createTask failed: {data.get('msg', data)}")
        task_id = (data.get("data") or {}).get("taskId")
        if not task_id:
            raise RuntimeError(f"KIE createTask returned no taskId: {data}")
        return await self._poll_task(task_id)

    async def _poll_task(self, task_id: str) -> str:
        """Poll GET recordInfo?taskId= until state success or fail; return resultUrls[0]."""
        auth = self._headers()
        last_state = None
        timeout_sec = max(60, int(getattr(app_settings, "kie_poll_timeout_seconds", 900) or 900))
        max_polls = max(12, timeout_sec // 5)
        for _ in range(max_polls):
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                r = await client.get(
                    f"{KIE_BASE}/jobs/recordInfo",
                    params={"taskId": task_id},
                    headers=auth,
                )
                r.raise_for_status()
                data = r.json()
            if data.get("code") != 200:
                raise RuntimeError(f"KIE recordInfo failed: {data.get('msg', data)}")
            info = data.get("data") or {}
            state = info.get("state")
            last_state = state
            if state == "success":
                result_json = info.get("resultJson") or "{}"
                try:
                    parsed = json.loads(result_json)
                except json.JSONDecodeError:
                    raise RuntimeError(f"KIE resultJson invalid: {result_json[:200]}")
                urls = parsed.get("resultUrls") or []
                if not urls or not isinstance(urls[0], str):
                    raise RuntimeError(f"KIE result has no video URL: {parsed}")
                return urls[0]
            if state == "fail":
                fail_msg = info.get("failMsg") or info.get("failCode") or "Unknown"
                raise RuntimeError(f"KIE generation failed: {fail_msg}")
            await asyncio.sleep(5)
        raise RuntimeError(f"KIE task {task_id} timed out after {timeout_sec}s (last state: {last_state})")
