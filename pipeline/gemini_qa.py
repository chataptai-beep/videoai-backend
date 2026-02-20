"""
Gemini QA: analyze stitched video transitions and return trim instructions.
"""

import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import httpx

from config import settings, sanitize_header_token

logger = logging.getLogger(__name__)

@dataclass
class TrimInstruction:
    scene_number: int
    trim_start: float
    trim_end: float
    reason: str = ""
    severity: str = ""

def _build_prompt(scene_count: int, scene_duration_seconds: float) -> str:
    duration = max(1, int(round(scene_duration_seconds)))
    return (
        "You are analyzing a stitched multi-scene commercial video. "
        f"Each scene is {duration} seconds long. Total scenes: {scene_count}.\n\n"
        "CRITICAL: Focus ONLY on transitions between scenes. Identify where one scene ends poorly "
        "and the next begins awkwardly.\n\n"
        "Respond with ONLY valid JSON, no markdown, no code blocks, no backticks, no explanations:\n\n"
        "{\n"
        "  \"overall_quality\": 0.0-1.0,\n"
        "  \"total_scenes_detected\": <number>,\n"
        "  \"transition_issues\": [\n"
        "    {\n"
        "      \"transition_between\": \"Scene X to Scene Y\",\n"
        "      \"timestamp_start\": \"MM:SS\",\n"
        "      \"timestamp_end\": \"MM:SS\",\n"
        "      \"issue\": \"describe the transition problem\",\n"
        "      \"severity\": \"critical|high|medium|low\",\n"
        "      \"recommended_trim\": {\n"
        "        \"scene\": \"X\",\n"
        "        \"trim_from_seconds\": <number>,\n"
        "        \"trim_to_seconds\": <number>,\n"
        "        \"reason\": \"why this trim fixes it\"\n"
        "      }\n"
        "    }\n"
        "  ],\n"
        "  \"approved_for_posting\": true|false,\n"
        "  \"next_steps\": \"brief instruction for what to do\",\n"
        "  \"notes\": \"any other observations\"\n"
        "}\n\n"
        "Evaluate each transition for:\n"
        "1. Pose continuity\n"
        "2. Action continuity\n"
        "3. Visual consistency\n"
        "4. Audio sync\n"
        "5. Timing\n"
        "For each problematic transition, give exact timestamps and a trim recommendation."
    )

def _clean_json(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    if cleaned.endswith("```"):
        cleaned = cleaned.rstrip("`").strip()
    return cleaned

async def analyze_transitions(
    video_path: str,
    scene_count: int,
    scene_duration_seconds: float,
) -> Optional[List[TrimInstruction]]:
    if not getattr(settings, "gemini_qa_enabled", True):
        return None
    api_key = sanitize_header_token(getattr(settings, "google_api_key", "") or "")
    if not api_key:
        logger.info("Gemini QA skipped: GOOGLE_API_KEY not configured")
        return None
    if not video_path:
        logger.warning("Gemini QA skipped: video path missing")
        return None
    if "://" in video_path:
        logger.warning("Gemini QA skipped: remote video URL (requires local file)")
        return None
    if not os.path.exists(video_path):
        logger.warning("Gemini QA skipped: video file not found")
        return None

    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    max_mb = max(1, int(getattr(settings, "gemini_max_video_mb", 20) or 20))
    if size_mb > max_mb:
        logger.warning("Gemini QA skipped: video size %.2fMB exceeds limit %sMB", size_mb, max_mb)
        return None

    prompt = _build_prompt(scene_count, scene_duration_seconds)
    with open(video_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "video/mp4", "data": data}},
                ],
            }
        ]
    }

    model = (getattr(settings, "gemini_model", "") or "gemini-2.5-flash").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, params={"key": api_key}, json=payload)
        response.raise_for_status()
        result = response.json()

    candidates = result.get("candidates") or []
    if not candidates:
        logger.warning("Gemini QA returned no candidates")
        return None
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    if not parts:
        logger.warning("Gemini QA returned empty content")
        return None
    text = parts[0].get("text") or ""
    cleaned = _clean_json(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("Gemini QA JSON parse failed: %s", e)
        return None

    return _extract_trim_instructions(parsed, scene_duration_seconds)

def _extract_trim_instructions(parsed: dict, scene_duration_seconds: float) -> List[TrimInstruction]:
    trims: List[TrimInstruction] = []
    issues = parsed.get("transition_issues") or []
    for issue in issues:
        recommended = issue.get("recommended_trim") or {}
        try:
            scene_number = int(str(recommended.get("scene", "")).strip())
        except ValueError:
            continue
        trim_from = float(recommended.get("trim_from_seconds", 0) or 0)
        trim_to = float(recommended.get("trim_to_seconds", 0) or 0)
        reason = str(recommended.get("reason", "") or "").strip()
        severity = str(issue.get("severity", "") or "").strip().lower()

        trim_start = 0.0
        trim_end = 0.0
        if trim_from <= 0.1 and trim_to > 0:
            trim_start = trim_to
        elif trim_to >= scene_duration_seconds - 0.1 and trim_from > 0:
            trim_end = max(0.0, scene_duration_seconds - trim_from)
        else:
            logger.info(
                "Gemini QA returned middle trim for scene %s (%s-%s). Skipping complex trim.",
                scene_number,
                trim_from,
                trim_to,
            )
            continue

        trims.append(
            TrimInstruction(
                scene_number=scene_number,
                trim_start=trim_start,
                trim_end=trim_end,
                reason=reason,
                severity=severity,
            )
        )
    return trims
