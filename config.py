"""
Configuration module for Videeo.ai Stage 1 Pipeline.
Loads environment variables from .env in project root and provides typed configuration.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings

# Load .env from project root (same dir as this file) so keys are found regardless of cwd
_CONFIG_DIR = Path(__file__).resolve().parent
_ENV_PATH = _CONFIG_DIR / ".env"
if _ENV_PATH.exists():
    with open(_ENV_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


def _strip_key(v: str) -> str:
    """Strip whitespace/newlines so keys never cause 'Illegal header value' in HTTP clients."""
    return (v or "").strip()


# Placeholder values from .env.example; treat as "no key" so we never send invalid headers
_PLACEHOLDER_KEY_PATTERNS = (
    "your_openai_api_key_here",
    "your_kie_api_key_here",
    "your_fal_key_here",
    "your_google_api_key_here",
    "your_cloud_name",
    "your-cloud-name",
    "your_api_key",
    "your_api_secret",
    "your-cloudinary-key",
)


def _is_placeholder_key(value: str) -> bool:
    """True if value looks like an example placeholder, not a real key."""
    v = (value or "").strip().lower()
    if not v or len(v) < 20:
        return bool(not v or v in ("your_cloud_name", "your_api_key", "your_api_secret"))
    return any(p in v for p in _PLACEHOLDER_KEY_PATTERNS)


def sanitize_header_token(value: str) -> str:
    """
    Return a token safe for HTTP Authorization header (avoids 'Illegal header value').
    Strips and removes control characters (\\r, \\n, etc.) and non-ASCII.
    Treats placeholder values from .env.example as empty.
    """
    s = (value or "").strip()
    if _is_placeholder_key(s):
        return ""
    # Only allow printable ASCII (space through ~)
    return "".join(c for c in s if 32 <= ord(c) <= 126)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys (validated/stripped to avoid "Illegal header value" in HTTP clients)
    openai_api_key: str = ""
    kie_api_key: str = ""
    fal_key: str = ""  # fal.ai (Kling 3.0 text-to-video)
    cloudinary_cloud_name: str = ""
    cloudinary_api_key: str = ""
    cloudinary_api_secret: str = ""

    # Optional: Google Gemini as alternative to OpenAI
    google_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Application Settings
    app_name: str = "Videeo.ai Stage 1 Pipeline"
    app_version: str = "1.0.0"
    debug: bool = False

    # Enterprise: observability and security
    rate_limit_per_hour: int = 50  # Max POST /generate per IP per hour (0 = disabled)
    cors_origins: str = "*"  # Comma-separated origins; * allows all (restrict in production)
    log_level: Optional[str] = None  # Override: DEBUG, INFO, WARNING, ERROR (default from debug)
    log_format: str = "default"  # "default" (human) or "json" (for log aggregators)
    request_id_header: str = "X-Request-ID"  # Header name for correlation id
    
    # Pipeline Settings
    default_scene_count: int = 5
    scene_duration_seconds: int = 4  # Faster pacing for Reels
    kling_shot_duration_seconds: int = 3  # One scene per action; ~3s per shot (e.g. 3 actions = ~9–10s video)
    kling_sound_retry: bool = False  # If True, on audio error we retry without sound (2nd KIE call). False = fail and show error on UI.
    # Kling 3.0: Standard 720p with audio = 30 credits/s. Pro 1080p = 40 credits/s. Use 720p to keep Standard pricing.
    kling_quality: str = "720p"  # "720p" (Standard) or "1080p" (Pro). Standard 720p + audio = 30 credits/s.
    video_width: int = 1080          # Vertical 9:16
    video_height: int = 1920         # Vertical 9:16
    video_fps: int = 30
    
    # Timeouts and Retries (0 = no retries; fail once and show error on UI)
    max_retries: int = 0
    pipeline_timeout_seconds: int = 300  # 5 minutes
    kie_poll_timeout_seconds: int = 900  # 15 min max wait for KIE task
    ffmpeg_timeout_seconds: int = 600  # 10 min for stitch/caption/audio
    api_poll_interval_seconds: int = 10
    
    # Storage
    output_dir: str = "./outputs"
    temp_dir: str = "./temp"
    
    # Redis (optional): set REDIS_URL for persistent job store
    redis_url: str = ""
    
    # Pipeline options
    parallel_scene_batch_size: int = 2
    use_last_frame_continuity: bool = True
    use_multi_prompt: bool = True
    scene_retry_limit: int = 1
    default_scene_trim_start_seconds: float = 1.0
    gemini_qa_enabled: bool = True
    gemini_max_video_mb: int = 20
    audio_polish_enabled: bool = True
    music_bed_path: str = ""
    music_bed_volume: float = 0.18
    music_bed_ducking: bool = True
    analytics_failures_path: str = "./logs/failures.jsonl"
    # Safety: optional prompt blocklist (comma-separated phrases); empty = no block
    prompt_blocklist: str = ""
    # Block webhook URLs to private/localhost (SSRF mitigation)
    block_webhook_private_ips: bool = True
    # For last-frame URLs when not using Cloudinary (kie.ai must fetch this)
    base_url: str = "http://localhost:8000"
    
    # Caption burner
    caption_font_path: str = ""
    caption_max_chars: int = 120  # Truncate long dialogue to avoid overflow
    caption_padding_seconds: float = 0.15  # Gap between caption segments to avoid overlap
    # Cleanup: delete temp/output files older than N days (0 = disabled)
    cleanup_temp_max_days: int = 0
    cleanup_outputs_max_days: int = 0

    # kie.ai specific
    kie_base_url: str = "https://api.kie.ai/api/v1"
    kie_image_urls_as_string: bool = False
    # Veo audio: kept for compatibility only. Pipeline always sends enableAudio=false so generation succeeds (Google often fails when audio requested).
    kie_veo_enable_audio: bool = False

    # Veo 3–optimized prompts: 8-element formula (subject, context, action, style, camera, lighting, audio, exclusions)
    veo_optimized: bool = True

    @field_validator(
        "openai_api_key",
        "kie_api_key",
        "fal_key",
        "cloudinary_cloud_name",
        "cloudinary_api_key",
        "cloudinary_api_secret",
        "google_api_key",
        mode="before",
    )
    @classmethod
    def strip_api_keys(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return v.strip()
        return v

    class Config:
        # Try project root (same dir as config.py) then cwd so .env is always found
        _config_dir = os.path.dirname(os.path.abspath(__file__))
        env_file = [
            os.path.join(_config_dir, ".env"),
            ".env",
        ]
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function to get settings
settings = get_settings()
