"""
Pydantic schemas for API requests, responses, and internal data models.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# === Enums ===

class JobStatus(str, Enum):
    """Job status states as per client requirements."""
    PENDING = "pending"
    GENERATING_SCRIPT = "generating_script"
    GENERATING_IMAGES = "generating_images"
    GENERATING_VIDEOS = "generating_videos"
    ASSEMBLING_VIDEO = "assembling_video"
    ADDING_CAPTIONS = "adding_captions"
    COMPLETE = "complete"
    ERROR = "error"


class AspectRatio(str, Enum):
    """Supported aspect ratios."""
    LANDSCAPE = "16:9"
    PORTRAIT = "9:16"
    SQUARE = "1:1"


# === API Request/Response Models ===

class GenerateRequest(BaseModel):
    """Request body for POST /generate endpoint. Only prompt is required; production choices have smart defaults."""
    prompt: str = Field(
        ...,
        min_length=10,
        max_length=20000,
        description="Text prompt describing the video (e.g. 'coffee morning routine', 'why I quit my 9-5'). Backend handles structure and style."
    )
    scenes: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of scenes (default: 3 for short-form). Optional; backend default is tuned for viral-style output."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.PORTRAIT,
        description="Aspect ratio; default 9:16 for vertical short-form."
    )
    webhook_url: Optional[str] = Field(
        default=None,
        description="URL to POST when job completes or fails (payload: job_id, status, video_url, error_message)"
    )
    vibe: Optional[str] = Field(
        default="literal",
        description="Tone: literal (default, best consistency), viral, documentary, ad, tutorial, cinematic."
    )
    tone: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Funny-serious slider: 0.0 = funny, 1.0 = serious. Used to bias dialogue and pacing."
    )
    brand_slider: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Personal (0) ↔ Business (1) brand. Influences voice and framing."
    )
    value_slider: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Promotional (0) ↔ Value (1). Promotional = salesy, CTA-heavy; Value = educational, helpful."
    )
    style_id: Optional[str] = Field(
        default=None,
        description="Optional hidden style preset id (ugc, real_life, cinematic, documentary, product)."
    )
    duration_seconds: Optional[int] = Field(
        default=None,
        ge=8,
        le=120,
        description="Optional. When set, number of scenes is derived as ceil(duration_seconds/8) for 8s Veo-optimized beats."
    )
    add_captions: bool = Field(
        default=False,
        description="If True, burn in text captions from the script. Default False; use Veo-native video only."
    )
    dry_run: bool = Field(
        default=False,
        description="If True, generate script and validate only; do not call KIE (no video credits used)."
    )
    auto_scenes: bool = Field(
        default=True,
        description="If True (default), backend infers number of scenes from the expanded brief (one scene per action; e.g. grab → peel → eat = 3 scenes, ~3s each)."
    )
    expectation: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional. One sentence for what the video should deliver; script will follow it."
    )
    template_id: Optional[str] = Field(
        default=None,
        description="Optional preset: product_ad, tutorial, vlog, cinematic. Overrides vibe/scenes/style when set."
    )

    @field_validator("webhook_url")
    @classmethod
    def webhook_url_http_or_https(cls, v: Optional[str]) -> Optional[str]:
        if not v or not v.strip():
            return None
        s = v.strip().lower()
        if s.startswith("https://") or s.startswith("http://"):
            return v.strip()
        raise ValueError("webhook_url must be an http or https URL")


class GenerateResponse(BaseModel):
    """Response body for POST /generate endpoint."""
    job_id: str
    status: JobStatus
    estimated_time_seconds: int = 180


class StatusResponse(BaseModel):
    """Response body for GET /status/{job_id} endpoint."""
    job_id: str
    status: JobStatus
    progress_percent: int = Field(ge=0, le=100)
    current_step: str
    created_at: datetime
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    virality_score: Optional[int] = None
    virality_suggestions: List[str] = Field(default_factory=list)
    storyboard: Optional["Storyboard"] = None
    script_preview: Optional[dict] = None  # Optional: core_concept, scene count for UI


class DownloadResponse(BaseModel):
    """Response body for GET /download/{job_id} endpoint."""
    job_id: str
    status: JobStatus
    video_url: Optional[str] = None
    duration_seconds: Optional[int] = None
    resolution: Optional[str] = None
    error_message: Optional[str] = None
    virality_score: Optional[int] = None
    virality_suggestions: List[str] = Field(default_factory=list)


# === Internal Data Models ===


class StoryboardScene(BaseModel):
    """Single scene in a storyboard (plan before script)."""
    scene_number: int = Field(ge=1, le=20)
    action: str = Field(..., description="Brief visual action for this scene")
    setting: str = Field(..., description="Where the scene takes place")
    characters: str = Field(..., description="Who appears in the scene")
    camera_suggestion: str = Field(..., description="Suggested camera angle")
    transition_from_previous: str = Field(..., description="How this scene connects to the previous")


class Storyboard(BaseModel):
    """Structured storyboard: scene-by-scene plan derived from the prompt."""
    scenes: List[StoryboardScene]


class Scene(BaseModel):
    """Single scene in the video script."""
    scene_number: int = Field(ge=1, le=20)
    visual_description: str = Field(
        ...,
        description="What to show visually in this scene"
    )
    dialogue: str = Field(
        ...,
        max_length=2000,
        description="Text overlay/dialogue for this scene"
    )
    camera_angle: Optional[str] = Field(
        default=None,
        description="Shot type for this scene: drone_wide, pov, close_up, tracking, low_angle, over_shoulder, dolly_in, high_angle, dutch_angle, medium_static"
    )
    # Generated during pipeline
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    last_frame_url: Optional[str] = None
    cloudinary_public_id: Optional[str] = None


class VideoScript(BaseModel):
    """Complete video script with all scenes."""
    core_concept: str = Field(
        ...,
        description="One-sentence summary of what the video MUST depict (user intent, no drift)"
    )
    character_description: str = Field(
        ...,
        description="Detailed description of the main character"
    )
    scenes: List[Scene]
    visual_style: Optional[str] = None
    background_theme: Optional[str] = None


class JobState(BaseModel):
    """Complete state for a video generation job."""
    job_id: str
    status: JobStatus = JobStatus.PENDING
    progress_percent: int = 0
    current_step: str = "Initializing..."
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Input
    prompt: str
    scene_count: int = 3
    aspect_ratio: AspectRatio = AspectRatio.PORTRAIT
    webhook_url: Optional[str] = None
    vibe: Optional[str] = None
    tone: float = 0.5
    brand_slider: float = 0.5
    value_slider: float = 0.5
    style_id: Optional[str] = None
    add_captions: bool = False
    dry_run: bool = False
    auto_scenes: bool = True
    user_expectation: Optional[str] = None  # What the video must show (user-provided or inferred)
    requested_duration_seconds: Optional[int] = None  # When set, scene_count is derived from this; auto_scenes won't overwrite

    # Generated data
    storyboard: Optional[Storyboard] = None
    script: Optional[VideoScript] = None
    reference_image_url: Optional[str] = None
    scene_durations: List[float] = Field(default_factory=list)
    scene_videos: List[str] = Field(default_factory=list)
    
    # Output
    video_url: Optional[str] = None
    duration_seconds: Optional[int] = None
    virality_score: Optional[int] = None  # 0–100, set when complete
    virality_suggestions: List[str] = Field(default_factory=list)
    
    # Error handling
    error_message: Optional[str] = None
    error_code: Optional[str] = None  # Structured: SCRIPT_FAILED, KIE_TIMEOUT, STITCH_FAILED, etc.
    retry_count: int = 0


# Resolve forward refs for StatusResponse.storyboard
StatusResponse.model_rebuild()
