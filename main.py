"""
Videeo.ai Stage 1 Pipeline - FastAPI Main Application

A production-ready backend that transforms text prompts into polished,
multi-scene AI videos with native video generation and frame continuity.

Author: AsaanAI
Version: 1.0.0
# Trigger redeploy: 2026-02-04 23:18
"""

import asyncio
import json
import logging
import math
import sys
import time
import uuid
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import settings, get_settings, sanitize_header_token
from models.schemas import (
    GenerateRequest,
    GenerateResponse,
    StatusResponse,
    DownloadResponse,
    JobStatus,
    AspectRatio
)
from services.job_manager import job_manager
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.validators import validate_request

# Configure logging (console + file); level and format from config
_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "app.log"

def _log_level():
    s = get_settings()
    if getattr(s, "log_level", None) and str(s.log_level).strip().upper() in ("DEBUG", "INFO", "WARNING", "ERROR"):
        return getattr(logging, str(s.log_level).strip().upper())
    return logging.DEBUG if s.debug else logging.INFO

def _log_format():
    return (getattr(get_settings(), "log_format", None) or "default").strip().lower() == "json"

class JsonLogFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }, ensure_ascii=False) + "\n"

_handlers = [
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(_LOG_FILE, encoding="utf-8"),
]
_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
for h in _handlers:
    h.setFormatter(JsonLogFormatter() if _log_format() else logging.Formatter(_format))
logging.basicConfig(level=_log_level(), handlers=_handlers)

logger = logging.getLogger(__name__)

# Project root: same folder as main.py (works regardless of cwd when running uvicorn)
PROJECT_ROOT = Path(__file__).resolve().parent

# Idempotency: key -> (job_id, timestamp); prune entries older than 24h
_idempotency_cache: dict = {}
_IDEMPOTENCY_TTL = 86400  # 24 hours

async def _idempotency_lookup(key: str) -> Optional[str]:
    if not key:
        return None
    now = time.time()
    entry = _idempotency_cache.get(key)
    if not entry:
        return None
    job_id, ts = entry
    if now - ts > _IDEMPOTENCY_TTL:
        _idempotency_cache.pop(key, None)
        return None
    return job_id

async def _idempotency_store(key: str, job_id: str) -> None:
    if not key:
        return
    _idempotency_cache[key] = (job_id, time.time())
    # Prune old entries when cache gets large
    if len(_idempotency_cache) > 1000:
        now = time.time()
        to_remove = [k for k, (_, ts) in _idempotency_cache.items() if now - ts > _IDEMPOTENCY_TTL]
        for k in to_remove:
            _idempotency_cache.pop(k, None)


# Job ID format: vid_<12 hex> or similar; reject empty/path/injection. Returns stripped id.
def _validate_job_id(job_id: str) -> str:
    if not job_id or not isinstance(job_id, str):
        raise HTTPException(status_code=400, detail="Invalid job_id")
    job_id = job_id.strip()
    if not job_id or len(job_id) > 64 or not job_id.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid job_id format")
    return job_id


# === Lifespan Events ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events for startup and shutdown."""
    import config as config_module
    import shutil
    # Ensure .env exists: copy from .env.example if missing (project root = parent of config)
    project_root = Path(config_module.__file__).resolve().parent
    env_path = project_root / ".env"
    env_example = project_root / ".env.example"
    if not env_path.exists() and env_example.exists():
        shutil.copy(env_example, env_path)
        logger.warning("Created .env from .env.example — add OPENAI_API_KEY and KIE_API_KEY, then restart.")
    # Reload config from .env so API keys are always fresh after restart
    config_module.get_settings.cache_clear()
    new_settings = config_module.get_settings()
    config_module.settings = new_settings
    global settings
    settings = new_settings

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Use project-root-relative paths so pipeline works no matter where uvicorn is run from
    if not Path(settings.output_dir).is_absolute():
        new_settings.output_dir = str(project_root / (settings.output_dir.strip().lstrip("./") or "outputs"))
    if not Path(settings.temp_dir).is_absolute():
        new_settings.temp_dir = str(project_root / (settings.temp_dir.strip().lstrip("./") or "temp"))
    config_module.settings = new_settings
    settings = new_settings
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.temp_dir).mkdir(parents=True, exist_ok=True)
    # Resolve analytics log path to project root when relative
    analytics_path = getattr(settings, "analytics_failures_path", "./logs/failures.jsonl") or "./logs/failures.jsonl"
    if not Path(analytics_path).is_absolute():
        new_settings.analytics_failures_path = str(project_root / analytics_path.lstrip("./"))
        config_module.settings = new_settings
        settings = new_settings
    Path(settings.analytics_failures_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        from services.cleanup import cleanup_old_files
        cleanup_old_files()
    except Exception as e:
        logger.warning(f"Cleanup on startup skipped: {e}")
    openai_ok = bool(sanitize_header_token(get_settings().openai_api_key or ""))
    kie_ok = bool(sanitize_header_token(get_settings().kie_api_key or ""))
    if not openai_ok:
        logger.warning("OPENAI_API_KEY not set or invalid — script generation will fail")
    if not kie_ok:
        logger.warning("KIE_API_KEY not set or invalid — kie.ai Kling 3.0 video generation will fail")
    if openai_ok and kie_ok:
        logger.info("API keys loaded — pipeline ready (kie.ai Kling 3.0 text-to-video)")
    yield
    
    # Shutdown
    logger.info("Shutting down Videeo.ai Pipeline...")


# === FastAPI App ===

app = FastAPI(
    title=settings.app_name,
    description="""
    ## Videeo.ai Pipeline — kie.ai Kling 3.0 text-to-video
    
    One prompt → script (N scenes) → **one kie.ai Kling 3.0** multi_prompt call → one video.
    No other video/image tools; pipeline uses kie.ai Kling only.
    
    ### Workflow
    1. `POST /generate` - Submit a text prompt
    2. `GET /status/{job_id}` - Poll status and progress
    3. `GET /download/{job_id}` - Get the final video URL
    """,
    version=settings.app_version,
    lifespan=lifespan
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return generic 500 to client; log full error (no stack trace to client). Include request_id when available."""
    if isinstance(exc, (HTTPException, RequestValidationError)):
        raise exc
    request_id = getattr(request.state, "request_id", None)
    logger.exception("Unhandled error: %s", exc)
    content = {"detail": "An internal error occurred. Check server logs."}
    if request_id:
        content["request_id"] = request_id
    response = JSONResponse(status_code=500, content=content)
    if request_id:
        response.headers["X-Request-ID"] = request_id
    return response


# === Request ID (correlation for logs and error responses) ===
_request_id_header = (getattr(settings, "request_id_header", None) or "X-Request-ID").strip() or "X-Request-ID"

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = (request.headers.get(_request_id_header) or "").strip() or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers[_request_id_header] = request_id
    return response


# === Rate limiting: configurable per IP per hour ===
_ip_request_history = {}  # {ip: [timestamp1, ...]}

def _prune_rate_limit_history():
    """Remove IPs with no requests in the last 2 hours to prevent unbounded memory growth."""
    if len(_ip_request_history) < 100:
        return
    now = datetime.utcnow().timestamp()
    two_hours_ago = now - 7200
    to_remove = [ip for ip, ts_list in _ip_request_history.items() if not ts_list or max(ts_list) < two_hours_ago]
    for ip in to_remove:
        del _ip_request_history[ip]

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    limit = max(0, getattr(settings, "rate_limit_per_hour", 50) or 0)
    if limit > 0 and request.url.path == "/generate" and request.method == "POST":
        _prune_rate_limit_history()
        client = getattr(request, "client", None)
        client_ip = (client.host if client else None) or (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() or "unknown"
        client_ip = str(client_ip or "unknown")
        now = datetime.utcnow().timestamp()
        one_hour_ago = now - 3600
        history = [ts for ts in _ip_request_history.get(client_ip, []) if ts > one_hour_ago]
        if len(history) >= limit:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded. Max {limit} requests per hour."}
            )
        history.append(now)
        _ip_request_history[client_ip] = history
    return await call_next(request)


# CORS: configurable origins (restrict in production)
_cors_origins = (getattr(settings, "cors_origins", None) or "*").strip()
_cors_origins_list = [o.strip() for o in _cors_origins.split(",") if o.strip()] if _cors_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static and outputs from project root (create dirs so mounts never 404)
_static_dir = PROJECT_ROOT / "static"
_outputs_dir = PROJECT_ROOT / "outputs"
_static_dir.mkdir(parents=True, exist_ok=True)
_outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
app.mount("/outputs", StaticFiles(directory=str(_outputs_dir)), name="outputs")


# === API Endpoints ===

@app.get("/", tags=["Frontend"], response_class=HTMLResponse)
async def root():
    """Serve the frontend application."""
    index_path = PROJECT_ROOT / "static" / "index.html"
    if not index_path.exists():
        return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness + dependency checks: API keys (sanitized), FFmpeg, dirs. Use /ready for readiness (Redis)."""
    s = get_settings()
    openai_ok = bool(sanitize_header_token(s.openai_api_key or ""))
    kie_ok = bool(sanitize_header_token(s.kie_api_key or ""))
    ffmpeg_ok = False
    ffmpeg_path = None
    try:
        import shutil
        ffmpeg_path = shutil.which("ffmpeg")
    except Exception:
        ffmpeg_path = None
    if ffmpeg_path:
        ffmpeg_ok = True
    else:
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg_ok = bool(ffmpeg_path and Path(ffmpeg_path).exists())
        except Exception:
            ffmpeg_ok = False
            ffmpeg_path = None
    redis_ok = None
    if getattr(s, "redis_url", None) and s.redis_url.strip():
        try:
            from services.job_store import store_ping
            redis_ok = await store_ping()
        except Exception:
            redis_ok = False
    return {
        "status": "healthy",
        "checks": {
            "openai_api_key": openai_ok,
            "kie_api_key": kie_ok,
            "ffmpeg_available": ffmpeg_ok,
            "redis": "ok" if redis_ok else ("unreachable" if redis_ok is False else "not_configured"),
            "cloudinary_configured": bool(
                (cn := (s.cloudinary_cloud_name or "").strip().lower())
                and "your_cloud" not in cn
                and "your-cloud" not in cn
                and sanitize_header_token(s.cloudinary_api_key or "")
                and "your_api_key" not in (s.cloudinary_api_key or "").lower()
            ),
        },
        "ready": openai_ok and kie_ok and ffmpeg_ok,
        "config": {
            "output_dir": s.output_dir,
            "scene_duration_seconds": s.scene_duration_seconds,
            "max_retries": s.max_retries,
            "ffmpeg_path": ffmpeg_path,
        }
    }


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness: 200 if app can accept work (Redis reachable when REDIS_URL set). Use for load balancer probes."""
    s = get_settings()
    if not (getattr(s, "redis_url", None) and s.redis_url.strip()):
        return {"status": "ready", "redis": "not_configured"}
    try:
        from services.job_store import store_ping
        if await store_ping():
            return {"status": "ready", "redis": "ok"}
        return JSONResponse(status_code=503, content={"status": "not_ready", "reason": "redis_unavailable"})
    except Exception as e:
        logger.warning("Readiness Redis ping failed: %s", e)
        return JSONResponse(status_code=503, content={"status": "not_ready", "reason": "redis_ping_failed", "detail": str(e)})


@app.post(
    "/generate",
    response_model=GenerateResponse,
    tags=["Video Generation"],
    summary="Start video generation",
    description="Submit a text prompt to generate a multi-scene video."
)
async def generate_video(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    req: Request,
):
    """
    Start a new video generation job.
    
    The job runs asynchronously in the background. Use the returned job_id
    to poll for status and eventually download the completed video.
    
    **Request Body:** Only `prompt` is required; production choices have smart defaults.
    - `prompt`: What the video is about (10-20000 chars). Short prompts are expanded into a full brief on the backend.
    - `scenes`: Optional; default 3 (short-form). Max 20.
    - `aspect_ratio`: Optional; default 9:16 (vertical).
    - `vibe`: Optional; default "literal" (real-life sequence). Or viral (hook/problem/insight/CTA), documentary, ad, tutorial, cinematic.
    
    **Response:**
    - `job_id`: Unique identifier for tracking the job
    - `status`: Current status (pending)
    - `estimated_time_seconds`: Estimated completion time
    """
    # Require only: OpenAI (script), KIE_API_KEY (Kling 3.0). Pipeline is Kling-only via kie.ai.
    s = get_settings()
    openai_key = sanitize_header_token(s.openai_api_key or "")
    kie_key = sanitize_header_token(s.kie_api_key or "")
    missing = []
    if not openai_key:
        missing.append("OPENAI_API_KEY")
    if not kie_key:
        missing.append("KIE_API_KEY")
    if missing:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Missing API keys: {', '.join(missing)}. "
                "Add OPENAI_API_KEY and KIE_API_KEY to .env (kie.ai Kling 3.0 text-to-video). Restart the server."
            ),
        )
    logger.info(f"New video generation request: {request.prompt[:50]}...")
    # Apply template if set (overrides vibe, scenes, style_id when not explicitly provided)
    template_id = (getattr(request, "template_id", None) or "").strip().lower()
    templates = {
        "product_ad": {"vibe": "ad", "scenes": 5, "style_id": "ugc"},
        "tutorial": {"vibe": "tutorial", "scenes": 4, "style_id": "real_life"},
        "vlog": {"vibe": "literal", "scenes": 3, "style_id": "ugc"},
        "cinematic": {"vibe": "cinematic", "scenes": 5, "style_id": "cinematic"},
    }
    if template_id and template_id in templates:
        t = templates[template_id]
        vibe = t.get("vibe", "literal")
        scene_count = t.get("scenes", request.scenes)
        style_id = t.get("style_id")
    else:
        vibe = getattr(request, "vibe", None)
        style_id = getattr(request, "style_id", None)
        scene_count = request.scenes
    # When duration_seconds is set, use 8s Veo-optimized beats: scenes = ceil(duration/8)
    if getattr(request, "duration_seconds", None) is not None:
        scene_count = min(20, max(1, math.ceil(request.duration_seconds / 8)))
    # Fail fast: validate request before creating job (no credits used)
    request_errors, request_warnings = validate_request(
        request.prompt,
        scene_count,
        request.aspect_ratio,
        style_id=style_id or getattr(request, "style_id", None),
        webhook_url=getattr(request, "webhook_url", None),
    )
    if request_errors:
        raise HTTPException(status_code=422, detail={"message": "Request validation failed", "errors": request_errors, "warnings": request_warnings})
    idempotency_key = (req.headers.get("X-Idempotency-Key") or "").strip()
    if idempotency_key and len(idempotency_key) <= 128:
        existing = await _idempotency_lookup(idempotency_key)
        if existing:
            job = await job_manager.get_job_async(existing)
            if job:
                logger.info("Idempotency hit: returning existing job %s", existing)
                return GenerateResponse(job_id=existing, status=job.status, estimated_time_seconds=job.scene_count * 30 + 60)
    job = await job_manager.create_job_async(
        prompt=request.prompt,
        scene_count=scene_count,
        aspect_ratio=request.aspect_ratio,
        webhook_url=getattr(request, "webhook_url", None),
        vibe=vibe or getattr(request, "vibe", None),
        tone=getattr(request, "tone", 0.5),
        brand_slider=getattr(request, "brand_slider", 0.5),
        value_slider=getattr(request, "value_slider", 0.5),
        style_id=style_id or getattr(request, "style_id", None),
        add_captions=getattr(request, "add_captions", False),
        dry_run=getattr(request, "dry_run", False),
        auto_scenes=getattr(request, "auto_scenes", True),
        user_expectation=getattr(request, "expectation", None),
        requested_duration_seconds=getattr(request, "duration_seconds", None),
    )
    orchestrator = PipelineOrchestrator()
    background_tasks.add_task(orchestrator.run_pipeline, job.job_id)
    
    # Calculate estimated time (rough estimate based on scene count)
    estimated_time = scene_count * 30 + 60  # ~30s per scene + 60s overhead
    
    logger.info(f"Job created: {job.job_id}")
    if idempotency_key:
        await _idempotency_store(idempotency_key, job.job_id)

    return GenerateResponse(
        job_id=job.job_id,
        status=JobStatus.PENDING,
        estimated_time_seconds=estimated_time
    )


@app.post(
    "/validate/request",
    tags=["Validation"],
    summary="Validate request (no API calls, no credits)",
    description="Check if prompt, scenes, and aspect_ratio would pass validation. No job created, no OpenAI/KIE calls.",
)
async def validate_request_endpoint(request: GenerateRequest):
    """Validate /generate request body. Returns valid: true if no errors, else errors list. Zero credits."""
    scene_count = request.scenes
    if getattr(request, "duration_seconds", None) is not None:
        scene_count = min(20, max(1, math.ceil(request.duration_seconds / 8)))
    errors, warnings = validate_request(
        request.prompt,
        scene_count,
        request.aspect_ratio,
        style_id=getattr(request, "style_id", None),
        webhook_url=getattr(request, "webhook_url", None),
    )
    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


@app.get(
    "/status/{job_id}",
    response_model=StatusResponse,
    tags=["Video Generation"],
    summary="Get job status",
    description="Get the current status and progress of a video generation job."
)
async def get_status(job_id: str):
    """
    Get the status of a video generation job.
    
    **Response:**
    - `job_id`: The job identifier
    - `status`: Current status (pending, generating_script, generating_images, etc.)
    - `progress_percent`: Progress as percentage (0-100)
    - `current_step`: Human-readable description of current step
    - `created_at`: Job creation timestamp
    - `error_message`: Error details if status is 'error'
    """
    job_id = _validate_job_id(job_id)
    job = await job_manager.get_job_async(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    script_preview = None
    if job.script:
        script_preview = {"core_concept": job.script.core_concept, "scene_count": len(job.script.scenes)}
    return StatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress_percent=max(0, min(100, job.progress_percent)),
        current_step=job.current_step,
        created_at=job.created_at,
        error_message=job.error_message,
        error_code=getattr(job, "error_code", None),
        virality_score=getattr(job, "virality_score", None),
        virality_suggestions=getattr(job, "virality_suggestions", None) or [],
        storyboard=getattr(job, "storyboard", None),
        script_preview=script_preview,
    )


@app.get(
    "/download/{job_id}",
    response_model=DownloadResponse,
    tags=["Video Generation"],
    summary="Get download URL",
    description="Get the download URL for a completed video."
)
async def get_download(job_id: str):
    """
    Get the download URL for a completed video.
    
    **Response:**
    - `job_id`: The job identifier
    - `status`: Current status (should be 'complete' for successful download)
    - `video_url`: URL to download the video
    - `duration_seconds`: Video duration
    - `resolution`: Video resolution (e.g., "1920x1080")
    - `error_message`: Error details if job failed
    """
    job_id = _validate_job_id(job_id)
    job = await job_manager.get_job_async(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status == JobStatus.ERROR:
        return DownloadResponse(
            job_id=job.job_id,
            status=job.status,
            error_message=job.error_message
        )
    
    if job.status != JobStatus.COMPLETE:
        return DownloadResponse(
            job_id=job.job_id,
            status=job.status,
            error_message=f"Video not ready. Current status: {job.status.value}"
        )
    # Ensure the UI gets a loadable URL: /outputs/filename or full https URL (never raw filesystem path)
    video_url = (job.video_url or "").strip()
    if video_url and not video_url.startswith("http"):
        from pathlib import Path as P
        base = P(video_url.replace("\\", "/")).name
        video_url = "/outputs/" + base if base else video_url
    if video_url and not video_url.startswith("/") and not video_url.startswith("http"):
        video_url = "/outputs/" + video_url.lstrip("/")

    return DownloadResponse(
        job_id=job.job_id,
        status=job.status,
        video_url=video_url,
        duration_seconds=job.duration_seconds,
        resolution=f"{settings.video_width}x{settings.video_height}",
        virality_score=getattr(job, "virality_score", None),
        virality_suggestions=getattr(job, "virality_suggestions", None) or [],
    )


# === Debug Endpoints (development only) ===

@app.get("/jobs", tags=["Debug"], include_in_schema=settings.debug)
async def list_jobs():
    """List all jobs (debug endpoint)."""
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Not found")
    
    jobs = await job_manager.get_all_jobs_async()
    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": j.job_id,
                "status": j.status.value,
                "progress": j.progress_percent,
                "created_at": j.created_at.isoformat()
            }
            for j in jobs.values()
        ]
    }


@app.delete("/jobs/{job_id}", tags=["Debug"], include_in_schema=settings.debug)
async def delete_job(job_id: str):
    """Delete a job (debug endpoint). Uses async so Redis is updated when enabled."""
    job_id = _validate_job_id(job_id)
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Not found")
    existing = await job_manager.get_job_async(job_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    await job_manager.delete_job_async(job_id)
    return {"deleted": job_id}


# === SSE: stream job progress ===

@app.get("/events/{job_id}", tags=["Video Generation"])
async def stream_job_events(job_id: str):
    """Server-Sent Events: stream job status until complete or error. Connect with EventSource."""
    job_id = _validate_job_id(job_id)
    try:
        from sse_starlette.sse import EventSourceResponse
    except ImportError:
        raise HTTPException(status_code=501, detail="Install sse-starlette for SSE support")
    import json
    _sse_max_duration_seconds = 3600
    _sse_poll_interval = 2

    async def event_generator():
        started = time.monotonic()
        while True:
            if time.monotonic() - started > _sse_max_duration_seconds:
                yield {"event": "timeout", "data": json.dumps({"detail": "SSE stream closed after max duration"})}
                return
            job = await job_manager.get_job_async(job_id)
            if not job:
                yield {"event": "error", "data": json.dumps({"detail": "Job not found"})}
                return
            data = {
                "job_id": job.job_id,
                "status": job.status.value,
                "progress_percent": max(0, min(100, job.progress_percent)),
                "current_step": job.current_step,
                "error_message": job.error_message,
                "error_code": getattr(job, "error_code", None),
                "video_url": getattr(job, "video_url", None),
                "storyboard": job.storyboard.model_dump(mode="json") if getattr(job, "storyboard", None) else None,
            }
            yield {"event": "status", "data": json.dumps(data)}
            if job.status in (JobStatus.COMPLETE, JobStatus.ERROR):
                return
            await asyncio.sleep(_sse_poll_interval)
    return EventSourceResponse(event_generator())


# === Analytics (failure log) ===

@app.get("/analytics/failures", tags=["Analytics"])
async def get_analytics_failures(limit: int = 100):
    """Read recent failure log entries (for debugging). Limit clamped to 1-500."""
    import json
    limit = max(1, min(500, limit))
    raw_path = getattr(get_settings(), "analytics_failures_path", "./logs/failures.jsonl") or "./logs/failures.jsonl"
    raw_path = raw_path.strip()
    path = Path(raw_path) if Path(raw_path).is_absolute() else (PROJECT_ROOT / raw_path.lstrip("./"))
    if not path.exists():
        return {"entries": [], "total": 0}
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    entries = []
    for line in reversed(lines[-limit:]):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
    return {"entries": entries, "total": len(entries)}


# === File Download (for local files) ===

@app.get("/files/{filename}", tags=["Files"])
async def download_local_file(filename: str):
    """Download a locally stored video file. Filename must not contain path traversal."""
    safe_name = Path(filename).name
    if not safe_name or safe_name in (".", "..") or safe_name != filename or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    out_dir = get_settings().output_dir
    out_dir_resolved = Path(out_dir).resolve() if Path(out_dir).is_absolute() else (PROJECT_ROOT / (out_dir.strip().lstrip("./") or "outputs"))
    file_path = out_dir_resolved / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        file_path.resolve().relative_to(out_dir_resolved.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return FileResponse(
        path=str(file_path),
        filename=safe_name,
        media_type="video/mp4"
    )


# === Run Server ===

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
