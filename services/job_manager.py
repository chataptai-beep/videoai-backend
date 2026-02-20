"""
Job Manager for tracking video generation jobs.
Uses Redis when REDIS_URL is set, else in-memory dict.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Optional

from config import settings
from models.schemas import JobState, JobStatus, AspectRatio

try:
    from services.job_store import store_get, store_set, store_delete, store_keys
except ImportError:
    store_get = store_set = store_delete = store_keys = None


def _redis_enabled() -> bool:
    return bool(getattr(settings, "redis_url", None) and settings.redis_url.strip())


class JobManager:
    """
    Manages video generation job states.
    Uses in-memory dict; when REDIS_URL is set, also persists to Redis.
    """

    def __init__(self):
        self._jobs: Dict[str, JobState] = {}

    async def create_job_async(
        self,
        prompt: str,
        scene_count: int = 3,
        aspect_ratio: AspectRatio = AspectRatio.PORTRAIT,
        webhook_url: Optional[str] = None,
        vibe: Optional[str] = "viral",
        tone: float = 0.5,
        brand_slider: float = 0.5,
        value_slider: float = 0.5,
        style_id: Optional[str] = None,
        add_captions: bool = False,
        dry_run: bool = False,
        auto_scenes: bool = True,
        user_expectation: Optional[str] = None,
        requested_duration_seconds: Optional[int] = None,
    ) -> JobState:
        """Create a new video generation job (async for Redis persistence)."""
        job_id = f"vid_{uuid.uuid4().hex[:12]}"
        job = JobState(
            job_id=job_id,
            prompt=prompt,
            scene_count=scene_count,
            aspect_ratio=aspect_ratio,
            webhook_url=webhook_url,
            vibe=vibe,
            tone=tone,
            brand_slider=brand_slider,
            value_slider=value_slider,
            style_id=style_id,
            add_captions=add_captions,
            dry_run=dry_run,
            auto_scenes=auto_scenes,
            user_expectation=user_expectation,
            requested_duration_seconds=requested_duration_seconds,
            status=JobStatus.PENDING,
            progress_percent=0,
            current_step="Job created, waiting to start...",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self._jobs[job_id] = job
        if _redis_enabled() and store_set:
            await store_set(job_id, job)
        return job

    def create_job(
        self,
        prompt: str,
        scene_count: int = 3,
        aspect_ratio: AspectRatio = AspectRatio.PORTRAIT,
        webhook_url: Optional[str] = None,
        vibe: Optional[str] = "viral",
        tone: float = 0.5,
        brand_slider: float = 0.5,
        value_slider: float = 0.5,
        style_id: Optional[str] = None,
        add_captions: bool = False,
        dry_run: bool = False,
        auto_scenes: bool = True,
        user_expectation: Optional[str] = None,
        requested_duration_seconds: Optional[int] = None,
    ) -> JobState:
        """Create a new video generation job (sync; use create_job_async from async code for Redis)."""
        job_id = f"vid_{uuid.uuid4().hex[:12]}"
        job = JobState(
            job_id=job_id,
            prompt=prompt,
            scene_count=scene_count,
            aspect_ratio=aspect_ratio,
            webhook_url=webhook_url,
            vibe=vibe,
            tone=tone,
            brand_slider=brand_slider,
            value_slider=value_slider,
            style_id=style_id,
            add_captions=add_captions,
            dry_run=dry_run,
            auto_scenes=auto_scenes,
            user_expectation=user_expectation,
            requested_duration_seconds=requested_duration_seconds,
            status=JobStatus.PENDING,
            progress_percent=0,
            current_step="Job created, waiting to start...",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self._jobs[job_id] = job
        return job

    async def get_job_async(self, job_id: str) -> Optional[JobState]:
        """Get a job by ID (checks Redis if enabled and not in memory)."""
        if job_id in self._jobs:
            return self._jobs[job_id]
        if _redis_enabled() and store_get:
            job = await store_get(job_id)
            if job:
                self._jobs[job_id] = job
                return job
        return None

    def get_job(self, job_id: str) -> Optional[JobState]:
        """Get a job by ID (memory only)."""
        return self._jobs.get(job_id)

    async def update_job_async(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress_percent: Optional[int] = None,
        current_step: Optional[str] = None,
        error_message: Optional[str] = None,
        **kwargs
    ) -> Optional[JobState]:
        """Update job state and persist to Redis if enabled."""
        job = self._jobs.get(job_id) or (await self.get_job_async(job_id) if _redis_enabled() else None)
        if not job:
            return None
        if status is not None:
            job.status = status
        if progress_percent is not None:
            job.progress_percent = max(0, min(100, progress_percent))
        if current_step is not None:
            job.current_step = current_step
        if error_message is not None:
            job.error_message = error_message
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        job.updated_at = datetime.utcnow()
        self._jobs[job_id] = job
        if _redis_enabled() and store_set:
            await store_set(job_id, job)
        return job

    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress_percent: Optional[int] = None,
        current_step: Optional[str] = None,
        error_message: Optional[str] = None,
        **kwargs
    ) -> Optional[JobState]:
        """Update job state (memory only)."""
        job = self._jobs.get(job_id)
        if not job:
            return None
        if status is not None:
            job.status = status
        if progress_percent is not None:
            job.progress_percent = max(0, min(100, progress_percent))
        if current_step is not None:
            job.current_step = current_step
        if error_message is not None:
            job.error_message = error_message
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        job.updated_at = datetime.utcnow()
        return job

    async def set_error_async(self, job_id: str, error_message: str, error_code: Optional[str] = None) -> Optional[JobState]:
        """Set job to error state (async for Redis)."""
        msg = (error_message or "Unknown error").strip()
        return await self.update_job_async(
            job_id,
            status=JobStatus.ERROR,
            error_message=msg,
            error_code=error_code,
            current_step=f"Error: {msg[:50]}..." if len(msg) > 50 else f"Error: {msg}",
        )

    def set_error(self, job_id: str, error_message: str) -> Optional[JobState]:
        """Set job to error state."""
        msg = (error_message or "Unknown error").strip()
        return self.update_job(
            job_id,
            status=JobStatus.ERROR,
            error_message=msg,
            current_step=f"Error: {msg[:50]}..." if len(msg) > 50 else f"Error: {msg}",
        )

    async def set_complete_async(
        self,
        job_id: str,
        video_url: str,
        duration_seconds: int,
    ) -> Optional[JobState]:
        """Set job to complete state (async for Redis)."""
        return await self.update_job_async(
            job_id,
            status=JobStatus.COMPLETE,
            progress_percent=100,
            current_step="Video generation complete!",
            video_url=video_url,
            duration_seconds=duration_seconds,
        )

    def set_complete(
        self,
        job_id: str,
        video_url: str,
        duration_seconds: int,
    ) -> Optional[JobState]:
        """Set job to complete state."""
        return self.update_job(
            job_id,
            status=JobStatus.COMPLETE,
            progress_percent=100,
            current_step="Video generation complete!",
            video_url=video_url,
            duration_seconds=duration_seconds,
        )

    def get_all_jobs(self) -> Dict[str, JobState]:
        """List all jobs from memory (debug). Use get_all_jobs_async when Redis is enabled for cross-instance list."""
        return self._jobs.copy()

    async def get_all_jobs_async(self) -> Dict[str, JobState]:
        """List all jobs: memory + Redis when REDIS_URL set (for multi-instance debug)."""
        result = dict(self._jobs)
        if _redis_enabled() and store_keys:
            for job_id in await store_keys():
                if job_id not in result and store_get:
                    job = await store_get(job_id)
                    if job:
                        result[job_id] = job
        return result

    async def delete_job_async(self, job_id: str) -> bool:
        """Delete a job (memory + Redis if enabled)."""
        if job_id in self._jobs:
            del self._jobs[job_id]
        if _redis_enabled() and store_delete:
            await store_delete(job_id)
        return True

    def delete_job(self, job_id: str) -> bool:
        """Delete a job from memory."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False


job_manager = JobManager()
