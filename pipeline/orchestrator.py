"""
Pipeline Orchestrator: one prompt → one perfect video.

Flow: expand short prompt → infer scene count → generate script → one Kling 3.0 call (Standard 720p, audio) → video.
No extra LLM calls; no expectation inference. Optional: captions, upload.
"""

import asyncio
import logging
from pathlib import Path

import httpx

from config import settings, _is_placeholder_key
from models.schemas import JobState, JobStatus, VideoScript, AspectRatio, Storyboard
from services.job_manager import job_manager
from services.analytics import log_failure, log_success
from services.virality_scorer import score_from_script
from .prompt_expander import expand_prompt, SHORT_PROMPT_MAX_CHARS, infer_scene_count_from_brief
from .storyboard_generator import StoryboardGenerator
from .script_generator import ScriptGenerator
from .style_presets import select_style_preset
from .video_generator import VideoGenerator
from .video_stitcher import VideoStitcher
from .caption_burner import CaptionBurner
from .audio_polish import AudioPolisher
from .gemini_qa import analyze_transitions
from .last_frame import extract_last_frame_as_url
from .validators import validate_script, validate_before_kie

logger = logging.getLogger(__name__)


def _remove_if_local_file(path: str) -> None:
    """Delete a local file if path is a filesystem path (no ://). Ignores errors."""
    if not path or "://" in path:
        return
    p = Path(path)
    if p.is_file():
        try:
            p.unlink()
            logger.debug("Removed intermediate file: %s", path)
        except OSError as e:
            logger.warning("Could not remove intermediate file %s: %s", path, e)


class PipelineOrchestrator:
    """
    Kling-only pipeline (kie.ai workflow).
    1. Script (LLM) – one prompt → N scene prompts for multi_prompt
    2. Video – kie.ai Kling 3.0 text-to-video (single multi_prompt call)
    3. Download – save Kling output to local file
    4. Captions (optional)
    5. Upload / finalize (optional Cloudinary)
    """

    def __init__(self):
        self.storyboard_generator = StoryboardGenerator()
        self.script_generator = ScriptGenerator()
        self.video_generator = VideoGenerator()
        self.video_stitcher = VideoStitcher()
        self.caption_burner = CaptionBurner()
        self.audio_polisher = AudioPolisher()
        self.cloudinary_cloud = settings.cloudinary_cloud_name
        self.cloudinary_key = settings.cloudinary_api_key
        self.cloudinary_secret = settings.cloudinary_api_secret

    async def run_pipeline(self, job_id: str) -> None:
        """Run pipeline: script → validate → Kling 3.0 (KIE) → download → optional captions → optional upload. If dry_run, stop after validation (no KIE)."""
        job = await job_manager.get_job_async(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        try:
            await self._stage_script_generation(job)
            job = await job_manager.get_job_async(job_id)
            if not job or not job.script:
                raise Exception("Script missing after script generation")
            # Validate before any KIE call (avoid burning credits on bad payload)
            script_errors = validate_script(job.script)
            if script_errors:
                raise Exception(f"Script validation failed: {'; '.join(script_errors)}")
            multi_prompt = self.video_generator.build_multi_prompt_only(job.script, job.aspect_ratio)
            kie_errors = validate_before_kie(job.script, job.aspect_ratio, multi_prompt)
            if kie_errors:
                raise Exception(f"Kling payload validation failed: {'; '.join(kie_errors)}")
            if getattr(job, "dry_run", False):
                await self._stage_dry_run_complete(job_id, multi_prompt)
                return
            await self._stage_video_generation(job)
            job = await job_manager.get_job_async(job_id)
            await self._stage_video_stitching(job)
            job = await job_manager.get_job_async(job_id)
            await self._stage_transition_qa(job)
            job = await job_manager.get_job_async(job_id)
            await self._stage_audio_polish(job)
            job = await job_manager.get_job_async(job_id)
            await self._stage_caption_burnin(job)
            job = await job_manager.get_job_async(job_id)
            await self._stage_upload_and_finalize(job)
        except Exception as e:
            logger.exception(f"Pipeline failed for job {job_id}: {e}")
            await self._handle_error_async(job_id, str(e))
    
    async def _stage_script_generation(self, job: JobState) -> None:
        """Stage 1: Expand brief → infer scene count → generate script. One prompt → one video."""
        await job_manager.update_job_async(
            job.job_id,
            status=JobStatus.GENERATING_SCRIPT,
            progress_percent=5,
            current_step="Generating video script..."
        )
        # Backend handles production: expand short prompts so one line becomes a full brief
        prompt_for_script = job.prompt
        if len((job.prompt or "").strip()) <= SHORT_PROMPT_MAX_CHARS:
            try:
                prompt_for_script = await expand_prompt(job.prompt)
            except Exception as e:
                logger.warning(f"Prompt expansion skipped, using original: {e}")
        # Auto scene count from brief (skip when user set duration_seconds — scene_count already derived)
        if getattr(job, "auto_scenes", True) and not getattr(job, "requested_duration_seconds", None):
            inferred = infer_scene_count_from_brief(prompt_for_script)
            await job_manager.update_job_async(job.job_id, scene_count=inferred)
            job = await job_manager.get_job_async(job.job_id) or job
        style_preset = select_style_preset(
            prompt_for_script,
            vibe=getattr(job, "vibe", None),
            style_id=getattr(job, "style_id", None),
        )
        # Generate storyboard (scene plan) before script
        await job_manager.update_job_async(
            job.job_id,
            style_id=style_preset.style_id,
            current_step="Creating storyboard...",
        )
        try:
            storyboard = await self._with_retry(
                lambda: self.storyboard_generator.generate(
                    prompt_for_script,
                    scene_count=job.scene_count,
                    vibe=getattr(job, "vibe", None),
                ),
                "Storyboard generation"
            )
        except Exception as e:
            logger.warning("Storyboard generation failed, continuing without storyboard: %s", e)
            storyboard = None
        await job_manager.update_job_async(
            job.job_id,
            storyboard=storyboard,
            progress_percent=8,
            current_step=f"Generating script ({style_preset.label} style)...",
        )
        job = await job_manager.get_job_async(job.job_id) or job
        script = await self._with_retry(
            lambda: self.script_generator.generate(
                prompt_for_script,
                job.scene_count,
                vibe=getattr(job, "vibe", None),
                tone=getattr(job, "tone", 0.5),
                brand_slider=getattr(job, "brand_slider", 0.5),
                value_slider=getattr(job, "value_slider", 0.5),
                user_expectation=getattr(job, "user_expectation", None),
                style_preset=style_preset,
                storyboard=job.storyboard,
            ),
            "Script generation"
        )
        await job_manager.update_job_async(
            job.job_id,
            script=script,
            progress_percent=15,
            current_step=f"Script generated: {len(script.scenes)} scenes"
        )
        logger.info(f"Job {job.job_id}: Script generated with {len(script.scenes)} scenes")

    async def _stage_video_generation(self, job: JobState) -> None:
        """Stage 2: kie.ai Kling 3.0 text-to-video (one multi_prompt call → one video)."""
        await job_manager.update_job_async(
            job.job_id,
            status=JobStatus.GENERATING_VIDEOS,
            progress_percent=25,
            current_step="Generating video (kie.ai Kling 3.0)..."
        )
        job = await job_manager.get_job_async(job.job_id)
        if not job or not job.script or not getattr(job.script, "scenes", None):
            raise Exception("Job or script/scenes missing before video generation")
        scenes = job.script.scenes
        if len(scenes) == 0:
            raise Exception("Script has no scenes; cannot generate video")
        use_last_frame = self._can_use_last_frame_continuity() and len(scenes) > 1
        max_shots = 6
        use_multi_prompt = (
            bool(getattr(settings, "use_multi_prompt", True))
            and not use_last_frame
            and len(scenes) <= max_shots
        )

        if use_multi_prompt:
            # No orchestrator retry for KIE: each attempt = 1 KIE call (credits). Avoid 2–3x spend on transient failures.
            video_url = await self._with_retry(
                lambda: self.video_generator.generate_video_multi_prompt(job.script, job.aspect_ratio),
                "Kling multi_prompt video generation",
                max_retries=0,
            )
            scene_videos = [video_url]
            for s in scenes:
                s.video_url = video_url
            await job_manager.update_job_async(
                job.job_id,
                scene_videos=scene_videos,
                progress_percent=70,
                current_step=f"Video generated ({len(scenes)} scenes in one clip)"
            )
            logger.info(f"Job {job.job_id}: Kling 3.0 multi_prompt video generated (1 URL, {len(scenes)} scenes)")
            return

        scene_videos = []
        last_frame_url = None
        retry_limit = max(0, int(getattr(settings, "scene_retry_limit", 1) or 0))
        total = len(scenes)
        for idx, scene in enumerate(scenes):
            scene_number = idx + 1
            reference_url = last_frame_url if (use_last_frame and last_frame_url) else ""
            progress = 25 + int(((idx + 1) / total) * 45)
            await job_manager.update_job_async(
                job.job_id,
                progress_percent=progress,
                current_step=f"Generating scene {scene_number}/{total}..."
            )
            video_url = await self._with_retry(
                lambda: self.video_generator.generate_scene_video(
                    scene=scene,
                    reference_image_url=reference_url,
                    aspect_ratio=job.aspect_ratio,
                    scene_index=idx,
                    character_description=job.script.character_description,
                    background_theme=job.script.background_theme or "",
                    core_concept=job.script.core_concept or "",
                    visual_style=job.script.visual_style or "",
                    original_prompt=job.prompt or "",
                    total_scenes=total,
                    previous_scenes_descriptions=[s.visual_description for s in scenes[:idx]],
                ),
                f"Kling scene {scene_number} generation",
                max_retries=retry_limit,
            )
            scene.video_url = video_url
            scene_videos.append(video_url)

            if use_last_frame:
                try:
                    last_frame_url = await extract_last_frame_as_url(
                        video_url,
                        job.job_id,
                        scene_number,
                        cloudinary_cloud=self.cloudinary_cloud,
                        cloudinary_key=self.cloudinary_key,
                        cloudinary_secret=self.cloudinary_secret,
                    )
                    scene.last_frame_url = last_frame_url
                except Exception as e:
                    logger.warning("Last-frame extraction failed for scene %s: %s", scene_number, e)
                    last_frame_url = None

        await job_manager.update_job_async(
            job.job_id,
            scene_videos=scene_videos,
            progress_percent=70,
            current_step=f"Generated {len(scene_videos)} scene clips"
        )
        logger.info(f"Job {job.job_id}: Generated {len(scene_videos)} scene clips")

    async def _stage_video_stitching(self, job: JobState) -> None:
        """Stage 3: Download Kling output (single URL) to local file."""
        await job_manager.update_job_async(
            job.job_id,
            status=JobStatus.ASSEMBLING_VIDEO,
            progress_percent=72,
            current_step="Saving video..."
        )
        job = await job_manager.get_job_async(job.job_id)
        if not job or not getattr(job, "scene_videos", None) or len(job.scene_videos) == 0:
            raise Exception("No scene videos to stitch; pipeline state invalid")
        try:
            output_filename = f"video_{job.job_id}"
            use_multi_prompt = bool(getattr(settings, "use_multi_prompt", True)) and len(job.scene_videos) == 1
            default_trim = float(getattr(settings, "default_scene_trim_start_seconds", 1.0) or 0.0)

            if use_multi_prompt:
                stitch_result = await self._with_retry(
                    lambda: self.video_stitcher.stitch_with_crossfade(
                        video_urls=job.scene_videos,
                        output_filename=output_filename
                    ),
                    "Video stitching"
                )
            else:
                stitch_result = await self._with_retry(
                    lambda: self.video_stitcher.stitch_with_trims(
                        video_urls=job.scene_videos,
                        output_filename=output_filename,
                        trim_instructions=None,
                        default_trim_start=default_trim
                    ),
                    "Video stitching"
                )
            stitched_path = stitch_result[0]
            scene_durations = stitch_result[1]
            
            # Store path temporarily (will be replaced with CDN URL)
            await job_manager.update_job_async(
                job.job_id,
                video_url=stitched_path,
                scene_durations=scene_durations,
                progress_percent=85,
                current_step="Video saved"
            )
            logger.info(f"Job {job.job_id}: Video saved to {stitched_path}")
        except Exception as e:
            raise Exception(f"Video save failed: {e}")

    async def _stage_transition_qa(self, job: JobState) -> None:
        """Stage 3.5: Gemini QA for transitions; optionally re-stitch with trims."""
        job = await job_manager.get_job_async(job.job_id)
        if not job or not getattr(job, "video_url", None):
            return
        if not getattr(settings, "gemini_qa_enabled", True):
            return
        if not job.script or not job.script.scenes or len(job.script.scenes) <= 1:
            return
        if not getattr(job, "scene_videos", None) or len(job.scene_videos) <= 1:
            return

        await job_manager.update_job_async(
            job.job_id,
            progress_percent=88,
            current_step="Running transition QA..."
        )
        scene_duration = 0.0
        if getattr(job, "scene_durations", None):
            scene_duration = sum(job.scene_durations) / max(1, len(job.scene_durations))
        if scene_duration <= 0:
            scene_duration = float(getattr(settings, "scene_duration_seconds", 4) or 4)

        try:
            trims = await analyze_transitions(
                job.video_url,
                scene_count=len(job.script.scenes),
                scene_duration_seconds=scene_duration,
            )
        except Exception as e:
            logger.warning("Gemini QA failed: %s", e)
            return

        if not trims:
            logger.info("Gemini QA: no trim changes requested")
            return

        try:
            default_trim = float(getattr(settings, "default_scene_trim_start_seconds", 1.0) or 0.0)
            stitch_result = await self.video_stitcher.stitch_with_trims(
                video_urls=job.scene_videos,
                output_filename=f"video_{job.job_id}_qa",
                trim_instructions=trims,
                default_trim_start=default_trim,
            )
            stitched_path, scene_durations = stitch_result
            await job_manager.update_job_async(
                job.job_id,
                video_url=stitched_path,
                scene_durations=scene_durations,
                progress_percent=90,
                current_step="Transition QA applied"
            )
            logger.info("Gemini QA trims applied for job %s", job.job_id)
        except Exception as e:
            logger.warning("Gemini QA restitch failed; using original stitch: %s", e)

    async def _stage_audio_polish(self, job: JobState) -> None:
        """Stage 3.6: Normalize audio + optional music bed/ducking."""
        job = await job_manager.get_job_async(job.job_id)
        if not job or not getattr(job, "video_url", None):
            return
        if not getattr(settings, "audio_polish_enabled", True):
            return

        await job_manager.update_job_async(
            job.job_id,
            progress_percent=92,
            current_step="Polishing audio..."
        )
        try:
            polished_path = await self.audio_polisher.polish(
                job.video_url,
                output_filename=f"video_{job.job_id}"
            )
            duration = await self.video_stitcher.get_video_duration(polished_path)
            await job_manager.update_job_async(
                job.job_id,
                video_url=polished_path,
                duration_seconds=int(duration) if duration else job.duration_seconds,
                progress_percent=94,
                current_step="Audio polished"
            )
        except Exception as e:
            logger.warning("Audio polish failed, continuing with original audio: %s", e)

    def _cloudinary_configured(self) -> bool:
        return bool(
            self.cloudinary_cloud
            and self.cloudinary_key
            and self.cloudinary_secret
            and not _is_placeholder_key(self.cloudinary_cloud)
            and not _is_placeholder_key(self.cloudinary_key or "")
            and not _is_placeholder_key(self.cloudinary_secret or "")
        )

    def _can_use_last_frame_continuity(self) -> bool:
        if not getattr(settings, "use_last_frame_continuity", True):
            return False
        if self._cloudinary_configured():
            return True
        base_url = (getattr(settings, "base_url", "") or "").lower()
        if base_url and "localhost" not in base_url and "127.0.0.1" not in base_url:
            return True
        logger.warning(
            "Last-frame continuity disabled: Cloudinary not configured and base_url is local (%s)",
            base_url or "<unset>",
        )
        return False
    
    async def _stage_caption_burnin(self, job: JobState) -> None:
        """Stage 4: Optionally burn captions; otherwise keep Kling video as final."""
        job = await job_manager.get_job_async(job.job_id)
        if not getattr(job, "add_captions", False):
            # No captions: use stitched video as final, get duration only
            await job_manager.update_job_async(
                job.job_id,
                status=JobStatus.ADDING_CAPTIONS,
                progress_percent=87,
                current_step="No captions requested..."
            )
            duration = await self.video_stitcher.get_video_duration(job.video_url or "")
            await job_manager.update_job_async(
                job.job_id,
                video_url=job.video_url,
                duration_seconds=int(duration) if duration else 0,
                progress_percent=95,
                current_step="Video ready"
            )
            logger.info(f"Job {job.job_id}: Using Kling video (no captions), duration: {duration}s")
            return

        await job_manager.update_job_async(
            job.job_id,
            status=JobStatus.ADDING_CAPTIONS,
            progress_percent=87,
            current_step="Adding captions..."
        )
        job = await job_manager.get_job_async(job.job_id)
        if not job or not job.script or not getattr(job.script, "scenes", None) or len(job.script.scenes) == 0:
            logger.warning("Caption burn skipped: script or scenes missing")
            duration = await self.video_stitcher.get_video_duration(job.video_url or "") if job else 0
            await job_manager.update_job_async(
                job.job_id,
                video_url=job.video_url,
                duration_seconds=int(duration) if duration else 0,
                progress_percent=95,
                current_step="Captions skipped (script missing)"
            )
            return
        try:
            stitched_path = (job.video_url or "").strip()  # intermediate video_xxx.mp4
            output_filename = f"final_{job.job_id}"
            scene_durations = getattr(job, "scene_durations", None) or []
            captioned_path = await self._with_retry(
                lambda: self.caption_burner.burn_captions(
                    input_video_path=stitched_path,
                    scenes=job.script.scenes,
                    output_filename=output_filename,
                    scene_duration=settings.scene_duration_seconds,
                    scene_durations=scene_durations if len(scene_durations) == len(job.script.scenes) else None
                ),
                "Caption burning"
            )
            duration = await self.video_stitcher.get_video_duration(captioned_path)
            await job_manager.update_job_async(
                job.job_id,
                video_url=captioned_path,
                duration_seconds=int(duration),
                progress_percent=95,
                current_step="Captions added successfully"
            )
            logger.info(f"Job {job.job_id}: Captions burned in, duration: {duration}s")
            # Remove intermediate stitched file so only one final video per job
            _remove_if_local_file(stitched_path)
        except Exception as e:
            logger.warning(f"Caption burning failed, falling back to original video: {e}")
            try:
                duration = await self.video_stitcher.get_video_duration(stitched_path) if stitched_path else 0
            except Exception:
                duration = 0
            await job_manager.update_job_async(
                job.job_id,
                video_url=stitched_path,
                duration_seconds=int(duration) if duration else 0,
                progress_percent=95,
                current_step="Captions failed; using video without captions"
            )
    
    async def _stage_upload_and_finalize(self, job: JobState) -> None:
        """Stage 5: Optional Cloudinary upload, finalize job, webhook, log success."""
        await job_manager.update_job_async(
            job.job_id,
            progress_percent=97,
            current_step="Uploading to CDN..."
        )
        job = await job_manager.get_job_async(job.job_id)
        # Skip upload if Cloudinary is not configured (avoid 401 from placeholder values)
        is_cloudinary_configured = self._cloudinary_configured()
        if not job.video_url or not job.video_url.strip():
            raise Exception("Video URL missing after caption burn-in")
        video_path = job.video_url.strip()
        if is_cloudinary_configured:
            if "://" not in video_path:
                p = Path(video_path)
                if not p.is_absolute():
                    p = Path(settings.output_dir) / p.name
                if not p.exists() or not p.is_file():
                    raise Exception(f"Video file missing before upload: {video_path}")
                video_path = str(p)
            cdn_url = await self._upload_to_cloudinary(video_path, job.job_id)
        else:
            # Local file: use basename for download URL (works on Windows too)
            raw = (job.video_url or "").strip()
            if "://" in raw:
                video_filename = raw.split("/")[-1] or "video.mp4"
            else:
                video_filename = Path(raw).name if raw else "video.mp4"
            cdn_url = f"/outputs/{video_filename}"
        await job_manager.set_complete_async(
            job.job_id,
            video_url=cdn_url,
            duration_seconds=job.duration_seconds or 30
        )
        # Virality score (0–100) and suggestions for completed video
        try:
            duration = job.duration_seconds or 30
            score, suggestions = score_from_script(
                job.script,
                video_duration_seconds=duration,
                pattern_interrupt_count=max(1, duration // 3),
                has_proof=False,
            )
            await job_manager.update_job_async(
                job.job_id,
                virality_score=score,
                virality_suggestions=suggestions,
            )
            logger.info(f"Job {job.job_id}: Virality score {score}/100")
        except Exception as e:
            logger.warning(f"Virality scoring skipped: {e}")
        logger.info(f"Job {job.job_id}: Pipeline complete! Video URL: {cdn_url}")
        log_success(job.job_id, job.prompt[:200], job.duration_seconds or 30)
        webhook_url = getattr(job, "webhook_url", None)
        if webhook_url and webhook_url.strip():
            await self._call_webhook(webhook_url, job.job_id, "complete", video_url=cdn_url, duration_seconds=job.duration_seconds)
    
    async def _upload_to_cloudinary(self, video_path: str, job_id: str) -> str:
        """Upload video to Cloudinary and return the URL. Refuses if credentials look like placeholders."""
        if _is_placeholder_key(self.cloudinary_cloud or ""):
            raise ValueError(
                "Cloudinary cloud name looks like a placeholder (e.g. your_cloud_name). "
                "Set real CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET in .env or leave unset to serve video locally."
            )
        if _is_placeholder_key(self.cloudinary_key or "") or _is_placeholder_key(self.cloudinary_secret or ""):
            raise ValueError("Cloudinary API key or secret looks like a placeholder. Set real values in .env or leave unset.")
        import base64
        import hashlib
        import time
        
        timestamp = int(time.time())
        public_id = f"videeo/{job_id}"
        
        # Generate signature
        params_to_sign = f"public_id={public_id}&timestamp={timestamp}{self.cloudinary_secret}"
        signature = hashlib.sha1(params_to_sign.encode()).hexdigest()
        
        # Upload using Cloudinary API
        async with httpx.AsyncClient(timeout=300.0) as client:
            with open(video_path, "rb") as f:
                files = {"file": f}
                data = {
                    "public_id": public_id,
                    "timestamp": timestamp,
                    "signature": signature,
                    "api_key": self.cloudinary_key
                }
                
                response = await client.post(
                    f"https://api.cloudinary.com/v1_1/{self.cloudinary_cloud}/video/upload",
                    files=files,
                    data=data
                )
                
                response.raise_for_status()
                result = response.json()
                
                return result.get("secure_url") or result.get("url")
    
    async def _with_retry(self, func, operation_name: str, max_retries: int = None):
        """Execute a function with retry logic."""
        max_retries = max_retries or settings.max_retries
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"{operation_name} failed after {max_retries + 1} attempts: {e}")
        
        raise last_error
    
    async def _stage_dry_run_complete(self, job_id: str, multi_prompt: list) -> None:
        """Mark job complete after dry run: script + Kling payload validated, no KIE call (no credits used)."""
        n = len(multi_prompt)
        preview = (multi_prompt[0].get("prompt") or "")[:60] + "..." if multi_prompt else ""
        await job_manager.update_job_async(
            job_id,
            status=JobStatus.COMPLETE,
            progress_percent=100,
            current_step=f"Dry run OK: script and {n} Kling prompt(s) validated; no video generated (no KIE credits used). First prompt: {preview}",
        )
        logger.info(f"Job {job_id}: Dry run complete (validated {n} prompts, no KIE call)")

    def _infer_error_code(self, error_message: str) -> str:
        """Map exception message to structured error code."""
        msg = (error_message or "").lower()
        if "script" in msg and ("validation" in msg or "missing" in msg or "failed" in msg):
            return "SCRIPT_FAILED"
        if "kling" in msg or "kie" in msg:
            if "timeout" in msg or "timed out" in msg:
                return "KIE_TIMEOUT"
            return "KIE_FAILED"
        if "stitch" in msg or "video save" in msg:
            return "STITCH_FAILED"
        if "caption" in msg:
            return "CAPTION_FAILED"
        if "cloudinary" in msg or "upload" in msg:
            return "UPLOAD_FAILED"
        return "PIPELINE_ERROR"

    async def _handle_error_async(self, job_id: str, error_message: str):
        """Handle pipeline error: update state, log analytics, call webhook."""
        job = await job_manager.get_job_async(job_id)
        if job:
            log_failure(job_id, (job.prompt or "")[:200], error_message=error_message, stage="pipeline")
        await job_manager.set_error_async(job_id, error_message, error_code=self._infer_error_code(error_message))
        logger.error(f"Job {job_id} failed: {error_message}")
        if job and getattr(job, "webhook_url", None) and job.webhook_url.strip():
            await self._call_webhook(job.webhook_url, job_id, "error", error_message=error_message)

    async def _call_webhook(self, url: str, job_id: str, status: str, **kwargs):
        """POST to webhook URL with job_id, status, and optional video_url, error_message, duration_seconds. Retries up to 3 times with backoff."""
        import json
        payload = {"job_id": job_id, "status": status, **kwargs}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    r = await client.post(url, json=payload)
                    r.raise_for_status()
                    return
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Webhook POST failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    logger.warning(f"Webhook POST failed after {max_retries} attempts: {e}")
