"""
Video Stitcher using FFmpeg.
Combines multiple scene videos into a single output with transitions.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import httpx

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_EXE = "ffmpeg"  # Fallback to system PATH

from config import settings

logger = logging.getLogger(__name__)


class VideoStitcher:
    """
    Stitches multiple video clips into a single video using FFmpeg.
    Supports crossfade transitions and trim operations.
    """
    
    def __init__(self):
        self.temp_dir = Path(settings.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(settings.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_path = FFMPEG_EXE
        logger.info(f"Using FFmpeg at: {self.ffmpeg_path}")
    
    async def stitch_videos(
        self,
        video_urls: List[str],
        output_filename: str,
        crossfade_duration: float = 0.5,
        trim_start_scenes_2_plus: float = 0.5
    ) -> str:
        """
        Stitch multiple video URLs into a single video.
        
        Args:
            video_urls: List of video URLs to stitch
            output_filename: Name for the output file (without extension)
            crossfade_duration: Duration of crossfade transitions in seconds
            trim_start_scenes_2_plus: Seconds to trim from start of scenes 2+ (for continuity)
        
        Returns:
            Path to the stitched video file
        
        Raises:
            Exception: If stitching fails
        """
        if not video_urls:
            raise Exception("No videos to stitch")
        
        if len(video_urls) == 1:
            # Single video, just download and return
            local_path = await self._download_video(video_urls[0], "scene_1.mp4")
            output_path = self.output_dir / f"{output_filename}.mp4"
            os.rename(local_path, output_path)
            return str(output_path)
        
        logger.info(f"Stitching {len(video_urls)} videos...")
        
        # Step 1: Download all videos
        local_videos = []
        for i, url in enumerate(video_urls):
            local_path = await self._download_video(url, f"scene_{i + 1}.mp4")
            local_videos.append(local_path)
        
        # Step 2: Create FFmpeg concat file
        concat_file = self._create_concat_file(local_videos, trim_start_scenes_2_plus)
        
        # Step 3: Run FFmpeg to stitch
        output_path = self.output_dir / f"{output_filename}.mp4"
        await self._run_ffmpeg_concat(concat_file, str(output_path), crossfade_duration)
        
        # Step 4: Cleanup temp files
        self._cleanup_temp_files(local_videos + [concat_file])
        
        logger.info(f"Stitched video saved to: {output_path}")
        return str(output_path)
    
    async def _download_video(self, url: str, filename: str) -> str:
        """Download a video from URL to temp directory."""
        local_path = self.temp_dir / filename
        
        logger.debug(f"Downloading video: {url}")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            with open(local_path, "wb") as f:
                f.write(response.content)
        
        return str(local_path)
    
    def _create_concat_file(
        self,
        video_paths: List[str],
        trim_start_seconds: float = 0.0
    ) -> str:
        """Create a concat demuxer file for FFmpeg."""
        concat_path = self.temp_dir / "concat.txt"
        
        with open(concat_path, "w") as f:
            for i, video_path in enumerate(video_paths):
                # Use absolute paths to avoid FFmpeg path resolution issues
                abs_path = os.path.abspath(video_path)
                # Escape single quotes in path for FFmpeg and use forward slashes
                escaped_path = abs_path.replace("\\", "/").replace("'", "'\\''")
                
                # For scenes 2+, we might want to trim the start
                # to avoid duplicate frames from continuity
                if i > 0 and trim_start_seconds > 0:
                    # We'll handle trimming in the filter complex instead
                    pass
                
                f.write(f"file '{escaped_path}'\n")
        
        return str(concat_path)
    
    async def _run_ffmpeg_concat(
        self,
        concat_file: str,
        output_path: str,
        crossfade_duration: float = 0.5
    ):
        """Run FFmpeg to concatenate videos."""
        
        # Simple concat without complex transitions
        # For MVP, use basic concatenation
        cmd = [
            self.ffmpeg_path,
            "-y",  # Overwrite output
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path
        ]
        
        logger.info(f"Running FFmpeg: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            logger.error(f"FFmpeg error: {error_msg}")
            raise Exception(f"FFmpeg concat failed: {error_msg[:200]}")
        
        logger.info("FFmpeg concat completed successfully")
    
    async def stitch_with_crossfade(
        self,
        video_urls: List[str],
        output_filename: str,
        crossfade_duration: float = 1.0
    ) -> tuple:
        """Returns (output_path_str, durations_list)."""
        if not video_urls:
            raise Exception("No videos to stitch")
        # 1. Download Async
        local_videos = []
        for i, url in enumerate(video_urls):
             # Ensure we get the duration
            path = await self._download_video(url, f"scene_{i+1}.mp4")
            local_videos.append(path)

        # 2. Run FFmpeg Heavy Lifting in Thread
        return await asyncio.to_thread(
            self._process_ffmpeg_sync,
            local_videos,
            output_filename,
            crossfade_duration
        )

    async def stitch_with_trims(
        self,
        video_urls: List[str],
        output_filename: str,
        trim_instructions: Optional[List[object]] = None,
        default_trim_start: float = 1.0,
    ) -> tuple:
        """Stitch videos with optional per-scene trims. Returns (output_path, durations)."""
        if not video_urls:
            raise Exception("No videos to stitch")
        local_videos = []
        for i, url in enumerate(video_urls):
            path = await self._download_video(url, f"scene_{i+1}.mp4")
            local_videos.append(path)
        return await asyncio.to_thread(
            self._process_ffmpeg_sync_with_trims,
            local_videos,
            output_filename,
            trim_instructions,
            default_trim_start,
        )

    def _process_ffmpeg_sync(self, local_videos, output_filename, crossfade_duration):
        import subprocess
        from config import settings
        if not local_videos:
            raise Exception("No local videos to process")
        # Get Durations & Speed Up Factor
        # User Request: Don't apply 2x speed. Use native speed.
        speed_factor = 1.0
        
        # Standardize (Vertical Crop + Speed Up)
        std_videos = []
        durations = [] # We will calculate durations of the processed videos
        target_w = settings.video_width
        target_h = settings.video_height
        
        for i, v in enumerate(local_videos):
            std_path = str(self.output_dir / f"std_scene_{i}.mp4")
            
            # VF chain:
            # 1. Scale to fill target (1080x1920) while preserving aspect ratio
            # 2. Crop to exactly 1080x1920 (center)
            # 3. Force SAR 1:1 to avoid aspect ratio weirdness in players
            
            vf = (f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                  f"crop={target_w}:{target_h}:(in_w-{target_w})/2:(in_h-{target_h})/2,"
                  f"setsar=1")

            input_args = ["-y"]
            
            # TRIM Logic: Cut first 1s for scenes > 0 (to remove static reference frame)
            if i > 0:
                input_args.extend(["-ss", "1.0"])
            
            input_args.extend(["-i", v])

            cmd = [
                self.ffmpeg_path, *input_args,
                "-threads", "1", # CRITICAL: Limit memory usage on Railway
                "-vf", vf,
                "-r", "30",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                std_path
            ]
            _t = max(60, int(getattr(settings, "ffmpeg_timeout_seconds", 600) or 600))
            subprocess.run(cmd, check=True, timeout=_t)
            std_videos.append(std_path)
            
            # Now get the duration of the standardized video
            cmd_dur = [self.ffmpeg_path, "-i", std_path]
            res_dur = subprocess.run(cmd_dur, capture_output=True, text=True)
            import re
            match = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2}\.\d+)", res_dur.stderr)
            if match:
                 h, m, s = map(float, match.groups())
                 durations.append(h*3600 + m*60 + s)
            else:
                 durations.append(4.0) # Fallback
            
        # Stitch using CONCAT DEMUXER (low-memory, no crossfade)
        # The xfade filter requires holding multiple frames in RAM, which OOMs on Railway.
        # Concat demuxer processes sequentially with minimal memory.
        output_path = self.output_dir / f"{output_filename}.mp4"
        n = len(std_videos)
        
        if n == 1:
            if os.path.exists(output_path): os.remove(output_path)
            os.rename(std_videos[0], output_path)
            return str(output_path), durations

        # Create concat list file
        concat_file = self.temp_dir / "concat_list.txt"
        with open(concat_file, "w") as f:
            for v in std_videos:
                # CRITICAL: Use absolute paths! FFmpeg resolves relative paths
                # from the concat file location, not the working directory.
                abs_path = os.path.abspath(v).replace("\\", "/")
                f.write(f"file '{abs_path}'\n")

        cmd = [
            self.ffmpeg_path, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-threads", "1",  # CRITICAL for Railway RAM
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_path)
        ]
        
        _timeout = max(60, int(getattr(settings, "ffmpeg_timeout_seconds", 600) or 600))
        logger.info(f"Running low-memory concat stitch for {n} clips...")
        subprocess.run(cmd, check=True, timeout=_timeout)
        
        # Cleanup concat file
        if os.path.exists(concat_file):
            os.remove(concat_file)

        return str(output_path), durations

    def _process_ffmpeg_sync_with_trims(
        self,
        local_videos,
        output_filename,
        trim_instructions,
        default_trim_start,
    ):
        import subprocess
        from config import settings
        if not local_videos:
            raise Exception("No local videos to process")

        trim_map = {}
        if trim_instructions:
            for item in trim_instructions:
                try:
                    scene_number = int(getattr(item, "scene_number", None) or item.get("scene_number"))
                except (TypeError, ValueError, AttributeError):
                    continue
                trim_start = float(getattr(item, "trim_start", None) or item.get("trim_start", 0))
                trim_end = float(getattr(item, "trim_end", None) or item.get("trim_end", 0))
                trim_map[scene_number] = (max(0.0, trim_start), max(0.0, trim_end))

        std_videos = []
        durations = []
        target_w = settings.video_width
        target_h = settings.video_height

        for i, v in enumerate(local_videos):
            scene_number = i + 1
            std_path = str(self.output_dir / f"std_scene_trim_{i}.mp4")
            if os.path.exists(std_path):
                os.remove(std_path)

            vf = (
                f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{target_h}:(in_w-{target_w})/2:(in_h-{target_h})/2,"
                "setsar=1"
            )

            duration = self._probe_duration_sync(v) or 0.0
            trim_start = 0.0 if scene_number == 1 else max(0.0, float(default_trim_start))
            trim_end = 0.0
            if scene_number in trim_map:
                trim_start = max(trim_start, trim_map[scene_number][0])
                trim_end = max(trim_end, trim_map[scene_number][1])

            target_duration = max(0.1, duration - trim_start - trim_end) if duration else 0.0

            cmd = [
                self.ffmpeg_path,
                "-y",
                "-i", v,
            ]
            if trim_start > 0:
                cmd.extend(["-ss", f"{trim_start:.3f}"])
            if target_duration > 0:
                cmd.extend(["-t", f"{target_duration:.3f}"])
            cmd.extend([
                "-threads", "1",
                "-vf", vf,
                "-r", "30",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                std_path,
            ])
            _t = max(60, int(getattr(settings, "ffmpeg_timeout_seconds", 600) or 600))
            subprocess.run(cmd, check=True, timeout=_t)
            std_videos.append(std_path)
            durations.append(target_duration if target_duration else duration)

        output_path = self.output_dir / f"{output_filename}.mp4"
        n = len(std_videos)
        if n == 1:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(std_videos[0], output_path)
            return str(output_path), durations

        concat_file = self.temp_dir / "concat_list_trim.txt"
        with open(concat_file, "w") as f:
            for v in std_videos:
                abs_path = os.path.abspath(v).replace("\\", "/")
                f.write(f"file '{abs_path}'\n")

        cmd = [
            self.ffmpeg_path, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-threads", "1",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_path)
        ]

        _t = max(60, int(getattr(settings, "ffmpeg_timeout_seconds", 600) or 600))
        logger.info(f"Running trimmed concat stitch for {n} clips...")
        subprocess.run(cmd, check=True, timeout=_t)

        if os.path.exists(concat_file):
            os.remove(concat_file)

        return str(output_path), durations

    def _probe_duration_sync(self, path: str) -> float:
        cmd = [self.ffmpeg_path, "-i", path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        import re
        match = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2}\.\d+)", res.stderr or "")
        if match:
            h, m, s = map(float, match.groups())
            return h * 3600 + m * 60 + s
        return 0.0
    
    def _cleanup_temp_files(self, files: List[str]):
        """Remove temporary files."""
        for file_path in files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
    
    async def get_video_duration(self, video_path: str) -> float:
        """Get the duration of a video in seconds."""
        # Try ffprobe first (if it exists)
        ffprobe_path = self.ffmpeg_path.replace("ffmpeg", "ffprobe")
        if os.path.exists(ffprobe_path) or ffprobe_path == "ffprobe":
            cmd = [
                ffprobe_path,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await process.communicate()
                if process.returncode == 0 and stdout:
                    return float(stdout.decode().strip())
            except Exception as e:
                logger.debug(f"ffprobe failed: {e}")

        # Fallback: Use ffmpeg -i
        cmd = [self.ffmpeg_path, "-i", video_path]
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()
            output = stderr.decode()
            # Look for "Duration: 00:00:05.50"
            import re
            match = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", output)
            if match:
                h, m, s = match.groups()
                return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")
        
        return 0.0
