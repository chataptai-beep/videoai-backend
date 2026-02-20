"""
Audio polish pipeline: normalize loudness and optionally mix a music bed with ducking.
"""

import asyncio
import logging
import os
import subprocess
from pathlib import Path

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_EXE = "ffmpeg"

from config import settings

logger = logging.getLogger(__name__)

class AudioPolisher:
    """Normalize audio and optionally add a ducked music bed."""

    def __init__(self):
        self.output_dir = Path(settings.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_path = FFMPEG_EXE

    async def polish(
        self,
        input_video_path: str,
        output_filename: str,
    ) -> str:
        """
        Return a new video with polished audio. If polishing is disabled or fails,
        returns the original path.
        """
        if not getattr(settings, "audio_polish_enabled", True):
            return input_video_path
        if not input_video_path:
            return input_video_path
        if "://" in input_video_path:
            logger.warning("Audio polish skipped for remote video path: %s", input_video_path)
            return input_video_path
        if not os.path.exists(input_video_path):
            logger.warning("Audio polish skipped (file missing): %s", input_video_path)
            return input_video_path

        output_path = self.output_dir / f"{output_filename}_audio.mp4"
        music_bed = (getattr(settings, "music_bed_path", "") or "").strip()
        if music_bed and os.path.exists(music_bed):
            try:
                await asyncio.to_thread(self._mix_with_music, input_video_path, music_bed, str(output_path))
                return str(output_path)
            except Exception as e:
                logger.warning("Audio polish (music mix) failed, falling back to normalize only: %s", e)

        try:
            await asyncio.to_thread(self._normalize_only, input_video_path, str(output_path))
            return str(output_path)
        except Exception as e:
            logger.warning("Audio normalize failed, using original audio: %s", e)
            return input_video_path

    def _normalize_only(self, input_video_path: str, output_path: str) -> None:
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", input_video_path,
            "-filter:a", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            output_path,
        ]
        timeout = max(60, int(getattr(settings, "ffmpeg_timeout_seconds", 600) or 600))
        logger.info("Running audio normalize: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, timeout=timeout)

    def _mix_with_music(self, input_video_path: str, music_bed_path: str, output_path: str) -> None:
        duration = self._probe_duration(input_video_path)
        volume = max(0.0, float(getattr(settings, "music_bed_volume", 0.18) or 0.18))
        ducking = bool(getattr(settings, "music_bed_ducking", True))
        music_trim = f"atrim=0:{max(0.1, duration)}" if duration else "atrim=0"

        if ducking:
            filter_complex = (
                "[0:a]loudnorm=I=-16:TP=-1.5:LRA=11[a0];"
                f"[1:a]volume={volume},{music_trim},asetpts=PTS-STARTPTS[a1];"
                "[a1][a0]sidechaincompress=threshold=0.1:ratio=10:attack=5:release=200[ducked];"
                "[a0][ducked]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )
        else:
            filter_complex = (
                "[0:a]loudnorm=I=-16:TP=-1.5:LRA=11[a0];"
                f"[1:a]volume={volume},{music_trim},asetpts=PTS-STARTPTS[a1];"
                "[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", input_video_path,
            "-stream_loop", "-1",
            "-i", music_bed_path,
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ]
        timeout = max(60, int(getattr(settings, "ffmpeg_timeout_seconds", 600) or 600))
        logger.info("Running audio mix: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, timeout=timeout)

    def _probe_duration(self, path: str) -> float:
        cmd = [self.ffmpeg_path, "-i", path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        import re
        match = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2}\.\d+)", res.stderr or "")
        if not match:
            return 0.0
        hours, minutes, seconds = map(float, match.groups())
        return hours * 3600 + minutes * 60 + seconds
