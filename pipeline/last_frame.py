"""
Extract last frame from a video URL and return a URL for use as reference image.
Uploads to Cloudinary if configured, else saves to outputs and returns /outputs/ path.
"""

import asyncio
import hashlib
import logging
import os
import re
import subprocess
import time
from pathlib import Path

import httpx

from config import settings, _is_placeholder_key

logger = logging.getLogger(__name__)

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_EXE = "ffmpeg"


async def extract_last_frame_as_url(
    video_url: str,
    job_id: str,
    scene_index: int,
    cloudinary_cloud: str = "",
    cloudinary_key: str = "",
    cloudinary_secret: str = "",
) -> str:
    """
    Download video from URL, extract last frame as image, return URL.
    Uses Cloudinary image upload if configured, else saves to outputs/ and returns /outputs/frame_xxx.jpg.
    """
    temp_dir = Path(settings.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download video to temp
    local_video = temp_dir / f"frame_src_{job_id}_{scene_index}.mp4"
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(video_url)
        r.raise_for_status()
        local_video.write_bytes(r.content)

    # Get duration
    cmd_dur = [FFMPEG_EXE, "-i", str(local_video)]
    res = await asyncio.to_thread(
        lambda: subprocess.run(cmd_dur, capture_output=True, text=True)
    )
    match = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2}\.\d+)", res.stderr or "")
    if not match:
        # Fallback: use -1 frame (last)
        seek = ""
    else:
        h, m, s = map(float, match.groups())
        total = h * 3600 + m * 60 + s
        seek = str(max(0, total - 0.1))  # 0.1s before end

    frame_path = output_dir / f"frame_{job_id}_scene{scene_index}.jpg"
    if seek:
        cmd = [
            FFMPEG_EXE, "-y",
            "-ss", seek,
            "-i", str(local_video),
            "-vframes", "1",
            "-q:v", "2",
            str(frame_path),
        ]
    else:
        cmd = [
            FFMPEG_EXE, "-y",
            "-sseof", "-1",
            "-i", str(local_video),
            "-vframes", "1",
            "-q:v", "2",
            str(frame_path),
        ]
    await asyncio.to_thread(lambda: subprocess.run(cmd, check=True, capture_output=True))

    try:
        os.remove(local_video)
    except Exception:
        pass

    # Upload to Cloudinary if configured (required for kie.ai to fetch the reference)
    if (
        cloudinary_cloud
        and not _is_placeholder_key(cloudinary_cloud or "")
        and cloudinary_key
        and not _is_placeholder_key(cloudinary_key or "")
        and cloudinary_secret
        and not _is_placeholder_key(cloudinary_secret or "")
    ):
        try:
            url = await _upload_image_to_cloudinary(
                str(frame_path),
                f"videeo/frames/{job_id}_{scene_index}",
                cloudinary_cloud,
                cloudinary_key,
                cloudinary_secret,
            )
            try:
                os.remove(frame_path)
            except Exception:
                pass
            return url
        except Exception as e:
            logger.warning(f"Cloudinary image upload failed, serving local: {e}")

    base = getattr(settings, "base_url", "") or "http://localhost:8000"
    return f"{base.rstrip('/')}/outputs/{frame_path.name}"


async def _upload_image_to_cloudinary(
    image_path: str,
    public_id: str,
    cloud_name: str,
    api_key: str,
    api_secret: str,
) -> str:
    """Upload image to Cloudinary and return secure_url."""
    timestamp = int(time.time())
    params_to_sign = f"public_id={public_id}&timestamp={timestamp}{api_secret}"
    signature = hashlib.sha1(params_to_sign.encode()).hexdigest()
    async with httpx.AsyncClient(timeout=30.0) as client:
        with open(image_path, "rb") as f:
            files = {"file": f}
            data = {
                "public_id": public_id,
                "timestamp": timestamp,
                "signature": signature,
                "api_key": api_key,
            }
            r = await client.post(
                f"https://api.cloudinary.com/v1_1/{cloud_name}/image/upload",
                files=files,
                data=data,
            )
            r.raise_for_status()
            result = r.json()
            return result.get("secure_url") or result.get("url", "")
