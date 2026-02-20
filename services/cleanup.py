"""
Cleanup old temp and output files to avoid disk full.
Called on startup when config cleanup_*_max_days > 0.
"""

import logging
import time
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


def cleanup_old_files() -> None:
    """Delete files in temp_dir and output_dir older than configured days."""
    now = time.time()
    temp_days = max(0, getattr(settings, "cleanup_temp_max_days", 0))
    out_days = max(0, getattr(settings, "cleanup_outputs_max_days", 0))
    if temp_days <= 0 and out_days <= 0:
        return
    temp_dir = Path(settings.temp_dir)
    output_dir = Path(settings.output_dir)
    deleted = 0
    if temp_days > 0 and temp_dir.exists():
        cutoff = now - temp_days * 86400
        for f in temp_dir.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                try:
                    f.unlink()
                    deleted += 1
                except Exception as e:
                    logger.warning(f"Cleanup failed for {f}: {e}")
    if out_days > 0 and output_dir.exists():
        cutoff = now - out_days * 86400
        for f in output_dir.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                try:
                    f.unlink()
                    deleted += 1
                except Exception as e:
                    logger.warning(f"Cleanup failed for {f}: {e}")
    if deleted:
        logger.info(f"Cleanup: removed {deleted} old file(s) from temp/outputs")
