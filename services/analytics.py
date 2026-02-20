"""
Log pipeline failures for analytics (e.g. which prompts/scenes fail).
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


def log_failure(
    job_id: str,
    prompt_preview: str,
    scene_index: Optional[int] = None,
    error_message: str = "",
    stage: str = "",
) -> None:
    """Append a failure record to the analytics log file."""
    path = getattr(settings, "analytics_failures_path", None) or "./logs/failures.jsonl"
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "job_id": job_id,
        "prompt_preview": (prompt_preview or "")[:200],
        "scene_index": scene_index,
        "stage": stage,
        "error": (error_message or "")[:500],
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Analytics log write failed: {e}")


def log_success(job_id: str, prompt_preview: str, duration_seconds: int) -> None:
    """Optionally log successful completions (for success-rate analytics)."""
    path = getattr(settings, "analytics_failures_path", None) or "./logs/failures.jsonl"
    log_path = Path(path).parent / "completions.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "job_id": job_id,
        "prompt_preview": (prompt_preview or "")[:200],
        "duration_seconds": duration_seconds,
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Analytics completion log failed: {e}")
