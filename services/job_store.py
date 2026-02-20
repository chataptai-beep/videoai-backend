"""
Persistent job store: Redis when REDIS_URL is set, else in-memory.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional

from pydantic import ValidationError

from config import settings
from models.schemas import JobState, JobStatus, AspectRatio

logger = logging.getLogger(__name__)

_redis = None


def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    if not getattr(settings, "redis_url", None) or not settings.redis_url.strip():
        return None
    try:
        import redis.asyncio as redis
        _redis = redis.from_url(settings.redis_url, decode_responses=True)
        return _redis
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        return None


def _job_to_dict(job: JobState) -> dict:
    d = job.model_dump(mode="json")
    d["aspect_ratio"] = job.aspect_ratio.value
    d["status"] = job.status.value
    return d


def _dict_to_job(d: dict) -> JobState:
    d = d.copy()
    try:
        d["aspect_ratio"] = AspectRatio(d.get("aspect_ratio", "9:16"))
    except (ValueError, TypeError):
        d["aspect_ratio"] = AspectRatio.PORTRAIT
    try:
        d["status"] = JobStatus(d.get("status", "pending"))
    except (ValueError, TypeError):
        d["status"] = JobStatus.PENDING
    for k in ("created_at", "updated_at"):
        if isinstance(d.get(k), str):
            try:
                d[k] = datetime.fromisoformat(d[k].replace("Z", "+00:00"))
            except Exception:
                pass
    return JobState.model_validate(d)


async def store_get(job_id: str) -> Optional[JobState]:
    r = _get_redis()
    if r is None:
        return None
    try:
        data = await r.get(f"job:{job_id}")
        if not data:
            return None
        parsed = json.loads(data)
        return _dict_to_job(parsed)
    except ValidationError as e:
        logger.warning(f"Redis job data invalid for {job_id}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Redis get failed: {e}")
        return None


async def store_set(job_id: str, job: JobState) -> None:
    r = _get_redis()
    if r is None:
        return
    try:
        await r.set(
            f"job:{job_id}",
            json.dumps(_job_to_dict(job), default=str),
            ex=86400 * 7,  # 7 days TTL
        )
    except Exception as e:
        logger.warning(f"Redis set failed: {e}")


async def store_delete(job_id: str) -> bool:
    r = _get_redis()
    if r is None:
        return False
    try:
        await r.delete(f"job:{job_id}")
        return True
    except Exception as e:
        logger.warning(f"Redis delete failed: {e}")
        return False


async def store_keys() -> list:
    r = _get_redis()
    if r is None:
        return []
    try:
        keys = await r.keys("job:*")
        return [k.replace("job:", "") for k in keys]
    except Exception as e:
        logger.warning(f"Redis keys failed: {e}")
        return []


async def store_ping() -> bool:
    """Ping Redis; return True if reachable. For readiness probes."""
    r = _get_redis()
    if r is None:
        return False
    try:
        await r.ping()
        return True
    except Exception:
        return False
