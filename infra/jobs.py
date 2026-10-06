"""Durable Redis Streams-backed investigation queue and job state."""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any
try:
    from redis.exceptions import ResponseError
except ImportError:
    class ResponseError(Exception):
        pass
from infra.redis_client import get_redis
from infra.observability import INVESTIGATION_QUEUE_DEPTH

QUEUE_KEY = os.getenv("INVESTIGATION_QUEUE_KEY", "razorrisk:investigations:stream")
QUEUE_GROUP = os.getenv("INVESTIGATION_QUEUE_GROUP", "investigation-workers")
JOB_PREFIX = os.getenv("INVESTIGATION_JOB_PREFIX", "razorrisk:investigations:job:")
JOB_TTL = int(os.getenv("INVESTIGATION_JOB_TTL_SECONDS", "86400"))
SLA_SECONDS = int(os.getenv("INVESTIGATION_SLA_SECONDS", str(2 * 60 * 60)))
RECLAIM_IDLE_MS = int(os.getenv("INVESTIGATION_RECLAIM_IDLE_MS", "300000"))


def job_key(job_id: str) -> str:
    return f"{JOB_PREFIX}{job_id}"

async def ensure_consumer_group() -> None:
    r = get_redis()
    try:
        await r.xgroup_create(QUEUE_KEY, QUEUE_GROUP, id="0-0", mkstream=True)
    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise

async def enqueue_investigation(transaction_id: str, force: bool = False) -> dict[str, Any]:
    r = get_redis()
    await ensure_consumer_group()
    existing_id = await r.get(f"{JOB_PREFIX}txn:{transaction_id}")
    if existing_id:
        existing = await get_job(existing_id)
        if existing and existing.get("status") in {"queued", "running"}:
            return existing

    job_id = f"JOB_{uuid.uuid4().hex[:16]}"
    now = time.time()
    job = {
        "job_id": job_id, "transaction_id": transaction_id, "force": force,
        "status": "queued", "created_at": now, "updated_at": now,
        "sla_deadline": now + SLA_SECONDS, "attempts": 0,
    }
    pipe = r.pipeline()
    pipe.set(job_key(job_id), json.dumps(job), ex=JOB_TTL)
    pipe.set(f"{JOB_PREFIX}txn:{transaction_id}", job_id, ex=JOB_TTL)
    pipe.xadd(QUEUE_KEY, {"job_id": job_id}, maxlen=100000, approximate=True)
    await pipe.execute()
    if INVESTIGATION_QUEUE_DEPTH is not None:
        try:
            INVESTIGATION_QUEUE_DEPTH.set(await r.xlen(QUEUE_KEY))
        except Exception:
            pass
    return job

async def get_job(job_id: str) -> dict[str, Any] | None:
    raw = await get_redis().get(job_key(job_id))
    return json.loads(raw) if raw else None

async def update_job(job_id: str, **updates: Any) -> dict[str, Any] | None:
    job = await get_job(job_id)
    if not job:
        return None
    job.update(updates)
    job["updated_at"] = time.time()
    await get_redis().set(job_key(job_id), json.dumps(job), ex=JOB_TTL)
    return job

async def read_job(consumer: str, timeout_ms: int = 5000) -> tuple[str | None, str | None]:
    """Read one pending/new stream message. Returns (stream_id, job_id)."""
    r = get_redis()
    await ensure_consumer_group()
    result = await r.xreadgroup(QUEUE_GROUP, consumer, {QUEUE_KEY: ">"}, count=1, block=timeout_ms)
    if not result:
        return None, None
    _stream, entries = result[0]
    stream_id, fields = entries[0]
    if INVESTIGATION_QUEUE_DEPTH is not None:
        try:
            INVESTIGATION_QUEUE_DEPTH.set(await r.xlen(QUEUE_KEY))
        except Exception:
            pass
    return stream_id, fields.get("job_id")

async def reclaim_job(consumer: str) -> tuple[str | None, str | None]:
    """Claim a stale pending message after a crashed worker."""
    r = get_redis()
    await ensure_consumer_group()
    result = await r.xautoclaim(QUEUE_KEY, QUEUE_GROUP, consumer, RECLAIM_IDLE_MS, start_id="0-0", count=1)
    # redis-py returns (next_start_id, [(message_id, fields)], deleted_ids)
    entries = result[1] if len(result) > 1 else []
    if not entries:
        return None, None
    stream_id, fields = entries[0]
    if INVESTIGATION_QUEUE_DEPTH is not None:
        try:
            INVESTIGATION_QUEUE_DEPTH.set(await r.xlen(QUEUE_KEY))
        except Exception:
            pass
    return stream_id, fields.get("job_id")

async def ack_job(stream_id: str) -> None:
    await get_redis().xack(QUEUE_KEY, QUEUE_GROUP, stream_id)

async def requeue_job(job_id: str) -> None:
    r = get_redis()
    await r.xadd(QUEUE_KEY, {"job_id": job_id}, maxlen=100000, approximate=True)
    if INVESTIGATION_QUEUE_DEPTH is not None:
        try:
            INVESTIGATION_QUEUE_DEPTH.set(await r.xlen(QUEUE_KEY))
        except Exception:
            pass
