"""Shared Redis connection for distributed rate limiting and job queues."""
from __future__ import annotations

import os
try:
    from redis.asyncio import Redis
except ImportError:  # Allows the core deterministic test suite to run without optional infra installed.
    Redis = None

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_REQUIRED = os.getenv("REDIS_REQUIRED", "false").lower() == "true"

_client: Redis | None = None


def get_redis() -> Redis:
    global _client
    if Redis is None:
        raise RuntimeError("redis package is not installed; install requirements.txt")
    if _client is None:
        _client = Redis.from_url(
            REDIS_URL, decode_responses=True, health_check_interval=30, socket_timeout=15
        )
    return _client


async def redis_health() -> bool:
    try:
        return bool(await get_redis().ping())
    except Exception:
        return False


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
