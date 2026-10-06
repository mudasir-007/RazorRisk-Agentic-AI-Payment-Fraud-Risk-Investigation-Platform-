"""Atomic, Redis-backed sliding-window rate limiter.

The Lua script makes check+increment one operation, so multiple API replicas
cannot race each other and exceed the configured limit.
"""
from __future__ import annotations

import time
from fastapi import HTTPException, Request
try:
    from redis.exceptions import RedisError
except ImportError:
    RedisError = Exception
from infra.redis_client import get_redis, REDIS_REQUIRED
from infra.observability import RATE_LIMIT_HITS, DEPENDENCY_FAILURES

_LUA = r"""
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count >= limit then
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  local retry = window
  if oldest[2] then retry = math.max(1, math.ceil(window - (now - tonumber(oldest[2])))) end
  return {0, count, retry}
end
redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, math.ceil(window + 5))
return {1, count + 1, 0}
"""

async def enforce_rate_limit(request: Request, scope: str, limit: int, window_seconds: int = 60) -> None:
    client_id = request.headers.get("X-Client-ID") or (request.client.host if request.client else "unknown")
    key = f"razorrisk:rl:{scope}:{client_id}"
    now = time.time()
    member = f"{now:.6f}:{id(request)}"
    try:
        allowed, _count, retry = await get_redis().eval(_LUA, 1, key, now, window_seconds, limit, member)
    except Exception:
        if DEPENDENCY_FAILURES is not None:
            DEPENDENCY_FAILURES.labels("redis_rate_limiter").inc()
        if REDIS_REQUIRED:
            raise HTTPException(status_code=503, detail="Rate limiting service unavailable")
        return
    if int(allowed) != 1:
        if RATE_LIMIT_HITS is not None:
            RATE_LIMIT_HITS.labels(scope).inc()
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded for {scope}", headers={"Retry-After": str(int(retry))})
