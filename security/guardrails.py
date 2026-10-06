
"""Security boundary for investigation/evidence tools.

The agent never gets arbitrary URL access or arbitrary SQL/API access.
Only allowlisted evidence tools can be invoked through this layer.
"""
from __future__ import annotations

import ipaddress
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable

from utils.logger import get_logger

logger = get_logger("security_guardrails")

_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")


class GuardrailViolation(ValueError):
    """Raised when an agent tool request violates the security policy."""


@dataclass(frozen=True)
class ToolPolicy:
    name: str
    max_calls_per_minute: int = 10
    requires_transaction_scope: bool = True


class SecurityGuard:
    """Allowlist, validate, rate-limit, and audit evidence-tool calls."""

    POLICIES = {
        "device_intelligence": ToolPolicy("device_intelligence", 10),
        "merchant_intelligence": ToolPolicy("merchant_intelligence", 20),
        "network_intelligence": ToolPolicy("network_intelligence", 20),
    }

    def __init__(self):
        self._calls: dict[str, deque[float]] = defaultdict(deque)

    @staticmethod
    def _validate_id(value: str, field: str) -> str:
        if not isinstance(value, str) or not _ID_RE.fullmatch(value):
            raise GuardrailViolation(f"Invalid {field} format")
        return value

    @staticmethod
    def _validate_ip(value: str) -> str:
        try:
            ipaddress.ip_address(value)
        except ValueError as exc:
            raise GuardrailViolation("Invalid IP address") from exc
        return value

    def authorize(
        self,
        tool_name: str,
        *,
        transaction_id: str | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        merchant_id: str | None = None,
        ip_address: str | None = None,
    ) -> dict[str, Any]:
        policy = self.POLICIES.get(tool_name)
        if policy is None:
            raise GuardrailViolation("Tool is not allowlisted")

        if policy.requires_transaction_scope:
            self._validate_id(transaction_id or "", "transaction_id")

        if user_id is not None:
            self._validate_id(user_id, "user_id")
        if device_id is not None:
            self._validate_id(device_id, "device_id")
        if merchant_id is not None:
            self._validate_id(merchant_id, "merchant_id")
        if ip_address is not None:
            self._validate_ip(ip_address)

        now = time.monotonic()
        calls = self._calls[tool_name]
        while calls and now - calls[0] >= 60:
            calls.popleft()
        if len(calls) >= policy.max_calls_per_minute:
            raise GuardrailViolation("Tool rate limit exceeded")
        calls.append(now)

        request = {
            "tool": tool_name,
            "transaction_id": transaction_id,
            "user_id": user_id,
            "device_id": device_id,
            "merchant_id": merchant_id,
            "ip_address": ip_address,
        }
        logger.info("[GUARDRAIL] Authorized evidence tool: %s", request)
        return request

    def audit_denial(self, tool_name: str, reason: str) -> None:
        logger.warning("[GUARDRAIL] Denied tool=%s reason=%s", tool_name, reason)


security_guard = SecurityGuard()
