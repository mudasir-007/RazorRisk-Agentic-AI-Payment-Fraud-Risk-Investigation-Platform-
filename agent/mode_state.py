"""
RazorRisk — runtime agent-mode override.

Holds a single in-memory value that lets the dashboard force which
investigation path runs next, without restarting the process or editing
.env: "auto" (default — provider priority order, same as before this
existed), a specific provider name to force that provider's LLM path,
or "deterministic" to force the rule-based fallback even if a key is
configured.

Deliberately process-local, in-memory, not persisted — this is an
operator toggle for demoing/debugging which path runs, not application
config. Restarting the server resets it to "auto".
"""

VALID_MODES = ("auto", "anthropic", "groq", "openai", "deterministic")

_current_mode = "auto"


def get_mode() -> str:
    return _current_mode


def set_mode(mode: str) -> str:
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid agent mode '{mode}'. Must be one of {VALID_MODES}.")
    global _current_mode
    _current_mode = mode
    return _current_mode


# Jev (TypeSafe) verification toggle. Off by default: it's a real API call
# with its own key requirement (config.TYPESAFE_API_KEY), not something
# that should silently start firing just because a key happens to be set.
# Same process-local, in-memory, not-persisted contract as _current_mode
# above — restarting the server resets it to off.
_jev_verification_enabled = False


def get_jev_verification_enabled() -> bool:
    return _jev_verification_enabled


def set_jev_verification_enabled(enabled: bool) -> bool:
    global _jev_verification_enabled
    _jev_verification_enabled = bool(enabled)
    return _jev_verification_enabled
