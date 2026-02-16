from __future__ import annotations

from typing import Literal

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]

_ALLOWED_THINKING_LEVELS: set[str] = {"off", "minimal", "low", "medium", "high", "xhigh"}


def normalize_thinking_level(value: str | None) -> ThinkingLevel | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized not in _ALLOWED_THINKING_LEVELS:
        msg = "thinking must be one of: off, minimal, low, medium, high, xhigh"
        raise ValueError(msg)

    return normalized  # type: ignore[return-value]


def to_openai_reasoning_effort(level: ThinkingLevel | None) -> str | None:
    if level is None:
        return None
    if level == "off":
        return "none"
    return level
