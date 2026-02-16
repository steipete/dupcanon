from __future__ import annotations

import pytest

from dupcanon.thinking import ThinkingLevel, normalize_thinking_level, to_openai_reasoning_effort


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("off", "off"),
        ("minimal", "minimal"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
        (" LOW ", "low"),
        ("MeDiuM", "medium"),
    ],
)
def test_normalize_thinking_level_accepts_all_supported_values(raw: str, normalized: str) -> None:
    assert normalize_thinking_level(raw) == normalized


def test_normalize_thinking_level_none_passthrough() -> None:
    assert normalize_thinking_level(None) is None


@pytest.mark.parametrize("empty_value", ["", " ", "  "])
def test_normalize_thinking_level_treats_empty_string_as_none(empty_value: str) -> None:
    assert normalize_thinking_level(empty_value) is None


def test_normalize_thinking_level_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="thinking must be one of"):
        normalize_thinking_level("turbo")


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (None, None),
        ("off", "none"),
        ("minimal", "minimal"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
    ],
)
def test_to_openai_reasoning_effort_maps_all_levels(
    level: ThinkingLevel | None,
    expected: str | None,
) -> None:
    assert to_openai_reasoning_effort(level) == expected
