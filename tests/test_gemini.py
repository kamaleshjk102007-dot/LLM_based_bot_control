"""Gemini boundary tests. Unit tests are mocked; live access is opt-in."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.config.settings import Settings
from app.llm.gemini_client import (
    GeminiCommandClient,
    GeminiCommandDraft,
    GeminiCommandError,
    GeminiEntityDraft,
    GeminiTaskDraft,
)


def settings() -> Settings:
    return Settings(gemini_api_key="test-key", gemini_model="test-model")


def test_provider_schema_has_no_default_values():
    # google-genai's Gemini response_schema path rejects Pydantic defaults.
    for model in (GeminiEntityDraft, GeminiTaskDraft, GeminiCommandDraft):
        assert all(field.is_required() for field in model.model_fields.values())


def test_mocked_structured_response_is_validated():
    sdk_client = Mock()
    sdk_client.models.generate_content.return_value = SimpleNamespace(
        parsed={
            "version": "1.0",
            "robot_id": None,
            "tasks": [{"action": "PICK", "object": {"type": "cube", "color": "red"}}],
        },
        text=None,
    )
    command = GeminiCommandClient(settings(), client=sdk_client).generate_command(
        "Pick the red cube"
    )
    assert command.tasks[0].action.value == "PICK"
    sdk_client.models.generate_content.assert_called_once()


def test_provider_model_is_normalized_before_strict_validation():
    sdk_client = Mock()
    sdk_client.models.generate_content.return_value = SimpleNamespace(
        parsed=GeminiCommandDraft(
            version="1.0",
            robot_id=None,
            tasks=[
                GeminiTaskDraft(
                    action="MOVE",
                    object=None,
                    target=None,
                    position=None,
                    direction="forward",
                    distance=20,
                    angle=None,
                    unit="cm",
                )
            ],
        ),
        text=None,
    )
    command = GeminiCommandClient(settings(), client=sdk_client).generate_command(
        "Move forward 20 cm"
    )
    assert command.tasks[0].distance == 20


def test_invalid_mocked_response_fails_safely():
    sdk_client = Mock()
    sdk_client.models.generate_content.return_value = SimpleNamespace(
        parsed={"version": "1.0", "tasks": [{"action": "PICK"}]},
        text=None,
    )
    with pytest.raises(GeminiCommandError):
        GeminiCommandClient(settings(), client=sdk_client).generate_command("Pick it")


def test_api_failure_is_wrapped_without_secrets():
    sdk_client = Mock()
    sdk_client.models.generate_content.side_effect = RuntimeError("provider unavailable")
    with pytest.raises(GeminiCommandError, match="Gemini API request failed"):
        GeminiCommandClient(settings(), client=sdk_client).generate_command("Stop")


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_GEMINI_TESTS") != "1",
    reason="set RUN_LIVE_GEMINI_TESTS=1 to enable the live Gemini test",
)
def test_live_gemini_move_command():
    command = GeminiCommandClient(Settings.from_env()).generate_command(
        "Move forward 20 centimeters."
    )
    assert command.tasks[0].action.value == "MOVE"
    assert command.tasks[0].distance == 20
