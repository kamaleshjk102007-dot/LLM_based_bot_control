"""Google Gemini implementation of the language-understanding boundary."""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types

from app.commands.models import UniversalCommand
from app.commands.validator import CommandValidationError, validate_command
from app.config.settings import Settings

SYSTEM_INSTRUCTION = """You are the language understanding component of a universal robot control system.

Convert human natural-language instructions only into the Universal Robot Command schema.
You do NOT control or execute a robot directly.

Never generate Python, executable code, motor commands, manufacturer-specific commands,
DOBOT commands, ROS commands, CAN frames, serial commands, vendor API calls, physical
coordinates, or invented robot capabilities.

Interpret intent conservatively. Never invent a robot ID, coordinate, object position, or
missing capability. Omit optional information that the user did not provide. If a request
is ambiguous, unsafe, unsupported, or cannot be represented by the schema, do not guess.

Supported actions: MOVE, ROTATE, STOP, HOME, PICK, PLACE, GRIP, RELEASE, NAVIGATE,
GET_STATUS. Return only schema-conforming structured data.
"""


class GeminiCommandError(RuntimeError):
    """Raised when Gemini cannot safely produce a usable command."""


class GeminiCommandClient:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client or genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=settings.gemini_timeout_ms),
        )

    def generate_command(self, instruction: str) -> UniversalCommand:
        instruction = instruction.strip()
        if not instruction:
            raise GeminiCommandError("Instruction cannot be empty.")

        try:
            response = self._client.models.generate_content(
                model=self._settings.gemini_model,
                contents=instruction,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=UniversalCommand,
                    temperature=0,
                ),
            )
        except TimeoutError as exc:
            raise GeminiCommandError("Gemini request timed out.") from exc
        except Exception as exc:
            raise GeminiCommandError(f"Gemini API request failed: {type(exc).__name__}") from exc

        parsed = getattr(response, "parsed", None)
        raw = getattr(response, "text", None)
        if parsed is None and not raw:
            raise GeminiCommandError("Gemini returned an empty or malformed response.")

        try:
            # Always perform a second application-side validation, even when the SDK parsed it.
            return validate_command(parsed if parsed is not None else raw)
        except CommandValidationError as exc:
            raise GeminiCommandError(str(exc)) from exc
