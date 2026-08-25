"""Google Gemini implementation of the language-understanding boundary."""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ConfigDict

from app.commands.models import Action, UniversalCommand
from app.commands.validator import CommandValidationError, validate_command
from app.config.settings import Settings

SYSTEM_INSTRUCTION = """You are the language understanding component of a universal robot control system.

Convert human natural-language instructions only into the Universal Robot Command schema.
You do NOT control or execute a robot directly.

Never generate Python, executable code, motor commands, manufacturer-specific commands,
DOBOT commands, ROS commands, CAN frames, serial commands, vendor API calls, physical
coordinates, or invented robot capabilities.

Interpret intent conservatively. Never invent a robot ID, coordinate, object position, or
missing capability. Use null for provider-schema fields the user did not specify. If a
request is ambiguous, unsafe, unsupported, or cannot be represented by the schema, do not
guess.

Supported actions: MOVE, ROTATE, STOP, HOME, PICK, PLACE, GRIP, RELEASE, NAVIGATE,
GET_STATUS. Return only schema-conforming structured data.
"""


class GeminiEntityDraft(BaseModel):
    """Provider schema with no defaults; the Gemini API rejects schema defaults."""

    model_config = ConfigDict(extra="forbid")

    type: str | None
    id: str | None
    color: str | None
    name: str | None


class GeminiTaskDraft(BaseModel):
    """Required-but-nullable fields keep Gemini's schema provider-compatible."""

    model_config = ConfigDict(extra="forbid")

    action: Action
    object: GeminiEntityDraft | None
    target: GeminiEntityDraft | None
    position: str | None
    direction: str | None
    distance: float | None
    angle: float | None
    unit: str | None


class GeminiCommandDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    robot_id: str | None
    tasks: list[GeminiTaskDraft]


class GeminiCommandError(RuntimeError):
    """Raised when Gemini cannot safely produce a usable command."""


def _safe_client_error(exc: errors.ClientError) -> str:
    """Classify provider errors without exposing keys or provider response bodies."""

    code = getattr(exc, "code", None)
    message = str(getattr(exc, "message", "")).lower()

    if "api key not valid" in message or "api_key_invalid" in message:
        reason = "GEMINI_API_KEY is invalid. Replace the GitHub Actions secret."
    elif "schema" in message or "invalid argument" in message:
        reason = "Gemini rejected the structured-output request schema."
    else:
        reasons = {
            400: "Gemini rejected the request.",
            401: "Gemini authentication failed. Check GEMINI_API_KEY.",
            403: "Gemini denied access. Check the API key and API permissions.",
            404: "The configured Gemini model is unavailable.",
            429: "Gemini quota or rate limit was exceeded.",
        }
        reason = reasons.get(code, "Gemini returned a client error.")

    suffix = f" (HTTP {code})" if code is not None else ""
    return f"{reason}{suffix}"


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
                    response_schema=GeminiCommandDraft,
                    temperature=0,
                ),
            )
        except TimeoutError as exc:
            raise GeminiCommandError("Gemini request timed out.") from exc
        except errors.ClientError as exc:
            raise GeminiCommandError(_safe_client_error(exc)) from exc
        except Exception as exc:
            raise GeminiCommandError(f"Gemini API request failed: {type(exc).__name__}") from exc

        parsed = getattr(response, "parsed", None)
        raw = getattr(response, "text", None)
        if parsed is None and not raw:
            raise GeminiCommandError("Gemini returned an empty or malformed response.")

        if isinstance(parsed, BaseModel):
            parsed = parsed.model_dump(exclude_none=True)

        try:
            # Always perform a second application-side validation, even when the SDK parsed it.
            return validate_command(parsed if parsed is not None else raw)
        except CommandValidationError as exc:
            raise GeminiCommandError(str(exc)) from exc
