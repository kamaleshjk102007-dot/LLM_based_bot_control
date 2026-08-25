"""Google Gemini implementation of the language-understanding boundary."""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

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
missing capability. Omit optional fields the user did not specify. If a request is
ambiguous, unsafe, unsupported, or cannot be represented by the schema, do not guess.

Supported actions: MOVE, ROTATE, STOP, HOME, PICK, PLACE, GRIP, RELEASE, NAVIGATE,
GET_STATUS. Return only schema-conforming structured data.
"""

_UNSUPPORTED_PROVIDER_KEYWORDS = {
    "additionalProperties",
    "default",
    "exclusiveMinimum",
    "maximum",
    "minItems",
    "minLength",
    "pattern",
    "title",
}


def _simplify_json_schema(node: Any) -> Any:
    """Reduce Pydantic JSON Schema to Gemini's portable structured-output subset."""

    if isinstance(node, list):
        return [_simplify_json_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    any_of = node.get("anyOf")
    if isinstance(any_of, list):
        non_null = [
            option
            for option in any_of
            if not (isinstance(option, dict) and option.get("type") == "null")
        ]
        if len(non_null) == 1:
            return _simplify_json_schema(non_null[0])

    return {
        key: _simplify_json_schema(value)
        for key, value in node.items()
        if key not in _UNSUPPORTED_PROVIDER_KEYWORDS
    }


def gemini_response_json_schema() -> dict[str, Any]:
    """Derive the provider schema from the authoritative Pydantic model."""

    return _simplify_json_schema(UniversalCommand.model_json_schema())


class GeminiCommandError(RuntimeError):
    """Raised when Gemini cannot safely produce a usable command."""


def _safe_client_error(exc: errors.ClientError) -> str:
    """Classify provider errors without exposing keys or provider response bodies."""

    code = getattr(exc, "code", None)
    message = str(getattr(exc, "message", "")).lower()

    if "api key not valid" in message or "api_key_invalid" in message:
        reason = "GEMINI_API_KEY is invalid. Replace the GitHub Actions secret."
    elif "schema" in message:
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
                    response_json_schema=gemini_response_json_schema(),
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
