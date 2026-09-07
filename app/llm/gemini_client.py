"""Google Gemini implementation of the language-understanding boundary."""

from __future__ import annotations

import json
import re
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
GET_STATUS. Linear units such as millimeters, centimeters, meters, and inches always
describe MOVE distance, even when the user says "turn". For signed Cartesian movement,
put the sign on direction (for example, direction "-X" and distance 5); distance must
always be positive. ROTATE requires an angular value in degrees or radians. Return only
schema-conforming structured data.
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


_LINEAR_UNITS = {
    "mm", "millimeter", "millimeters", "millimetre", "millimetres",
    "cm", "centimeter", "centimeters", "centimetre", "centimetres",
    "m", "meter", "meters", "metre", "metres",
    "in", "inch", "inches",
}


_EXPLICIT_AXIS = re.compile(
    r"\b(?:on|along)\s*([+-]?\s*[xyz])\b",
    re.IGNORECASE,
)

_EXPLICIT_SIGNED_AXIS_DISTANCE = re.compile(
    r"([+-])\s*(\d+(?:\.\d+)?)\s*"
    r"(?:mm|millimeters?|millimetres?|cm|centimeters?|centimetres?|"
    r"m|meters?|metres?|in|inches?)\s*"
    r"(?:on|along)\s*[+-]?\s*([xyz])\b",
    re.IGNORECASE,
)


def _repair_explicit_axis(payload: Any, instruction: str) -> Any:
    """Map explicitly written Cartesian clauses to MOVE tasks in order."""
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        return payload

    directives = []
    for match in _EXPLICIT_SIGNED_AXIS_DISTANCE.finditer(instruction):
        sign, magnitude, axis = match.groups()
        normalized_axis = axis.upper()
        if sign == "-":
            normalized_axis = "-" + normalized_axis
        directives.append((normalized_axis, float(magnitude)))

    move_indices = [
        index
        for index, item in enumerate(payload["tasks"])
        if isinstance(item, dict) and item.get("action") == "MOVE"
    ]

    # Sequential repair is safe only when each MOVE has one explicit Cartesian
    # clause. This prevents the first axis from leaking into later tasks.
    if directives and len(directives) == len(move_indices):
        repaired = {**payload, "tasks": [
            dict(item) if isinstance(item, dict) else item
            for item in payload["tasks"]
        ]}
        for task_index, (axis, magnitude) in zip(move_indices, directives):
            task = repaired["tasks"][task_index]
            task["direction"] = axis
            try:
                provider_magnitude = abs(float(task.get("distance")))
            except (TypeError, ValueError):
                provider_magnitude = None
            if (
                provider_magnitude is not None
                and abs(provider_magnitude - magnitude) <= 1e-9
            ):
                task["distance"] = magnitude
        return repaired

    # Preserve the conservative single-task fallback when Gemini omitted only
    # the axis and the instruction contains one unambiguous "on/along X/Y/Z".
    if len(move_indices) != 1:
        return payload
    match = _EXPLICIT_AXIS.search(instruction)
    if match is None:
        return payload
    axis = match.group(1).replace(" ", "").upper()
    if axis.startswith("+"):
        axis = axis[1:]
    repaired = {**payload, "tasks": [
        dict(item) if isinstance(item, dict) else item
        for item in payload["tasks"]
    ]}
    task = repaired["tasks"][move_indices[0]]
    if (
        not task.get("direction")
        and not task.get("position")
        and not task.get("target")
    ):
        task["direction"] = axis
    return repaired


def _repair_linear_motion(payload: Any) -> Any:
    """Correct a provider ROTATE classification when the fields are linear."""
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        return payload
    repaired = {**payload, "tasks": []}
    for item in payload["tasks"]:
        task = dict(item) if isinstance(item, dict) else item
        if (
            isinstance(task, dict)
            and task.get("action") == "ROTATE"
            and "distance" in task
            and "angle" not in task
            and str(task.get("unit", "")).strip().lower() in _LINEAR_UNITS
        ):
            task["action"] = "MOVE"
        repaired["tasks"].append(task)
    return repaired


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
            if parsed is not None:
                payload = parsed
            else:
                try:
                    payload = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    payload = raw
            payload = _repair_linear_motion(payload)
            payload = _repair_explicit_axis(payload, instruction)
            return validate_command(payload)
        except CommandValidationError as exc:
            raise GeminiCommandError(str(exc)) from exc
