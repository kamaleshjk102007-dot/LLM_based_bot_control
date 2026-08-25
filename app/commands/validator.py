"""Application-side validation; LLM output is never trusted directly."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.commands.models import UniversalCommand


class CommandValidationError(ValueError):
    """Safe, user-facing validation error."""


def validate_command(payload: UniversalCommand | dict[str, Any] | str) -> UniversalCommand:
    try:
        if isinstance(payload, UniversalCommand):
            # Re-validate a serialized copy so every boundary follows the same path.
            return UniversalCommand.model_validate(payload.model_dump())
        if isinstance(payload, str):
            return UniversalCommand.model_validate_json(payload)
        return UniversalCommand.model_validate(payload)
    except (ValidationError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CommandValidationError(f"Invalid Universal Robot Command: {exc}") from exc
