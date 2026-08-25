"""JSON Schema export for integrations and documentation."""

from __future__ import annotations

from typing import Any

from app.commands.models import UniversalCommand


def universal_command_json_schema() -> dict[str, Any]:
    return UniversalCommand.model_json_schema()
