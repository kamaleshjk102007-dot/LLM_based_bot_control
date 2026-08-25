"""Terminal interface for Phase 1."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from app.commands.models import normalized_command
from app.config.settings import ConfigurationError, Settings
from app.llm.gemini_client import GeminiCommandClient, GeminiCommandError


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for field in ("instruction", "llm_status", "validation_status", "error"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("universal_robot_control")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def run() -> int:
    logger = configure_logging()
    print("=" * 40)
    print(" Universal Robot Command System - Phase 1")
    print("=" * 40)
    print("\nEnter robot instruction:")
    try:
        instruction = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 1

    if not instruction:
        print("\nError: instruction cannot be empty.")
        return 1

    logger.info(
        "command_request",
        extra={"instruction": instruction, "llm_status": "pending"},
    )

    try:
        client = GeminiCommandClient(Settings.from_env())
        command = client.generate_command(instruction)
    except (ConfigurationError, GeminiCommandError) as exc:
        logger.error(
            "command_rejected",
            extra={
                "instruction": instruction,
                "llm_status": "failed",
                "validation_status": "invalid",
                "error": str(exc),
            },
        )
        print(f"\nError: {exc}\n\nStatus: INVALID")
        return 1

    logger.info(
        "command_accepted",
        extra={
            "instruction": instruction,
            "llm_status": "success",
            "validation_status": "valid",
        },
    )
    print("\nUniversal Command:\n")
    print(json.dumps(normalized_command(command), indent=2, ensure_ascii=False))
    print("\nStatus: VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
