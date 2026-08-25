"""Terminal interface for the Universal Robot Control platform through Phase 2."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from app.commands.models import normalized_command
from app.config.settings import ConfigurationError, Settings
from app.gateway.command_router import PlanStatus
from app.gateway.gateway import UniversalGateway
from app.gateway.robot_registry import RobotRegistry
from app.llm.gemini_client import GeminiCommandClient, GeminiCommandError
from app.robots.models import Robot


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for field in (
            "instruction",
            "llm_status",
            "validation_status",
            "gateway_status",
            "robot_id",
            "error",
        ):
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


def build_demo_gateway() -> UniversalGateway:
    registry = RobotRegistry()
    robots = [
        {
            "robot_id": "robot_001",
            "name": "Demo Arm",
            "robot_type": "robotic_arm",
            "manufacturer": "generic",
            "model": "demo",
            "adapter_type": "mock",
            "capabilities": [
                "MOVE", "PICK", "PLACE", "GRIP", "RELEASE",
                "HOME", "STOP", "GET_STATUS",
            ],
            "status": "ONLINE",
            "priority": 10,
        },
        {
            "robot_id": "robot_002",
            "name": "Mobile Bot",
            "robot_type": "mobile_robot",
            "manufacturer": "generic",
            "model": "demo",
            "adapter_type": "mock",
            "capabilities": [
                "MOVE", "ROTATE", "NAVIGATE", "STOP", "GET_STATUS",
            ],
            "status": "ONLINE",
            "priority": 20,
        },
        {
            "robot_id": "robot_003",
            "name": "Demo Drone",
            "robot_type": "drone",
            "manufacturer": "generic",
            "model": "demo",
            "adapter_type": "mock",
            "capabilities": [
                "MOVE", "ROTATE", "NAVIGATE", "STOP", "GET_STATUS",
            ],
            "status": "OFFLINE",
            "priority": 30,
        },
    ]
    for data in robots:
        registry.register(Robot.model_validate(data))
    return UniversalGateway(registry)


def _print_registry(gateway: UniversalGateway) -> None:
    print("\nRegistered Robots:\n")
    for index, robot in enumerate(gateway.registry.list_all(), start=1):
        print(
            f"{index}. {robot.robot_id} - {robot.name} - {robot.status.value}"
        )


def run() -> int:
    logger = configure_logging()
    gateway = build_demo_gateway()

    print("=" * 40)
    print(" UNIVERSAL ROBOT CONTROL - PHASE 2")
    print("=" * 40)
    _print_registry(gateway)
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
                "gateway_status": PlanStatus.INVALID.value,
                "error": str(exc),
            },
        )
        print(f"\nError: {exc}\n\nGateway Status: INVALID")
        return 1

    print("\nUniversal Command:\n")
    print(json.dumps(normalized_command(command), indent=2, ensure_ascii=False))
    print("\nLLM Validation:\nVALID")

    plan = gateway.process(command)
    if plan.robot_id:
        print(f"\nSelected Robot:\n{plan.robot_id}")
    if plan.capability_checks:
        print("\nCapabilities:")
        for action, passed in plan.capability_checks.items():
            print(f"{action} {'✓' if passed else '✗'}")
    print(f"\nSafety:\n{'PASSED' if plan.safety_passed else 'NOT PASSED'}")
    if plan.adapter_type:
        print(f"\nAdapter:\n{plan.adapter_type}")
    print(f"\nExecution:\n{'SIMULATED' if plan.simulated else 'NOT STARTED'}")
    for result in plan.results:
        print(result)
    if plan.reason:
        print(f"\nReason:\n{plan.reason}")
    print(f"\nGateway Status:\n{plan.status.value}")

    level = logging.INFO if plan.status is PlanStatus.READY else logging.ERROR
    logger.log(
        level,
        "gateway_result",
        extra={
            "instruction": instruction,
            "llm_status": "success",
            "validation_status": "valid",
            "gateway_status": plan.status.value,
            "robot_id": plan.robot_id,
            "error": plan.reason,
        },
    )
    return 0 if plan.status is PlanStatus.READY else 1


if __name__ == "__main__":
    raise SystemExit(run())
