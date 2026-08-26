"""CLI for simulation and explicitly confirmed DOBOT Magician Lite operation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from app.commands.models import UniversalCommand, normalized_command
from app.config.settings import ConfigurationError, Settings
from app.gateway.adapter_manager import AdapterManager
from app.gateway.command_router import PlanStatus
from app.gateway.gateway import UniversalGateway
from app.gateway.robot_registry import RobotRegistry
from app.llm.gemini_client import GeminiCommandClient, GeminiCommandError
from app.robots.models import Robot


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for field in (
            "instruction", "llm_status", "validation_status",
            "gateway_status", "robot_id", "error",
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
            "robot_id": "robot_001", "name": "Demo Arm",
            "robot_type": "robotic_arm", "manufacturer": "generic",
            "model": "demo", "adapter_type": "mock",
            "capabilities": [
                "MOVE", "PICK", "PLACE", "GRIP", "RELEASE",
                "HOME", "STOP", "GET_STATUS",
            ],
            "status": "ONLINE", "priority": 10,
        },
        {
            "robot_id": "robot_002", "name": "Mobile Bot",
            "robot_type": "mobile_robot", "manufacturer": "generic",
            "model": "demo", "adapter_type": "mock",
            "capabilities": [
                "MOVE", "ROTATE", "NAVIGATE", "STOP", "GET_STATUS",
            ],
            "status": "ONLINE", "priority": 20,
        },
        {
            "robot_id": "robot_003", "name": "Demo Drone",
            "robot_type": "drone", "manufacturer": "generic",
            "model": "demo", "adapter_type": "mock",
            "capabilities": [
                "MOVE", "ROTATE", "NAVIGATE", "STOP", "GET_STATUS",
            ],
            "status": "OFFLINE", "priority": 30,
        },
    ]
    for data in robots:
        registry.register(Robot.model_validate(data))
    return UniversalGateway(registry)


def _confirm_physical(action: str, detail: str) -> bool:
    print("\nWARNING: REAL DOBOT HARDWARE OPERATION")
    print(f"Action: {action}")
    print(f"Command: {detail}")
    print("Clear the workspace, supervise the robot, and keep emergency controls ready.")
    try:
        answer = input("Type YES to execute this physical operation: ").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer == "YES"


def _build_real_gateway():
    from app.adapters.dobot.adapter import DobotMagicianLiteAdapter
    from app.adapters.dobot.capabilities import build_dobot_robot
    from app.adapters.dobot.client import DobotLinkClient
    from app.adapters.dobot.config import DobotConfig

    config = DobotConfig.from_env("real")
    client = DobotLinkClient(config)
    robot = build_dobot_robot()
    registry = RobotRegistry()
    registry.register(robot)  # required initial UNKNOWN registration
    client.connect()
    robot = registry.update_status(robot.robot_id, "ONLINE")
    adapter = DobotMagicianLiteAdapter(robot, client, config, _confirm_physical)
    manager = AdapterManager(register_mock=False)
    manager.register(robot.adapter_type, lambda _: adapter)
    return UniversalGateway(registry, manager), client


def _print_registry(gateway: UniversalGateway) -> None:
    print("\nRegistered Robots:\n")
    for index, robot in enumerate(gateway.registry.list_all(), start=1):
        print(f"{index}. {robot.robot_id} - {robot.name} - {robot.status.value}")


def _dobot_test(name: str) -> int:
    from app.adapters.dobot.adapter import DobotMagicianLiteAdapter
    from app.adapters.dobot.capabilities import build_dobot_robot
    from app.adapters.dobot.client import DobotLinkClient
    from app.adapters.dobot.config import DobotConfig
    from app.adapters.base import RobotAdapterError
    from app.adapters.dobot.exceptions import DobotError

    print("WARNING: --dobot-test uses a physical Magician Lite in real mode.")
    config = DobotConfig.from_env("real")
    client = DobotLinkClient(config)
    try:
        client.connect()
        if name == "connection":
            print(json.dumps(client.get_status(), indent=2, default=str))
            return 0
        action = {
            "status": {"action": "GET_STATUS"},
            "home": {"action": "HOME"},
            "move": {"action": "MOVE", "position": "configured_safe_test_position"},
            "grip": {"action": "GRIP"},
            "release": {"action": "RELEASE"},
        }[name]
        command = UniversalCommand.model_validate({
            "robot_id": "dobot_001", "tasks": [action],
        })
        adapter = DobotMagicianLiteAdapter(
            build_dobot_robot(), client, config, _confirm_physical
        )
        print("\n".join(adapter.execute(command)))
        return 0
    except (DobotError, RobotAdapterError, ValueError) as exc:
        print(f"\nDOBOT test failed safely: {exc}", file=sys.stderr)
        return 1
    finally:
        if client.is_connected():
            try:
                client.disconnect()
            except DobotError as exc:
                print(f"Disconnect warning: {exc}", file=sys.stderr)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal Robot Control")
    parser.add_argument(
        "--mode", choices=("simulation", "real"), default="simulation",
        help="simulation is the safe default; real explicitly enables DobotLink",
    )
    parser.add_argument(
        "--dobot-test",
        choices=("connection", "status", "home", "move", "grip", "release"),
        help="run one supervised real-hardware diagnostic",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dobot_test:
        if args.mode != "real":
            print("--dobot-test requires --mode real.", file=sys.stderr)
            return 2
        return _dobot_test(args.dobot_test)

    logger = configure_logging()
    client = None
    try:
        if args.mode == "real":
            gateway, client = _build_real_gateway()
        else:
            gateway = build_demo_gateway()
    except Exception as exc:
        print(f"\nReal DOBOT startup failed safely: {exc}", file=sys.stderr)
        return 1

    print("=" * 48)
    print(f" UNIVERSAL ROBOT CONTROL - PHASE 3 ({args.mode.upper()})")
    print("=" * 48)
    _print_registry(gateway)
    print("\nEnter robot instruction:")

    try:
        instruction = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 1
    finally:
        # Connection remains open through processing; cleanup occurs below.
        pass

    if not instruction:
        print("\nError: instruction cannot be empty.")
        return 1

    logger.info(
        "command_request",
        extra={"instruction": instruction, "llm_status": "pending"},
    )

    try:
        gemini = GeminiCommandClient(Settings.from_env())
        command = gemini.generate_command(instruction)
    except (ConfigurationError, GeminiCommandError) as exc:
        logger.error(
            "command_rejected",
            extra={
                "instruction": instruction, "llm_status": "failed",
                "validation_status": "invalid",
                "gateway_status": PlanStatus.INVALID.value, "error": str(exc),
            },
        )
        print(f"\nError: {exc}\n\nGateway Status: INVALID")
        if client and client.is_connected():
            client.disconnect()
        return 1

    print("\nUniversal Command:\n")
    print(json.dumps(normalized_command(command), indent=2, ensure_ascii=False))
    print("\nLLM Validation:\nVALID")

    try:
        plan = gateway.process(command)
    finally:
        if client and client.is_connected():
            client.disconnect()

    if plan.robot_id:
        print(f"\nSelected Robot:\n{plan.robot_id}")
    if plan.capability_checks:
        print("\nCapabilities:")
        for action, passed in plan.capability_checks.items():
            print(f"{action} {'✓' if passed else '✗'}")
    print(f"\nSafety:\n{'PASSED' if plan.safety_passed else 'NOT PASSED'}")
    if plan.adapter_type:
        print(f"\nAdapter:\n{plan.adapter_type}")
    execution = "SIMULATED" if plan.simulated else (
        "REAL" if plan.status is PlanStatus.READY else "NOT STARTED"
    )
    print(f"\nExecution:\n{execution}")
    for result in plan.results:
        print(result)
    if plan.reason:
        print(f"\nReason:\n{plan.reason}")
    print(f"\nGateway Status:\n{plan.status.value}")

    level = logging.INFO if plan.status is PlanStatus.READY else logging.ERROR
    logger.log(
        level, "gateway_result",
        extra={
            "instruction": instruction, "llm_status": "success",
            "validation_status": "valid", "gateway_status": plan.status.value,
            "robot_id": plan.robot_id, "error": plan.reason,
        },
    )
    return 0 if plan.status is PlanStatus.READY else 1


if __name__ == "__main__":
    raise SystemExit(run())
