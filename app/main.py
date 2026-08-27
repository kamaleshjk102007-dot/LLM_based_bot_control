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



def _build_webots_gateway() -> UniversalGateway:
    """Build a visual-simulation gateway with no physical hardware path."""
    from app.adapters.webots import WebotsRobotAdapter

    robot = Robot.model_validate({
        "robot_id": "webots_001",
        "name": "Virtual Magician Lite",
        "robot_type": "robotic_arm",
        "manufacturer": "DOBOT-inspired",
        "model": "simplified_visual_model",
        "adapter_type": "webots",
        "capabilities": ["MOVE", "ROTATE", "HOME", "STOP", "GET_STATUS"],
        "status": "ONLINE",
        "priority": 1,
    })
    registry = RobotRegistry()
    registry.register(robot)
    manager = AdapterManager(register_mock=False)
    manager.register("webots", WebotsRobotAdapter)
    return UniversalGateway(registry, manager)


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
        try:
            client.disconnect()
        except DobotError as exc:
            print(f"Disconnect warning: {exc}", file=sys.stderr)



def _dobot_calibrate(axis: str, delta_mm: float) -> int:
    """Run one explicitly confirmed, bounded, single-axis calibration move."""
    from app.adapters.dobot.client import DobotLinkClient
    from app.adapters.dobot.config import DobotConfig
    from app.adapters.dobot.exceptions import DobotError

    print("WARNING: calibration moves the physical Magician Lite.")
    config = DobotConfig.from_env("real")
    client = DobotLinkClient(config)
    try:
        client.connect()
        before, target = client.calibration_preview(axis, delta_mm)
        detail = json.dumps({
            "axis": axis.upper(),
            "delta_mm": delta_mm,
            "before": before.as_dict(),
            "target": target.as_dict(),
            "speed_ratio": config.calibration_speed_ratio,
            "acceleration_ratio": config.calibration_acceleration_ratio,
            "hard_max_step_mm": config.calibration_max_step_mm,
        }, sort_keys=True)
        if not _confirm_physical("CALIBRATE", detail):
            print("Calibration cancelled; no movement command was sent.")
            return 1
        result = client.calibrate(axis, delta_mm, before)
        print(f"[DOBOT REAL] CALIBRATION VERIFIED: {json.dumps(result, sort_keys=True)}")
        return 0
    except (DobotError, ValueError) as exc:
        print(f"\nDOBOT calibration failed safely: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            client.disconnect()
        except DobotError as exc:
            print(f"Disconnect warning: {exc}", file=sys.stderr)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal Robot Control")
    parser.add_argument(
        "--mode", choices=("simulation", "webots", "real"), default="simulation",
        help="simulation is text-only, webots is visual, and real explicitly enables DobotLink",
    )
    parser.add_argument(
        "--dobot-calibrate-axis", choices=("x", "y", "z"),
        help="guarded real calibration axis; requires --dobot-calibrate-mm",
    )
    parser.add_argument(
        "--dobot-calibrate-mm", type=float,
        help="signed calibration distance, hard-limited to 5 mm",
    )
    parser.add_argument(
        "--dobot-test",
        choices=("connection", "status", "home", "move", "grip", "release"),
        help="run one supervised real-hardware diagnostic",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (args.dobot_calibrate_axis is None) != (args.dobot_calibrate_mm is None):
        print("--dobot-calibrate-axis and --dobot-calibrate-mm are required together.", file=sys.stderr)
        return 2
    if args.dobot_calibrate_axis is not None:
        if args.mode != "real":
            print("Calibration requires --mode real.", file=sys.stderr)
            return 2
        return _dobot_calibrate(args.dobot_calibrate_axis, args.dobot_calibrate_mm)
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
        elif args.mode == "webots":
            gateway = _build_webots_gateway()
        else:
            gateway = build_demo_gateway()
    except Exception as exc:
        print(f"\nStartup failed safely: {exc}", file=sys.stderr)
        return 1

    print("=" * 48)
    print(f" UNIVERSAL ROBOT CONTROL - PHASE 4 ({args.mode.upper()})")
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
        if client:
            client.disconnect()
        return 1

    print("\nUniversal Command:\n")
    print(json.dumps(normalized_command(command), indent=2, ensure_ascii=False))
    print("\nLLM Validation:\nVALID")

    try:
        plan = gateway.process(command)
    finally:
        if client:
            client.disconnect()

    if plan.robot_id:
        print(f"\nSelected Robot:\n{plan.robot_id}")
    if plan.capability_checks:
        print("\nCapabilities:")
        for action, passed in plan.capability_checks.items():
            print(f"{action} {'PASS' if passed else 'FAIL'}")
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
