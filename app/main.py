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
from app.perception.manager import PerceptionManager
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


def _build_real_gateway(perception: Any | None = None):
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
    adapter = DobotMagicianLiteAdapter(
        robot, client, config, _confirm_physical, perception=perception
    )
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
        if name == "limits":
            status = client.get_status()
            limits = config.safety_limits
            print("=" * 55)
            print(" DOBOT MAGICIAN LITE PHYSICAL LIMITS & STATUS")
            print("=" * 55)
            if limits:
                print(f"  X Range (Front/Back) : [{limits.min_x:.1f}, {limits.max_x:.1f}] mm")
                print(f"  Y Range (Right/Left) : [{limits.min_y:.1f} (Right), {limits.max_y:.1f} (Left)] mm")
                print(f"  Z Range (Down/Up)    : [{limits.min_z:.1f} (Table), {limits.max_z:.1f} (High)] mm")
                print(f"  R Range (Rotation)   : [{limits.min_r:.1f}°, {limits.max_r:.1f}°]")
                print(f"  Max Spherical Reach  : 340.0 mm (sqrt(X^2 + Y^2) <= 340)")
            print(f"  Robot State          : {status.get('state', 'UNKNOWN')}")
            pose = status.get("pose")
            if pose:
                print(
                    f"  Current Pose         : X={pose.get('x', 0):.1f}, "
                    f"Y={pose.get('y', 0):.1f}, Z={pose.get('z', 0):.1f}, "
                    f"R={pose.get('r', 0):.1f}"
                )
            print("=" * 55)
            return 0

        action_map = {
            "status": {"action": "GET_STATUS"},
            "home": {"action": "HOME"},
            "move": {"action": "MOVE", "position": "configured_safe_test_position"},
            "center": {"action": "MOVE", "position": "configured_safe_test_position"},
            "right": {
                "action": "MOVE",
                "position": "extreme_right",
                "parameters": {"x": 110.0, "y": -320.0, "z": 20.0, "r": 0.0},
            },
            "left": {
                "action": "MOVE",
                "position": "extreme_left",
                "parameters": {"x": 110.0, "y": 320.0, "z": 20.0, "r": 0.0},
            },
            "up": {
                "action": "MOVE",
                "position": "extreme_up",
                "parameters": {"x": 200.0, "y": 0.0, "z": 120.0, "r": 0.0},
            },
            "down": {
                "action": "MOVE",
                "position": "extreme_down",
                "parameters": {"x": 200.0, "y": 0.0, "z": 0.0, "r": 0.0},
            },
            "forward": {
                "action": "MOVE",
                "position": "extreme_forward",
                "parameters": {"x": 280.0, "y": 0.0, "z": 20.0, "r": 0.0},
            },
            "backward": {
                "action": "MOVE",
                "position": "extreme_backward",
                "parameters": {"x": 150.0, "y": 0.0, "z": 20.0, "r": 0.0},
            },
            "grip": {"action": "GRIP"},
            "release": {"action": "RELEASE"},
        }
        action = action_map[name]
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


def _dobot_move_to_coords(x: float, y: float, z: float, r: float = 0.0) -> int:
    """Move physical robot directly to coordinates X Y Z with confirmation."""
    from app.adapters.dobot.client import DobotLinkClient
    from app.adapters.dobot.config import DobotConfig, DobotPosition
    from app.adapters.dobot.exceptions import DobotError

    print("WARNING: moving the physical Magician Lite to target coordinates.")
    config = DobotConfig.from_env("real")
    client = DobotLinkClient(config)
    try:
        client.connect()
        target = DobotPosition(x, y, z, r)
        if config.safety_limits:
            config.safety_limits.validate(target, label="Target position")
        before = client._read_position()
        detail = json.dumps({
            "target": target.as_dict(),
            "current": before.as_dict(),
        }, sort_keys=True)
        if not _confirm_physical("MOVE_TO_COORDINATES", detail):
            print("Movement cancelled; no command was sent.")
            return 1
        result = client.move(target)
        print(f"[DOBOT REAL] MOVED AND VERIFIED: {json.dumps(result, sort_keys=True)}")
        return 0
    except (DobotError, ValueError) as exc:
        print(f"\nDOBOT move failed safely: {exc}", file=sys.stderr)
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
        "--blocks", default="config/detected_blocks.json",
        help="path to detected blocks JSON file from vision system (Member 3)",
    )
    parser.add_argument(
        "--dobot-move-to", nargs=3, type=float, metavar=("X", "Y", "Z"),
        help="move physical robot directly to coordinates X Y Z in mm",
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
        choices=(
            "connection", "status", "home", "move", "center",
            "right", "left", "up", "down", "forward", "backward",
            "grip", "release", "limits",
        ),
        help="run one supervised real-hardware diagnostic (home, move, right, left, up, down, forward, backward, limits)",
    )
    parser.add_argument(
        "--auto-track", action="store_true",
        help="continually follow/move to blocks detected by Member 3 camera in real time",
    )
    parser.add_argument(
        "--track-color", default="red",
        help="block color to track when using --auto-track (default: red)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="parse natural language locally using rule engine without calling cloud Gemini",
    )
    return parser


def _parse_offline_instruction(instruction: str) -> UniversalCommand | None:
    """Fallback parser for standard commands when cloud LLM is unavailable."""
    import re
    from app.commands.models import Action, Task, UniversalCommand

    clauses = re.split(r",?\s+(?:then|and)\s+|,", instruction.strip(), flags=re.IGNORECASE)
    tasks: list[Task] = []

    for clause in clauses:
        c = clause.strip().lower()
        if not c:
            continue

        if re.search(r"\b(status|get status|get the robot status)\b", c):
            tasks.append(Task(action=Action.GET_STATUS))
            continue

        if re.search(r"\b(go home|home|return home)\b", c):
            tasks.append(Task(action=Action.HOME))
            continue

        if re.search(r"\b(stop|halt|emergency stop)\b", c):
            tasks.append(Task(action=Action.STOP))
            continue

        if re.search(r"\b(open gripper|release|drop|open the gripper)\b", c):
            tasks.append(Task(action=Action.RELEASE))
            continue

        if re.search(r"\b(close gripper|grip|grasp|close the gripper)\b", c):
            tasks.append(Task(action=Action.GRIP))
            continue

        rot_match = re.search(r"(?:rotate|turn)\s+(?:by\s+)?([+-]?\d+(?:\.\d+)?)\s*(degrees?|deg)?", c)
        if rot_match:
            angle = abs(float(rot_match.group(1)))
            tasks.append(Task(action=Action.ROTATE, angle=angle, unit="deg"))
            continue

        move_match = re.search(
            r"(?:move|translate|drive)\s+(?:the\s+(?:arm|robot)\s+)?(?:to\s+the\s+)?([a-z+-]+)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*([a-z]+)?",
            c,
        )
        if move_match:
            raw_dir = move_match.group(1).strip().lower()
            dist = float(move_match.group(2))
            unit = (move_match.group(3) or "mm").strip()
            if unit in ("centimeters", "centimeter", "cms"):
                unit = "cm"
            elif unit in ("millimeters", "millimeter"):
                unit = "mm"
            elif unit in ("meters", "meter"):
                unit = "m"
            direction_map = {
                "right": "-y",
                "left": "+y",
                "forward": "+x",
                "front": "+x",
                "backward": "-x",
                "back": "-x",
                "up": "upward",
                "down": "downward",
            }
            direction = direction_map.get(raw_dir, raw_dir)
            tasks.append(Task(action=Action.MOVE, direction=direction, distance=dist, unit=unit))
            continue

        # Match "pick up the red block", "pick red block", "grab the blue block"
        pick_match = re.search(r"(?:pick\s*up|pick|grab|grasp)\s+(?:the\s+)?([a-z]+)\s*(?:block|cube|object)?", c)
        if pick_match:
            color = pick_match.group(1).strip().lower()
            from app.commands.models import ObjectReference
            tasks.append(Task(action=Action.PICK, object=ObjectReference(color=color, type="block")))
            continue

        # Match "place at ...", "place block", "place the red block"
        place_match = re.search(r"(?:place|put\s*down|drop)\s+(?:the\s+)?([a-z]+)?\s*(?:block|cube|object)?", c)
        if place_match and not re.search(r"\b(release|open)\b", c):
            color = (place_match.group(1) or "").strip().lower()
            from app.commands.models import ObjectReference
            obj = ObjectReference(color=color, type="block") if color else None
            tasks.append(Task(action=Action.PLACE, object=obj))
            continue

        # Match "move to the red block", "go to red block", "move to red"
        goto_match = re.search(r"(?:move\s+to|go\s+to)\s+(?:the\s+)?([a-z]+)\s*(?:block|cube|object)?", c)
        if goto_match and not re.search(r"\b(right|left|forward|backward|up|down)\b", goto_match.group(1)):
            color = goto_match.group(1).strip().lower()
            from app.commands.models import ObjectReference
            tasks.append(Task(action=Action.MOVE, object=ObjectReference(color=color, type="block")))
            continue

    if tasks:
        return UniversalCommand(version="1.0", tasks=tasks)
    return None


def run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dobot_move_to is not None:
        if args.mode != "real":
            print("--dobot-move-to requires --mode real.", file=sys.stderr)
            return 2
        return _dobot_move_to_coords(
            args.dobot_move_to[0], args.dobot_move_to[1], args.dobot_move_to[2]
        )
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

    perception = PerceptionManager.from_file(args.blocks)
    logger = configure_logging()
    client = None
    try:
        if args.mode == "real":
            gateway, client = _build_real_gateway(perception)
        elif args.mode == "webots":
            gateway = _build_webots_gateway()
        else:
            gateway = build_demo_gateway()
    except Exception as exc:
        print(f"\nStartup failed safely: {exc}", file=sys.stderr)
        return 1

    print("=" * 48)
    print(f" UNIVERSAL ROBOT CONTROL ({args.mode.upper()})")
    print("=" * 48)
    _print_registry(gateway)
    if perception.blocks:
        print("\nPerception (Detected Blocks):")
        for b in perception.blocks:
            print(f"  • {b.color.upper()} {b.type} [{b.id}] at X={b.x:.1f}, Y={b.y:.1f}, Z={b.z:.1f} mm")
    if getattr(args, "auto_track", False):
        print(f"\n[AUTO-TRACK] Started live tracking for '{args.track_color}' block...")
        print("Watching config/detected_blocks.json from Member 3 camera...")
        print("Press Ctrl+C to stop.\n")
        last_coords = None
        try:
            while True:
                perception.reload()
                block = perception.find(color=args.track_color)
                if block is not None:
                    curr_coords = (round(block.x, 1), round(block.y, 1), round(block.z, 1))
                    if last_coords is None or (
                        abs(curr_coords[0] - last_coords[0]) > 8.0 or
                        abs(curr_coords[1] - last_coords[1]) > 8.0
                    ):
                        print(f"\n[AUTO-TRACK] Camera detected {block.color.upper()} block at X={block.x:.1f}, Y={block.y:.1f}, Z={block.z:.1f} mm")
                        from app.commands.models import ObjectReference
                        cmd = UniversalCommand(version="1.0", tasks=[
                            Task(action=Action.MOVE, object=ObjectReference(color=block.color, type="block"))
                        ])
                        plan = gateway.process(cmd)
                        print(f"  --> Bot moving along with block! Gateway status: {plan.status.value}")
                        last_coords = curr_coords
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[AUTO-TRACK] Stopped.")
        finally:
            if client:
                client.disconnect()
        return 0

    print("\nEnter robot instruction:")

    try:
        instruction = input("> ").strip()
        perception.reload()  # Refresh with latest camera coordinates
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

    command: UniversalCommand | None = None
    if instruction.startswith("{"):
        try:
            command = UniversalCommand.model_validate_json(instruction)
        except Exception:
            pass

    if command is None and getattr(args, "offline", False):
        command = _parse_offline_instruction(instruction)
        if command:
            print("\n[OFFLINE PARSER] Parsed instruction using local rule engine.")
        else:
            print(f"\nError: Could not parse instruction locally: {instruction}")
            if client:
                client.disconnect()
            return 1

    if command is None:
        try:
            gemini = GeminiCommandClient(Settings.from_env())
            perception_context = perception.to_prompt_context() if perception.blocks else None
            command = gemini.generate_command(instruction, perception_context=perception_context)
        except (ConfigurationError, GeminiCommandError) as exc:
            command = _parse_offline_instruction(instruction)
            if command:
                print("\n[LOCAL PARSER] Cloud Gemini was unavailable; parsed instruction locally.")
            else:
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
