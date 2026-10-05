from __future__ import annotations

"""
Member 3  -->  Member 2 Integration Bridge
=========================================
This script completes the full pipeline:

  CAMERA  -->  MEMBER 3 (Vision)  -->  X, Y, Z (mm in dobot_base)
                                           |
                                           v
                               MEMBER 4 (Motion Planner)
                                  breaks into small steps
                                           |
                                           v
                               MEMBER 2 (Universal Gateway)
                                  sends each step to bot

Modes
-----
  py examples\\vision_to_bot.py              - Live webcam, simulation gateway
  py examples\\vision_to_bot.py --mock       - MockCamera, simulation gateway (no webcam needed)
  py examples\\vision_to_bot.py --mock --real-bot  - MockCamera, real Dobot via DobotLink

Keyboard shortcuts (live webcam mode only)
------------------------------------------
  Q       : Quit
  1/2/3/4 : Switch target color (1=Red 2=Blue 3=Green 4=Yellow)
  SPACE   : Lock current target and send to bot now
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

# Suppress OpenCV DSHOW backend warning spam on Windows
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2

# --- Path setup so we can import from the workspace root ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------- Member 3 (Vision) imports ----------
from vision.camera import MockCamera, USBCamera
from vision.calibration import TabletopCalibration
from vision.coordinate_transform import CoordinateTransformer
from vision.detector import ColorShapeDetector
from vision.models import TargetStatus, RobotTarget, Point3D
from vision.pipeline import VisionPipeline
from vision.visualization import VisionVisualizer
from vision.target import create_invalid_robot_target

# ---------- Member 4 (Motion Planning) imports ----------
from motion.models import (
    RobotPosition,
    RobotTarget as MotionRobotTarget,
)
from motion.planner import MotionPlanner, MotionPlanningPolicy, MotionPlanningError

# ---------- Member 2 (Gateway) imports ----------
from app.adapters.mock import MockRobotAdapter
from app.commands.models import Action
from app.gateway.adapter_manager import AdapterManager
from app.gateway.command_router import PlanStatus
from app.gateway.gateway import UniversalGateway
from app.gateway.robot_registry import RobotRegistry
from app.robots.models import Robot

SEP = "=" * 60
LINE = "-" * 60


# ------------------------------------------------------------
# Vision --> Motion conversion (Member 3 --> Member 4 boundary)
# ------------------------------------------------------------

class VisionConversionError(ValueError):
    """Raised when a vision RobotTarget cannot cross to motion planning."""


def vision_to_motion_target(vision_target: RobotTarget) -> MotionRobotTarget:
    """
    Converts Member 3's vision.models.RobotTarget to
    Member 4's motion.models.RobotTarget.

    Only VALID targets in the dobot_base frame are allowed through.
    """
    if not vision_target.valid or vision_target.status is not TargetStatus.VALID:
        raise VisionConversionError(
            f"Only VALID vision targets can be sent to the bot. "
            f"Current status: {vision_target.status.value} — {vision_target.message}"
        )
    if vision_target.coordinate_frame != "dobot_base":
        raise VisionConversionError(
            f"Target must be in dobot_base frame, got: {vision_target.coordinate_frame}"
        )
    pos = vision_target.position
    import math
    if any(not math.isfinite(v) for v in (pos.x, pos.y, pos.z)):
        raise VisionConversionError("Vision target contains non-finite coordinates.")

    return MotionRobotTarget(
        target_id=vision_target.target_id,
        class_name=vision_target.class_name,
        position=RobotPosition(x=pos.x, y=pos.y, z=pos.z),
        confidence=vision_target.confidence,
        timestamp=datetime.fromisoformat(vision_target.timestamp),
        coordinate_frame=vision_target.coordinate_frame,
        valid=True,
    )


# ------------------------------------------------------------
# Member 2 - Gateway setup (simulation mode, no hardware)
# ------------------------------------------------------------

def build_simulation_gateway(robot_id: str = "dobot_001") -> tuple[UniversalGateway, Robot]:
    """
    Build a fully simulated Member 2 gateway that records
    what it would send to the bot without opening DobotLink.
    """
    robot = Robot(
        robot_id=robot_id,
        name="DOBOT Magician Lite (Simulation)",
        robot_type="robotic_arm",
        adapter_type="vision-sim",
        capabilities=frozenset({Action.MOVE, Action.ROTATE, Action.HOME,
                                 Action.STOP, Action.GRIP, Action.RELEASE,
                                 Action.GET_STATUS}),
        status="ONLINE",
    )
    registry = RobotRegistry()
    registry.register(robot)

    adapters = AdapterManager(register_mock=False)
    adapters.register("vision-sim", lambda r: MockRobotAdapter(r))

    gateway = UniversalGateway(registry=registry, adapter_manager=adapters)
    return gateway, robot


def build_real_dobot_gateway() -> tuple[UniversalGateway, Robot, Any]:
    """
    Build a real Member 2 gateway connected to Dobot Magician Lite via DobotLink.
    """
    from app.adapters.dobot.adapter import DobotMagicianLiteAdapter
    from app.adapters.dobot.capabilities import build_dobot_robot
    from app.adapters.dobot.client import DobotLinkClient
    from app.adapters.dobot.config import DobotConfig

    config = DobotConfig.from_env("real")
    client = DobotLinkClient(config)
    robot = build_dobot_robot()
    registry = RobotRegistry()
    registry.register(robot)
    client.connect()
    robot = registry.update_status(robot.robot_id, "ONLINE")
    adapter = DobotMagicianLiteAdapter(robot, client, config, lambda act, det: True)
    manager = AdapterManager(register_mock=False)
    manager.register(robot.adapter_type, lambda _: adapter)
    return UniversalGateway(registry, manager), robot, client


# ?????????????????????????????????????????????????????????????
# Member 4 – Motion Planner setup
# ?????????????????????????????????????????????????????????????

def build_planner(max_step_mm: float = 5.0) -> MotionPlanner:
    """
    Create Member 4's motion planner with the agreed-upon step size.
    Steps larger than max_step_mm are split into multiple sub-commands.
    """
    policy = MotionPlanningPolicy(
        max_translation_step_mm=max_step_mm,
        max_rotation_step_degrees=5.0,
        coordinate_frame="dobot_base",
    )
    return MotionPlanner(policy)


# ?????????????????????????????????????????????????????????????
# Core send-to-bot logic
# ?????????????????????????????????????????????????????????????

def send_to_bot(
    vision_target: RobotTarget,
    current_position: RobotPosition,
    planner: MotionPlanner,
    gateway: UniversalGateway,
    robot: Robot,
    verbose: bool = True,
) -> RobotPosition:
    """
    Takes a validated RobotTarget from Member 3, plans motion steps
    via Member 4, and dispatches each step through Member 2's gateway.

    Returns the estimated new robot position after all steps.
    """
    print()
    print(SEP)
    print("  SENDING TO BOT")
    print(SEP)
    print(f"  Target class : {vision_target.class_name}")
    print(f"  X (mm)       : {vision_target.position.x:+.2f}")
    print(f"  Y (mm)       : {vision_target.position.y:+.2f}")
    print(f"  Z (mm)       : {vision_target.position.z:+.2f}")
    print(f"  Confidence   : {vision_target.confidence * 100:.1f}%")
    print(f"  Stability    : {vision_target.stability_score * 100:.1f}%")
    print(LINE)

    # Step 1: Convert to motion target
    try:
        motion_target = vision_to_motion_target(vision_target)
    except VisionConversionError as exc:
        print(f"  [ERROR] Cannot send to bot: {exc}")
        return current_position

    # Step 2: Plan motion steps
    try:
        plan = planner.plan(current_position, motion_target, robot.robot_id)
    except MotionPlanningError as exc:
        print(f"  [ERROR] Motion planning failed: {exc}")
        return current_position

    if not plan.steps:
        print("  [INFO] Robot is already at the target position. No movement needed.")
        return current_position

    print(f"  Total steps  : {len(plan.steps)}")
    print(f"  Plan ID      : {plan.plan_id}")
    print(LINE)

    # Step 3: Dispatch each step through Member 2's gateway
    new_pos = RobotPosition(x=current_position.x, y=current_position.y,
                            z=current_position.z, r=current_position.r)
    all_ok = True
    for step in plan.steps:
        # Build and send the universal command
        task_dict = step.model_dump(mode="json", exclude={"sequence"}, exclude_none=True)
        command_dict = {
            "version": "1.0",
            "robot_id": robot.robot_id,
            "tasks": [task_dict],
        }
        result = gateway.process(command_dict)

        direction = step.direction or ""
        amount = step.distance if step.distance else step.angle
        unit = step.unit or ""

        if result.status is PlanStatus.READY:
            # Update simulated position
            axis = direction[-1].upper() if direction else "?"
            sign = 1.0 if direction.startswith("+") else -1.0
            if axis == "X":
                new_pos = RobotPosition(x=new_pos.x + sign * (step.distance or 0),
                                        y=new_pos.y, z=new_pos.z, r=new_pos.r)
            elif axis == "Y":
                new_pos = RobotPosition(x=new_pos.x,
                                        y=new_pos.y + sign * (step.distance or 0),
                                        z=new_pos.z, r=new_pos.r)
            elif axis == "Z":
                new_pos = RobotPosition(x=new_pos.x, y=new_pos.y,
                                        z=new_pos.z + sign * (step.distance or 0),
                                        r=new_pos.r)
            if verbose:
                print(f"  [OK] MOVE {direction} {amount:.2f} {unit}")
        else:
            print(f"  [FAIL] MOVE {direction} {amount:.2f} {unit}  --> {result.reason}")
            all_ok = False
            break

    print(LINE)
    if all_ok:
        print("  [[OK]] ALL STEPS DISPATCHED SUCCESSFULLY")
        print(f"  Final simulated position:")
        print(f"    X={new_pos.x:+.2f} mm  Y={new_pos.y:+.2f} mm  Z={new_pos.z:+.2f} mm")
        # Save to output/ and config/
        _save_output(vision_target, plan, new_pos)
    else:
        print("  [[FAIL]] DISPATCH FAILED -- see error above")
    print(SEP)
    return new_pos


def _save_output(vision_target, plan, final_pos) -> None:
    """Save the RobotTarget JSON and motion plan to output/, and blocks to config/ for Member 2."""
    os.makedirs("output", exist_ok=True)
    target_path = "output/member4_target.json"
    plan_path = "output/member4_motion_plan.json"

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(vision_target.model_dump(), f, indent=4, default=str)

    plan_data = {
        "plan_id": plan.plan_id,
        "robot_id": plan.robot_id,
        "target_id": plan.target_id,
        "steps": [
            {
                "sequence": s.sequence,
                "action": s.action.value,
                "direction": s.direction,
                "distance": s.distance,
                "angle": s.angle,
                "unit": s.unit,
            }
            for s in plan.steps
        ],
        "final_position": {
            "x": final_pos.x, "y": final_pos.y, "z": final_pos.z, "r": final_pos.r
        }
    }
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan_data, f, indent=4)

    # Also save to config/detected_blocks.json for Member 2's PerceptionManager & LLM
    blocks_path = "config/detected_blocks.json"
    os.makedirs("config", exist_ok=True)
    raw_class = vision_target.class_name
    color = raw_class.split("_")[0] if "_" in raw_class else raw_class
    blocks_data = {
        "blocks": [
            {
                "id": f"{color}_block_1",
                "color": color,
                "type": "block",
                "x": round(float(vision_target.position.x), 1),
                "y": round(float(vision_target.position.y), 1),
                "z": round(float(vision_target.position.z), 1),
                "r": 0.0,
            }
        ]
    }
    with open(blocks_path, "w", encoding="utf-8") as f:
        json.dump(blocks_data, f, indent=2)

    print(f"  Saved target JSON        --> {target_path}")
    print(f"  Saved motion plan        --> {plan_path}")
    print(f"  Saved Member 2 blocks    --> {blocks_path}")


# ------------------------------------------------------------
# Mock-camera mode (no webcam needed)
# ------------------------------------------------------------

def run_mock_mode(target_class: str, current_position: RobotPosition,
                  planner: MotionPlanner, gateway: UniversalGateway, robot: Robot) -> None:
    """
    Run with MockCamera + simulated blocks (no webcam required).
    Useful for CI, demo, and testing without physical hardware.
    """
    print(SEP)
    print("  MOCK CAMERA MODE")
    print("  (No webcam needed — using simulated colored blocks)")
    print(SEP)

    cam = MockCamera(width=640, height=480)
    cam.open()
    cam.add_object("red_block",    center=(320.0, 250.0), size=(60, 60))
    cam.add_object("blue_block",   center=(180.0, 200.0), size=(60, 60))
    cam.add_object("green_block",  center=(460.0, 200.0), size=(60, 60))
    cam.add_object("yellow_block", center=(320.0, 350.0), size=(60, 60))

    pipeline = VisionPipeline(camera=cam, stability_samples=3)

    print(f"  Looking for: {target_class}  (stabilizing over 4 frames)")
    print()

    final_target = None
    for i in range(1, 5):
        target, _ = pipeline.capture_and_process(target_class)
        status_icon = "[OK]" if target.valid else "[--]"
        print(f"  Frame {i}: {status_icon} {target.status.value}  | {target.message}")
        if target.valid:
            final_target = target

    if final_target is not None:
        send_to_bot(final_target, current_position, planner, gateway, robot)
    else:
        print()
        print(f"  [!!] Could not get a stable target for '{target_class}'.")
        print("  Available blocks: red_block, blue_block, green_block, yellow_block")

    cam.release()


# ------------------------------------------------------------
# Live webcam mode
# ------------------------------------------------------------

def run_live_mode(cam_idx: int, current_position: RobotPosition,
                  planner: MotionPlanner, gateway: UniversalGateway, robot: Robot,
                  auto_mode: bool = False) -> None:
    """
    Live webcam mode: detect objects in real time.
    Press SPACE to lock the current target and send X,Y,Z to the bot,
    or press A to enable Auto-Follow so the bot tracks blocks automatically!
    """
    print(SEP)
    print("  LIVE WEBCAM MODE — MEMBER 3 --> MEMBER 2 BRIDGE")
    print(SEP)
    print("  Controls:")
    print("  [A]       Toggle Auto-Follow (bot automatically moves to cube as it moves!)")
    print("  [SPACE]   Lock current VALID target and SEND to bot")
    print("  [1/2/3/4] Switch color target (1=Red 2=Blue 3=Green 4=Yellow)")
    print("  [Q]       Quit")
    print()
    print(f"  Connecting to camera index {cam_idx}...")

    camera = USBCamera(device_index=cam_idx, width=640, height=480)
    if not camera.open():
        print(f"  [ERROR] Cannot open camera {cam_idx}. Try: py examples\\vision_to_bot.py --mock")
        return

    print(f"  [OK] Camera {cam_idx} opened!")
    print()

    pipeline = VisionPipeline(
        camera=camera,
        stability_samples=3,
        disambiguation_strategy="leftmost",
    )
    visualizer = VisionVisualizer()

    requested_class = "red_block"
    last_target: RobotTarget | None = None
    frame_count = 0
    send_count = 0
    current_pos = current_position
    auto_follow = auto_mode
    last_auto_send = 0.0

    print(f"  Currently looking for: {requested_class}  (press 1/2/3/4 to change)")
    if auto_follow:
        print("  [AUTO-FOLLOW] ACTIVE -- Bot will automatically move to tracked blocks!")
    print(SEP)

    try:
        while True:
            target, hud = pipeline.capture_and_process(requested_class, visualize=True)
            last_target = target
            frame_count += 1

            # --- Auto-Follow Logic ---
            now = time.time()
            if auto_follow and (now - last_auto_send > 1.2):
                if target.valid:
                    dx = target.position.x - current_pos.x
                    dy = target.position.y - current_pos.y
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist > 8.0:  # Block moved > 8 mm
                        print(f"\n  [AUTO-FOLLOW] Block moved ({dist:.1f}mm) -- Bot following to X={target.position.x:+.1f}, Y={target.position.y:+.1f} mm...")
                        current_pos = send_to_bot(target, current_pos, planner, gateway, robot)
                        send_count += 1
                        last_auto_send = now

            if hud is not None:
                # Draw info bar at bottom
                h, w = hud.shape[:2]
                status_color = (0, 220, 80) if target.valid else (60, 60, 200)
                cv2.rectangle(hud, (0, h - 50), (w, h), (20, 20, 20), -1)
                follow_str = "AUTO: ON" if auto_follow else "AUTO: OFF ([A])"
                cv2.putText(hud,
                    f"Target: {requested_class} | {follow_str} | [SPACE]=Send [A]=Auto [1-4]=Color [Q]=Quit",
                    (6, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
                if target.valid:
                    pos_text = (f"X={target.position.x:+.1f}mm  "
                                f"Y={target.position.y:+.1f}mm  "
                                f"Z={target.position.z:+.1f}mm  "
                                f"Conf={target.confidence * 100:.0f}%  Stab={target.stability_score * 100:.0f}%")
                else:
                    pos_text = target.message or "Waiting..."
                cv2.putText(hud, pos_text, (6, h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1, cv2.LINE_AA)

                cv2.imshow("Member 3 --> Member 2 Bridge (SPACE=Send, A=Auto, Q=Quit)", hud)

            # Console log every 15 frames
            if frame_count % 15 == 0:
                icon = "[OK]" if target.valid else "[--]"
                if target.valid:
                    print(f"  {icon} {target.status.value:12s} | "
                          f"X={target.position.x:+7.1f}  Y={target.position.y:+7.1f}  "
                          f"Z={target.position.z:+6.1f} mm | "
                          f"Conf={target.confidence * 100:.0f}%")
                else:
                    print(f"  {icon} {target.status.value:12s} | {target.message}")

            key_ex = cv2.waitKeyEx(1)
            key = key_ex & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                print("\n  [Q] Quitting...")
                break

            elif key == ord('a') or key == ord('A'):
                auto_follow = not auto_follow
                state = "ENABLED" if auto_follow else "DISABLED"
                print(f"\n  [A] Auto-Follow mode: {state}")

            elif key == ord(' '):  # SPACE – send to bot
                if last_target is not None and last_target.valid:
                    print(f"\n  [SPACE] Locking target and sending to bot...")
                    current_pos = send_to_bot(
                        last_target, current_pos, planner, gateway, robot
                    )
                    send_count += 1
                    print(f"\n  Total sends this session: {send_count}")
                    print(SEP)
                else:
                    status = last_target.status.value if last_target else "NONE"
                    print(f"\n  [!!] Cannot send — target not VALID yet. Status: {status}")

            elif key == ord('1'):
                requested_class = "red_block"
                pipeline.reset_history()
                print(f"\n  [1] Switched to: red_block")
            elif key == ord('2'):
                requested_class = "blue_block"
                pipeline.reset_history()
                print(f"\n  [2] Switched to: blue_block")
            elif key == ord('3'):
                requested_class = "green_block"
                pipeline.reset_history()
                print(f"\n  [3] Switched to: green_block")
            elif key == ord('4'):
                requested_class = "yellow_block"
                pipeline.reset_history()
                print(f"\n  [4] Switched to: yellow_block")

    finally:
        try:
            camera.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print(f"\n  Session complete. Total sends: {send_count}")


# ------------------------------------------------------------
# CLI Entry point
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Member 3 Vision --> Member 2 Gateway Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mock", action="store_true",
                        help="Use MockCamera with simulated blocks (no webcam needed)")
    parser.add_argument("--auto", action="store_true",
                        help="Enable auto-follow mode: bot automatically moves along with detected blocks")
    parser.add_argument("--real-bot", action="store_true",
                        help="Connect to real DOBOT Magician Lite hardware via DobotLink")
    parser.add_argument("--target", default="red_block",
                        choices=["red_block", "blue_block", "green_block", "yellow_block"],
                        help="Object class to detect in mock mode (default: red_block)")
    parser.add_argument("--cam", type=int, default=None,
                        help="Camera index override for live mode (default: auto-detect)")
    parser.add_argument("--start-x", type=float, default=200.0,
                        help="Robot current X position in mm (default: 200.0)")
    parser.add_argument("--start-y", type=float, default=0.0,
                        help="Robot current Y position in mm (default: 0.0)")
    parser.add_argument("--start-z", type=float, default=50.0,
                        help="Robot current Z position in mm (default: 50.0)")
    parser.add_argument("--step-mm", type=float, default=5.0,
                        help="Max translation step size in mm per gateway command (default: 5.0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print()
    print(SEP)
    print("  MEMBER 3 --> MEMBER 2 INTEGRATION BRIDGE")
    print("  Vision Coordinates --> Bot Movement")
    print(SEP)
    print()

    # Robot starting position (current estimated pose before move)
    current_position = RobotPosition(
        x=args.start_x,
        y=args.start_y,
        z=args.start_z,
        r=0.0,
    )
    print(f"  Robot start position:")
    print(f"    X={current_position.x:+.1f} mm  Y={current_position.y:+.1f} mm  Z={current_position.z:+.1f} mm")
    print()

    client = None
    if getattr(args, "real_bot", False):
        try:
            gateway, robot, client = build_real_dobot_gateway()
            print(f"  [OK] Connected to REAL DOBOT Magician Lite (robot_id: {robot.robot_id})")
        except Exception as e:
            print("  [INFO] DobotLink not reachable at 127.0.0.1:9090.")
            print("         --> Falling back cleanly to SIMULATION mode.")
            print("         (To connect real bot, ensure DobotLink is installed and running.)")
            gateway, robot = build_simulation_gateway()
    else:
        gateway, robot = build_simulation_gateway()
        print(f"  [OK] Member 2 gateway ready  (mode: SIMULATION, robot_id: {robot.robot_id})")

    # Build Member 4 planner
    planner = build_planner(max_step_mm=args.step_mm)
    print(f"  [OK] Member 4 motion planner ready  (max_step={args.step_mm} mm)")
    print()

    try:
        if args.mock:
            # --- Mock camera mode (no webcam) ---
            run_mock_mode(args.target, current_position, planner, gateway, robot)
        else:
            # --- Live webcam mode ---
            cam_idx = args.cam
            if cam_idx is None:
                # Auto-detect: try index 1 first (DOBOT cam), else 0 (laptop cam)
                import cv2 as _cv
                if hasattr(_cv, "setLogLevel"):
                    _cv.setLogLevel(0)
                elif hasattr(_cv, "utils") and hasattr(_cv.utils, "logging"):
                    _cv.utils.logging.setLogLevel(_cv.utils.logging.LOG_LEVEL_SILENT)
                backend = _cv.CAP_DSHOW if sys.platform.startswith("win") else _cv.CAP_ANY
                test = _cv.VideoCapture(1, backend)
                if test.isOpened():
                    test.release()
                    cam_idx = 1
                    print("  [INFO] Found external camera at index 1 (DOBOT/USB)")
                else:
                    cam_idx = 0
                    print("  [INFO] Using laptop webcam at index 0")
            run_live_mode(cam_idx, current_position, planner, gateway, robot, auto_mode=args.auto)
    except KeyboardInterrupt:
        print("\n  [INFO] Stopped by user (Ctrl+C). Exiting cleanly.")
    finally:
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
