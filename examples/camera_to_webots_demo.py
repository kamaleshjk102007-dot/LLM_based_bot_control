"""Simulation-only camera-to-Webots demonstration.

Start ``simulation/webots/worlds/magician_lite.wbt`` and press Play first.
The demo never imports physical DOBOT or DobotLink code.  It requires an
explicitly supplied, measured Webots camera calibration before motion can be
submitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.adapters.webots import WebotsClient, WebotsRobotAdapter
from app.commands.models import Action
from app.gateway.adapter_manager import AdapterManager
from app.gateway.gateway import UniversalGateway
from app.gateway.robot_registry import RobotRegistry
from app.robots.models import Robot
from motion.controller import MotionController
from motion.models import RobotPosition
from motion.planner import MotionPlanner, MotionPlanningPolicy
from vision.calibration import TabletopCalibration
from vision.camera import WebotsTcpCamera
from vision.coordinate_transform import CoordinateTransformer
from vision.detector import ColorShapeDetector
from vision.pipeline import VisionPipeline
from vision.target import MotionTargetConversionError, to_motion_target

# This visual demo uses a simplified arm, not a calibrated digital twin.  Keep
# the camera-derived route inside the range independently exercised by the
# Webots Cartesian integration tests.  These are demo preflight bounds, not
# universal robot safety limits.
def create_webots_controller(client: WebotsClient) -> MotionController:
    """Build the already-tested gateway route for the virtual robot only."""
    robot = Robot(
        robot_id="webots_001",
        name="Virtual Magician Lite",
        robot_type="robotic_arm",
        manufacturer="DOBOT-inspired",
        model="simplified_visual_model",
        adapter_type="webots",
        capabilities=frozenset({Action.MOVE, Action.ROTATE, Action.HOME, Action.STOP, Action.GET_STATUS}),
        status="ONLINE",
    )
    registry = RobotRegistry()
    registry.register(robot)
    adapters = AdapterManager(register_mock=False)
    adapter = WebotsRobotAdapter(robot, client)
    adapters.register("webots", lambda configured_robot: adapter)
    gateway = UniversalGateway(registry, adapters)
    planner = MotionPlanner(MotionPlanningPolicy(
        max_translation_step_mm=5.0,
        max_rotation_step_degrees=5.0,
        coordinate_frame="dobot_base",
        max_axis_translation_mm=20.0,
        max_total_translation_mm=30.0,
    ))
    return MotionController(planner, gateway)


def measured_position(client: WebotsClient) -> RobotPosition:
    status = client.request({"type": "status"})
    point = status["end_effector_position_mm"]
    return RobotPosition(
        x=float(point["x"]),
        y=float(point["y"]),
        z=float(point["z"]),
        r=float(status["measured_wrist_r_degrees"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the simulation-only camera-to-Webots demo")
    parser.add_argument("--calibration", type=Path, required=True,
                        help="Measured Webots-camera calibration JSON created with TabletopCalibration.save().")
    parser.add_argument("--approach-z-mm", type=float, required=True,
                        help="Reviewed, reachable end-effector approach height in Webots coordinates.")
    parser.add_argument("--class-name", default="red_block")
    parser.add_argument("--execute", action="store_true",
                        help="Submit the validated plan through the existing Webots gateway.")
    args = parser.parse_args()

    calibration = TabletopCalibration()
    calibration.load(str(args.calibration))
    client = WebotsClient(timeout=90.0)
    camera = WebotsTcpCamera(client.request)
    pipeline = VisionPipeline(
        camera=camera,
        detector=ColorShapeDetector(
            color_ranges={args.class_name: [((0, 100, 180), (12, 255, 255)), ((168, 100, 180), (180, 255, 255))]}
        ),
        calibration=calibration,
        transformer=CoordinateTransformer(
            calibration, table_z_mm=args.approach_z_mm,
            object_heights={args.class_name: 0.0, "default": 0.0},
        ),
    )
    target = None
    for _ in range(3):
        target, _ = pipeline.capture_and_process(args.class_name, visualize=False)
    assert target is not None
    print(target.model_dump_json(indent=2))
    if not target.valid:
        print("No command submitted: vision target is not valid.")
        return 2
    if not args.execute:
        print("Dry run complete. Re-run with --execute only after reviewing this target and calibration.")
        return 0
    try:
        motion_target = to_motion_target(target)
    except MotionTargetConversionError as exc:
        print(f"No command submitted: {exc}")
        return 2
    current_position = measured_position(client)
    controller = create_webots_controller(client)
    result = controller.execute_target(current_position, motion_target, "webots_001")
    print(result.model_dump_json(indent=2))
    return 0 if result.successful else 3


if __name__ == "__main__":
    raise SystemExit(main())
