"""Controller for the simplified Magician Lite Webots model.

The model is for visual software testing only. It is not a calibrated digital twin.
"""

import base64
import json
import math
import socket

from controller import Supervisor

from cartesian_motion import (
    axis_displacement_report,
    damped_xyz_step,
    displacement_report,
    requested_axis_metres,
)
from rotation_motion import angular_report, requested_r_radians


TIME_STEP = 32
HOST = "127.0.0.1"
PORT = 8765
LIMITS = {
    "base_motor": (-math.radians(135), math.radians(135)),
    "shoulder_motor": (-math.radians(5), math.radians(80)),
    "elbow_motor": (-math.radians(10), math.radians(85)),
    "wrist_motor": (-math.radians(145), math.radians(145)),
}
HOME = {
    "base_motor": 0.0,
    "shoulder_motor": math.radians(35),
    "elbow_motor": math.radians(35),
    "wrist_motor": 0.0,
}


class Simulator:
    def __init__(self):
        self.robot = Supervisor()
        self.motors = {name: self.robot.getDevice(name) for name in LIMITS}
        self.nodes = {
            name: self.robot.getFromDef(name)
            for name in ("ARM_BASE", "SHOULDER_LINK", "ELBOW_LINK", "END_EFFECTOR")
        }
        missing = [name for name, node in self.nodes.items() if node is None]
        if missing:
            raise RuntimeError(
                "Webots world is missing Cartesian nodes: " + ", ".join(missing)
            )
        self.visual_nodes = {
            name: self.robot.getFromDef(name)
            for name in (
                "MOVE_START_MARKER",
                "MOVE_END_MARKER",
                "MOVE_TRAJECTORY_COORD",
            )
        }
        missing_visuals = [
            name for name, node in self.visual_nodes.items() if node is None
        ]
        if missing_visuals:
            raise RuntimeError(
                "Webots world is missing movement indicators: "
                + ", ".join(missing_visuals)
            )
        self.targets = dict(HOME)
        self.wrist_sensor = self.robot.getDevice("wrist_sensor")
        self.joint_sensors = {
            motor: self.robot.getDevice(sensor)
            for motor, sensor in {
                "base_motor": "base_sensor",
                "shoulder_motor": "shoulder_sensor",
                "elbow_motor": "elbow_sensor",
            }.items()
        }
        self.joint_sensors["wrist_motor"] = self.wrist_sensor
        missing_sensors = [
            motor for motor, sensor in self.joint_sensors.items() if sensor is None
        ]
        if missing_sensors:
            raise RuntimeError(
                "Webots world is missing joint sensors: "
                + ", ".join(missing_sensors)
            )
        for sensor in self.joint_sensors.values():
            sensor.enable(TIME_STEP)
        self.camera = self.robot.getDevice("tabletop_camera")
        if self.camera is None:
            raise RuntimeError("Webots world is missing tabletop_camera.")
        self.camera.enable(TIME_STEP)
        self.calibration_markers = {
            name: self.robot.getFromDef(def_name)
            for name, def_name in {
                "magenta_marker": "CALIBRATION_MAGENTA",
                "cyan_marker": "CALIBRATION_CYAN",
                "green_marker": "CALIBRATION_GREEN",
                "yellow_marker": "CALIBRATION_YELLOW",
            }.items()
        }
        if any(node is None for node in self.calibration_markers.values()):
            raise RuntimeError("Webots world is missing calibration markers.")
        self.demo_target = self.robot.getFromDef("DEMO_RED_BLOCK")
        if self.demo_target is None:
            raise RuntimeError("Webots world is missing DEMO_RED_BLOCK.")
        self.stopped = False
        for name, motor in self.motors.items():
            motor.setVelocity(0.7)
            motor.setPosition(self.targets[name])

    @staticmethod
    def clamp(name, value):
        low, high = LIMITS[name]
        return max(low, min(high, value))

    def set_targets(self):
        for name, target in self.targets.items():
            self.targets[name] = self.clamp(name, target)
            self.motors[name].setVelocity(0.7)
            self.motors[name].setPosition(self.targets[name])

    def advance(self, steps):
        for _ in range(steps):
            if self.robot.step(TIME_STEP) == -1:
                raise RuntimeError("Webots stopped during Cartesian movement.")

    def end_effector_position(self):
        return tuple(self.nodes["END_EFFECTOR"].getPosition())

    def settle_cartesian(self):
        """Wait until measured pose and joint feedback are both stationary."""
        previous = self.end_effector_position()
        sensors = getattr(self, "joint_sensors", {})
        previous_joints = {
            name: sensor.getValue() for name, sensor in sensors.items()
        }
        stable = 0
        for _ in range(240):
            self.advance(1)
            current = self.end_effector_position()
            current_joints = {
                name: sensor.getValue() for name, sensor in sensors.items()
            }
            pose_stable = math.dist(previous, current) <= 1e-6
            joints_stable = all(
                abs(current_joints[name] - previous_joints[name]) <= 1e-5
                for name in sensors
            )
            stable = stable + 1 if pose_stable and joints_stable else 0
            if stable >= 8:
                return
            previous = current
            previous_joints = current_joints
        raise ValueError("Cartesian joints did not settle.")

    def xyz_columns(self, end_position):
        """World-space Jacobian for the base, shoulder and elbow joints."""
        base = self.nodes["ARM_BASE"]
        orientation = base.getOrientation()
        base_axis = (orientation[2], orientation[5], orientation[8])
        arm_axis = (orientation[1], orientation[4], orientation[7])

        def column(axis, pivot):
            radius = tuple(a - b for a, b in zip(end_position, pivot))
            return (
                axis[1] * radius[2] - axis[2] * radius[1],
                axis[2] * radius[0] - axis[0] * radius[2],
                axis[0] * radius[1] - axis[1] * radius[0],
            )

        return (
            column(base_axis, base.getPosition()),
            column(arm_axis, self.nodes["SHOULDER_LINK"].getPosition()),
            column(arm_axis, self.nodes["ELBOW_LINK"].getPosition()),
        )

    def show_motion_indicator(self, before, after, report):
        # Pins are anchored at the exact measured positions. Their tops and the
        # trajectory are lifted equally for visibility, preserving 5 mm spacing.
        marker_height = 0.04
        self.visual_nodes["MOVE_START_MARKER"].getField(
            "translation"
        ).setSFVec3f(list(before))
        self.visual_nodes["MOVE_END_MARKER"].getField(
            "translation"
        ).setSFVec3f(list(after))
        start_top = [before[0], before[1], before[2] + marker_height]
        end_top = [after[0], after[1], after[2] + marker_height]
        points = self.visual_nodes["MOVE_TRAJECTORY_COORD"].getField("point")
        points.setMFVec3f(0, start_top)
        points.setMFVec3f(1, end_top)
        axis = report.get("axis", "x").lower()
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        self.robot.setLabel(
            0,
            f"CARTESIAN {axis.upper()}: "
            + ("PASS" if report["verified"] else "FAIL")
            + "\n"
            f"GREEN start {axis.upper()}: "
            f"{report['before_mm'][axis_index]:.3f} mm\n"
            f"RED final {axis.upper()}: "
            f"{report['after_mm'][axis_index]:.3f} mm\n"
            f"Requested: {report['requested_mm']:+.3f} mm\n"
            f"Measured: {report['actual_mm']:+.3f} mm\n"
            f"Error: {report['error_mm']:+.3f} mm\n"
            "Yellow line: exact measured trajectory",
            0.015,
            0.04,
            0.035,
            0xFFFFFF,
            0.0,
            "Arial",
        )

    def move_cartesian_x(self, requested_x_m):
        return self.move_cartesian_axis("x", requested_x_m)

    def move_cartesian_y(self, requested_y_m):
        return self.move_cartesian_axis("y", requested_y_m)

    def move_cartesian_z(self, requested_z_m):
        return self.move_cartesian_axis("z", requested_z_m)

    def move_cartesian_axis(self, axis, requested_m):
        if not math.isfinite(requested_m) or abs(requested_m) > 0.020:
            raise ValueError(f"Webots Cartesian {axis.upper()} MOVE is limited to 20 mm.")
        self.settle_cartesian()
        before = self.end_effector_position()
        desired = list(before)
        desired[("x", "y", "z").index(axis)] += requested_m
        original_targets = dict(self.targets)
        try:
            for _ in range(80):
                current = self.end_effector_position()
                error = tuple(goal - value for goal, value in zip(desired, current))
                if math.hypot(*error) <= 0.00025:
                    break
                steps = damped_xyz_step(self.xyz_columns(current), error)
                joint_steps = dict(
                    zip(("base_motor", "shoulder_motor", "elbow_motor"), steps)
                )
                proposed = {
                    name: self.targets[name] + step
                    for name, step in joint_steps.items()
                }
                limited_joints = [
                    name for name, value in proposed.items()
                    if not LIMITS[name][0] <= value <= LIMITS[name][1]
                ]
                if limited_joints:
                    raise ValueError(
                        "Cartesian target reaches Webots joint limit; "
                        "command rejected before further correction: "
                        + ", ".join(limited_joints)
                    )
                self.targets.update(proposed)
                self.set_targets()
                self.settle_cartesian()

            after = self.end_effector_position()
            report = (displacement_report(before, after, requested_m)
                      if axis == "x" else
                      axis_displacement_report(before, after, axis, requested_m))
            for index, other_axis in enumerate(("x", "y", "z")):
                if other_axis != axis:
                    drift = (after[index] - before[index]) * 1000.0
                    report[f"{other_axis}_drift_mm"] = drift
                    report["verified"] = bool(
                        report["verified"] and abs(drift) <= report["tolerance_mm"]
                    )
            self.show_motion_indicator(before, after, report)
            if not report["verified"]:
                raise ValueError(
                    "Cartesian MOVE verification failed: "
                    + json.dumps(report, sort_keys=True)
                )
        except Exception:
            self.targets = original_targets
            self.set_targets()
            self.advance(40)
            raise
        self.stopped = False
        return report

    def move(self, task):
        requested_axis = requested_axis_metres(task)
        if requested_axis is not None:
            axis, requested_m = requested_axis
            if axis == "x":
                return self.move_cartesian_x(requested_m)
            if axis == "y":
                return self.move_cartesian_y(requested_m)
            if axis == "z":
                return self.move_cartesian_z(requested_m)
        direction = str(task.get("direction", "")).lower()
        distance = float(task.get("distance", 10.0))
        unit = str(task.get("unit", "centimeters")).lower()
        scale = distance / 100.0 if unit.startswith("cent") else distance / 1000.0
        step = max(0.02, min(0.30, scale))
        if direction in {"forward", "ahead"}:
            self.targets["shoulder_motor"] -= step
        elif direction in {"backward", "back"}:
            self.targets["shoulder_motor"] += step
        elif direction in {"left"}:
            self.targets["base_motor"] += step
        elif direction in {"right"}:
            self.targets["base_motor"] -= step
        elif direction in {"up", "upward"}:
            self.targets["elbow_motor"] -= step
        elif direction in {"down", "downward"}:
            self.targets["elbow_motor"] += step
        elif task.get("position"):
            raise ValueError(
                "The visual model currently supports directional MOVE. "
                "Use forward, backward, left, right, up, or down."
            )
        else:
            raise ValueError("MOVE requires a supported direction.")
        self.stopped = False
        self.set_targets()

    def show_rotation_indicator(self, report):
        self.robot.setLabel(
            0,
            "ROTATION R: "
            + ("PASS" if report["verified"] else "FAIL")
            + "\n"
            f"Start R: {report['before_r_deg']:+.3f} deg\n"
            f"Final R: {report['after_r_deg']:+.3f} deg\n"
            f"Requested: {report['requested_r_deg']:+.3f} deg\n"
            f"Measured: {report['actual_r_deg']:+.3f} deg\n"
            f"Error: {report['error_deg']:+.3f} deg",
            0.015,
            0.04,
            0.035,
            0xFFFFFF,
            0.0,
            "Arial",
        )

    def rotate_wrist(self, requested_r_rad):
        hard_limit = math.radians(5.0)
        if abs(requested_r_rad) > hard_limit + 1e-12:
            raise ValueError("Webots R ROTATE is limited to 5 degrees.")
        self.advance(10)
        before_r = self.wrist_sensor.getValue()
        before_position = self.end_effector_position()
        target_r = before_r + requested_r_rad
        low, high = LIMITS["wrist_motor"]
        if not low <= target_r <= high:
            raise ValueError("Requested R rotation exceeds the wrist joint limit.")
        original_target = self.targets["wrist_motor"]

        self.targets["wrist_motor"] = target_r
        self.motors["wrist_motor"].setVelocity(0.3)
        self.motors["wrist_motor"].setPosition(target_r)
        for _ in range(120):
            if abs(self.wrist_sensor.getValue() - target_r) <= math.radians(0.10):
                break
            self.advance(1)
        self.advance(10)

        after_r = self.wrist_sensor.getValue()
        after_position = self.end_effector_position()
        report = angular_report(before_r, after_r, requested_r_rad)
        report["translation_drift_mm"] = math.dist(
            before_position, after_position
        ) * 1000.0
        report["verified"] = bool(
            report["verified"] and report["translation_drift_mm"] <= 0.75
        )
        self.show_rotation_indicator(report)
        if not report["verified"]:
            self.targets["wrist_motor"] = original_target
            self.motors["wrist_motor"].setPosition(original_target)
            self.advance(40)
            raise ValueError(
                "Wrist ROTATE verification failed: "
                + json.dumps(report, sort_keys=True)
            )
        self.stopped = False
        return report

    def execute_task(self, task):
        action = task.get("action")
        if action == "MOVE":
            report = self.move(task)
            if report is not None:
                return (
                    f"[WEBOTS] MOVE {report.get('axis', 'x').upper()} verified "
                    + json.dumps(report, sort_keys=True)
                )
        elif action == "ROTATE":
            requested_r_rad = requested_r_radians(task)
            if requested_r_rad is not None:
                report = self.rotate_wrist(requested_r_rad)
                return "[WEBOTS] ROTATE R verified " + json.dumps(
                    report, sort_keys=True
                )
            angle = math.radians(float(task.get("angle", 0.0)))
            direction = str(task.get("direction", "left")).lower()
            self.targets["base_motor"] += angle if direction == "left" else -angle
            self.stopped = False
            self.set_targets()
        elif action == "HOME":
            self.targets = dict(HOME)
            self.stopped = False
            self.set_targets()
        elif action == "STOP":
            self.stopped = True
            for motor in self.motors.values():
                motor.setVelocity(0.0)
        elif action == "GET_STATUS":
            pass
        else:
            raise ValueError(f"{action} is not supported by the visual model.")
        return f"[WEBOTS] {action} accepted"

    def status(self):
        position = self.end_effector_position()
        return {
            "ok": True,
            "state": "STOPPED" if self.stopped else "READY",
            "end_effector_position_mm": {
                axis: round(position[index] * 1000.0, 4)
                for index, axis in enumerate(("x", "y", "z"))
            },
            "measured_wrist_r_degrees": round(
                math.degrees(self.wrist_sensor.getValue()), 4
            ),
            "joint_targets_degrees": {
                name: round(math.degrees(value), 2)
                for name, value in self.targets.items()
            },
            "measured_joint_degrees": {
                name: round(math.degrees(sensor.getValue()), 2)
                for name, sensor in self.joint_sensors.items()
            },
        }

    def camera_frame(self):
        """Return one raw simulated camera frame for the application demo."""
        image = self.camera.getImage()
        if image is None:
            raise ValueError("Webots camera has not produced a frame yet.")
        return {
            "ok": True,
            "width": self.camera.getWidth(),
            "height": self.camera.getHeight(),
            "image_bgra_base64": base64.b64encode(image).decode("ascii"),
        }
    def calibration_references(self):
        """Expose known simulated marker coordinates for camera calibration."""
        markers = {}
        for name, node in self.calibration_markers.items():
            position = node.getPosition()
            markers[name] = {"x": position[0] * 1000.0, "y": position[1] * 1000.0}
        target_position = self.demo_target.getPosition()
        return {
            "ok": True,
            "coordinate_frame": "dobot_base",
            "markers": markers,
            "target_plane_z_mm": target_position[2] * 1000.0,
            "validation_target": {
                "x": target_position[0] * 1000.0,
                "y": target_position[1] * 1000.0,
            },
        }
def main():
    sim = Simulator()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(4)
    server.setblocking(False)
    print(f"Webots LLM controller listening on {HOST}:{PORT}")

    while sim.robot.step(TIME_STEP) != -1:
        try:
            client, _ = server.accept()
        except BlockingIOError:
            continue
        client.settimeout(1.0)
        with client:
            try:
                data = bytearray()
                while b"\n" not in data:
                    chunk = client.recv(65536)
                    if not chunk:
                        break
                    data.extend(chunk)
                request = json.loads(bytes(data).split(b"\n", 1)[0])
                if request.get("type") == "status":
                    response = sim.status()
                elif request.get("type") == "camera_frame":
                    response = sim.camera_frame()
                elif request.get("type") == "calibration_references":
                    response = sim.calibration_references()
                elif request.get("type") == "execute":
                    tasks = request.get("tasks", [])
                    if not isinstance(tasks, list) or not tasks:
                        raise ValueError("At least one task is required.")
                    results = [sim.execute_task(task) for task in tasks]
                    response = {
                        "ok": True,
                        "state": sim.status()["state"],
                        "results": results,
                        "status": sim.status(),
                    }
                else:
                    raise ValueError("Unknown request type.")
            except Exception as exc:
                response = {"ok": False, "error": str(exc)}
            try:
                client.sendall((json.dumps(response) + "\n").encode("utf-8"))
            except OSError as exc:
                # The caller may have reached an older/shorter timeout while this
                # measured move was completing. Keep the simulator controller alive.
                print(f"Webots response client disconnected: {type(exc).__name__}")


if __name__ == "__main__":
    main()
