"""Controller for the simplified Magician Lite Webots model.

The model is for visual software testing only. It is not a calibrated digital twin.
"""

import json
import math
import socket

from controller import Supervisor

from cartesian_motion import (
    axis_displacement_report,
    damped_scalar_step,
    damped_xz_step,
    displacement_report,
    requested_axis_metres,
)


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

    def xz_columns(self, end_position):
        orientation = self.nodes["ARM_BASE"].getOrientation()
        axis = (orientation[1], orientation[4], orientation[7])

        def column(pivot):
            radius = tuple(a - b for a, b in zip(end_position, pivot))
            cross = (
                axis[1] * radius[2] - axis[2] * radius[1],
                axis[2] * radius[0] - axis[0] * radius[2],
                axis[0] * radius[1] - axis[1] * radius[0],
            )
            return cross[0], cross[2]

        return (
            column(self.nodes["SHOULDER_LINK"].getPosition()),
            column(self.nodes["ELBOW_LINK"].getPosition()),
        )

    def base_y_jacobian(self, end_position):
        """Return d(end-effector Y)/d(base angle) in world coordinates."""
        orientation = self.nodes["ARM_BASE"].getOrientation()
        axis = (orientation[2], orientation[5], orientation[8])
        pivot = self.nodes["ARM_BASE"].getPosition()
        radius = tuple(a - b for a, b in zip(end_position, pivot))
        cross = (
            axis[1] * radius[2] - axis[2] * radius[1],
            axis[2] * radius[0] - axis[0] * radius[2],
            axis[0] * radius[1] - axis[1] * radius[0],
        )
        return cross[1]

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
        if abs(requested_x_m) > 0.020:
            raise ValueError("Webots Cartesian X MOVE is limited to 20 mm.")
        self.advance(20)
        before = self.end_effector_position()
        desired = (before[0] + requested_x_m, before[2])
        original_targets = dict(self.targets)

        for _ in range(80):
            current = self.end_effector_position()
            error = (desired[0] - current[0], desired[1] - current[2])
            if math.hypot(*error) <= 0.00075:
                break
            shoulder_step, elbow_step = damped_xz_step(
                self.xz_columns(current), error
            )
            self.targets["shoulder_motor"] += shoulder_step
            self.targets["elbow_motor"] += elbow_step
            self.set_targets()
            self.advance(8)

        self.advance(20)
        after = self.end_effector_position()
        report = displacement_report(before, after, requested_x_m)
        report["y_drift_mm"] = (after[1] - before[1]) * 1000.0
        report["z_drift_mm"] = (after[2] - before[2]) * 1000.0
        report["verified"] = bool(
            report["verified"]
            and abs(report["y_drift_mm"]) <= report["tolerance_mm"]
            and abs(report["z_drift_mm"]) <= report["tolerance_mm"]
        )
        self.show_motion_indicator(before, after, report)
        if not report["verified"]:
            self.targets = original_targets
            self.set_targets()
            self.advance(40)
            raise ValueError(
                "Cartesian MOVE verification failed: "
                + json.dumps(report, sort_keys=True)
            )
        self.stopped = False
        return report

    def move_cartesian_z(self, requested_z_m):
        if abs(requested_z_m) > 0.020:
            raise ValueError("Webots Cartesian Z MOVE is limited to 20 mm.")
        self.advance(20)
        before = self.end_effector_position()
        desired = (before[0], before[2] + requested_z_m)
        original_targets = dict(self.targets)

        for _ in range(80):
            current = self.end_effector_position()
            error = (desired[0] - current[0], desired[1] - current[2])
            if math.hypot(*error) <= 0.00075:
                break
            shoulder_step, elbow_step = damped_xz_step(
                self.xz_columns(current), error
            )
            self.targets["shoulder_motor"] += shoulder_step
            self.targets["elbow_motor"] += elbow_step
            self.set_targets()
            self.advance(8)

        self.advance(20)
        after = self.end_effector_position()
        report = axis_displacement_report(before, after, "z", requested_z_m)
        report["x_drift_mm"] = (after[0] - before[0]) * 1000.0
        report["y_drift_mm"] = (after[1] - before[1]) * 1000.0
        report["verified"] = bool(
            report["verified"]
            and abs(report["x_drift_mm"]) <= report["tolerance_mm"]
            and abs(report["y_drift_mm"]) <= report["tolerance_mm"]
        )
        self.show_motion_indicator(before, after, report)
        if not report["verified"]:
            self.targets = original_targets
            self.set_targets()
            self.advance(40)
            raise ValueError(
                "Cartesian MOVE verification failed: "
                + json.dumps(report, sort_keys=True)
            )
        self.stopped = False
        return report

    def move_cartesian_y(self, requested_y_m):
        if abs(requested_y_m) > 0.020:
            raise ValueError("Webots Cartesian Y MOVE is limited to 20 mm.")
        self.advance(20)
        before = self.end_effector_position()
        desired_y = before[1] + requested_y_m
        original_targets = dict(self.targets)

        for _ in range(80):
            current = self.end_effector_position()
            error_y = desired_y - current[1]
            if abs(error_y) <= 0.00075:
                break
            base_step = damped_scalar_step(
                self.base_y_jacobian(current), error_y
            )
            self.targets["base_motor"] += base_step
            self.set_targets()
            self.advance(8)

        self.advance(20)
        after = self.end_effector_position()
        report = axis_displacement_report(before, after, "y", requested_y_m)
        report["x_drift_mm"] = (after[0] - before[0]) * 1000.0
        report["z_drift_mm"] = (after[2] - before[2]) * 1000.0
        report["verified"] = bool(
            report["verified"]
            and abs(report["x_drift_mm"]) <= report["tolerance_mm"]
            and abs(report["z_drift_mm"]) <= report["tolerance_mm"]
        )
        self.show_motion_indicator(before, after, report)
        if not report["verified"]:
            self.targets = original_targets
            self.set_targets()
            self.advance(40)
            raise ValueError(
                "Cartesian MOVE verification failed: "
                + json.dumps(report, sort_keys=True)
            )
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
            "joint_targets_degrees": {
                name: round(math.degrees(value), 2)
                for name, value in self.targets.items()
            },
        }


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
        client.sendall((json.dumps(response) + "\n").encode("utf-8"))
