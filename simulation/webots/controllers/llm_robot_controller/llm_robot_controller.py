"""Controller for the simplified Magician Lite Webots model.

The model is for visual software testing only. It is not a calibrated digital twin.
"""

import json
import math
import socket

from controller import Robot


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
        self.robot = Robot()
        self.motors = {name: self.robot.getDevice(name) for name in LIMITS}
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

    def move(self, task):
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
            self.move(task)
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
        return {
            "ok": True,
            "state": "STOPPED" if self.stopped else "READY",
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
                response = {"ok": True, "state": sim.status()["state"], "results": results}
            else:
                raise ValueError("Unknown request type.")
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        client.sendall((json.dumps(response) + "\n").encode("utf-8"))
