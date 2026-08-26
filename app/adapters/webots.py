"""TCP adapter for the Webots visual simulator.

This module never imports DOBOT code and cannot communicate with physical hardware.
"""

from __future__ import annotations

import json
import socket
from typing import Any

from app.adapters.base import RobotAdapter, RobotAdapterError
from app.commands.models import UniversalCommand
from app.commands.validator import CommandValidationError, validate_command


class WebotsClient:
    """Small newline-delimited JSON client for the Webots controller."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = (json.dumps(payload) + "\n").encode("utf-8")
        try:
            with socket.create_connection((self.host, self.port), self.timeout) as connection:
                connection.settimeout(self.timeout)
                connection.sendall(encoded)
                chunks = bytearray()
                while b"\n" not in chunks:
                    part = connection.recv(65536)
                    if not part:
                        raise RobotAdapterError("Webots closed the connection without a response.")
                    chunks.extend(part)
        except (OSError, TimeoutError) as exc:
            raise RobotAdapterError(
                f"Could not connect to Webots at {self.host}:{self.port}. "
                "Open simulation/webots/worlds/magician_lite.wbt and press Play."
            ) from exc

        try:
            response = json.loads(bytes(chunks).split(b"\n", 1)[0])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RobotAdapterError("Webots returned an invalid response.") from exc
        if not isinstance(response, dict):
            raise RobotAdapterError("Webots response must be a JSON object.")
        if not response.get("ok", False):
            raise RobotAdapterError(str(response.get("error", "Webots rejected the command.")))
        return response


class WebotsRobotAdapter(RobotAdapter):
    """Runs validated universal commands in a visual, non-hardware simulator."""

    simulated = True

    def __init__(self, robot, client: WebotsClient | None = None) -> None:
        super().__init__(robot)
        self.client = client or WebotsClient()

    def validate(self, command: UniversalCommand) -> bool:
        try:
            validate_command(command)
        except CommandValidationError:
            return False
        return True

    def prepare(self, command: UniversalCommand) -> list[dict[str, Any]]:
        return [
            task.model_dump(mode="json", exclude_none=True)
            for task in command.tasks
        ]

    def execute(self, command: UniversalCommand) -> list[str]:
        response = self.client.request({"type": "execute", "tasks": self.prepare(command)})
        results = response.get("results")
        if not isinstance(results, list) or not all(isinstance(item, str) for item in results):
            raise RobotAdapterError("Webots response did not contain text results.")
        return results

    def get_status(self) -> str:
        response = self.client.request({"type": "status"})
        return str(response.get("state", "UNKNOWN"))
