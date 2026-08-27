"""Bounded JSON-RPC client for DobotLink's Magician Lite module."""

from __future__ import annotations

import json
import time
from enum import Enum
from typing import Any, Callable

from app.adapters.dobot.config import DobotConfig, DobotPosition, OperationMode
from app.adapters.dobot.exceptions import (
    DobotConfigurationError,
    DobotConnectionError,
    DobotEndEffectorError,
    DobotError,
    DobotNotReadyError,
    DobotSafetyError,
    DobotTimeoutError,
)


class ConnectionState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    READY = "READY"
    ERROR = "ERROR"


class _JsonRpcModule:
    def __init__(self, backend: "_WebSocketDobotLinkBackend", module: str) -> None:
        self._backend = backend
        self._module = module

    def __getattr__(self, method: str):
        def call(**params):
            return self._backend.call(
                f"dobotlink.{self._module}.{method}", params
            )
        return call


class _WebSocketDobotLinkBackend:
    """Small synchronous JSON-RPC 2.0 transport matching installed DobotLink."""

    def __init__(self, config: DobotConfig) -> None:
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise DobotConfigurationError(
                "Real mode requires websockets>=13; install requirements-hardware.txt."
            ) from exc
        try:
            self._socket = connect(
                f"ws://{config.host}:{config.rpc_port}",
                open_timeout=config.connect_timeout_seconds,
                close_timeout=2,
            )
        except TimeoutError as exc:
            raise DobotTimeoutError(
                f"Timed out connecting to DobotLink at "
                f"{config.host}:{config.rpc_port}."
            ) from exc
        except Exception as exc:
            raise DobotConnectionError(
                f"Could not connect to DobotLink at "
                f"{config.host}:{config.rpc_port}: {exc}"
            ) from exc
        self._next_id = 0
        self._timeout = config.command_timeout_ms / 1000
        self.MagicianLite = _JsonRpcModule(self, "MagicianLite")

    def call(self, method: str, params: dict[str, Any]) -> Any:
        self._next_id += 1
        request_id = self._next_id
        packet = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        try:
            self._socket.send(json.dumps(packet))
            while True:
                response = json.loads(self._socket.recv(timeout=self._timeout))
                if response.get("id") != request_id:
                    continue
                if response.get("error") is not None:
                    raise DobotError(
                        f"DobotLink {method} failed: {response['error']!r}"
                    )
                return response.get("result")
        except TimeoutError as exc:
            raise DobotTimeoutError(f"DobotLink {method} timed out.") from exc
        except DobotError:
            raise
        except Exception as exc:
            raise DobotConnectionError(
                f"DobotLink communication failed during {method}: {exc}"
            ) from exc

    def close(self) -> None:
        self._socket.close()


BackendFactory = Callable[[DobotConfig], Any]


class DobotLinkClient:
    """Owns lifecycle and translates transport failures into stable errors."""

    def __init__(
        self,
        config: DobotConfig,
        backend_factory: BackendFactory | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.state = ConnectionState.DISCONNECTED
        self.port_name: str | None = None
        self._backend: Any = None
        self._backend_factory = backend_factory or _WebSocketDobotLinkBackend
        self._sleep = sleep
        self._clock = clock
        self.last_error: str | None = None

    @staticmethod
    def _result_failed(result: Any) -> bool:
        if result is False or result is None:
            return True
        if isinstance(result, dict):
            if result.get("error"):
                return True
            code = result.get("code")
            return isinstance(code, (int, float)) and code < 0
        return False

    @staticmethod
    def _extract_ports(result: Any) -> list[str]:
        payload = result.get("result", result) if isinstance(result, dict) else result
        if not isinstance(payload, list):
            return []
        ports: list[str] = []
        for item in payload:
            if isinstance(item, str):
                ports.append(item)
            elif isinstance(item, dict):
                port = item.get("portName") or item.get("port")
                if port:
                    ports.append(str(port))
        return ports

    def _close_backend(self) -> None:
        close = getattr(self._backend, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        self._backend = None

    def connect(self) -> None:
        if self.config.mode is not OperationMode.REAL:
            raise DobotConfigurationError("Simulation mode never connects to DobotLink.")
        if self.state is ConnectionState.READY:
            return

        self.state = ConnectionState.CONNECTING
        self.last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                self._backend = self._backend_factory(self.config)
                module = self._backend.MagicianLite
                detected = self._extract_ports(module.SearchDobot())
                if self.config.robot_port:
                    if detected and self.config.robot_port not in detected:
                        raise DobotConnectionError(
                            f"Configured DOBOT_PORT_NAME {self.config.robot_port!r} "
                            "was not detected."
                        )
                    self.port_name = self.config.robot_port
                elif len(detected) == 1:
                    self.port_name = detected[0]
                elif not detected:
                    raise DobotConnectionError(
                        "No Magician Lite was detected by DobotLink."
                    )
                else:
                    raise DobotConfigurationError(
                        "Multiple DOBOT devices detected; "
                        "set DOBOT_PORT_NAME explicitly."
                    )

                result = module.ConnectDobot(
                    portName=self.port_name,
                    queueStart=True,
                    isQueued=False,
                )
                if self._result_failed(result):
                    raise DobotConnectionError(f"ConnectDobot failed: {result!r}")
                self.state = ConnectionState.CONNECTED
                pose = module.GetPose(portName=self.port_name, isQueued=False)
                if self._result_failed(pose):
                    raise DobotConnectionError(
                        f"Connected but GetPose failed: {pose!r}"
                    )
                self.state = ConnectionState.READY
                self.last_error = None
                return
            except DobotConfigurationError:
                self._close_backend()
                self.state = ConnectionState.ERROR
                raise
            except Exception as exc:
                self.last_error = str(exc)
                self._close_backend()
                self.state = ConnectionState.ERROR
                if attempt >= self.config.max_retries:
                    if isinstance(exc, DobotError):
                        raise
                    raise DobotConnectionError(
                        f"DOBOT connection failed after {attempt + 1} "
                        f"attempt(s): {exc}"
                    ) from exc
                self.state = ConnectionState.CONNECTING
                self._sleep(min(0.5 * (attempt + 1), 2.0))

    def disconnect(self) -> None:
        error: Exception | None = None
        try:
            if self._backend is not None and self.port_name:
                self._backend.MagicianLite.DisconnectDobot(
                    portName=self.port_name,
                    queueStop=True,
                    queueClear=True,
                    isQueued=False,
                )
        except Exception as exc:
            error = exc
            self.last_error = str(exc)
        finally:
            self._close_backend()
            self.port_name = None
            self.state = (
                ConnectionState.ERROR if error else ConnectionState.DISCONNECTED
            )
        if error:
            raise DobotConnectionError(f"DisconnectDobot failed: {error}") from error

    def is_connected(self) -> bool:
        return self.state in {ConnectionState.CONNECTED, ConnectionState.READY}

    def _module(self) -> Any:
        if (
            self.state is not ConnectionState.READY
            or self._backend is None
            or not self.port_name
        ):
            raise DobotNotReadyError(
                f"DOBOT is not READY (state={self.state.value})."
            )
        return self._backend.MagicianLite

    def get_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "state": self.state.value,
            "connected": self.is_connected(),
            "port_name": self.port_name,
            "last_error": self.last_error,
        }
        if self.state is ConnectionState.READY:
            status["pose"] = self._module().GetPose(
                portName=self.port_name, isQueued=False
            )
        return status

    def home(self) -> Any:
        return self._module().SetHOMECmd(
            portName=self.port_name,
            isQueued=True,
            isWaitForFinish=True,
            timeout=self.config.command_timeout_ms,
        )

    @staticmethod
    def _position_from_pose(pose: Any) -> DobotPosition:
        payload = pose.get("result", pose) if isinstance(pose, dict) else pose
        if not isinstance(payload, dict):
            raise DobotSafetyError(f"Invalid GetPose response: {pose!r}")
        try:
            return DobotPosition(
                x=float(payload["x"]),
                y=float(payload["y"]),
                z=float(payload["z"]),
                r=float(payload["r"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DobotSafetyError(
                f"Incomplete GetPose response: {pose!r}"
            ) from exc

    def _read_position(self) -> DobotPosition:
        pose = self._module().GetPose(
            portName=self.port_name, isQueued=False
        )
        return self._position_from_pose(pose)

    def _target_matches(
        self, actual: DobotPosition, target: DobotPosition
    ) -> bool:
        return (
            abs(actual.x - target.x) <= self.config.position_tolerance_mm
            and abs(actual.y - target.y) <= self.config.position_tolerance_mm
            and abs(actual.z - target.z) <= self.config.position_tolerance_mm
            and abs(actual.r - target.r)
            <= self.config.rotation_tolerance_degrees
        )

    def _stop_and_clear_queue(self) -> None:
        module = self._backend.MagicianLite
        try:
            module.QueuedCmdStop(
                portName=self.port_name, forceStop=True, isQueued=False
            )
        finally:
            module.QueuedCmdClear(
                portName=self.port_name, isQueued=False
            )

    def move(self, position: DobotPosition) -> dict[str, Any]:
        module = self._module()
        before = self._read_position()
        try:
            accepted = module.SetPTPCmd(
                portName=self.port_name,
                ptpMode=self.config.ptp_mode,
                x=position.x,
                y=position.y,
                z=position.z,
                r=position.r,
                isQueued=True,
                # DobotLink may never send the completion response for queued PTP
                # commands. Accept the queue response immediately, then verify the
                # live pose below with a bounded deadline.
                isWaitForFinish=False,
            )
        except Exception as exc:
            stop_error = None
            try:
                self._stop_and_clear_queue()
            except Exception as cleanup_exc:
                stop_error = cleanup_exc
            self.state = ConnectionState.ERROR
            message = (
                "SetPTPCmd failed before completion confirmation; software stop "
                f"and queue clear attempted. error={exc!r}"
            )
            if stop_error is not None:
                message += f", cleanup_error={stop_error!r}"
            self.last_error = message
            raise DobotSafetyError(message) from exc
        if self._result_failed(accepted):
            self._stop_and_clear_queue()
            raise DobotSafetyError(
                f"SetPTPCmd was not accepted: {accepted!r}"
            )

        self._sleep(self.config.verification_start_delay_seconds)
        deadline = self._clock() + self.config.verification_timeout_seconds
        matching_samples = 0
        final = before
        while self._clock() <= deadline:
            final = self._read_position()
            if self._target_matches(final, position):
                matching_samples += 1
                if matching_samples >= self.config.verification_samples:
                    module.QueuedCmdClear(
                        portName=self.port_name, isQueued=False
                    )
                    return {
                        "verified": True,
                        "accepted": accepted,
                        "before": before.as_dict(),
                        "target": position.as_dict(),
                        "final": final.as_dict(),
                    }
            else:
                matching_samples = 0
            self._sleep(0.1)

        try:
            self._stop_and_clear_queue()
        finally:
            self.state = ConnectionState.ERROR
        message = (
            "MOVE final-pose verification failed; software stop and queue "
            f"clear issued. before={before.as_dict()}, "
            f"target={position.as_dict()}, final={final.as_dict()}"
        )
        self.last_error = message
        raise DobotSafetyError(message)

    def calibration_preview(
        self, axis: str, delta_mm: float
    ) -> tuple[DobotPosition, DobotPosition]:
        """Create a bounded single-axis target from the live Cartesian pose."""
        normalized_axis = axis.lower()
        if normalized_axis not in {"x", "y", "z"}:
            raise DobotSafetyError("Calibration axis must be X, Y, or Z.")
        if delta_mm == 0 or abs(delta_mm) > self.config.calibration_max_step_mm:
            raise DobotSafetyError(
                "Calibration movement must be non-zero and no more than "
                f"{self.config.calibration_max_step_mm:g} mm."
            )
        if self.config.safety_limits is None:
            raise DobotConfigurationError(
                "Calibration is disabled until all DOBOT_MIN/MAX_X/Y/Z/R "
                "limits are configured."
            )
        before = self._read_position()
        values = before.as_dict()
        values[normalized_axis] += delta_mm
        target = DobotPosition(**values)
        self.config.safety_limits.validate(target)
        return before, target

    def calibrate(
        self,
        axis: str,
        delta_mm: float,
        expected_before: DobotPosition,
    ) -> dict[str, Any]:
        """Execute one low-speed, single-axis calibration move."""
        current = self._read_position()
        if not self._target_matches(current, expected_before):
            raise DobotSafetyError(
                "Robot pose changed after confirmation; calibration cancelled. "
                f"preview={expected_before.as_dict()}, current={current.as_dict()}"
            )
        normalized_axis = axis.lower()
        values = current.as_dict()
        values[normalized_axis] += delta_mm
        target = DobotPosition(**values)
        if self.config.safety_limits is None:
            raise DobotConfigurationError("Calibration safety limits are missing.")
        self.config.safety_limits.validate(target)

        module = self._module()
        speed_result = module.SetPTPCommonParams(
            portName=self.port_name,
            velocityRatio=self.config.calibration_speed_ratio,
            accelerationRatio=self.config.calibration_acceleration_ratio,
            isQueued=False,
        )
        if self._result_failed(speed_result):
            self._stop_and_clear_queue()
            raise DobotSafetyError(
                f"Low-speed calibration setup failed: {speed_result!r}"
            )
        result = self.move(target)
        result.update({
            "calibration_axis": normalized_axis,
            "requested_delta_mm": delta_mm,
            "speed_ratio": self.config.calibration_speed_ratio,
            "acceleration_ratio": self.config.calibration_acceleration_ratio,
        })
        return result

    def stop(self) -> Any:
        return self._module().QueuedCmdStop(
            portName=self.port_name,
            forceStop=True,
            isQueued=False,
        )

    def set_gripper(self, on: bool) -> Any:
        module = self._module()
        effector = module.GetEndEffectorType(
            portName=self.port_name, isQueued=False
        )
        payload = (
            effector.get("result", effector)
            if isinstance(effector, dict)
            else effector
        )
        if isinstance(payload, dict):
            payload = payload.get("type")
        if payload not in (2, "2", "gripper", "Gripper"):
            raise DobotEndEffectorError(
                "A gripper was not detected "
                f"(end-effector response: {effector!r})."
            )
        return module.SetEndEffectorGripper(
            portName=self.port_name,
            enable=True,
            on=on,
            isQueued=True,
        )
