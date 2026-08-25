"""Bounded DobotLink RPC client for the installed Magician Lite interface."""

from __future__ import annotations

import asyncio
import importlib
import sys
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
    DobotTimeoutError,
)


class ConnectionState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    READY = "READY"
    ERROR = "ERROR"


BackendFactory = Callable[[DobotConfig], Any]


class DobotLinkClient:
    """Owns lifecycle and translates transport failures into stable errors."""

    def __init__(
        self,
        config: DobotConfig,
        backend_factory: BackendFactory | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.state = ConnectionState.DISCONNECTED
        self.port_name: str | None = None
        self._backend: Any = None
        self._backend_factory = backend_factory or self._load_dobotlink_backend
        self._sleep = sleep
        self.last_error: str | None = None

    @staticmethod
    def _load_dobotlink_backend(config: DobotConfig) -> Any:
        if config.mode is not OperationMode.REAL:
            raise DobotConfigurationError("DobotLink may only be loaded in real mode.")
        if config.sdk_path and config.sdk_path not in sys.path:
            sys.path.insert(0, config.sdk_path)
        try:
            module = importlib.import_module("DobotRPC")
        except ImportError as exc:
            raise DobotConfigurationError(
                "DobotRPC is unavailable. Install the verified optional SDK or set "
                "DOBOT_SDK_PATH to its site-packages directory."
            ) from exc

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            rpc = module.RPCClient(ip=config.host, port=config.rpc_port, loop=loop)
            loop.run_until_complete(
                asyncio.wait_for(
                    rpc.wait_for_connected(),
                    timeout=config.connect_timeout_seconds,
                )
            )
            return module.DobotlinkAdapter(rpc, is_sync=True)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise DobotTimeoutError(
                f"Timed out connecting to DobotLink at {config.host}:{config.rpc_port}."
            ) from exc
        except DobotError:
            raise
        except Exception as exc:
            raise DobotConnectionError(f"Could not initialize DobotLink: {exc}") from exc

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
                            f"Configured DOBOT_PORT_NAME {self.config.robot_port!r} was not detected."
                        )
                    self.port_name = self.config.robot_port
                elif len(detected) == 1:
                    self.port_name = detected[0]
                elif not detected:
                    raise DobotConnectionError("No Magician Lite was detected by DobotLink.")
                else:
                    raise DobotConfigurationError(
                        "Multiple DOBOT devices detected; set DOBOT_PORT_NAME explicitly."
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
                    raise DobotConnectionError(f"Connected but GetPose failed: {pose!r}")
                self.state = ConnectionState.READY
                return
            except DobotConfigurationError:
                self.state = ConnectionState.ERROR
                raise
            except Exception as exc:
                self.last_error = str(exc)
                self.state = ConnectionState.ERROR
                if attempt >= self.config.max_retries:
                    if isinstance(exc, DobotError):
                        raise
                    raise DobotConnectionError(
                        f"DOBOT connection failed after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                self.state = ConnectionState.CONNECTING
                self._sleep(min(0.5 * (attempt + 1), 2.0))

    def disconnect(self) -> None:
        try:
            if self._backend is not None and self.port_name:
                self._backend.MagicianLite.DisconnectDobot(
                    portName=self.port_name,
                    queueStop=True,
                    queueClear=True,
                    isQueued=False,
                )
        except Exception as exc:
            self.last_error = str(exc)
            self.state = ConnectionState.ERROR
            raise DobotConnectionError(f"DisconnectDobot failed: {exc}") from exc
        finally:
            self._backend = None
            self.port_name = None
            if self.state is not ConnectionState.ERROR:
                self.state = ConnectionState.DISCONNECTED

    def is_connected(self) -> bool:
        return self.state in {ConnectionState.CONNECTED, ConnectionState.READY}

    def _module(self) -> Any:
        if self.state is not ConnectionState.READY or self._backend is None or not self.port_name:
            raise DobotNotReadyError(f"DOBOT is not READY (state={self.state.value}).")
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

    def move(self, position: DobotPosition) -> Any:
        return self._module().SetPTPCmd(
            portName=self.port_name,
            ptpMode=self.config.ptp_mode,
            x=position.x, y=position.y, z=position.z, r=position.r,
            isQueued=True,
            isWaitForFinish=True,
            timeout=self.config.command_timeout_ms,
        )

    def stop(self) -> Any:
        return self._module().QueuedCmdStop(
            portName=self.port_name, forceStop=True, isQueued=False
        )

    def set_gripper(self, on: bool) -> Any:
        module = self._module()
        effector = module.GetEndEffectorType(
            portName=self.port_name, isQueued=False
        )
        payload = effector.get("result", effector) if isinstance(effector, dict) else effector
        if isinstance(payload, dict):
            payload = payload.get("type")
        if payload not in (2, "2", "gripper", "Gripper"):
            raise DobotEndEffectorError(
                f"A gripper was not detected (end-effector response: {effector!r})."
            )
        return module.SetEndEffectorGripper(
            portName=self.port_name,
            enable=True,
            on=on,
            isQueued=True,
        )
