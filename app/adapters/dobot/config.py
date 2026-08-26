"""Environment-backed DOBOT configuration with fail-closed movement safety."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

from app.adapters.dobot.exceptions import DobotConfigurationError, DobotSafetyError


class OperationMode(str, Enum):
    SIMULATION = "simulation"
    REAL = "real"


@dataclass(frozen=True)
class DobotPosition:
    x: float
    y: float
    z: float
    r: float

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "r": self.r}


@dataclass(frozen=True)
class SafetyLimits:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    min_r: float
    max_r: float

    def validate(self, position: DobotPosition) -> None:
        for axis in ("x", "y", "z", "r"):
            value = getattr(position, axis)
            lower = getattr(self, f"min_{axis}")
            upper = getattr(self, f"max_{axis}")
            if lower > upper:
                raise DobotConfigurationError(
                    f"DOBOT_{axis.upper()} minimum exceeds maximum."
                )
            if not lower <= value <= upper:
                raise DobotSafetyError(
                    f"Configured test {axis.upper()}={value} is outside "
                    f"the allowed range [{lower}, {upper}]."
                )


@dataclass(frozen=True)
class DobotConfig:
    mode: OperationMode = OperationMode.SIMULATION
    host: str = "127.0.0.1"
    rpc_port: int = 9090
    robot_port: str | None = None
    connect_timeout_seconds: float = 10.0
    command_timeout_ms: int = 30_000
    max_retries: int = 2
    ptp_mode: int = 1
    verification_timeout_seconds: float = 5.0
    verification_start_delay_seconds: float = 0.5
    position_tolerance_mm: float = 1.0
    rotation_tolerance_degrees: float = 1.0
    verification_samples: int = 3
    test_position: DobotPosition | None = None
    safety_limits: SafetyLimits | None = None

    @classmethod
    def from_env(
        cls, mode: OperationMode | str = OperationMode.SIMULATION
    ) -> "DobotConfig":
        load_dotenv()
        resolved_mode = OperationMode(mode)
        position_names = (
            "DOBOT_TEST_X", "DOBOT_TEST_Y", "DOBOT_TEST_Z", "DOBOT_TEST_R"
        )
        limit_names = (
            "DOBOT_MIN_X", "DOBOT_MAX_X", "DOBOT_MIN_Y", "DOBOT_MAX_Y",
            "DOBOT_MIN_Z", "DOBOT_MAX_Z", "DOBOT_MIN_R", "DOBOT_MAX_R",
        )

        def optional_group(names: tuple[str, ...], factory):
            values = [os.getenv(name) for name in names]
            if not any(value not in (None, "") for value in values):
                return None
            missing = [
                name for name, value in zip(names, values)
                if value in (None, "")
            ]
            if missing:
                raise DobotConfigurationError(
                    "Incomplete DOBOT configuration; missing: "
                    + ", ".join(missing)
                )
            try:
                return factory(*(float(value) for value in values))
            except ValueError as exc:
                raise DobotConfigurationError(
                    "DOBOT coordinates and limits must be numeric."
                ) from exc

        try:
            config = cls(
                mode=resolved_mode,
                host=os.getenv("DOBOTLINK_HOST", "127.0.0.1"),
                rpc_port=int(os.getenv("DOBOTLINK_PORT", "9090")),
                robot_port=os.getenv("DOBOT_PORT_NAME") or None,
                connect_timeout_seconds=float(
                    os.getenv("DOBOT_CONNECT_TIMEOUT_SECONDS", "10")
                ),
                command_timeout_ms=int(
                    os.getenv("DOBOT_COMMAND_TIMEOUT_MS", "30000")
                ),
                max_retries=int(os.getenv("DOBOT_MAX_RETRIES", "2")),
                ptp_mode=int(os.getenv("DOBOT_PTP_MODE", "1")),
                verification_timeout_seconds=float(
                    os.getenv("DOBOT_VERIFY_TIMEOUT_SECONDS", "5")
                ),
                verification_start_delay_seconds=float(
                    os.getenv("DOBOT_VERIFY_START_DELAY_SECONDS", "0.5")
                ),
                position_tolerance_mm=float(
                    os.getenv("DOBOT_POSITION_TOLERANCE_MM", "1")
                ),
                rotation_tolerance_degrees=float(
                    os.getenv("DOBOT_ROTATION_TOLERANCE_DEGREES", "1")
                ),
                verification_samples=int(
                    os.getenv("DOBOT_VERIFY_SAMPLES", "3")
                ),
                test_position=optional_group(position_names, DobotPosition),
                safety_limits=optional_group(limit_names, SafetyLimits),
            )
        except ValueError as exc:
            raise DobotConfigurationError(
                f"Invalid DOBOT configuration: {exc}"
            ) from exc

        if config.max_retries < 0 or config.max_retries > 5:
            raise DobotConfigurationError(
                "DOBOT_MAX_RETRIES must be between 0 and 5."
            )
        if config.connect_timeout_seconds <= 0 or config.command_timeout_ms <= 0:
            raise DobotConfigurationError("DOBOT timeouts must be positive.")
        if config.verification_timeout_seconds <= 0:
            raise DobotConfigurationError(
                "DOBOT_VERIFY_TIMEOUT_SECONDS must be positive."
            )
        if config.verification_start_delay_seconds < 0:
            raise DobotConfigurationError(
                "DOBOT_VERIFY_START_DELAY_SECONDS cannot be negative."
            )
        if (
            config.position_tolerance_mm <= 0
            or config.rotation_tolerance_degrees <= 0
        ):
            raise DobotConfigurationError(
                "DOBOT verification tolerances must be positive."
            )
        if not 1 <= config.verification_samples <= 10:
            raise DobotConfigurationError(
                "DOBOT_VERIFY_SAMPLES must be between 1 and 10."
            )
        if config.ptp_mode not in {0, 1, 2}:
            raise DobotConfigurationError(
                "Real Cartesian MOVE requires DOBOT_PTP_MODE 0, 1, or 2."
            )
        return config

    def require_safe_test_position(self) -> DobotPosition:
        if self.test_position is None:
            raise DobotConfigurationError(
                "MOVE is disabled until DOBOT_TEST_X/Y/Z/R are all configured."
            )
        if self.safety_limits is None:
            raise DobotConfigurationError(
                "MOVE is disabled until all DOBOT_MIN/MAX_X/Y/Z/R limits "
                "are configured."
            )
        self.safety_limits.validate(self.test_position)
        return self.test_position
