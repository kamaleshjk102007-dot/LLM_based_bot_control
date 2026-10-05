"""Environment-backed DOBOT configuration with fail-closed movement safety."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

from app.adapters.dobot.exceptions import (
    DobotConfigurationError,
    DobotSafetyError,
)


class OperationMode(str, Enum):
    SIMULATION = "simulation"
    REAL = "real"


@dataclass(frozen=True)
class DobotPosition:
    """Cartesian position of the DOBOT Magician Lite."""

    x: float
    y: float
    z: float
    r: float

    def as_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "r": self.r,
        }


@dataclass(frozen=True)
class SafetyLimits:
    """Configured safety envelope for the physical DOBOT."""

    min_x: float
    max_x: float

    min_y: float
    max_y: float

    min_z: float
    max_z: float

    min_r: float
    max_r: float

    def validate(
        self,
        position: DobotPosition,
        *,
        label: str = "Position",
    ) -> None:
        """
        Validate a position against the configured safety envelope.

        This method is intentionally fail-closed:
        if even one axis is outside the configured range,
        the movement is rejected.
        """

        for axis in ("x", "y", "z", "r"):
            value = getattr(position, axis)

            lower = getattr(
                self,
                f"min_{axis}",
            )

            upper = getattr(
                self,
                f"max_{axis}",
            )

            # Configuration itself must be valid.
            if lower > upper:
                raise DobotConfigurationError(
                    f"DOBOT_{axis.upper()} minimum exceeds maximum."
                )

            # Position must remain inside the safety envelope.
            if not lower <= value <= upper:
                raise DobotSafetyError(
                    f"{label} {axis.upper()}={value} is outside "
                    f"the allowed range [{lower}, {upper}]."
                )


@dataclass(frozen=True)
class DobotConfig:
    """Complete environment-backed DOBOT configuration."""

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

    calibration_max_step_mm: float = 5.0

    rotation_max_step_degrees: float = 5.0

    calibration_speed_ratio: float = 5.0

    calibration_acceleration_ratio: float = 5.0

    hover_height_mm: float = 40.0

    # Explicit physical test position.
    test_position: DobotPosition | None = None

    # Explicit physical safety envelope.
    safety_limits: SafetyLimits | None = None

    @classmethod
    def from_env(
        cls,
        mode: OperationMode | str = OperationMode.SIMULATION,
    ) -> "DobotConfig":
        """
        Load DOBOT configuration from environment variables.

        Missing test-position and safety-limit groups are allowed
        during configuration loading, but real MOVE/calibration
        operations must explicitly reject missing safety settings.
        """

        load_dotenv()

        resolved_mode = OperationMode(mode)

        # ---------------------------------------------------------
        # TEST POSITION
        # ---------------------------------------------------------

        position_names = (
            "DOBOT_TEST_X",
            "DOBOT_TEST_Y",
            "DOBOT_TEST_Z",
            "DOBOT_TEST_R",
        )

        # ---------------------------------------------------------
        # SAFETY LIMITS
        # ---------------------------------------------------------

        limit_names = (
            "DOBOT_MIN_X",
            "DOBOT_MAX_X",
            "DOBOT_MIN_Y",
            "DOBOT_MAX_Y",
            "DOBOT_MIN_Z",
            "DOBOT_MAX_Z",
            "DOBOT_MIN_R",
            "DOBOT_MAX_R",
        )

        def optional_group(
            names: tuple[str, ...],
            factory,
        ):
            """
            Load an all-or-nothing group of environment variables.

            Behavior:

            Nothing configured:
                return None

            Partially configured:
                raise configuration error

            Everything configured:
                parse and return the requested object
            """

            values = [
                os.getenv(name)
                for name in names
            ]

            # Nothing configured at all.
            if not any(
                value not in (None, "")
                for value in values
            ):
                return None

            # Some values configured but others missing.
            missing = [
                name
                for name, value in zip(
                    names,
                    values,
                )
                if value in (None, "")
            ]

            if missing:
                raise DobotConfigurationError(
                    "Incomplete DOBOT configuration; missing: "
                    + ", ".join(missing)
                )

            # Convert all values to float.
            try:
                return factory(
                    *(
                        float(value)
                        for value in values
                    )
                )

            except (TypeError, ValueError) as exc:
                raise DobotConfigurationError(
                    "DOBOT coordinates and limits must be numeric."
                ) from exc

        # ---------------------------------------------------------
        # CREATE CONFIGURATION
        # ---------------------------------------------------------

        try:
            config = cls(
                mode=resolved_mode,

                host=os.getenv(
                    "DOBOTLINK_HOST",
                    "127.0.0.1",
                ),

                rpc_port=int(
                    os.getenv(
                        "DOBOTLINK_PORT",
                        "9090",
                    )
                ),

                robot_port=(
                    os.getenv("DOBOT_PORT_NAME")
                    or None
                ),

                connect_timeout_seconds=float(
                    os.getenv(
                        "DOBOT_CONNECT_TIMEOUT_SECONDS",
                        "10",
                    )
                ),

                command_timeout_ms=int(
                    os.getenv(
                        "DOBOT_COMMAND_TIMEOUT_MS",
                        "30000",
                    )
                ),

                max_retries=int(
                    os.getenv(
                        "DOBOT_MAX_RETRIES",
                        "2",
                    )
                ),

                ptp_mode=int(
                    os.getenv(
                        "DOBOT_PTP_MODE",
                        "1",
                    )
                ),

                verification_timeout_seconds=float(
                    os.getenv(
                        "DOBOT_VERIFY_TIMEOUT_SECONDS",
                        "5",
                    )
                ),

                verification_start_delay_seconds=float(
                    os.getenv(
                        "DOBOT_VERIFY_START_DELAY_SECONDS",
                        "0.5",
                    )
                ),

                position_tolerance_mm=float(
                    os.getenv(
                        "DOBOT_POSITION_TOLERANCE_MM",
                        "1",
                    )
                ),

                rotation_tolerance_degrees=float(
                    os.getenv(
                        "DOBOT_ROTATION_TOLERANCE_DEGREES",
                        "1",
                    )
                ),

                verification_samples=int(
                    os.getenv(
                        "DOBOT_VERIFY_SAMPLES",
                        "3",
                    )
                ),

                calibration_max_step_mm=float(
                    os.getenv(
                        "DOBOT_CALIBRATION_MAX_STEP_MM",
                        "5",
                    )
                ),

                rotation_max_step_degrees=float(
                    os.getenv(
                        "DOBOT_ROTATION_MAX_STEP_DEGREES",
                        "5",
                    )
                ),

                calibration_speed_ratio=float(
                    os.getenv(
                        "DOBOT_CALIBRATION_SPEED_RATIO",
                        "5",
                    )
                ),

                calibration_acceleration_ratio=float(
                    os.getenv(
                        "DOBOT_CALIBRATION_ACCELERATION_RATIO",
                        "5",
                    )
                ),

                hover_height_mm=float(
                    os.getenv(
                        "DOBOT_HOVER_HEIGHT_MM",
                        "40.0",
                    )
                ),

                test_position=optional_group(
                    position_names,
                    DobotPosition,
                ),

                safety_limits=optional_group(
                    limit_names,
                    SafetyLimits,
                ),
            )

        except ValueError as exc:
            raise DobotConfigurationError(
                f"Invalid DOBOT configuration: {exc}"
            ) from exc

        # ---------------------------------------------------------
        # GENERAL VALIDATION
        # ---------------------------------------------------------

        if not 0 <= config.max_retries <= 5:
            raise DobotConfigurationError(
                "DOBOT_MAX_RETRIES must be between 0 and 5."
            )

        if (
            config.connect_timeout_seconds <= 0
            or config.command_timeout_ms <= 0
        ):
            raise DobotConfigurationError(
                "DOBOT timeouts must be positive."
            )

        if config.verification_timeout_seconds <= 0:
            raise DobotConfigurationError(
                "DOBOT_VERIFY_TIMEOUT_SECONDS must be positive."
            )

        if config.verification_start_delay_seconds < 0:
            raise DobotConfigurationError(
                "DOBOT_VERIFY_START_DELAY_SECONDS "
                "cannot be negative."
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

        if not 0 < config.calibration_max_step_mm <= 50:
            raise DobotConfigurationError(
                "DOBOT_CALIBRATION_MAX_STEP_MM must be "
                "greater than 0 and at most 50."
            )

        if not 0 < config.rotation_max_step_degrees <= 30:
            raise DobotConfigurationError(
                "DOBOT_ROTATION_MAX_STEP_DEGREES must be "
                "greater than 0 and at most 30."
            )

        if not 1 <= config.calibration_speed_ratio <= 10:
            raise DobotConfigurationError(
                "DOBOT_CALIBRATION_SPEED_RATIO must be "
                "between 1 and 10."
            )

        if not 1 <= config.calibration_acceleration_ratio <= 10:
            raise DobotConfigurationError(
                "DOBOT_CALIBRATION_ACCELERATION_RATIO must be "
                "between 1 and 10."
            )

        if config.ptp_mode not in {0, 1, 2}:
            raise DobotConfigurationError(
                "Real Cartesian MOVE requires "
                "DOBOT_PTP_MODE 0, 1, or 2."
            )

        return config

    def require_safe_test_position(
        self,
    ) -> DobotPosition:
        """
        Return the explicitly configured physical test position.

        MOVE is fail-closed:

        1. Test position must exist.
        2. Safety limits must exist.
        3. Test position must be inside those limits.

        If any requirement fails, MOVE is rejected.
        """

        # ---------------------------------------------------------
        # SAFETY CHECK 1
        # ---------------------------------------------------------

        if self.test_position is None:
            raise DobotConfigurationError(
                "MOVE is disabled until "
                "DOBOT_TEST_X/Y/Z/R are all configured."
            )

        # ---------------------------------------------------------
        # SAFETY CHECK 2
        # ---------------------------------------------------------

        if self.safety_limits is None:
            raise DobotConfigurationError(
                "MOVE is disabled until all "
                "DOBOT_MIN/MAX_X/Y/Z/R limits are configured."
            )

        # ---------------------------------------------------------
        # SAFETY CHECK 3
        # ---------------------------------------------------------

        self.safety_limits.validate(
            self.test_position,
            label="Configured test position",
        )

        return self.test_position