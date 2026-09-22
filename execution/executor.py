from __future__ import annotations

import json
from uuid import uuid4

from app.commands.models import Action, UniversalCommand
from robots.interface import RobotInterface

from execution.logger import ExecutionLogger
from execution.models import (
    ActualMovement,
    Pose,
    RequestedMovement,
)
from execution.result import ExecutionResult, ExecutionStatus
from execution.verifier import MovementVerifier


class ExecutionEngine:
    """
    Robot-agnostic execution, verification, and logging engine.
    """

    def __init__(
        self,
        robot: RobotInterface,
        tolerance: float = 0.2,
        logger: ExecutionLogger | None = None,
    ) -> None:
        self.robot = robot
        self.verifier = MovementVerifier(
            tolerance=tolerance
        )
        self.logger = logger

    def execute(
        self,
        command: UniversalCommand,
    ) -> list[ExecutionResult]:

        results: list[ExecutionResult] = []

        robot_id = (
            command.robot_id
            or self.robot.robot.robot_id
        )

        # ========================================================
        # STEP 1: VALIDATE COMMAND
        # ========================================================

        if not self.robot.validate(command):

            for task in command.tasks:

                result = ExecutionResult(
                    execution_id=str(uuid4()),
                    robot_id=robot_id,
                    action=task.action,
                    status=ExecutionStatus.REJECTED,
                    message="Robot rejected the command",
                    failure_reason="Robot validation failed",
                )

                results.append(result)
                self._log(result)

            return results

        # ========================================================
        # STEP 2: EXECUTE EACH TASK
        # ========================================================

        for task in command.tasks:

            execution_id = str(uuid4())

            # ----------------------------------------------------
            # STEP 2A: INITIAL POSE
            # ----------------------------------------------------

            try:

                initial_status = self.robot.get_status()

                initial_pose = self._extract_pose(
                    initial_status
                )

            except Exception as exc:

                result = ExecutionResult(
                    execution_id=execution_id,
                    robot_id=robot_id,
                    action=task.action,
                    status=ExecutionStatus.FAILED,
                    message="Unable to read initial robot state",
                    failure_reason=str(exc),
                )

                results.append(result)
                self._log(result)

                continue

            # ----------------------------------------------------
            # STEP 2B: SINGLE TASK COMMAND
            # ----------------------------------------------------

            single_task_command = UniversalCommand(
                version=command.version,
                robot_id=command.robot_id,
                tasks=[task],
            )

            # ----------------------------------------------------
            # STEP 2C: EXECUTE TASK
            # ----------------------------------------------------

            try:

                messages = self.robot.execute(
                    single_task_command
                )

                message = (
                    messages[0]
                    if messages
                    else "Command executed successfully"
                )

            except Exception as exc:

                result = ExecutionResult(
                    execution_id=execution_id,
                    robot_id=robot_id,
                    action=task.action,
                    status=ExecutionStatus.FAILED,
                    message="Robot execution failed",

                    # Store initial pose
                    initial_x=initial_pose.x,
                    initial_y=initial_pose.y,
                    initial_z=initial_pose.z,
                    initial_r=initial_pose.r,

                    failure_reason=str(exc),
                )

                results.append(result)
                self._log(result)

                continue

            # ----------------------------------------------------
            # STEP 2D: FINAL POSE
            # ----------------------------------------------------

            try:

                final_status = self.robot.get_status()

                final_pose = self._extract_pose(
                    final_status
                )

            except Exception as exc:

                result = ExecutionResult(
                    execution_id=execution_id,
                    robot_id=robot_id,
                    action=task.action,
                    status=ExecutionStatus.FAILED,
                    message="Unable to read final robot state",

                    # Store initial pose
                    initial_x=initial_pose.x,
                    initial_y=initial_pose.y,
                    initial_z=initial_pose.z,
                    initial_r=initial_pose.r,

                    failure_reason=str(exc),
                )

                results.append(result)
                self._log(result)

                continue

            # ====================================================
            # STEP 3: MOVE VERIFICATION
            # ====================================================

            if task.action == Action.MOVE:

                result = self._verify_move(
                    task=task,
                    initial_pose=initial_pose,
                    final_pose=final_pose,
                    execution_id=execution_id,
                    robot_id=robot_id,
                )

                results.append(result)
                self._log(result)

                continue

            # ====================================================
            # STEP 4: STOP
            # ====================================================

            if task.action == Action.STOP:

                result = ExecutionResult(
                    execution_id=execution_id,
                    robot_id=robot_id,
                    action=task.action,
                    status=ExecutionStatus.STOPPED,
                    message=message,

                    # Store initial pose
                    initial_x=initial_pose.x,
                    initial_y=initial_pose.y,
                    initial_z=initial_pose.z,
                    initial_r=initial_pose.r,

                    # Store final pose
                    final_x=final_pose.x,
                    final_y=final_pose.y,
                    final_z=final_pose.z,
                    final_r=final_pose.r,
                )

                results.append(result)
                self._log(result)

                continue

            # ====================================================
            # STEP 5: OTHER ACTIONS
            # ====================================================

            result = ExecutionResult(
                execution_id=execution_id,
                robot_id=robot_id,
                action=task.action,
                status=ExecutionStatus.SUCCESS,
                message=message,

                # Store initial pose
                initial_x=initial_pose.x,
                initial_y=initial_pose.y,
                initial_z=initial_pose.z,
                initial_r=initial_pose.r,

                # Store final pose
                final_x=final_pose.x,
                final_y=final_pose.y,
                final_z=final_pose.z,
                final_r=final_pose.r,
            )

            results.append(result)
            self._log(result)

        return results

    # ============================================================
    # MOVE VERIFICATION
    # ============================================================

    def _verify_move(
        self,
        task,
        initial_pose: Pose,
        final_pose: Pose,
        execution_id: str,
        robot_id: str,
    ) -> ExecutionResult:

        try:

            requested = self._build_requested_movement(
                task
            )

            actual = self._calculate_actual_movement(
                initial=initial_pose,
                final=final_pose,
                requested=requested,
            )

            verification = self.verifier.verify(
                requested,
                actual,
            )

        except Exception as exc:

            return ExecutionResult(
                execution_id=execution_id,
                robot_id=robot_id,
                action=task.action,
                status=ExecutionStatus.FAILED,
                message="Movement verification failed",

                # Store initial pose
                initial_x=initial_pose.x,
                initial_y=initial_pose.y,
                initial_z=initial_pose.z,
                initial_r=initial_pose.r,

                # Store final pose
                final_x=final_pose.x,
                final_y=final_pose.y,
                final_z=final_pose.z,
                final_r=final_pose.r,

                failure_reason=str(exc),
            )

        # --------------------------------------------------------
        # VERIFICATION FAILED
        # --------------------------------------------------------

        if not verification.passed:

            return ExecutionResult(
                execution_id=execution_id,
                robot_id=robot_id,
                action=task.action,
                status=ExecutionStatus.VERIFICATION_FAILED,
                message=verification.message,

                requested_axis=requested.axis,
                requested_distance=requested.distance,
                requested_unit=requested.unit,

                actual_axis=actual.axis,
                actual_distance=actual.distance,
                actual_unit=actual.unit,

                error=verification.error,
                tolerance=verification.tolerance,
                verification=False,

                # Store initial pose
                initial_x=initial_pose.x,
                initial_y=initial_pose.y,
                initial_z=initial_pose.z,
                initial_r=initial_pose.r,

                # Store final pose
                final_x=final_pose.x,
                final_y=final_pose.y,
                final_z=final_pose.z,
                final_r=final_pose.r,

                failure_reason=verification.message,
            )

        # --------------------------------------------------------
        # VERIFICATION SUCCESS
        # --------------------------------------------------------

        return ExecutionResult(
            execution_id=execution_id,
            robot_id=robot_id,
            action=task.action,
            status=ExecutionStatus.SUCCESS,
            message=verification.message,

            requested_axis=requested.axis,
            requested_distance=requested.distance,
            requested_unit=requested.unit,

            actual_axis=actual.axis,
            actual_distance=actual.distance,
            actual_unit=actual.unit,

            error=verification.error,
            tolerance=verification.tolerance,
            verification=True,

            # Store initial pose
            initial_x=initial_pose.x,
            initial_y=initial_pose.y,
            initial_z=initial_pose.z,
            initial_r=initial_pose.r,

            # Store final pose
            final_x=final_pose.x,
            final_y=final_pose.y,
            final_z=final_pose.z,
            final_r=final_pose.r,
        )

    # ============================================================
    # LOGGING
    # ============================================================

    def _log(
        self,
        result: ExecutionResult,
    ) -> None:

        if self.logger is not None:
            self.logger.log(result)

    
        # ============================================================
    # POSE EXTRACTION
    # ============================================================

    def _extract_pose(
        self,
        status,
    ) -> Pose:

        # --------------------------------------------------------
        # FORMAT 1: Pose object
        # --------------------------------------------------------

        if isinstance(status, Pose):
            return status

        # --------------------------------------------------------
        # FORMAT 2: JSON string (e.g. Dobot adapter get_status())
        # --------------------------------------------------------

        if isinstance(status, str):
            try:
                parsed = json.loads(status)
                if isinstance(parsed, dict):
                    status = parsed
            except (ValueError, json.JSONDecodeError):
                pass

        # --------------------------------------------------------
        # FORMAT 3: Dictionary (supports flat & nested 'pose' dict)
        # --------------------------------------------------------

        if isinstance(status, dict):
            pose_data = status.get("pose", status)
            if isinstance(pose_data, dict):
                inner = pose_data.get("result", pose_data)
                if isinstance(inner, dict) and all(k in inner for k in ("x", "y", "z", "r")):
                    return Pose(
                        x=float(inner["x"]),
                        y=float(inner["y"]),
                        z=float(inner["z"]),
                        r=float(inner["r"]),
                    )

            if all(k in status for k in ("x", "y", "z", "r")):
                return Pose(
                    x=float(status.get("x", 0.0)),
                    y=float(status.get("y", 0.0)),
                    z=float(status.get("z", 0.0)),
                    r=float(status.get("r", 0.0)),
                )

        # --------------------------------------------------------
        # FORMAT 4: Webots status string
        # Example:
        # "X=0.0 Y=0.0 Z=0.0 R=0.0"
        # --------------------------------------------------------

        if isinstance(status, str):

            values = {}

            for item in status.split():

                if "=" not in item:
                    continue

                key, value = item.split("=", 1)

                key = key.strip().lower()
                value = value.strip()

                if key in {"x", "y", "z", "r"}:
                    values[key] = float(value)

            required = {"x", "y", "z", "r"}

            if required.issubset(values):

                return Pose(
                    x=values["x"],
                    y=values["y"],
                    z=values["z"],
                    r=values["r"],
                )

        # --------------------------------------------------------
        # Unsupported status format
        # --------------------------------------------------------

        raise ValueError(
            "Robot status does not contain a supported pose format"
        )

    # ============================================================
    # REQUESTED MOVEMENT
    # ============================================================

    def _build_requested_movement(
        self,
        task,
    ) -> RequestedMovement:

        direction_text = (
            task.direction or ""
        ).lower().strip()

        direction_aliases = {
            "up": "z",
            "upward": "z",
            "upwards": "z",
            "down": "-z",
            "downward": "-z",
            "downwards": "-z",
        }
        direction_text = direction_aliases.get(direction_text, direction_text)

        # --------------------------------------------------------
        # GET AXIS
        # --------------------------------------------------------

        axis = (
            direction_text
            .replace("+", "")
            .replace("-", "")
        )

        if axis not in {"x", "y", "z"}:

            raise ValueError(
                f"Unsupported movement direction: {task.direction}"
            )

        # --------------------------------------------------------
        # GET DIRECTION SIGN
        # --------------------------------------------------------

        if direction_text.startswith("-"):
            direction_sign = -1
        else:
            direction_sign = 1

        return RequestedMovement(
            axis=axis.upper(),
            distance=float(task.distance),
            unit=task.unit or "mm",
            direction=direction_sign,
        )

    # ============================================================
    # ACTUAL MOVEMENT
    # ============================================================

    def _calculate_actual_movement(
        self,
        initial: Pose,
        final: Pose,
        requested: RequestedMovement,
    ) -> ActualMovement:

        axis = requested.axis.upper()

        # --------------------------------------------------------
        # CALCULATE SIGNED MOVEMENT
        # --------------------------------------------------------

        if axis == "X":

            signed_distance = final.x - initial.x

        elif axis == "Y":

            signed_distance = final.y - initial.y

        elif axis == "Z":

            signed_distance = final.z - initial.z

        else:

            raise ValueError(
                f"Unsupported verification axis: {axis}"
            )

        # --------------------------------------------------------
        # DETERMINE ACTUAL DIRECTION
        # --------------------------------------------------------

        if signed_distance < 0:
            direction_sign = -1
        elif signed_distance > 0:
            direction_sign = 1
        else:
            direction_sign = 0

        # --------------------------------------------------------
        # DISTANCE MAGNITUDE
        # --------------------------------------------------------

        distance = abs(signed_distance)

        return ActualMovement(
            axis=axis,
            distance=distance,
            unit=requested.unit,
            direction=direction_sign,
        )