"""Thin orchestration from planning inputs to the existing gateway."""

from __future__ import annotations

from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from app.commands.models import UniversalCommand
from app.commands.validator import CommandValidationError, validate_command
from app.gateway.command_router import ExecutionPlan, PlanStatus
from motion.models import MotionPlan, MotionStep, RobotPosition, RobotTarget
from motion.planner import MotionPlanner, MotionPlanningError
from motion.trajectory import Trajectory, TrajectoryError
from motion.validator import MotionValidationResult, MotionValidator


class Gateway(Protocol):
    """The existing gateway operation used by the orchestrator."""

    def process(self, command: UniversalCommand) -> ExecutionPlan:
        ...


class ControllerStatus(str, Enum):
    PLANNING_FAILED = "PLANNING_FAILED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    SUCCESS = "SUCCESS"
    NO_MOTION = "NO_MOTION"


class ControllerResult(BaseModel):
    """Controller outcome without physical-state verification claims."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ControllerStatus
    plan: MotionPlan | None = None
    trajectory: Trajectory | None = None
    prepared_commands: tuple[UniversalCommand, ...] = ()
    submitted_commands: tuple[UniversalCommand, ...] = ()
    gateway_results: tuple[ExecutionPlan, ...] = ()
    failed_step_sequence: int | None = None
    remaining_step_sequences: tuple[int, ...] = ()
    error: str | None = None

    @property
    def successful(self) -> bool:
        return self.status in {
            ControllerStatus.SUCCESS,
            ControllerStatus.NO_MOTION,
        }


class MotionController:
    """Plan, validate, convert, and submit through the existing gateway."""

    def __init__(
        self,
        planner: MotionPlanner,
        gateway: Gateway,
        validator: type[MotionValidator] = MotionValidator,
    ) -> None:
        self.planner = planner
        self.gateway = gateway
        self.validator = validator

    def execute_target(
        self,
        current: RobotPosition,
        target: RobotTarget,
        robot_id: str,
    ) -> ControllerResult:
        """Orchestrate one target and stop submission after the first failure."""

        try:
            plan = self.planner.plan(current, target, robot_id)
        except (MotionPlanningError, TypeError, ValueError) as exc:
            return ControllerResult(
                status=ControllerStatus.PLANNING_FAILED,
                error=f"Motion planning failed: {exc}",
            )

        try:
            trajectory = Trajectory.from_plan(plan)
        except (TrajectoryError, TypeError, ValueError) as exc:
            return ControllerResult(
                status=ControllerStatus.VALIDATION_FAILED,
                plan=plan,
                error=f"Trajectory creation failed: {exc}",
            )

        validation = self.validator.validate(plan, trajectory)
        if not validation.valid:
            return ControllerResult(
                status=ControllerStatus.VALIDATION_FAILED,
                plan=plan,
                trajectory=trajectory,
                error=self._validation_error(validation),
            )

        try:
            commands = tuple(
                self._command_for_step(step, plan.robot_id)
                for step in trajectory.steps
            )
        except (CommandValidationError, TypeError, ValueError) as exc:
            return ControllerResult(
                status=ControllerStatus.VALIDATION_FAILED,
                plan=plan,
                trajectory=trajectory,
                error=f"Universal command validation failed: {exc}",
            )

        if not commands:
            return ControllerResult(
                status=ControllerStatus.NO_MOTION,
                plan=plan,
                trajectory=trajectory,
            )

        submitted: list[UniversalCommand] = []
        results: list[ExecutionPlan] = []
        for index, command in enumerate(commands):
            submitted.append(command)
            try:
                result = self.gateway.process(command)
            except Exception as exc:
                return self._execution_failure(
                    plan,
                    trajectory,
                    commands,
                    submitted,
                    results,
                    index,
                    f"Gateway execution raised {type(exc).__name__}: {exc}",
                )
            if not isinstance(result, ExecutionPlan):
                return self._execution_failure(
                    plan,
                    trajectory,
                    commands,
                    submitted,
                    results,
                    index,
                    "Gateway returned an invalid execution result.",
                )
            results.append(result)
            if result.status is not PlanStatus.READY:
                detail = result.reason or f"Gateway returned {result.status.value}."
                return self._execution_failure(
                    plan,
                    trajectory,
                    commands,
                    submitted,
                    results,
                    index,
                    detail,
                )

        return ControllerResult(
            status=ControllerStatus.SUCCESS,
            plan=plan,
            trajectory=trajectory,
            prepared_commands=commands,
            submitted_commands=tuple(submitted),
            gateway_results=tuple(results),
        )

    @staticmethod
    def _command_for_step(
        step: MotionStep,
        robot_id: str,
    ) -> UniversalCommand:
        task = step.model_dump(
            mode="json",
            exclude={"sequence"},
            exclude_none=True,
        )
        return validate_command({
            "version": "1.0",
            "robot_id": robot_id,
            "tasks": [task],
        })

    @staticmethod
    def _validation_error(validation: MotionValidationResult) -> str:
        return "Motion validation failed: " + "; ".join(validation.errors)

    @staticmethod
    def _execution_failure(
        plan: MotionPlan,
        trajectory: Trajectory,
        commands: tuple[UniversalCommand, ...],
        submitted: list[UniversalCommand],
        results: list[ExecutionPlan],
        failed_index: int,
        error: str,
    ) -> ControllerResult:
        steps = trajectory.steps
        return ControllerResult(
            status=ControllerStatus.EXECUTION_FAILED,
            plan=plan,
            trajectory=trajectory,
            prepared_commands=commands,
            submitted_commands=tuple(submitted),
            gateway_results=tuple(results),
            failed_step_sequence=steps[failed_index].sequence,
            remaining_step_sequences=tuple(
                step.sequence for step in steps[failed_index + 1 :]
            ),
            error=error,
        )
