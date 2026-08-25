"""Pure gateway routing pipeline producing a typed ExecutionPlan."""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.adapters.base import RobotAdapterError
from app.commands.models import Task, UniversalCommand
from app.gateway.adapter_manager import (
    AdapterManager,
    AdapterManagerError,
    AdapterNotFoundError,
)
from app.gateway.capability_manager import CapabilityManager
from app.gateway.robot_selector import (
    NoRobotError,
    RobotSelector,
    RobotUnavailableError,
    UnsupportedRobotError,
)
from app.gateway.safety_validator import SafetyError, SafetyValidator


class PlanStatus(str, Enum):
    READY = "READY"
    REJECTED = "REJECTED"
    NO_ROBOT = "NO_ROBOT"
    UNSUPPORTED = "UNSUPPORTED"
    UNSAFE = "UNSAFE"
    INVALID = "INVALID"


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=1)
    robot_id: str | None = None
    adapter_type: str | None = None
    status: PlanStatus
    tasks: list[Task] = Field(default_factory=list)
    reason: str | None = None
    capability_checks: dict[str, bool] = Field(default_factory=dict)
    safety_passed: bool = False
    simulated: bool = False
    prepared_tasks: list[dict[str, Any]] = Field(default_factory=list)
    results: list[str] = Field(default_factory=list)


class CommandRouter:
    def __init__(
        self,
        selector: RobotSelector,
        capabilities: CapabilityManager,
        safety: SafetyValidator,
        adapters: AdapterManager,
    ) -> None:
        self.selector = selector
        self.capabilities = capabilities
        self.safety = safety
        self.adapters = adapters

    @staticmethod
    def _plan(
        command: UniversalCommand,
        status: PlanStatus,
        reason: str,
    ) -> ExecutionPlan:
        return ExecutionPlan(
            plan_id=f"plan-{uuid4()}",
            status=status,
            tasks=command.tasks,
            reason=reason,
        )

    def route(self, command: UniversalCommand) -> ExecutionPlan:
        try:
            robot = self.selector.select(command)
        except NoRobotError as exc:
            return self._plan(command, PlanStatus.NO_ROBOT, str(exc))
        except RobotUnavailableError as exc:
            return self._plan(command, PlanStatus.REJECTED, str(exc))
        except UnsupportedRobotError as exc:
            return self._plan(command, PlanStatus.UNSUPPORTED, str(exc))

        checks = {
            task.action.value: self.capabilities.has_capability(robot, task.action)
            for task in command.tasks
        }

        try:
            self.safety.validate(command, robot)
        except SafetyError as exc:
            plan = self._plan(command, PlanStatus.UNSAFE, str(exc))
            return plan.model_copy(update={
                "robot_id": robot.robot_id,
                "adapter_type": robot.adapter_type,
                "capability_checks": checks,
            })

        try:
            adapter = self.adapters.get(robot)
        except (AdapterNotFoundError, AdapterManagerError) as exc:
            plan = self._plan(command, PlanStatus.UNSUPPORTED, str(exc))
            return plan.model_copy(update={
                "robot_id": robot.robot_id,
                "adapter_type": robot.adapter_type,
                "capability_checks": checks,
                "safety_passed": True,
            })

        if not adapter.validate(command):
            plan = self._plan(
                command, PlanStatus.INVALID, "Adapter rejected the command."
            )
            return plan.model_copy(update={
                "robot_id": robot.robot_id,
                "adapter_type": robot.adapter_type,
                "capability_checks": checks,
                "safety_passed": True,
            })

        try:
            prepared = adapter.prepare(command)
            results = adapter.execute(command)
        except RobotAdapterError as exc:
            plan = self._plan(command, PlanStatus.REJECTED, str(exc))
            return plan.model_copy(update={
                "robot_id": robot.robot_id,
                "adapter_type": robot.adapter_type,
                "capability_checks": checks,
                "safety_passed": True,
            })

        return ExecutionPlan(
            plan_id=f"plan-{uuid4()}",
            robot_id=robot.robot_id,
            adapter_type=robot.adapter_type,
            status=PlanStatus.READY,
            tasks=command.tasks,
            capability_checks=checks,
            safety_passed=True,
            simulated=adapter.simulated,
            prepared_tasks=prepared,
            results=results,
        )
