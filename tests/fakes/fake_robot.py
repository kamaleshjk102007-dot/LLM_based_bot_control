"""In-memory RobotInterface implementation for gateway integration tests."""

from __future__ import annotations

import json
from dataclasses import dataclass

from app.adapters.base import RobotAdapter, RobotAdapterError
from app.commands.models import Action, Task, UniversalCommand
from app.commands.validator import CommandValidationError, validate_command
from motion.models import RobotPosition


@dataclass(frozen=True)
class FakeExecutionRecord:
    """One attempted fake execution, including deterministic failure details."""

    execution_number: int
    action: Action
    direction: str | None
    distance: float | None
    angle: float | None
    unit: str | None
    succeeded: bool
    error: str | None = None


class FakeRobotAdapter(RobotAdapter):
    """Synchronous in-memory adapter that never accesses a robot or network."""

    simulated = True
    SUPPORTED_ACTIONS = frozenset({
        Action.MOVE,
        Action.ROTATE,
        Action.HOME,
        Action.STOP,
        Action.GET_STATUS,
        Action.GRIP,
        Action.RELEASE,
    })
    MOVE_DIRECTIONS = frozenset({"+X", "-X", "+Y", "-Y", "+Z", "-Z"})
    ROTATE_DIRECTIONS = frozenset({"+R", "-R"})

    def __init__(
        self,
        robot,
        initial_position: RobotPosition,
        *,
        fail_on_execution: int | None = None,
    ) -> None:
        super().__init__(robot)
        self.position = RobotPosition.model_validate(initial_position.model_dump())
        # HOME returns to the explicitly supplied initial fake pose.
        self.home_position = RobotPosition.model_validate(initial_position.model_dump())
        self.fail_on_execution = fail_on_execution
        self.execution_history: list[FakeExecutionRecord] = []
        self.stopped = False
        self.gripped = False

    def validate(self, command: UniversalCommand) -> bool:
        try:
            validated = validate_command(command)
        except CommandValidationError:
            return False
        if len(validated.tasks) != 1:
            return False
        task = validated.tasks[0]
        if task.action not in self.SUPPORTED_ACTIONS:
            return False
        if task.action is Action.MOVE:
            return task.direction in self.MOVE_DIRECTIONS and task.unit == "mm"
        if task.action is Action.ROTATE:
            return task.direction in self.ROTATE_DIRECTIONS and task.unit == "degrees"
        return True

    def prepare(self, command: UniversalCommand) -> list[dict[str, object]]:
        if not self.validate(command):
            raise RobotAdapterError("Fake robot rejected the command.")
        task = command.tasks[0]
        return [task.model_dump(mode="json", exclude_none=True)]

    def execute(self, command: UniversalCommand) -> list[str]:
        if not self.validate(command):
            raise RobotAdapterError("Fake robot rejected the command.")

        task = command.tasks[0]
        execution_number = len(self.execution_history) + 1
        if execution_number == self.fail_on_execution:
            error = f"Injected failure on execution {execution_number}."
            self._record(execution_number, task, succeeded=False, error=error)
            raise RobotAdapterError(error)

        self._apply(task)
        self._record(execution_number, task, succeeded=True)
        if task.action is Action.GET_STATUS:
            return [self.get_status()]
        return [f"[FAKE] {task.action.value} accepted"]

    def get_status(self) -> str:
        payload = {
            "gripped": self.gripped,
            "position": self.position.model_dump(mode="json"),
            "state": "STOPPED" if self.stopped else "READY",
        }
        return json.dumps(payload, sort_keys=True)

    def _apply(self, task: Task) -> None:
        values = self.position.model_dump()
        if task.action is Action.MOVE:
            axis = task.direction[-1].lower()
            sign = 1 if task.direction.startswith("+") else -1
            values[axis] += sign * task.distance
            self.position = RobotPosition.model_validate(values)
            self.stopped = False
        elif task.action is Action.ROTATE:
            sign = 1 if task.direction.startswith("+") else -1
            values["r"] += sign * task.angle
            self.position = RobotPosition.model_validate(values)
            self.stopped = False
        elif task.action is Action.HOME:
            self.position = self.home_position.model_copy(deep=True)
            self.stopped = False
        elif task.action is Action.STOP:
            self.stopped = True
        elif task.action is Action.GRIP:
            self.gripped = True
        elif task.action is Action.RELEASE:
            self.gripped = False

    def _record(
        self,
        execution_number: int,
        task: Task,
        *,
        succeeded: bool,
        error: str | None = None,
    ) -> None:
        self.execution_history.append(FakeExecutionRecord(
            execution_number=execution_number,
            action=task.action,
            direction=task.direction,
            distance=task.distance,
            angle=task.angle,
            unit=task.unit,
            succeeded=succeeded,
            error=error,
        ))

