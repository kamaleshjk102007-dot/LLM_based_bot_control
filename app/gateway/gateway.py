"""Facade coordinating the Universal Gateway Core."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.commands.models import UniversalCommand
from app.commands.validator import CommandValidationError, validate_command
from app.gateway.adapter_manager import AdapterManager
from app.gateway.capability_manager import CapabilityManager
from app.gateway.command_router import CommandRouter, ExecutionPlan, PlanStatus
from app.gateway.robot_registry import RobotRegistry
from app.gateway.robot_selector import RobotSelector
from app.gateway.safety_validator import SafetyValidator


class UniversalGateway:
    def __init__(
        self,
        registry: RobotRegistry | None = None,
        adapter_manager: AdapterManager | None = None,
    ) -> None:
        self.registry = registry or RobotRegistry()
        self.capabilities = CapabilityManager()
        self.selector = RobotSelector(self.registry, self.capabilities)
        self.safety = SafetyValidator(self.capabilities)
        self.adapters = adapter_manager or AdapterManager()
        self.router = CommandRouter(
            self.selector, self.capabilities, self.safety, self.adapters
        )

    def process(
        self, command: UniversalCommand | dict[str, Any] | str
    ) -> ExecutionPlan:
        try:
            validated = validate_command(command)
        except CommandValidationError as exc:
            return ExecutionPlan(
                plan_id=f"plan-{uuid4()}",
                status=PlanStatus.INVALID,
                reason=str(exc),
            )
        return self.router.route(validated)
