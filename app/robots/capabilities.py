"""Capability helpers shared by gateway components."""

from __future__ import annotations

from collections.abc import Iterable

from app.commands.models import Action, Task


def required_actions(tasks: Iterable[Task]) -> frozenset[Action]:
    return frozenset(task.action for task in tasks)


def missing_actions(
    supported: Iterable[Action], required: Iterable[Action]
) -> frozenset[Action]:
    return frozenset(required) - frozenset(supported)
