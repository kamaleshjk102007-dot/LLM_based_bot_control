"""Invalid command and validator tests."""

import pytest

from app.commands.validator import CommandValidationError, validate_command


@pytest.mark.parametrize(
    "task",
    [
        {"action": "PICK"},
        {"action": "PLACE"},
        {"action": "ROTATE", "angle": -1, "unit": "degrees"},
        {"action": "FLY"},
        {"action": "STOP", "target": {"name": "station"}},
        {"action": "MOVE", "direction": "forward", "distance": 20},
    ],
)
def test_invalid_tasks_are_rejected(task):
    with pytest.raises(CommandValidationError):
        validate_command({"version": "1.0", "robot_id": None, "tasks": [task]})


def test_malformed_json_is_rejected():
    with pytest.raises(CommandValidationError):
        validate_command('{"version": "1.0", "tasks": [')


def test_empty_task_list_is_rejected():
    with pytest.raises(CommandValidationError):
        validate_command({"version": "1.0", "robot_id": None, "tasks": []})


def test_unknown_fields_are_rejected():
    with pytest.raises(CommandValidationError):
        validate_command(
            {"version": "1.0", "tasks": [{"action": "STOP", "motor_speed": 100}]}
        )
