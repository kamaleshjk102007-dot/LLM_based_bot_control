import pytest

from app.commands.models import Action, Task, UniversalCommand
from app.gateway.capability_manager import CapabilityManager
from app.gateway.safety_validator import SafetyError, SafetyValidator
from conftest import command, robot


def validator():
    return SafetyValidator(CapabilityManager())


def test_valid_logical_safety():
    validator().validate(
        command({"action": "MOVE", "direction": "forward"}),
        robot(capabilities=("MOVE",)),
    )


@pytest.mark.parametrize("status", ["OFFLINE", "BUSY", "ERROR"])
def test_non_online_robot_is_unsafe(status):
    with pytest.raises(SafetyError, match="not online"):
        validator().validate(command({"action": "STOP"}), robot(status=status))


def test_empty_tasks_and_bad_version_are_unsafe():
    empty = UniversalCommand.model_construct(version="1.0", robot_id=None, tasks=[])
    with pytest.raises(SafetyError):
        validator().validate(empty, robot())
    bad_version = UniversalCommand.model_construct(
        version="2.0",
        robot_id=None,
        tasks=[Task(action="STOP")],
    )
    with pytest.raises(SafetyError, match="version"):
        validator().validate(bad_version, robot())


def test_invalid_required_parameters_are_unsafe():
    invalid = UniversalCommand.model_construct(
        version="1.0",
        robot_id=None,
        tasks=[Task.model_construct(action=Action.PICK)],
    )
    with pytest.raises(SafetyError):
        validator().validate(invalid, robot())


def test_unsupported_action_and_impossible_stop_sequence_are_unsafe():
    with pytest.raises(SafetyError, match="does not support"):
        validator().validate(
            command({"action": "NAVIGATE", "target": {"name": "table"}}),
            robot(capabilities=("MOVE",)),
        )
    with pytest.raises(SafetyError, match="STOP cannot"):
        validator().validate(
            command({"action": "STOP"}, {"action": "GET_STATUS"}),
            robot(capabilities=("STOP", "GET_STATUS")),
        )
