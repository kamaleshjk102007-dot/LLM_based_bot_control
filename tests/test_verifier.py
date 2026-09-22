from execution.models import ActualMovement, RequestedMovement
from execution.verifier import MovementVerifier


def test_movement_within_tolerance_passes():
    verifier = MovementVerifier(tolerance=0.2)

    requested = RequestedMovement(
        axis="X",
        distance=5.0,
        unit="mm",
        direction=1,
    )

    actual = ActualMovement(
        axis="X",
        distance=4.8,
        unit="mm",
        direction=1,
    )

    result = verifier.verify(requested, actual)

    assert result.passed is True
    assert result.error == 0.2


def test_movement_outside_tolerance_fails():
    verifier = MovementVerifier(tolerance=0.2)

    requested = RequestedMovement(
        axis="X",
        distance=5.0,
        unit="mm",
        direction=1,
    )

    actual = ActualMovement(
        axis="X",
        distance=4.7,
        unit="mm",
        direction=1,
    )

    result = verifier.verify(requested, actual)

    assert result.passed is False


def test_opposite_direction_fails():
    verifier = MovementVerifier(tolerance=0.2)

    requested = RequestedMovement(
        axis="X",
        distance=5.0,
        unit="mm",
        direction=1,
    )

    actual = ActualMovement(
        axis="X",
        distance=4.8,
        unit="mm",
        direction=-1,
    )

    result = verifier.verify(requested, actual)

    assert result.passed is False
    assert "direction" in result.message.lower()


def test_wrong_axis_fails():
    verifier = MovementVerifier(tolerance=0.2)

    requested = RequestedMovement(
        axis="X",
        distance=5.0,
        unit="mm",
        direction=1,
    )

    actual = ActualMovement(
        axis="Y",
        distance=5.0,
        unit="mm",
        direction=1,
    )

    result = verifier.verify(requested, actual)

    assert result.passed is False
    assert "movement on" in result.message.lower()


def test_wrong_direction_fails():
    verifier = MovementVerifier(tolerance=0.2)

    requested = RequestedMovement(
        axis="X",
        distance=5.0,
        unit="mm",
        direction=1,
    )

    actual = ActualMovement(
        axis="X",
        distance=5.0,
        unit="mm",
        direction=-1,
    )

    result = verifier.verify(requested, actual)

    assert result.passed is False
    assert "direction" in result.message.lower()


def test_correct_negative_direction_passes():
    verifier = MovementVerifier(tolerance=0.2)

    requested = RequestedMovement(
        axis="X",
        distance=5.0,
        unit="mm",
        direction=-1,
    )

    actual = ActualMovement(
        axis="X",
        distance=5.0,
        unit="mm",
        direction=-1,
    )

    result = verifier.verify(requested, actual)

    assert result.passed is True
    assert result.error == 0


def test_no_movement_fails_verification():
    verifier = MovementVerifier(tolerance=0.2)

    requested = RequestedMovement(
        axis="X",
        distance=5.0,
        unit="mm",
        direction=1,
    )

    actual = ActualMovement(
        axis="X",
        distance=0.0,
        unit="mm",
        direction=0,
    )

    result = verifier.verify(
        requested,
        actual,
    )

    assert result.passed is False
    assert result.error == 5.0
    assert result.tolerance == 0.2
    assert "exceeds tolerance" in result.message.lower()