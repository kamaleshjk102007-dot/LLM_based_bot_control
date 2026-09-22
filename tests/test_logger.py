from app.commands.models import Action
from execution.logger import ExecutionLogger
from execution.result import ExecutionResult, ExecutionStatus


# ============================================================
# TEST 1: LOGGER CREATES LOG FILE
# ============================================================

def test_logger_creates_log_file(tmp_path):

    log_file = tmp_path / "execution_log.json"

    logger = ExecutionLogger(
        log_file=str(log_file)
    )

    assert log_file.exists()


# ============================================================
# TEST 2: LOGGER STORES ONE RESULT
# ============================================================

def test_logger_stores_result(tmp_path):

    log_file = tmp_path / "execution_log.json"

    logger = ExecutionLogger(
        log_file=str(log_file)
    )

    result = ExecutionResult(
        execution_id="test-001",
        robot_id="fake_robot",
        action=Action.MOVE,
        status=ExecutionStatus.SUCCESS,
        message="Movement successful",

        requested_axis="X",
        requested_distance=5.0,
        requested_unit="mm",

        actual_axis="X",
        actual_distance=5.0,
        actual_unit="mm",

        error=0.0,
        tolerance=0.2,
        verification=True,
    )

    logger.log(result)

    logs = logger.read_logs()

    assert len(logs) == 1

    assert logs[0]["execution_id"] == "test-001"
    assert logs[0]["robot_id"] == "fake_robot"
    assert logs[0]["action"] == "MOVE"
    assert logs[0]["status"] == "SUCCESS"

    assert logs[0]["requested_distance"] == 5.0
    assert logs[0]["actual_distance"] == 5.0

    assert logs[0]["error"] == 0.0
    assert logs[0]["tolerance"] == 0.2

    assert logs[0]["verification"] is True


# ============================================================
# TEST 3: LOGGER STORES VERIFICATION FAILURE
# ============================================================

def test_logger_stores_verification_failure(tmp_path):

    log_file = tmp_path / "execution_log.json"

    logger = ExecutionLogger(
        log_file=str(log_file)
    )

    result = ExecutionResult(
        execution_id="test-002",
        robot_id="fake_robot",
        action=Action.MOVE,
        status=ExecutionStatus.VERIFICATION_FAILED,
        message="Movement error exceeds tolerance",

        requested_axis="X",
        requested_distance=5.0,
        requested_unit="mm",

        actual_axis="X",
        actual_distance=4.5,
        actual_unit="mm",

        error=0.5,
        tolerance=0.2,
        verification=False,

        failure_reason="Movement error exceeds tolerance",
    )

    logger.log(result)

    logs = logger.read_logs()

    assert len(logs) == 1

    assert logs[0]["status"] == "VERIFICATION_FAILED"

    assert logs[0]["requested_distance"] == 5.0
    assert logs[0]["actual_distance"] == 4.5

    assert logs[0]["error"] == 0.5
    assert logs[0]["tolerance"] == 0.2

    assert logs[0]["verification"] is False

    assert (
        logs[0]["failure_reason"]
        == "Movement error exceeds tolerance"
    )


# ============================================================
# TEST 4: LOGGER STORES MULTIPLE RESULTS
# ============================================================

def test_logger_stores_multiple_results(tmp_path):

    log_file = tmp_path / "execution_log.json"

    logger = ExecutionLogger(
        log_file=str(log_file)
    )

    result1 = ExecutionResult(
        execution_id="test-001",
        robot_id="fake_robot",
        action=Action.HOME,
        status=ExecutionStatus.SUCCESS,
        message="HOME executed",
    )

    result2 = ExecutionResult(
        execution_id="test-002",
        robot_id="fake_robot",
        action=Action.MOVE,
        status=ExecutionStatus.SUCCESS,
        message="MOVE executed",

        requested_axis="X",
        requested_distance=5.0,
        requested_unit="mm",

        actual_axis="X",
        actual_distance=5.0,
        actual_unit="mm",

        error=0.0,
        tolerance=0.2,
        verification=True,
    )

    logger.log_many(
        [result1, result2]
    )

    logs = logger.read_logs()

    assert len(logs) == 2

    assert logs[0]["execution_id"] == "test-001"
    assert logs[0]["action"] == "HOME"

    assert logs[1]["execution_id"] == "test-002"
    assert logs[1]["action"] == "MOVE"


# ============================================================
# TEST 5: LOGGER RECORDS DIRECTION VERIFICATION FAILURE
# ============================================================

def test_logger_records_direction_verification_failure(tmp_path):

    log_file = tmp_path / "execution_log.json"

    logger = ExecutionLogger(
        log_file=str(log_file)
    )

    result = ExecutionResult(
        execution_id="test-failure-001",
        robot_id="dobot_001",
        action=Action.MOVE,
        status=ExecutionStatus.VERIFICATION_FAILED,
        message="Wrong movement direction",

        requested_axis="X",
        requested_distance=5.0,
        requested_unit="mm",

        actual_axis="X",
        actual_distance=5.0,
        actual_unit="mm",

        error=0.0,
        tolerance=0.2,
        verification=False,

        failure_reason="Expected +X but actual movement was -X",
    )

    logger.log(result)

    logs = logger.read_logs()

    assert len(logs) == 1

    assert logs[0]["execution_id"] == "test-failure-001"
    assert logs[0]["robot_id"] == "dobot_001"
    assert logs[0]["action"] == "MOVE"
    assert logs[0]["status"] == "VERIFICATION_FAILED"

    assert logs[0]["requested_axis"] == "X"
    assert logs[0]["requested_distance"] == 5.0

    assert logs[0]["actual_axis"] == "X"
    assert logs[0]["actual_distance"] == 5.0

    assert logs[0]["verification"] is False

    assert (
        logs[0]["failure_reason"]
        == "Expected +X but actual movement was -X"
    )


# ============================================================
# TEST 6: LOGGER STORES INITIAL AND FINAL POSE
# ============================================================

def test_logger_stores_initial_and_final_pose(tmp_path):

    log_file = tmp_path / "execution_log.json"

    logger = ExecutionLogger(
        log_file=str(log_file)
    )

    result = ExecutionResult(
        execution_id="pose-test-001",
        robot_id="fake_robot",
        action=Action.MOVE,
        status=ExecutionStatus.SUCCESS,
        message="Movement successful",

        requested_axis="X",
        requested_distance=5.0,
        requested_unit="mm",

        actual_axis="X",
        actual_distance=5.0,
        actual_unit="mm",

        error=0.0,
        tolerance=0.2,
        verification=True,

        # ----------------------------------------------------
        # INITIAL POSE
        # ----------------------------------------------------

        initial_x=100.0,
        initial_y=50.0,
        initial_z=30.0,
        initial_r=0.0,

        # ----------------------------------------------------
        # FINAL POSE
        # ----------------------------------------------------

        final_x=105.0,
        final_y=50.0,
        final_z=30.0,
        final_r=0.0,
    )

    logger.log(result)

    logs = logger.read_logs()

    assert len(logs) == 1

    # --------------------------------------------------------
    # CHECK INITIAL POSE
    # --------------------------------------------------------

    assert logs[0]["initial_x"] == 100.0
    assert logs[0]["initial_y"] == 50.0
    assert logs[0]["initial_z"] == 30.0
    assert logs[0]["initial_r"] == 0.0

    # --------------------------------------------------------
    # CHECK FINAL POSE
    # --------------------------------------------------------

    assert logs[0]["final_x"] == 105.0
    assert logs[0]["final_y"] == 50.0
    assert logs[0]["final_z"] == 30.0
    assert logs[0]["final_r"] == 0.0