
from app.commands.models import Action, Task, UniversalCommand
from app.robots.models import Robot, RobotType
from execution.executor import ExecutionEngine
from execution.result import ExecutionStatus
from robots.interface import RobotInterface


# ============================================================
# CREATE FAKE ROBOT
# ============================================================

def create_fake_robot():
    return Robot(
        robot_id="fake_robot",
        name="Fake Robot",
        robot_type=RobotType.ROBOTIC_ARM,
        manufacturer="test",
        model="fake",
        adapter_type="fake",
        capabilities=frozenset({
            Action.MOVE,
            Action.HOME,
            Action.STOP,
            Action.GET_STATUS,
        }),
    )


# ============================================================
# FAKE ROBOT
# ============================================================

class FakeRobot(RobotInterface):
    simulated = True

    def __init__(self, robot):
        super().__init__(robot)

        self.pose = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "r": 0.0,
        }

    def validate(self, command):
        return True

    def prepare(self, command):
        return []

    def execute(self, command):

        for task in command.tasks:

            if task.action == Action.MOVE:

                direction = task.direction.lower()

                # X positive
                if direction in ("x", "+x", "x+"):
                    self.pose["x"] += task.distance

                # X negative
                elif direction in ("-x", "x-"):
                    self.pose["x"] -= task.distance

                # Y positive
                elif direction in ("y", "+y", "y+"):
                    self.pose["y"] += task.distance

                # Y negative
                elif direction in ("-y", "y-"):
                    self.pose["y"] -= task.distance

                # Z positive
                elif direction in ("z", "+z", "z+", "up"):
                    self.pose["z"] += task.distance

                # Z negative
                elif direction in ("-z", "z-", "down"):
                    self.pose["z"] -= task.distance

        return [
            f"{task.action.value} executed"
            for task in command.tasks
        ]

    def get_status(self):
        return self.pose


# ============================================================
# TEST 1: NORMAL EXECUTION SUCCESS
# ============================================================

def test_executor_success():

    robot = FakeRobot(create_fake_robot())

    engine = ExecutionEngine(robot)

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            {
                "action": "HOME"
            }
        ]
    )

    results = engine.execute(command)

    assert len(results) == 1
    assert results[0].status == ExecutionStatus.SUCCESS
    assert results[0].action == Action.HOME


# ============================================================
# TEST 2: COMMAND REJECTED
# ============================================================

def test_executor_rejected_command():

    class RejectingRobot(FakeRobot):

        def validate(self, command):
            return False

    robot = RejectingRobot(create_fake_robot())

    engine = ExecutionEngine(robot)

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            {
                "action": "HOME"
            }
        ]
    )

    results = engine.execute(command)

    assert len(results) == 1
    assert results[0].status == ExecutionStatus.REJECTED


# ============================================================
# TEST 3: ROBOT EXECUTION FAILURE
# ============================================================

def test_executor_execution_failure():

    class FailingRobot(FakeRobot):

        def execute(self, command):
            raise RuntimeError("Robot execution failed")

    robot = FailingRobot(create_fake_robot())

    engine = ExecutionEngine(robot)

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            {
                "action": "HOME"
            }
        ]
    )

    results = engine.execute(command)

    assert len(results) == 1
    assert results[0].status == ExecutionStatus.FAILED
    assert results[0].failure_reason == "Robot execution failed"


# ============================================================
# TEST 4: MOVE VERIFICATION SUCCESS
# ============================================================

def test_executor_move_verification_success():

    robot = FakeRobot(create_fake_robot())

    engine = ExecutionEngine(robot)

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            {
                "action": "MOVE",
                "direction": "X",
                "distance": 5,
                "unit": "mm",
            }
        ]
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    assert result.action == Action.MOVE
    assert result.status == ExecutionStatus.SUCCESS

    assert result.requested_axis == "X"
    assert result.requested_distance == 5.0
    assert result.requested_unit == "mm"

    assert result.actual_axis == "X"
    assert result.actual_distance == 5.0
    assert result.actual_unit == "mm"

    assert result.error == 0.0
    assert result.verification is True


# ============================================================
# TEST 5: MOVE VERIFICATION FAILURE
# ============================================================

def test_executor_move_verification_failure():

    class InaccurateRobot(FakeRobot):

        def execute(self, command):

            for task in command.tasks:

                if task.action == Action.MOVE:

                    direction = task.direction.lower()

                    # Robot moves only 4.5 mm
                    # even though 5 mm was requested

                    if direction in ("x", "+x", "x+"):
                        self.pose["x"] += 4.5

                    elif direction in ("y", "+y", "y+"):
                        self.pose["y"] += 4.5

                    elif direction in ("z", "+z", "z+", "up"):
                        self.pose["z"] += 4.5

            return [
                f"{task.action.value} executed"
                for task in command.tasks
            ]

    robot = InaccurateRobot(create_fake_robot())

    engine = ExecutionEngine(robot)

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            {
                "action": "MOVE",
                "direction": "X",
                "distance": 5,
                "unit": "mm",
            }
        ]
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    assert result.action == Action.MOVE

    assert result.status == ExecutionStatus.VERIFICATION_FAILED

    assert result.requested_distance == 5.0
    assert result.actual_distance == 4.5

    assert result.error == 0.5
    assert result.tolerance == 0.2

    assert result.verification is False

    assert result.failure_reason is not None
    assert "exceeds tolerance" in result.failure_reason.lower()


# ============================================================
# TEST 6: EXECUTOR WRITES RESULT TO LOGGER
# ============================================================

def test_executor_logs_result(tmp_path):

    from execution.logger import ExecutionLogger

    log_file = tmp_path / "execution_log.json"

    logger = ExecutionLogger(
        log_file=str(log_file)
    )

    robot = FakeRobot(create_fake_robot())

    engine = ExecutionEngine(
        robot,
        logger=logger,
    )

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            {
                "action": "MOVE",
                "direction": "X",
                "distance": 5,
                "unit": "mm",
            }
        ]
    )

    results = engine.execute(command)

    assert len(results) == 1
    assert results[0].status == ExecutionStatus.SUCCESS

    logs = logger.read_logs()

    assert len(logs) == 1

    assert logs[0]["robot_id"] == "fake_robot"
    assert logs[0]["action"] == "MOVE"
    assert logs[0]["status"] == "SUCCESS"

    assert logs[0]["requested_axis"] == "X"
    assert logs[0]["requested_distance"] == 5.0

    assert logs[0]["actual_axis"] == "X"
    assert logs[0]["actual_distance"] == 5.0

    assert logs[0]["verification"] is True


# ============================================================
# TEST 7: MULTI-STEP EXECUTION
# ============================================================

def test_executor_multi_step_move():

    robot = FakeRobot(create_fake_robot())

    engine = ExecutionEngine(robot)

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            {
                "action": "MOVE",
                "direction": "X",
                "distance": 5,
                "unit": "mm",
            },
            {
                "action": "MOVE",
                "direction": "Y",
                "distance": 5,
                "unit": "mm",
            },
            {
                "action": "MOVE",
                "direction": "Z",
                "distance": 5,
                "unit": "mm",
            },
        ],
    )

    results = engine.execute(command)

    assert len(results) == 3

    assert results[0].action == Action.MOVE
    assert results[0].status == ExecutionStatus.SUCCESS
    assert results[0].requested_axis == "X"
    assert results[0].requested_distance == 5.0
    assert results[0].actual_distance == 5.0
    assert results[0].verification is True

    assert results[1].action == Action.MOVE
    assert results[1].status == ExecutionStatus.SUCCESS
    assert results[1].requested_axis == "Y"
    assert results[1].requested_distance == 5.0
    assert results[1].actual_distance == 5.0
    assert results[1].verification is True

    assert results[2].action == Action.MOVE
    assert results[2].status == ExecutionStatus.SUCCESS
    assert results[2].requested_axis == "Z"
    assert results[2].requested_distance == 5.0
    assert results[2].actual_distance == 5.0
    assert results[2].verification is True

    assert robot.pose["x"] == 5.0
    assert robot.pose["y"] == 5.0
    assert robot.pose["z"] == 5.0


# ============================================================
# TEST 8: REPEATED AXIS MOVEMENTS
# ============================================================

def test_executor_repeated_axis_moves():

    robot = FakeRobot(create_fake_robot())

    engine = ExecutionEngine(robot)

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            {
                "action": "MOVE",
                "direction": "X",
                "distance": 5,
                "unit": "mm",
            },
            {
                "action": "MOVE",
                "direction": "X",
                "distance": 10,
                "unit": "mm",
            },
            {
                "action": "MOVE",
                "direction": "Y",
                "distance": 5,
                "unit": "mm",
            },
        ],
    )

    results = engine.execute(command)

    assert len(results) == 3

    assert results[0].status == ExecutionStatus.SUCCESS
    assert results[0].requested_axis == "X"
    assert results[0].requested_distance == 5.0
    assert results[0].actual_distance == 5.0
    assert results[0].verification is True

    assert results[1].status == ExecutionStatus.SUCCESS
    assert results[1].requested_axis == "X"
    assert results[1].requested_distance == 10.0
    assert results[1].actual_distance == 10.0
    assert results[1].verification is True

    assert results[2].status == ExecutionStatus.SUCCESS
    assert results[2].requested_axis == "Y"
    assert results[2].requested_distance == 5.0
    assert results[2].actual_distance == 5.0
    assert results[2].verification is True

    assert robot.pose["x"] == 15.0
    assert robot.pose["y"] == 5.0


# ============================================================
# TEST 9: VERIFICATION FAILURE IS WRITTEN TO LOGGER
# ============================================================

def test_executor_logs_verification_failure(tmp_path):

    from execution.logger import ExecutionLogger

    class InaccurateRobot(FakeRobot):

        def execute(self, command):

            for task in command.tasks:

                if task.action == Action.MOVE:

                    direction = task.direction.lower()

                    if direction in ("x", "+x", "x+"):
                        self.pose["x"] += 4.5

                    elif direction in ("y", "+y", "y+"):
                        self.pose["y"] += 4.5

                    elif direction in ("z", "+z", "z+", "up"):
                        self.pose["z"] += 4.5

            return [
                f"{task.action.value} executed"
                for task in command.tasks
            ]

    log_file = tmp_path / "execution_log.json"

    logger = ExecutionLogger(
        log_file=str(log_file)
    )

    robot = InaccurateRobot(create_fake_robot())

    engine = ExecutionEngine(
        robot,
        logger=logger,
    )

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            {
                "action": "MOVE",
                "direction": "X",
                "distance": 5,
                "unit": "mm",
            }
        ]
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    assert result.status == ExecutionStatus.VERIFICATION_FAILED
    assert result.verification is False
    assert result.requested_distance == 5.0
    assert result.actual_distance == 4.5
    assert result.error == 0.5

    logs = logger.read_logs()

    assert len(logs) == 1

    assert logs[0]["robot_id"] == "fake_robot"
    assert logs[0]["action"] == "MOVE"
    assert logs[0]["status"] == "VERIFICATION_FAILED"

    assert logs[0]["requested_axis"] == "X"
    assert logs[0]["requested_distance"] == 5.0

    assert logs[0]["actual_axis"] == "X"
    assert logs[0]["actual_distance"] == 4.5

    assert logs[0]["error"] == 0.5
    assert logs[0]["tolerance"] == 0.2

    assert logs[0]["verification"] is False

    assert logs[0]["failure_reason"] is not None
    assert "exceeds tolerance" in logs[0]["failure_reason"].lower()


# ============================================================
# TEST 10: NEGATIVE X MOVEMENT VERIFICATION
# ============================================================

def test_executor_negative_x_move():

    robot = FakeRobot(create_fake_robot())

    engine = ExecutionEngine(robot)

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            {
                "action": "MOVE",
                "direction": "-X",
                "distance": 5,
                "unit": "mm",
            }
        ]
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    assert result.action == Action.MOVE
    assert result.status == ExecutionStatus.SUCCESS

    assert result.requested_axis == "X"
    assert result.requested_distance == 5.0
    assert result.requested_unit == "mm"

    assert result.actual_axis == "X"
    assert result.actual_distance == 5.0
    assert result.actual_unit == "mm"

    assert result.error == 0.0
    assert result.verification is True

    assert robot.pose["x"] == -5.0


# ============================================================
# TEST 11: NEGATIVE Y MOVEMENT VERIFICATION
# ============================================================

def test_executor_negative_y_move():

    robot = FakeRobot(create_fake_robot())

    engine = ExecutionEngine(robot)

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            {
                "action": "MOVE",
                "direction": "-Y",
                "distance": 5,
                "unit": "mm",
            }
        ]
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    assert result.action == Action.MOVE
    assert result.status == ExecutionStatus.SUCCESS

    assert result.requested_axis == "Y"
    assert result.requested_distance == 5.0
    assert result.requested_unit == "mm"

    assert result.actual_axis == "Y"
    assert result.actual_distance == 5.0
    assert result.actual_unit == "mm"

    assert result.error == 0.0
    assert result.verification is True

    assert robot.pose["y"] == -5.0


# ============================================================
# TEST 12: NEGATIVE Z MOVEMENT VERIFICATION
# ============================================================

def test_executor_negative_z_move():

    robot = FakeRobot(create_fake_robot())

    engine = ExecutionEngine(robot)

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            {
                "action": "MOVE",
                "direction": "-Z",
                "distance": 5,
                "unit": "mm",
            }
        ]
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    assert result.action == Action.MOVE
    assert result.status == ExecutionStatus.SUCCESS

    assert result.requested_axis == "Z"
    assert result.requested_distance == 5.0
    assert result.requested_unit == "mm"

    assert result.actual_axis == "Z"
    assert result.actual_distance == 5.0
    assert result.actual_unit == "mm"

    assert result.error == 0.0
    assert result.verification is True

    assert robot.pose["z"] == -5.0


# ============================================================
# TEST 13: WRONG DIRECTION VERIFICATION FAILURE
# ============================================================

def test_executor_wrong_direction_move():

    class WrongDirectionRobot(FakeRobot):

        def execute(self, command):

            for task in command.tasks:

                if task.action == Action.MOVE:

                    # Requested -X
                    # Robot incorrectly moves +X

                    if task.direction.lower() in ("-x", "x-"):
                        self.pose["x"] += 5.0

            return [
                f"{task.action.value} executed"
                for task in command.tasks
            ]

    robot = WrongDirectionRobot(create_fake_robot())

    engine = ExecutionEngine(robot)

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            {
                "action": "MOVE",
                "direction": "-X",
                "distance": 5,
                "unit": "mm",
            }
        ]
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    assert result.action == Action.MOVE
    assert result.status == ExecutionStatus.VERIFICATION_FAILED

    assert result.requested_axis == "X"
    assert result.requested_distance == 5.0

    assert result.actual_axis == "X"
    assert result.actual_distance == 5.0

    assert result.error is None

    assert result.verification is False

    assert result.failure_reason is not None
    assert "direction" in result.failure_reason.lower()

    assert robot.pose["x"] == 5.0


# ============================================================
# TEST 14: WRONG AXIS VERIFICATION FAILURE
# ============================================================

def test_executor_wrong_axis_move():

    class WrongAxisRobot(FakeRobot):

        def execute(self, command):

            for task in command.tasks:

                if task.action == Action.MOVE:

                    # Requested X movement
                    # Robot incorrectly moves Y

                    if task.direction.lower() in ("x", "+x", "x+"):
                        self.pose["y"] += 5.0

            return [
                f"{task.action.value} executed"
                for task in command.tasks
            ]

    robot = WrongAxisRobot(create_fake_robot())

    engine = ExecutionEngine(robot)

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            {
                "action": "MOVE",
                "direction": "X",
                "distance": 5,
                "unit": "mm",
            }
        ]
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    assert result.action == Action.MOVE

    assert result.status == ExecutionStatus.VERIFICATION_FAILED

    assert result.requested_axis == "X"
    assert result.requested_distance == 5.0

    # Current executor detects no X movement
    assert result.actual_axis == "X"
    assert result.actual_distance == 0.0

    assert result.verification is False

    assert result.failure_reason is not None
    assert "movement" in result.failure_reason.lower()

    assert robot.pose["x"] == 0.0
    assert robot.pose["y"] == 5.0


# ============================================================
# TEST 15: ZERO MOVEMENT VERIFICATION FAILURE
# ============================================================

def test_zero_movement_is_verification_failure():

    class NoMovementFakeRobot(FakeRobot):

        def execute(self, command):

            # Pretend that the robot accepted and executed
            # the command, but intentionally do not change
            # the robot pose.

            return [
                f"{task.action.value} executed"
                for task in command.tasks
            ]

    robot = NoMovementFakeRobot(create_fake_robot())

    engine = ExecutionEngine(
        robot,
        tolerance=0.2,
    )

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            Task(
                action=Action.MOVE,
                direction="+x",
                distance=5,
                unit="mm",
            )
        ],
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    # --------------------------------------------------------
    # CHECK VERIFICATION FAILURE
    # --------------------------------------------------------

    assert result.status == ExecutionStatus.VERIFICATION_FAILED

    # --------------------------------------------------------
    # CHECK REQUESTED MOVEMENT
    # --------------------------------------------------------

    assert result.requested_axis == "X"
    assert result.requested_distance == 5.0
    assert result.requested_unit == "mm"

    # --------------------------------------------------------
    # CHECK ACTUAL MOVEMENT
    # --------------------------------------------------------

    assert result.actual_axis == "X"
    assert result.actual_distance == 0.0

    # --------------------------------------------------------
    # CHECK VERIFICATION
    # --------------------------------------------------------

    assert result.verification is False

    # Requested = 5 mm
    # Actual = 0 mm
    # Error = 5 mm

    assert result.error == 5.0

    # --------------------------------------------------------
    # CHECK FAILURE REASON
    # --------------------------------------------------------

    assert result.failure_reason is not None
    assert "no movement" in result.failure_reason.lower()

    # --------------------------------------------------------
    # CONFIRM ROBOT DID NOT MOVE
    # --------------------------------------------------------

    assert robot.pose["x"] == 0.0
    assert robot.pose["y"] == 0.0
    assert robot.pose["z"] == 0.0
    # ============================================================
# TEST 16: EXCESSIVE MOVEMENT VERIFICATION FAILURE
# ============================================================

def test_excessive_movement_is_verification_failure():

    class ExcessiveMovementFakeRobot(FakeRobot):

        def execute(self, command):

            # Pretend the robot executed the command,
            # but intentionally move 8 mm instead of requested 5 mm.

            for task in command.tasks:

                if task.action == Action.MOVE:

                    direction = task.direction.lower()

                    if direction in {"+x", "x+"}:
                        self.pose["x"] += 8.0

                    elif direction in {"-x", "x-"}:
                        self.pose["x"] -= 8.0

                    elif direction in {"+y", "y+"}:
                        self.pose["y"] += 8.0

                    elif direction in {"-y", "y-"}:
                        self.pose["y"] -= 8.0

                    elif direction in {"+z", "z+"}:
                        self.pose["z"] += 8.0

                    elif direction in {"-z", "z-"}:
                        self.pose["z"] -= 8.0

            return [
                f"{task.action.value} executed"
                for task in command.tasks
            ]

    robot = ExcessiveMovementFakeRobot(create_fake_robot())

    engine = ExecutionEngine(
        robot,
        tolerance=0.2,
    )

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            Task(
                action=Action.MOVE,
                direction="+x",
                distance=5,
                unit="mm"
            )
        ],
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    # Verification must detect that 8 mm was moved
    # instead of the requested 5 mm.
    assert result.status == ExecutionStatus.VERIFICATION_FAILED

    assert result.requested_axis == "X"
    assert result.requested_distance == 5.0
    assert result.requested_unit == "mm"

    assert result.actual_axis == "X"
    assert result.actual_distance == 8.0

    # Error = |5 - 8| = 3 mm
    assert result.error == 3.0

    assert result.tolerance == 0.2

    assert result.verification is False

    assert result.failure_reason is not None
    assert "exceeds tolerance" in result.failure_reason.lower()

    # Confirm fake robot actually moved 8 mm.
    assert robot.pose["x"] == 8.0
    # ============================================================
# TEST 17: WRONG DIRECTION VERIFICATION FAILURE
# ============================================================

def test_wrong_direction_is_verification_failure():

    class WrongDirectionFakeRobot(FakeRobot):

        def execute(self, command):

            for task in command.tasks:

                if task.action == Action.MOVE:

                    direction = task.direction.lower()

                    # Requested +X, but intentionally move -X.
                    if direction in {"+x", "x+"}:
                        self.pose["x"] -= 5.0

                    elif direction in {"-x", "x-"}:
                        self.pose["x"] += 5.0

            return [
                f"{task.action.value} executed"
                for task in command.tasks
            ]

    robot = WrongDirectionFakeRobot(create_fake_robot())

    engine = ExecutionEngine(
        robot,
        tolerance=0.2,
    )

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            Task(
                action=Action.MOVE,
                direction="+x",
                distance=5,
                unit="mm"
            )
        ],
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    assert result.status == ExecutionStatus.VERIFICATION_FAILED

    assert result.requested_axis == "X"
    assert result.requested_distance == 5.0
    assert result.requested_unit == "mm"

    assert result.actual_axis == "X"
    assert result.actual_distance == 5.0

    assert result.verification is False

    assert result.failure_reason is not None
    assert "direction" in result.failure_reason.lower()

    # Robot actually moved in the negative X direction.
    assert robot.pose["x"] == -5.0
    # ============================================================
# TEST 18: EXECUTION RESULT IS LOGGED
# ============================================================

def test_execution_result_is_logged(tmp_path):

    from execution.logger import ExecutionLogger

    log_file = tmp_path / "execution_log.json"

    logger = ExecutionLogger(
        log_file=str(log_file)
    )

    robot = FakeRobot(create_fake_robot())

    engine = ExecutionEngine(
        robot,
        tolerance=0.2,
        logger=logger,
    )

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            Task(
                action=Action.MOVE,
                direction="+x",
                distance=5,
                unit="mm"
            )
        ],
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    # Execution itself must succeed.
    assert result.status == ExecutionStatus.SUCCESS

    # An execution ID must be generated.
    assert result.execution_id

    # Read the saved log.
    logs = logger.read_logs()

    assert len(logs) == 1

    logged_result = logs[0]

    # Logged execution must match the returned execution result.
    assert logged_result["execution_id"] == result.execution_id
    assert logged_result["robot_id"] == "fake_robot"
    assert logged_result["action"] == "MOVE"
    assert logged_result["status"] == "SUCCESS"

    # Verify movement information was logged.
    assert logged_result["requested_axis"] == "X"
    assert logged_result["requested_distance"] == 5.0
    assert logged_result["requested_unit"] == "mm"

    assert logged_result["actual_axis"] == "X"
    assert logged_result["actual_distance"] == 5.0

    assert logged_result["verification"] is True

    # Confirm the log file was created.
    assert log_file.exists()
    # ============================================================
# TEST 19: FAILED EXECUTION IS LOGGED
# ============================================================

def test_failed_execution_is_logged(tmp_path):

    from execution.logger import ExecutionLogger

    class FailingFakeRobot(FakeRobot):

        def execute(self, command):
            raise RuntimeError("Simulated robot execution failure")

    log_file = tmp_path / "execution_log.json"

    logger = ExecutionLogger(
        log_file=str(log_file)
    )

    robot = FailingFakeRobot(create_fake_robot())

    engine = ExecutionEngine(
        robot,
        tolerance=0.2,
        logger=logger,
    )

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            Task(
                action=Action.MOVE,
                direction="+x",
                distance=5,
                unit="mm"
            )
        ],
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    # Execution must be marked as FAILED.
    assert result.status == ExecutionStatus.FAILED

    # Failure reason must be recorded.
    assert result.failure_reason is not None
    assert "simulated robot execution failure" in result.failure_reason.lower()

    # Read execution log.
    logs = logger.read_logs()

    assert len(logs) == 1

    logged_result = logs[0]

    # Verify failed execution was logged.
    assert logged_result["execution_id"] == result.execution_id
    assert logged_result["robot_id"] == "fake_robot"
    assert logged_result["action"] == "MOVE"
    assert logged_result["status"] == "FAILED"

    assert logged_result["failure_reason"] is not None
    assert "simulated robot execution failure" in logged_result["failure_reason"].lower()

    # Log file must exist.
    assert log_file.exists()
    # ============================================================
# TEST 20: VERIFICATION FAILURE IS LOGGED
# ============================================================

def test_verification_failure_is_logged(tmp_path):

    from execution.logger import ExecutionLogger

    class WrongMovementFakeRobot(FakeRobot):

        def execute(self, command):

            # Robot accepts the command but moves the wrong
            # distance: 8 mm instead of requested 5 mm.

            for task in command.tasks:

                if task.action == Action.MOVE:

                    direction = task.direction.lower()

                    if direction in {"+x", "x+"}:
                        self.pose["x"] += 8.0

                    elif direction in {"-x", "x-"}:
                        self.pose["x"] -= 8.0

            return [
                f"{task.action.value} executed"
                for task in command.tasks
            ]

    log_file = tmp_path / "execution_log.json"

    logger = ExecutionLogger(
        log_file=str(log_file)
    )

    robot = WrongMovementFakeRobot(create_fake_robot())

    engine = ExecutionEngine(
        robot,
        tolerance=0.2,
        logger=logger,
    )

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            Task(
                action=Action.MOVE,
                direction="+x",
                distance=5,
                unit="mm"
            )
        ],
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    # Execution itself succeeded, but verification failed.
    assert result.status == ExecutionStatus.VERIFICATION_FAILED

    assert result.verification is False

    assert result.requested_axis == "X"
    assert result.requested_distance == 5.0

    assert result.actual_axis == "X"
    assert result.actual_distance == 8.0

    assert result.error == 3.0

    # Check the log.
    logs = logger.read_logs()

    assert len(logs) == 1

    logged_result = logs[0]

    assert logged_result["execution_id"] == result.execution_id
    assert logged_result["robot_id"] == "fake_robot"
    assert logged_result["action"] == "MOVE"

    assert logged_result["status"] == "VERIFICATION_FAILED"

    assert logged_result["verification"] is False

    assert logged_result["requested_distance"] == 5.0
    assert logged_result["actual_distance"] == 8.0
    assert logged_result["error"] == 3.0

    assert logged_result["failure_reason"] is not None
    assert "exceeds tolerance" in logged_result["failure_reason"].lower()

    assert log_file.exists()
    # ============================================================
# TEST 21: ROBOT VALIDATION REJECTION
# ============================================================

def test_robot_validation_rejects_command():

    class RejectingFakeRobot(FakeRobot):

        def validate(self, command):
            return False

        def execute(self, command):
            raise AssertionError(
                "Robot execute() must not be called after validation failure"
            )

    robot = RejectingFakeRobot(create_fake_robot())

    engine = ExecutionEngine(
        robot,
        tolerance=0.2,
    )

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            Task(
                action=Action.MOVE,
                direction="+x",
                distance=5,
                unit="mm"
            )
        ],
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    # Command must be rejected before execution.
    assert result.status == ExecutionStatus.REJECTED

    assert result.robot_id == "fake_robot"

    assert result.action == Action.MOVE

    assert result.message == "Robot rejected the command"

    assert result.failure_reason == "Robot validation failed"

    # Robot pose must remain unchanged.
    assert robot.pose["x"] == 0.0
    assert robot.pose["y"] == 0.0
    assert robot.pose["z"] == 0.0
    # ============================================================
# TEST 22: INITIAL ROBOT STATE FAILURE
# ============================================================

def test_initial_state_failure_prevents_execution():

    class InitialStateFailureFakeRobot(FakeRobot):

        def get_status(self):
            raise RuntimeError("Initial robot state unavailable")

        def execute(self, command):
            raise AssertionError(
                "Robot execute() must not be called when initial state is unavailable"
            )

    robot = InitialStateFailureFakeRobot(create_fake_robot())

    engine = ExecutionEngine(
        robot,
        tolerance=0.2,
    )

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            Task(
                action=Action.MOVE,
                direction="+x",
                distance=5,
                unit="mm"
            )
        ],
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    # Execution must fail safely.
    assert result.status == ExecutionStatus.FAILED

    assert result.robot_id == "fake_robot"

    assert result.action == Action.MOVE

    assert result.message == "Unable to read initial robot state"

    assert result.failure_reason is not None
    assert "initial robot state unavailable" in result.failure_reason.lower()

    # No initial pose should be recorded because
    # the robot state could not be read.
    assert result.initial_x is None
    assert result.initial_y is None
    assert result.initial_z is None
    assert result.initial_r is None

    # Robot pose must remain unchanged.
    assert robot.pose["x"] == 0.0
    assert robot.pose["y"] == 0.0
    assert robot.pose["z"] == 0.0
    # ============================================================
# TEST 23: FINAL ROBOT STATE FAILURE
# ============================================================

def test_final_state_failure_prevents_false_verification():

    class FinalStateFailureFakeRobot(FakeRobot):

        def __init__(self, robot):
            super().__init__(robot)
            self.status_calls = 0

        def get_status(self):
            self.status_calls += 1

            # First call = initial state: available.
            if self.status_calls == 1:
                return super().get_status()

            # Second call = final state: unavailable.
            raise RuntimeError("Final robot state unavailable")

        def execute(self, command):
            # Robot really moves 5 mm.
            for task in command.tasks:

                if task.action == Action.MOVE:

                    direction = task.direction.lower()

                    if direction in {"+x", "x+"}:
                        self.pose["x"] += 5.0

                    elif direction in {"-x", "x-"}:
                        self.pose["x"] -= 5.0

            return [
                f"{task.action.value} executed"
                for task in command.tasks
            ]

    robot = FinalStateFailureFakeRobot(create_fake_robot())

    engine = ExecutionEngine(
        robot,
        tolerance=0.2,
    )

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            Task(
                action=Action.MOVE,
                direction="+x",
                distance=5,
                unit="mm"
            )
        ],
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    # Execution result must be FAILED because
    # final state could not be obtained.
    assert result.status == ExecutionStatus.FAILED

    assert result.robot_id == "fake_robot"
    assert result.action == Action.MOVE

    assert result.message == "Unable to read final robot state"

    assert result.failure_reason is not None
    assert "final robot state unavailable" in result.failure_reason.lower()

    # Initial pose was successfully captured.
    assert result.initial_x == 0.0
    assert result.initial_y == 0.0
    assert result.initial_z == 0.0

    # Final pose must NOT be recorded because
    # the final state could not be read.
    assert result.final_x is None
    assert result.final_y is None
    assert result.final_z is None
    assert result.final_r is None

    # Most importantly: execution must NOT claim
    # that movement verification succeeded.
    assert result.verification is None

    # The robot did actually move, but we cannot verify
    # the final state through the interface.
    assert robot.pose["x"] == 5.0
    # ============================================================
# TEST 24: STOP COMMAND
# ============================================================

def test_stop_command_is_recorded_as_stopped():

    robot = FakeRobot(create_fake_robot())

    engine = ExecutionEngine(
        robot,
        tolerance=0.2,
    )

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            Task(
                action=Action.STOP
            )
        ],
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    # STOP must have its own execution status.
    assert result.status == ExecutionStatus.STOPPED

    assert result.robot_id == "fake_robot"

    assert result.action == Action.STOP

    # STOP is not a movement command,
    # therefore movement verification is not performed.
    assert result.verification is None

    assert result.requested_axis is None
    assert result.requested_distance is None
    assert result.actual_axis is None
    assert result.actual_distance is None

    # Robot pose must remain unchanged.
    assert robot.pose["x"] == 0.0
    assert robot.pose["y"] == 0.0
    assert robot.pose["z"] == 0.0
    # ============================================================
# TEST 25: HOME COMMAND
# ============================================================

def test_home_command_is_successful():

    robot = FakeRobot(create_fake_robot())

    engine = ExecutionEngine(
        robot,
        tolerance=0.2,
    )

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            Task(
                action=Action.HOME
            )
        ],
    )

    results = engine.execute(command)

    assert len(results) == 1

    result = results[0]

    # HOME should execute successfully.
    assert result.status == ExecutionStatus.SUCCESS

    assert result.robot_id == "fake_robot"

    assert result.action == Action.HOME

    assert result.execution_id

    # HOME does not have movement-distance verification.
    assert result.verification is None

    assert result.requested_axis is None
    assert result.requested_distance is None

    assert result.actual_axis is None
    assert result.actual_distance is None

    # Initial state should have been captured.
    assert result.initial_x == 0.0
    assert result.initial_y == 0.0
    assert result.initial_z == 0.0
    assert result.initial_r == 0.0

    # Final state should also be available.
    assert result.final_x == 0.0
    assert result.final_y == 0.0
    assert result.final_z == 0.0
    assert result.final_r == 0.0
    # ============================================================
# TEST 26: MULTI-TASK EXECUTION
# ============================================================

def test_multiple_tasks_are_executed_and_recorded():

    robot = FakeRobot(create_fake_robot())

    engine = ExecutionEngine(
        robot,
        tolerance=0.2,
    )

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            Task(
                action=Action.MOVE,
                direction="+x",
                distance=5,
                unit="mm"
            ),
            Task(
                action=Action.MOVE,
                direction="+y",
                distance=5,
                unit="mm"
            ),
            Task(
                action=Action.HOME
            ),
        ],
    )

    results = engine.execute(command)

    # Three tasks must produce three results.
    assert len(results) == 3

    # --------------------------------------------------------
    # Task 1: MOVE +X 5 mm
    # --------------------------------------------------------

    result_1 = results[0]

    assert result_1.action == Action.MOVE
    assert result_1.status == ExecutionStatus.SUCCESS
    assert result_1.verification is True

    assert result_1.requested_axis == "X"
    assert result_1.requested_distance == 5.0
    assert result_1.actual_axis == "X"
    assert result_1.actual_distance == 5.0

    # --------------------------------------------------------
    # Task 2: MOVE +Y 5 mm
    # --------------------------------------------------------

    result_2 = results[1]

    assert result_2.action == Action.MOVE
    assert result_2.status == ExecutionStatus.SUCCESS
    assert result_2.verification is True

    assert result_2.requested_axis == "Y"
    assert result_2.requested_distance == 5.0
    assert result_2.actual_axis == "Y"
    assert result_2.actual_distance == 5.0

    # --------------------------------------------------------
    # Task 3: HOME
    # --------------------------------------------------------

    result_3 = results[2]

    assert result_3.action == Action.HOME
    assert result_3.status == ExecutionStatus.SUCCESS

    # HOME is not movement-distance verification.
    assert result_3.verification is None

    # --------------------------------------------------------
    # Execution IDs must be unique.
    # --------------------------------------------------------

    execution_ids = {
        result.execution_id
        for result in results
    }

    assert len(execution_ids) == 3

    # --------------------------------------------------------
    # Final robot state should be available.
    # --------------------------------------------------------

    assert result_3.final_x is not None
    assert result_3.final_y is not None
    assert result_3.final_z is not None
    assert result_3.final_r is not None
    # ============================================================
# TEST 27: MULTI-TASK EXECUTION LOGGING
# ============================================================

def test_multiple_tasks_are_logged_individually(tmp_path):

    from execution.logger import ExecutionLogger

    log_file = tmp_path / "execution_log.json"

    logger = ExecutionLogger(
        log_file=str(log_file)
    )

    robot = FakeRobot(create_fake_robot())

    engine = ExecutionEngine(
        robot,
        tolerance=0.2,
        logger=logger,
    )

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            Task(
                action=Action.MOVE,
                direction="+x",
                distance=5,
                unit="mm"
            ),
            Task(
                action=Action.MOVE,
                direction="+y",
                distance=5,
                unit="mm"
            ),
            Task(
                action=Action.HOME
            ),
        ],
    )

    results = engine.execute(command)

    # Three tasks must produce three results.
    assert len(results) == 3

    # Read the execution log.
    logs = logger.read_logs()

    # Every task must have its own log entry.
    assert len(logs) == 3

    # --------------------------------------------------------
    # Verify execution IDs are unique and preserved.
    # --------------------------------------------------------

    result_ids = {
        result.execution_id
        for result in results
    }

    log_ids = {
        log["execution_id"]
        for log in logs
    }

    assert len(result_ids) == 3
    assert len(log_ids) == 3
    assert result_ids == log_ids

    # --------------------------------------------------------
    # Verify task order is preserved.
    # --------------------------------------------------------

    assert logs[0]["action"] == "MOVE"
    assert logs[0]["requested_axis"] == "X"
    assert logs[0]["requested_distance"] == 5.0
    assert logs[0]["status"] == "SUCCESS"

    assert logs[1]["action"] == "MOVE"
    assert logs[1]["requested_axis"] == "Y"
    assert logs[1]["requested_distance"] == 5.0
    assert logs[1]["status"] == "SUCCESS"

    assert logs[2]["action"] == "HOME"
    assert logs[2]["status"] == "SUCCESS"

    # HOME does not have movement verification.
    assert logs[2]["verification"] is None

    # Log file must exist.
    assert log_file.exists()
    # ============================================================
# TEST 28: LOG PERSISTENCE / APPENDING
# ============================================================

def test_execution_log_persists_between_runs(tmp_path):

    from execution.logger import ExecutionLogger

    log_file = tmp_path / "execution_log.json"

    logger = ExecutionLogger(
        log_file=str(log_file)
    )

    robot = FakeRobot(create_fake_robot())

    engine = ExecutionEngine(
        robot,
        tolerance=0.2,
        logger=logger,
    )

    # --------------------------------------------------------
    # FIRST EXECUTION
    # --------------------------------------------------------

    command1 = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            Task(
                action=Action.MOVE,
                direction="+x",
                distance=5,
                unit="mm"
            )
        ],
    )

    results1 = engine.execute(command1)

    assert len(results1) == 1

    logs_after_first_run = logger.read_logs()

    assert len(logs_after_first_run) == 1

    first_execution_id = logs_after_first_run[0]["execution_id"]

    # --------------------------------------------------------
    # SECOND EXECUTION
    # --------------------------------------------------------

    command2 = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            Task(
                action=Action.MOVE,
                direction="+y",
                distance=5,
                unit="mm"
            )
        ],
    )

    results2 = engine.execute(command2)

    assert len(results2) == 1

    logs_after_second_run = logger.read_logs()

    # Previous log must still exist.
    assert len(logs_after_second_run) == 2

    # First execution must not be overwritten.
    assert logs_after_second_run[0]["execution_id"] == first_execution_id

    # Second execution must have a different ID.
    assert logs_after_second_run[1]["execution_id"] != first_execution_id

    # Verify order.
    assert logs_after_second_run[0]["requested_axis"] == "X"
    assert logs_after_second_run[1]["requested_axis"] == "Y"

    # Verify both executions succeeded.
    assert logs_after_second_run[0]["status"] == "SUCCESS"
    assert logs_after_second_run[1]["status"] == "SUCCESS"

    # Log file must still exist.
    assert log_file.exists()
    # ============================================================
# TEST 29: END-TO-END EXECUTION + VERIFICATION + LOGGING
# ============================================================

def test_end_to_end_execution_verification_and_logging(tmp_path):

    from execution.logger import ExecutionLogger

    # --------------------------------------------------------
    # Create execution logger
    # --------------------------------------------------------

    log_file = tmp_path / "execution_log.json"

    logger = ExecutionLogger(
        log_file=str(log_file)
    )

    # --------------------------------------------------------
    # Create fake robot
    # --------------------------------------------------------

    robot = FakeRobot(create_fake_robot())

    # --------------------------------------------------------
    # Create execution engine
    # --------------------------------------------------------

    engine = ExecutionEngine(
        robot,
        tolerance=0.2,
        logger=logger,
    )

    # --------------------------------------------------------
    # Create Universal Command
    # --------------------------------------------------------

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            Task(
                action=Action.MOVE,
                direction="+x",
                distance=5,
                unit="mm"
            )
        ],
    )

    # --------------------------------------------------------
    # Execute command
    # --------------------------------------------------------

    results = engine.execute(command)

    # --------------------------------------------------------
    # STEP 1: Execution result exists
    # --------------------------------------------------------

    assert len(results) == 1

    result = results[0]

    # --------------------------------------------------------
    # STEP 2: Execution succeeded
    # --------------------------------------------------------

    assert result.status == ExecutionStatus.SUCCESS

    # --------------------------------------------------------
    # STEP 3: Correct action
    # --------------------------------------------------------

    assert result.action == Action.MOVE

    # --------------------------------------------------------
    # STEP 4: Requested movement is recorded
    # --------------------------------------------------------

    assert result.requested_axis == "X"
    assert result.requested_distance == 5.0
    assert result.requested_unit == "mm"

    # --------------------------------------------------------
    # STEP 5: Actual movement is recorded
    # --------------------------------------------------------

    assert result.actual_axis == "X"
    assert result.actual_distance == 5.0
    assert result.actual_unit == "mm"

    # --------------------------------------------------------
    # STEP 6: Verification passed
    # --------------------------------------------------------

    assert result.verification is True
    assert result.error is not None
    assert result.error <= 0.2

    # --------------------------------------------------------
    # STEP 7: Initial and final poses are recorded
    # --------------------------------------------------------

    assert result.initial_x is not None
    assert result.initial_y is not None
    assert result.initial_z is not None
    assert result.initial_r is not None

    assert result.final_x is not None
    assert result.final_y is not None
    assert result.final_z is not None
    assert result.final_r is not None

    # --------------------------------------------------------
    # STEP 8: Execution ID exists
    # --------------------------------------------------------

    assert result.execution_id

    # --------------------------------------------------------
    # STEP 9: Verify log was created
    # --------------------------------------------------------

    assert log_file.exists()

    logs = logger.read_logs()

    assert len(logs) == 1

    log = logs[0]

    # --------------------------------------------------------
    # STEP 10: Logged result matches execution result
    # --------------------------------------------------------

    assert log["execution_id"] == result.execution_id
    assert log["robot_id"] == "fake_robot"
    assert log["action"] == "MOVE"
    assert log["status"] == "SUCCESS"

    # --------------------------------------------------------
    # STEP 11: Logged verification information
    # --------------------------------------------------------

    assert log["requested_axis"] == "X"
    assert log["requested_distance"] == 5.0
    assert log["actual_axis"] == "X"
    assert log["actual_distance"] == 5.0
    assert log["verification"] is True

    # --------------------------------------------------------
    # END-TO-END PIPELINE PASSED
    # --------------------------------------------------------
    # ============================================================
# TEST 30: VERIFIER TOLERANCE BOUNDARY
# ============================================================

def test_movement_verification_at_exact_tolerance():

    from execution.models import ActualMovement, RequestedMovement
    from execution.verifier import MovementVerifier

    verifier = MovementVerifier(tolerance=0.2)

    requested = RequestedMovement(
        axis="X",
        distance=5,
        unit="mm",
        direction=1,
    )

    actual = ActualMovement(
        axis="X",
        distance=4.8,
        unit="mm",
        direction=1,
    )

    result = verifier.verify(
        requested=requested,
        actual=actual,
    )

    # Error is exactly the allowed tolerance.
    assert result.error == 0.2

    # Movement at the tolerance boundary must pass.
    assert result.passed is True

    assert result.tolerance == 0.2
    # ============================================================
# TEST 31: VERIFIER JUST OUTSIDE TOLERANCE
# ============================================================

def test_movement_verification_just_outside_tolerance():

    from execution.models import ActualMovement, RequestedMovement
    from execution.verifier import MovementVerifier

    verifier = MovementVerifier(tolerance=0.2)

    requested = RequestedMovement(
        axis="X",
        distance=5,
        unit="mm",
        direction=1,
    )

    actual = ActualMovement(
        axis="X",
        distance=4.79,
        unit="mm",
        direction=1,
    )

    result = verifier.verify(
        requested=requested,
        actual=actual,
    )

    # Error is greater than the allowed tolerance.
    assert result.error == 0.21

    # Movement outside tolerance must fail.
    assert result.passed is False

    assert result.tolerance == 0.2
   # ============================================================
# TEST 32: ROBOT-INDEPENDENT EXECUTION
# ============================================================

def test_execution_engine_works_with_robot_interface(tmp_path):

    from execution.logger import ExecutionLogger

    logger = ExecutionLogger(
        log_file="logs/test_robot_independent.json"
    )

    # FakeRobot implements RobotInterface.
    robot = FakeRobot(create_fake_robot())

    engine = ExecutionEngine(
        robot,
        tolerance=0.2,
        logger=logger,
    )

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            Task(
                action=Action.MOVE,
                direction="+z",
                distance=5,
                unit="mm"
            )
        ],
    )

    results = engine.execute(command)

    # One command should produce one result.
    assert len(results) == 1

    result = results[0]

    # Execution must succeed.
    assert result.status == ExecutionStatus.SUCCESS

    # Requested movement.
    assert result.requested_axis == "Z"
    assert result.requested_distance == 5.0

    # Actual movement.
    assert result.actual_axis == "Z"
    assert result.actual_distance == 5.0

    # Verification must pass.
    assert result.verification is True

    # Execution ID must exist.
    assert result.execution_id

    # Robot ID must be preserved.
    assert result.robot_id == "fake_robot"
    # ============================================================
# TEST 32: ROBOT-INDEPENDENT EXECUTION
# ============================================================

def test_execution_engine_works_with_robot_interface(tmp_path):

    from execution.logger import ExecutionLogger
    # Keep test output out of the repository working tree.
    log_file = tmp_path / "test_robot_independent.json"

    logger = ExecutionLogger(
        log_file=str(log_file)
    )

    # --------------------------------------------------------
    # FakeRobot implements RobotInterface
    # --------------------------------------------------------

    robot = FakeRobot(
        create_fake_robot()
    )

    # --------------------------------------------------------
    # Create Member 2 execution engine
    # --------------------------------------------------------

    engine = ExecutionEngine(
        robot,
        tolerance=0.2,
        logger=logger,
    )

    # --------------------------------------------------------
    # Universal command
    # --------------------------------------------------------

    command = UniversalCommand(
        robot_id="fake_robot",
        tasks=[
            Task(
                action=Action.MOVE,
                direction="+z",
                distance=5,
                unit="mm",
            )
        ],
    )

    # --------------------------------------------------------
    # Execute command
    # --------------------------------------------------------

    results = engine.execute(command)

    # --------------------------------------------------------
    # Verify result count
    # --------------------------------------------------------

    assert len(results) == 1

    result = results[0]

    # --------------------------------------------------------
    # Verify execution success
    # --------------------------------------------------------

    assert result.status == ExecutionStatus.SUCCESS

    # --------------------------------------------------------
    # Verify requested movement
    # --------------------------------------------------------

    assert result.requested_axis == "Z"

    assert result.requested_distance == 5.0

    assert result.requested_unit == "mm"

    # --------------------------------------------------------
    # Verify actual movement
    # --------------------------------------------------------

    assert result.actual_axis == "Z"

    assert result.actual_distance == 5.0

    assert result.actual_unit == "mm"

    # --------------------------------------------------------
    # Verify movement verification
    # --------------------------------------------------------

    assert result.verification is True

    assert result.error == 0.0

    assert result.tolerance == 0.2

    # --------------------------------------------------------
    # Verify execution ID
    # --------------------------------------------------------

    assert result.execution_id

    # --------------------------------------------------------
    # Verify robot ID
    # --------------------------------------------------------

    assert result.robot_id == "fake_robot"

    # --------------------------------------------------------
    # Verify initial pose
    # --------------------------------------------------------

    assert result.initial_x == 0.0

    assert result.initial_y == 0.0

    assert result.initial_z == 0.0

    assert result.initial_r == 0.0

    # --------------------------------------------------------
    # Verify final pose
    # --------------------------------------------------------

    assert result.final_x == 0.0

    assert result.final_y == 0.0

    assert result.final_z == 5.0

    assert result.final_r == 0.0

    # --------------------------------------------------------
    # Verify execution was logged
    # --------------------------------------------------------

    logs = logger.read_logs()

    assert len(logs) == 1

    assert logs[0]["robot_id"] == "fake_robot"

    assert logs[0]["action"] == "MOVE"

    assert logs[0]["status"] == "SUCCESS"

    assert logs[0]["verification"] is True
    # ============================================================
# TEST 33: WEBOTS ADAPTER + EXECUTION ENGINE INTEGRATION
# ============================================================

def test_execution_engine_works_with_webots_adapter(tmp_path):

    from app.adapters.webots import WebotsRobotAdapter
    from execution.logger import ExecutionLogger

    # --------------------------------------------------------
    # Fake Webots client
    # --------------------------------------------------------

    class FakeWebotsClient:

        def __init__(self):
            self.timeout = 5.0

            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.r = 0.0

        def request(self, payload, response_timeout=None):

            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if payload["type"] == "status":

                return {
                    "ok": True,
                    "state": (
                        f"X={self.x} "
                        f"Y={self.y} "
                        f"Z={self.z} "
                        f"R={self.r}"
                    ),
                }

            # ------------------------------------------------
            # EXECUTE
            # ------------------------------------------------

            if payload["type"] == "execute":

                for task in payload["tasks"]:

                    if task["action"] == "MOVE":

                        direction = task.get(
                            "direction",
                            ""
                        ).lower()

                        distance = float(
                            task["distance"]
                        )

                        if direction == "+x":
                            self.x += distance

                        elif direction == "-x":
                            self.x -= distance

                        elif direction == "+y":
                            self.y += distance

                        elif direction == "-y":
                            self.y -= distance

                        elif direction == "+z":
                            self.z += distance

                        elif direction == "-z":
                            self.z -= distance

                return {
                    "ok": True,
                    "results": [
                        "MOVE executed successfully"
                    ],
                }

            raise RuntimeError(
                "Unknown Webots request"
            )

    # --------------------------------------------------------
    # Create generic robot metadata
    # --------------------------------------------------------

    webots_robot = Robot(
        robot_id="webots_001",
        name="Webots Magician Lite",
        robot_type=RobotType.ROBOTIC_ARM,
        manufacturer="Webots",
        model="Magician Lite",
        adapter_type="webots",
        capabilities=frozenset({
            Action.MOVE,
            Action.ROTATE,
            Action.STOP,
            Action.HOME,
            Action.GET_STATUS,
        }),
    )

    # --------------------------------------------------------
    # Create fake Webots client
    # --------------------------------------------------------

    client = FakeWebotsClient()

    # --------------------------------------------------------
    # Create Webots adapter
    # --------------------------------------------------------

    robot = WebotsRobotAdapter(
        webots_robot,
        client=client,
    )

    # --------------------------------------------------------
    # Create execution logger
    # --------------------------------------------------------

    # Keep test output out of the repository working tree.
    log_file = tmp_path / "test_webots_execution.json"

    logger = ExecutionLogger(
        log_file=str(log_file)
    )

    # --------------------------------------------------------
    # Create Member 2 execution engine
    # --------------------------------------------------------

    engine = ExecutionEngine(
        robot,
        tolerance=0.2,
        logger=logger,
    )

    # --------------------------------------------------------
    # Universal command
    # --------------------------------------------------------

    command = UniversalCommand(
        robot_id="webots_001",
        tasks=[
            Task(
                action=Action.MOVE,
                direction="+x",
                distance=5,
                unit="mm",
            )
        ],
    )

    # --------------------------------------------------------
    # Execute command
    # --------------------------------------------------------

    results = engine.execute(command)

    # --------------------------------------------------------
    # Basic result verification
    # --------------------------------------------------------

    assert len(results) == 1

    result = results[0]

    assert result.status == ExecutionStatus.SUCCESS

    assert result.robot_id == "webots_001"

    assert result.action == Action.MOVE

    # --------------------------------------------------------
    # Requested movement verification
    # --------------------------------------------------------

    assert result.requested_axis == "X"

    assert result.requested_distance == 5.0

    assert result.requested_unit == "mm"

    # --------------------------------------------------------
    # Actual movement verification
    # --------------------------------------------------------

    assert result.actual_axis == "X"

    assert result.actual_distance == 5.0

    assert result.actual_unit == "mm"

    # --------------------------------------------------------
    # Movement verification
    # --------------------------------------------------------

    assert result.verification is True

    assert result.error == 0.0

    assert result.tolerance == 0.2

    # --------------------------------------------------------
    # Verify Webots state changed
    # --------------------------------------------------------

    assert client.x == 5.0

    assert client.y == 0.0

    assert client.z == 0.0

    assert client.r == 0.0

    # --------------------------------------------------------
    # Verify initial pose
    # --------------------------------------------------------

    assert result.initial_x == 0.0

    assert result.initial_y == 0.0

    assert result.initial_z == 0.0

    assert result.initial_r == 0.0

    # --------------------------------------------------------
    # Verify final pose
    # --------------------------------------------------------

    assert result.final_x == 5.0

    assert result.final_y == 0.0

    assert result.final_z == 0.0

    assert result.final_r == 0.0

    # --------------------------------------------------------
    # Verify execution log
    # --------------------------------------------------------

    logs = logger.read_logs()

    assert len(logs) == 1

    assert logs[0]["robot_id"] == "webots_001"

    assert logs[0]["action"] == "MOVE"

    assert logs[0]["status"] == "SUCCESS"

    assert logs[0]["verification"] is True
