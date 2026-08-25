"""Clear exception boundary around DobotLink and the Magician Lite."""

class DobotError(RuntimeError):
    """Base error for all DOBOT adapter failures."""


class DobotConfigurationError(DobotError):
    """Real-mode configuration is absent or unsafe."""


class DobotConnectionError(DobotError):
    """DobotLink or the robot could not be connected."""


class DobotTimeoutError(DobotConnectionError):
    """A bounded DOBOT operation timed out."""


class DobotNotReadyError(DobotError):
    """An operation was requested outside READY state."""


class DobotSafetyError(DobotError):
    """A requested physical movement failed configured safety checks."""


class DobotUnsupportedActionError(DobotError):
    """A universal action has no Phase 3 DOBOT mapping."""


class DobotEndEffectorError(DobotError):
    """The required gripper is not attached or did not respond."""
