# Universal LLM Robot Control Platform — Phase 3

Phase 3 adds an isolated, fail-closed DOBOT Magician Lite adapter to the
robot-independent language and gateway layers built in Phases 1 and 2.

> **Simulation is always the default. Real mode is explicit, requires local
> DobotLink plus a connected robot, and asks for operator confirmation before
> HOME, MOVE, GRIP, or RELEASE.**

## Architecture

```text
Natural language -> Gemini -> Pydantic Universal Command
                                  |
                                  v
                         Universal Gateway
                    selection / capability / safety
                         |                 |
                         v                 v
                    Mock adapter      DOBOT adapter
                    (simulation)       (real only)
                                             |
                                             v
                              DobotLink WebSocket RPC :9090
                                             |
                                             v
                                  Magician Lite / Magic Box
```

Vendor code is confined to `app/adapters/dobot/`. The universal command,
registry, selector, safety validator, and generic adapter contract contain no
DOBOT transport details. Importing the package does not load the SDK or touch
hardware.

## Phase 3 support boundary

The registered real robot is `dobot_001`, initially in `UNKNOWN` state.
Only a successful lifecycle transition makes it eligible:

```text
DISCONNECTED -> CONNECTING -> CONNECTED -> READY
                       failure -> ERROR
```

Supported Phase 3 actions:

- `GET_STATUS`
- `HOME`
- `MOVE` to one operator-configured test pose
- `STOP` using DobotLink's software queue stop
- `GRIP`
- `RELEASE`

`PICK` and `PLACE` are intentionally unsupported because they need
perception, task-space planning, and object-specific safety behavior. Commands
are fully mapped before the first physical action, so an unsupported step
rejects the entire multi-step command without partial execution.

The Phase 3 STOP is a **software stop**, not a certified emergency stop.
Always keep the manufacturer's emergency/safety controls available.

## Verified DOBOT interface

The implementation targets the interface verified from DobotLab 2.4.0 and its
bundled DobotLink/DobotRPC installation:

- DobotLink WebSocket RPC at `127.0.0.1:9090`
- Installed DobotRPC 4.8.5 source was used to verify the JSON-RPC envelope and module naming
- Direct `dobotlink.MagicianLite.*` JSON-RPC over the supported DobotLink WebSocket
- `SearchDobot`, `ConnectDobot`, `DisconnectDobot`, `GetPose`
- `SetHOMECmd`, `SetPTPCmd`, `QueuedCmdStop`
- `GetEndEffectorType`, `SetEndEffectorGripper`

Official references:

- [DOBOT DobotLink repository and architecture](https://github.com/Dobot-Arm/DobotLink)
- [Official Magician Lite DobotLab user manual (PDF)](https://download.dobot.cc/product-manual/magician-lite/cn/Dobot%20Magician%20Lite%20%E7%94%A8%E6%88%B7%E6%89%8B%E5%86%8C%EF%BC%88DobotLab%E7%89%88%EF%BC%89.pdf)

No serial protocol, USB packet format, or undocumented command was guessed.

## Requirements

Base/simulation:

- Python 3.11+
- Gemini API key
- `pip install -r requirements.txt`

Real Magician Lite:

- Windows computer physically connected to the Magician Lite/Magic Box
- DobotLab/DobotLink installed and running
- Modern synchronous WebSocket transport: `pip install -r requirements-hardware.txt`

The incompatible legacy DobotRPC dependency is not installed. Real mode uses the verified DobotLink JSON-RPC calls directly; simulation never opens a hardware connection.

## Configuration

Copy `.env.example` to the uncommitted `.env` and set
`GEMINI_API_KEY`. The real `.env` is ignored by Git.

Connection settings:

```text
DOBOTLINK_HOST=127.0.0.1
DOBOTLINK_PORT=9090
DOBOT_PORT_NAME=                 # optional when exactly one device is detected
DOBOT_CONNECT_TIMEOUT_SECONDS=10
DOBOT_COMMAND_TIMEOUT_MS=30000
DOBOT_MAX_RETRIES=2
DOBOT_PTP_MODE=1
DOBOT_VERIFY_TIMEOUT_SECONDS=5
DOBOT_VERIFY_START_DELAY_SECONDS=0.5
DOBOT_POSITION_TOLERANCE_MM=1
DOBOT_ROTATION_TOLERANCE_DEGREES=1
DOBOT_VERIFY_SAMPLES=3
```

Real MOVE is disabled until all values below are present:

```text
DOBOT_TEST_X=
DOBOT_TEST_Y=
DOBOT_TEST_Z=
DOBOT_TEST_R=
DOBOT_MIN_X=
DOBOT_MAX_X=
DOBOT_MIN_Y=
DOBOT_MAX_Y=
DOBOT_MIN_Z=
DOBOT_MAX_Z=
DOBOT_MIN_R=
DOBOT_MAX_R=
```

Coordinates are never accepted from an LLM for physical movement in Phase 3. The adapter uses only this preconfigured pose and rejects it unless every axis is within the explicit bounds. After MOVE, it requires three consecutive GetPose samples within the configured tolerances. A mismatch triggers software queue stop and clear, marks the client ERROR, and reports before, target, and final poses.

## Run

Safe default simulation:

```bash
python -m app.main
```

Explicit real gateway:

```bash
python -m app.main --mode real
```

Focused real diagnostics:

```bash
python -m app.main --mode real --dobot-test connection
python -m app.main --mode real --dobot-test status
python -m app.main --mode real --dobot-test home
python -m app.main --mode real --dobot-test move
python -m app.main --mode real --dobot-test grip
python -m app.main --mode real --dobot-test release
```

HOME, MOVE, GRIP, and RELEASE print the exact prepared operation and execute
only when the supervising operator types `YES`. Connection and status tests
do not move the robot. Simulation never opens a WebSocket or contacts DobotLink.

## Run simulation on GitHub

GitHub-hosted runners can test the language, gateway, mapping, lifecycle, and
safety logic, but they cannot reach a USB robot/DobotLink running on your PC.

1. Add `GEMINI_API_KEY` under **Settings → Secrets and variables → Actions**.
2. Open **Actions → Run Universal Robot Command**.
3. Select **Run workflow** and enter an instruction.
4. Inspect **Convert and validate instruction**.

Use a self-hosted runner physically attached to the robot only if you knowingly
want GitHub Actions to access that hardware. Never put real movement
confirmation flags into an ordinary hosted workflow.

## Testing

Unit suite (no Gemini, DobotLink, or hardware):

```bash
pytest -m "not live and not hardware"
```

The DOBOT unit tests inject fake RPC modules and verify lifecycle transitions,
bounded retries, exact method mapping, simulation isolation, confirmations,
software STOP labeling, safety limits, unsupported actions, multi-step fail-closed behavior, verified final-pose success, and mismatch stop/clear behavior.

Live Gemini remains opt-in:

```bash
RUN_LIVE_GEMINI_TESTS=1 pytest -m live
```

Physical connection/status integration is separately marked and requires two
deliberate environment flags after clearing the robot workspace:

```bash
RUN_DOBOT_HARDWARE_TESTS=1 DOBOT_HARDWARE_CONFIRMED=YES \
  pytest -m hardware -s
```

This test does not issue HOME, MOVE, GRIP, or RELEASE. Use the interactive CLI
for those actions so each operation receives an immediate human confirmation.

## Safety limitations

This software provides logical validation and configured bounds only. It is not
a safety-rated controller and does not implement collision avoidance,
trajectory planning, vision, payload checks, speed/acceleration validation, or
workspace sensing. The operator remains responsible for the physical workspace,
tooling, fixtures, people nearby, and manufacturer procedures.
