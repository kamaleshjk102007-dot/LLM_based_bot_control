# Universal LLM Robot Control Platform — Phase 2

A robot-independent platform that converts human instructions into a strict Universal Robot Command, validates it, and routes it through a generic Universal Gateway to a simulated adapter.

> **Phase 2 does NOT connect to or physically control any robot. All execution is simulated.**

## Architecture

```text
Phase 1:
Natural language -> Gemini -> Universal Command -> Pydantic validation

Phase 2:
Universal Command -> Robot Selection -> Capability Check
                  -> Logical Safety -> Execution Plan
```

```text
LLM
 |
 v
Universal Command
 |
 v
Universal Gateway
 |-- Robot Registry
 |-- Capability Manager
 |-- Robot Selector
 |-- Safety Validator
 |-- Command Router
 '-- Adapter Manager
 |
 v
RobotAdapter Interface
 |
 v
Mock Adapter
 |
 v
SIMULATED EXECUTION
```

The LLM only performs language understanding. Robot selection is deterministic Python logic. The gateway knows capabilities and abstract adapter types, never transport protocols or vendor APIs.

## Phase 2 components

- **Robot Registry** — in-memory registration, lookup, filtering, and status updates
- **Capability Manager** — all-or-nothing validation of every task
- **Robot Selector** — explicit selection or deterministic automatic selection
- **Safety Validator** — logical schema, status, capability, and task-combination checks
- **Command Router** — selection → capability → safety → adapter → plan
- **Adapter Manager** — extensible mapping from adapter type to implementation
- **RobotAdapter** — generic abstract interface
- **MockRobotAdapter** — in-memory simulated preparation and execution
- **ExecutionPlan** — typed output with status and simulation results

Plan statuses are `READY`, `REJECTED`, `NO_ROBOT`, `UNSUPPORTED`, `UNSAFE`, and `INVALID`.

`READY` means the command passed gateway checks and the mock adapter simulated acceptance. It does not mean a physical robot executed anything.

## Robot selection

Only `ONLINE` robots are automatically eligible. A robot must support every action in the command. Selection order is deterministic:

1. Explicit `robot_id`, when supplied
2. All required capabilities
3. Lowest priority number
4. Alphabetically lowest `robot_id` as the tie-break

Gemini never selects a robot, and multi-task commands are never partially approved.

## Requirements

- Python 3.11+
- Google Gemini API key
- Dependencies in `requirements.txt`

No robot SDK, ROS, serial, CAN, USB, or vendor dependency is included.

## Local installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env
```

Set the local key in the uncommitted `.env`:

```text
GEMINI_API_KEY=your_api_key_here
```

The real `.env` is excluded by `.gitignore`.

## Run locally

```bash
python -m app.main
```

The CLI registers three demo definitions in memory:

- `robot_001`: online generic demo arm using `mock`
- `robot_002`: online generic mobile robot using `mock`
- `robot_003`: offline generic drone using `mock`

Example instruction:

```text
Pick the red cube, place it in box A, then move forward 20 cm.
```

The expected gateway result selects `robot_001`, passes `PICK`, `PLACE`, and `MOVE`, and returns `Gateway Status: READY` with mock-only messages.

## Run entirely on GitHub

1. Add `GEMINI_API_KEY` under **Settings → Secrets and variables → Actions**.
2. Open **Actions → Run Universal Robot Command**.
3. Select **Run workflow**.
4. Enter a natural-language instruction.
5. Open **Convert and validate instruction** in the completed job.

The encrypted secret is injected only at runtime and is never committed or printed.

## Universal command example

```json
{
  "version": "1.0",
  "robot_id": null,
  "tasks": [
    {
      "action": "NAVIGATE",
      "target": {
        "type": "location",
        "id": "table"
      }
    }
  ]
}
```

Supported actions: `MOVE`, `ROTATE`, `STOP`, `HOME`, `PICK`, `PLACE`, `GRIP`, `RELEASE`, `NAVIGATE`, `GET_STATUS`.

## Execution plan example

```json
{
  "plan_id": "plan-generated-id",
  "robot_id": "robot_002",
  "adapter_type": "mock",
  "status": "READY",
  "tasks": [
    {
      "action": "NAVIGATE",
      "target": {
        "type": "location",
        "id": "table"
      }
    }
  ],
  "capability_checks": {
    "NAVIGATE": true
  },
  "safety_passed": true,
  "simulated": true,
  "results": [
    "[MOCK] NAVIGATE accepted"
  ]
}
```

## Testing

Unit tests never require Gemini:

```bash
pytest -m "not live"
```

The suite covers Phase 1, registry behavior, capabilities, deterministic selection, logical safety, adapter management, routing, gateway scenarios, and the no-hardware architectural guard.

The live Gemini test remains opt-in:

```bash
RUN_LIVE_GEMINI_TESTS=1 pytest
```

GitHub Actions runs the unit suite on Python 3.11, 3.12, and 3.13.

## Scope and limitations

Phase 2 uses in-memory definitions only. It implements no database, physical collision avoidance, motion planning, vision, camera access, network endpoint, robot connection, or physical execution.

It contains no DOBOT-specific code, ROS node, serial port, CAN bus, USB driver, motor control, GPIO, or wireless robot communication.

## Future phases

```text
Phase 3 -> DOBOT Adapter
Phase 4 -> Real DOBOT Control
Phase 5 -> Vision
Phase 6 -> Planning
Phase 7 -> ROS 2 Adapter
Phase 8 -> Additional Robot Adapters
Phase 9 -> Wireless Receiver
```

Those future adapters and integrations are intentionally not implemented in Phase 2.
