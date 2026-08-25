# Universal Robot Command System — Phase 1

A robot-independent language-understanding layer that converts human instructions into a strict Universal Robot Command using Google Gemini, then validates the result again with Pydantic.

> **Phase 1 does NOT control any physical robot.**

## Architecture

```text
Natural language -> Gemini structured output -> Universal Command -> Pydantic validation
```

The LLM is isolated behind `GeminiCommandClient`. The command models and validator do not depend on Gemini, so another LLM can replace it without changing the universal schema or future gateway/adapters.

## Requirements

- Python 3.11+
- A Google Gemini API key

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env  # Windows
# cp .env.example .env  # macOS/Linux
```

Set your key in `.env`:

```text
GEMINI_API_KEY=your_api_key_here
```

Never commit `.env`.

## Run

```bash
python -m app.main
```

Examples:

- `Move forward 20 centimeters.`
- `Pick the red cube and put it in box A.`
- `Stop immediately.`
- `Pick the red cube, place it in box A, then move forward 20 cm.`

The CLI prints normalized JSON and `Status: VALID`, or safely rejects the instruction.

## Run entirely on GitHub

No local installation is required:

1. Open repository **Settings > Secrets and variables > Actions**.
2. Create a repository secret named `GEMINI_API_KEY`.
3. Open **Actions > Run Universal Robot Command**.
4. Select **Run workflow**, enter the instruction, and run it.
5. Open the completed job to read the validated Universal Command.

The secret is injected only at runtime and is never printed or committed.

## Universal command schema

```json
{
  "version": "1.0",
  "robot_id": null,
  "tasks": [
    {
      "action": "MOVE",
      "direction": "forward",
      "distance": 20,
      "unit": "cm"
    }
  ]
}
```

Supported actions: `MOVE`, `ROTATE`, `STOP`, `HOME`, `PICK`, `PLACE`, `GRIP`, `RELEASE`, `NAVIGATE`, `GET_STATUS`.

Models reject unknown fields, unknown actions, missing action-specific fields, invalid measurements, and fields incompatible with an action. They contain no manufacturer, ROS, motor, serial, coordinate-system, or hardware API concepts.

## Testing

Unit tests do not call Gemini:

```bash
pytest -m "not live"
```

Run all tests, including the opt-in live API test:

```bash
RUN_LIVE_GEMINI_TESTS=1 pytest
```

The live test requires `GEMINI_API_KEY`.

## Logging

Structured JSON logs include timestamp, instruction, LLM request status, validation status, and errors. API keys and response credentials are never logged.

## Current limitations

- No physical robot connection or command execution
- No gateway, adapter, capability discovery, planning, or safety controller
- Interpretation quality depends on Gemini and the supported schema
- Ambiguous, unsafe, unsupported, or unrepresentable requests are rejected rather than guessed

## Future phases

The validated command is intended to flow through a future universal gateway, capability/safety checks, and robot adapters:

```text
LLM -> Universal Command -> Gateway -> Adapters -> Robots
```

Phase 2 and robot-specific integrations are intentionally out of scope.
