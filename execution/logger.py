from __future__ import annotations

import json
from pathlib import Path

from execution.result import ExecutionResult


class ExecutionLogger:
    """Stores execution results in a JSON log file."""

    def __init__(self, log_file: str = "logs/execution_log.json"):
        self.log_file = Path(log_file)

        # Create logs directory automatically
        self.log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Create empty JSON file if it does not exist
        if not self.log_file.exists():
            self.log_file.write_text(
                "[]",
                encoding="utf-8",
            )

    def log(self, result: ExecutionResult) -> None:
        """Store one execution result."""

        logs = self._read_logs()

        logs.append(
            result.model_dump(mode="json")
        )

        self.log_file.write_text(
            json.dumps(
                logs,
                indent=4,
            ),
            encoding="utf-8",
        )

    def log_many(
        self,
        results: list[ExecutionResult],
    ) -> None:
        """Store multiple execution results."""

        for result in results:
            self.log(result)

    def read_logs(self) -> list[dict]:
        """Read all execution logs."""

        return self._read_logs()

    def _read_logs(self) -> list[dict]:

        try:
            content = self.log_file.read_text(
                encoding="utf-8"
            )

            return json.loads(content)

        except (FileNotFoundError, json.JSONDecodeError):
            return []