"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    gemini_timeout_ms: int = 30_000

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == "your_api_key_here":
            raise ConfigurationError(
                "GEMINI_API_KEY is missing. Copy .env.example to .env and add your key."
            )

        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        try:
            timeout_ms = int(os.getenv("GEMINI_TIMEOUT_MS", "30000"))
        except ValueError as exc:
            raise ConfigurationError("GEMINI_TIMEOUT_MS must be an integer.") from exc
        if timeout_ms <= 0:
            raise ConfigurationError("GEMINI_TIMEOUT_MS must be positive.")

        return cls(
            gemini_api_key=api_key,
            gemini_model=model,
            gemini_timeout_ms=timeout_ms,
        )
