"""Base session manager for LLM CLI providers."""

import asyncio
import logging
import os
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from history.store import HistoryStore, TopicHistory

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a provider health check."""
    provider: str
    status: str        # ok | not_installed | auth_error | network_error | timeout | error
    latency_ms: int | None
    error: str | None  # human-readable error detail
    fix_hint: str | None  # actionable fix instruction


class BaseSession(ABC):
    """Abstract contract for all LLM session providers."""

    provider_name: str = "base"

    def __init__(self, history_store: HistoryStore, language: str | None = None):
        self.history_store = history_store
        self.available = True
        self._cli_not_found: bool = False
        self.language = language
        self.capabilities: set[str] = set()

    def _language_instruction(self) -> str | None:
        """Return a language instruction string, or None if no language is set."""
        if not self.language:
            return None
        _LANG_NAMES = {
            "zh": "Chinese (中文)", "en": "English", "ja": "Japanese (日本語)",
            "ko": "Korean (한국어)", "fr": "French", "de": "German",
            "es": "Spanish", "ru": "Russian", "pt": "Portuguese",
        }
        lang_name = _LANG_NAMES.get(self.language, self.language)
        return f"Please respond in {lang_name}."

    @abstractmethod
    async def send(self, message: str, topic: str, timeout: int = 300) -> str:
        """Send a message and return a response."""
        ...

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """Test real connectivity and authentication with a lightweight request."""
        ...


# Auth-failure strings to detect in CLI output (lowercase match)
_CLI_AUTH_STRINGS = (
    "not logged in",
    "authentication",
    "unauthorized",
    "401",
    "api key",
    "please log in",
    "login required",
    "oauth",
)


class CLISession(BaseSession):
    """Base class for LLM providers that use a CLI subprocess."""

    # Subclasses set these
    _AUTH_STRINGS: tuple[str, ...] = _CLI_AUTH_STRINGS

    def _prepend_language(self, prompt: str) -> str:
        """Prepend language instruction to a prompt if language is set."""
        instruction = self._language_instruction()
        if instruction:
            return f"{instruction}\n\n{prompt}"
        return prompt

    @abstractmethod
    def _build_command(self, message: str, history: TopicHistory) -> list[str]:
        """Build the CLI command to execute."""
        ...

    @abstractmethod
    def _parse_output(self, stdout: str, stderr: str) -> str:
        """Parse CLI output into clean response text."""
        ...

    def _health_check_command(self) -> list[str]:
        """Return the minimal CLI command used for health checking. Subclasses should override."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _health_check_command()")

    def _get_cli_path(self, name: str) -> str | None:
        """Find CLI executable. Returns None and disables provider if not found."""
        path = shutil.which(name)
        if path is None:
            self.available = False
            self._cli_not_found = True
            logger.warning(f"[{self.provider_name}] CLI '{name}' not found in PATH — provider disabled.")
        return path

    def _is_auth_error(self, stdout: str, stderr: str) -> bool:
        combined = (stdout + stderr).lower()
        return any(s in combined for s in self._AUTH_STRINGS)

    async def send(self, message: str, topic: str, timeout: int = 300) -> str:
        """Send a message and get a response, maintaining conversation context."""
        if not self.available:
            return f"[{self.provider_name} unavailable — CLI not installed or not in PATH]"

        history = self.history_store.get_or_create(self.provider_name, topic)
        history.add("user", message)

        cmd = self._build_command(message, history)
        logger.info(f"[{self.provider_name}] Executing: {' '.join(str(c) for c in cmd[:3])}...")

        response = await self._run_subprocess(cmd, timeout, attempts=2)

        if not response.startswith("Error:"):
            history.add("assistant", response)
            self.history_store.save(self.provider_name, topic)
        return response

    async def _run_subprocess(self, cmd: list[str], timeout: int, attempts: int = 1) -> str:
        """Run a subprocess with optional retry on timeout. Detects auth errors."""
        last_error: str = ""
        for attempt in range(attempts):
            proc = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=os.environ.copy(),
                )
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")

                if stderr.strip():
                    logger.warning(f"[{self.provider_name}] stderr: {stderr.strip()}")

                if self._is_auth_error(stdout, stderr):
                    self.available = False
                    hint = self._auth_fix_hint()
                    return f"Error: [{self.provider_name}] Authentication failed. {hint}"

                if proc.returncode != 0:
                    error_msg = f"[{self.provider_name}] Process exited with code {proc.returncode}"
                    logger.error(error_msg)
                    if stdout.strip():
                        return self._parse_output(stdout, stderr)
                    return f"Error: {error_msg}\nstderr: {stderr.strip()}"

                return self._parse_output(stdout, stderr)

            except asyncio.TimeoutError:
                last_error = f"{self.provider_name} did not respond within {timeout}s"
                logger.warning(f"[{self.provider_name}] Timeout on attempt {attempt + 1}")
                if proc is not None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                if attempt + 1 < attempts:
                    continue

            except Exception as e:
                logger.exception(f"[{self.provider_name}] Unexpected error")
                return f"Error: {e}"

        return f"Error: {last_error}"

    def _auth_fix_hint(self) -> str:
        """Subclasses override to provide a fix hint for auth errors."""
        return "Check your login status."

    async def health_check(self) -> HealthCheckResult:
        """Test provider with a real lightweight request."""
        if self._cli_not_found:
            return HealthCheckResult(
                provider=self.provider_name,
                status="not_installed",
                latency_ms=None,
                error="Binary not found in PATH",
                fix_hint=self._install_hint(),
            )

        cmd = self._health_check_command()
        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=30
            )
            latency_ms = int((time.monotonic() - start) * 1000)
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            if self._is_auth_error(stdout, stderr):
                self.available = False
                return HealthCheckResult(
                    provider=self.provider_name,
                    status="auth_error",
                    latency_ms=latency_ms,
                    error="Authentication failed",
                    fix_hint=self._auth_fix_hint(),
                )

            if proc.returncode != 0 and not stdout.strip():
                return HealthCheckResult(
                    provider=self.provider_name,
                    status="error",
                    latency_ms=latency_ms,
                    error=f"Exit {proc.returncode}: {stderr.strip()[:200]}",
                    fix_hint=None,
                )

            return HealthCheckResult(
                provider=self.provider_name,
                status="ok",
                latency_ms=latency_ms,
                error=None,
                fix_hint=None,
            )

        except asyncio.TimeoutError:
            return HealthCheckResult(
                provider=self.provider_name,
                status="timeout",
                latency_ms=int((time.monotonic() - start) * 1000),
                error="Did not respond within 30s",
                fix_hint=None,
            )
        except Exception as e:
            return HealthCheckResult(
                provider=self.provider_name,
                status="error",
                latency_ms=None,
                error=str(e),
                fix_hint=None,
            )

    def _install_hint(self) -> str | None:
        """Subclasses override to provide an install hint."""
        return None
