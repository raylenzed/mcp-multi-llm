# Provider Health Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `health_check_providers()` and `dismiss_provider()` MCP tools that test every provider with a real lightweight request, surface actionable fix hints, and let the user skip unavailable providers — plus robustness improvements (retry, auth detection, better error messages) across all sessions.

**Architecture:** Each session class gains a `health_check()` method that runs a real minimal request and returns a `HealthCheckResult` dataclass. `server.py` gains two new tools that call these methods in parallel and cache results. `CLISession` gets a shared base implementation with subclass-provided test commands. API sessions get retry logic (up to 2 retries on 5xx/network errors) and auth-error disabling.

**Tech Stack:** Python 3.13, asyncio, httpx, FastMCP — no new dependencies.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `sessions/base.py` | Modify | Add `HealthCheckResult` dataclass; add abstract `health_check()` to `BaseSession`; add `_health_check_command()` + `health_check()` to `CLISession`; add auth-error detection + one retry on timeout to `CLISession.send()` |
| `sessions/claude_session.py` | Modify | Implement `_health_check_command()`; define Claude-specific auth error strings |
| `sessions/codex_session.py` | Modify | Implement `_health_check_command()`; define Codex-specific auth error strings |
| `sessions/gemini_session.py` | Modify | Implement `_health_check_command()`; define Gemini-specific auth error strings |
| `sessions/openai_compat_session.py` | Modify | Add `health_check()`; add retry + auth-disable + rate-limit message to `send()` |
| `sessions/anthropic_compat_session.py` | Modify | Add `health_check()`; add retry + auth-disable to `send()` |
| `server.py` | Modify | Add `_dismissed` set + `_health_cache` dict; add `health_check_providers()` + `dismiss_provider()` tools; update `list_available_providers()` tip |
| `tests/test_health_check.py` | Create | Unit tests for `HealthCheckResult`, CLI not-installed path, API auth-error path |
| `~/Github/app-factory/skills/new-app/SKILL.md` | Modify | Phase 1c Step 0: replace `list_available_providers` with `health_check_providers` + decision instructions |
| `~/Github/app-factory/skills/new-app-freeserver/SKILL.md` | Modify | Same |
| `~/Github/app-factory/skills/new-app-paidserver/SKILL.md` | Same | Same |

---

### Task 1: Add `HealthCheckResult` and abstract `health_check()` to `sessions/base.py`

**Files:**
- Modify: `sessions/base.py`

- [ ] **Step 1: Add `HealthCheckResult` dataclass and update `BaseSession`**

Replace the top of `sessions/base.py` with:

```python
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

    def __init__(self, history_store: HistoryStore):
        self.history_store = history_store
        self.available = True

    @abstractmethod
    async def send(self, message: str, topic: str, timeout: int = 300) -> str:
        """Send a message and return a response."""
        ...

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """Test real connectivity and authentication with a lightweight request."""
        ...
```

- [ ] **Step 2: Add `CLISession` with auth detection, one timeout retry, and `health_check()`**

Append the full `CLISession` class after `BaseSession`:

```python
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

    @abstractmethod
    def _build_command(self, message: str, history: TopicHistory) -> list[str]:
        """Build the CLI command to execute."""
        ...

    @abstractmethod
    def _parse_output(self, stdout: str, stderr: str) -> str:
        """Parse CLI output into clean response text."""
        ...

    @abstractmethod
    def _health_check_command(self) -> list[str]:
        """Return the minimal CLI command used for health checking."""
        ...

    def _get_cli_path(self, name: str) -> str | None:
        """Find CLI executable. Returns None and disables provider if not found."""
        path = shutil.which(name)
        if path is None:
            self.available = False
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

        history.add("assistant", response)
        self.history_store.save(self.provider_name, topic)
        return response

    async def _run_subprocess(self, cmd: list[str], timeout: int, attempts: int = 1) -> str:
        """Run a subprocess, retry once on timeout, detect auth errors."""
        last_error: str = ""
        for attempt in range(attempts):
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
        if not self.available:
            return HealthCheckResult(
                provider=self.provider_name,
                status="not_installed",
                latency_ms=None,
                error=f"Binary not found in PATH",
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
                latency_ms=30000,
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
```

- [ ] **Step 3: Verify the file looks correct**

```bash
cd /Users/raylenzed/Documents/80_89_Github/app/mcp-multi-llm
python -c "from sessions.base import BaseSession, CLISession, HealthCheckResult; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add sessions/base.py
git commit -m "feat: add HealthCheckResult dataclass, health_check() to BaseSession/CLISession, auth detection + timeout retry to CLISession.send()"
```

---

### Task 2: Implement `_health_check_command()` in CLI session subclasses

**Files:**
- Modify: `sessions/claude_session.py`
- Modify: `sessions/codex_session.py`
- Modify: `sessions/gemini_session.py`

- [ ] **Step 1: Update `sessions/claude_session.py`**

```python
"""Claude Code CLI session provider."""

import re
from sessions.base import CLISession
from history.store import HistoryStore, TopicHistory


class ClaudeSession(CLISession):
    provider_name = "claude"

    _AUTH_STRINGS = (
        "not logged in",
        "authentication",
        "unauthorized",
        "401",
        "api key",
        "login",
    )

    def __init__(self, history_store: HistoryStore, model: str = "claude-opus-4-6"):
        super().__init__(history_store)
        self.model = model
        self.cli_path = self._get_cli_path("claude")

    def _build_prompt_with_context(self, message: str, history: TopicHistory) -> str:
        if len(history.messages) <= 1:
            return message
        context = history.format_for_prompt(max_messages=10)
        return (
            f"Previous conversation context:\n{context}\n\n"
            f"Current question/request:\n{message}"
        )

    def _build_command(self, message: str, history: TopicHistory) -> list[str]:
        prompt = self._build_prompt_with_context(message, history)
        return [
            self.cli_path,
            "-p", prompt,
            "--model", self.model,
            "--no-session-persistence",
            "--permission-mode", "bypassPermissions",
        ]

    def _health_check_command(self) -> list[str]:
        return [
            self.cli_path,
            "-p", "Reply with the single word: ok",
            "--model", self.model,
            "--no-session-persistence",
            "--permission-mode", "bypassPermissions",
        ]

    def _auth_fix_hint(self) -> str:
        return "Run 'claude' in your terminal to log in."

    def _install_hint(self) -> str:
        return "Install Claude Code: https://claude.ai/download"

    def _parse_output(self, stdout: str, stderr: str) -> str:
        text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', stdout)
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")):
                continue
            lines.append(line)
        return "\n".join(lines).strip() or "(no output)"
```

- [ ] **Step 2: Update `sessions/codex_session.py`**

```python
"""Codex CLI session provider."""

import re
from sessions.base import CLISession
from history.store import HistoryStore, TopicHistory


class CodexSession(CLISession):
    provider_name = "codex"

    _AUTH_STRINGS = (
        "not logged in",
        "unauthorized",
        "401",
        "api key",
        "openai_api_key",
        "authentication",
        "login",
    )

    def __init__(self, history_store: HistoryStore, model: str | None = None):
        super().__init__(history_store)
        self.model = model
        self.cli_path = self._get_cli_path("codex")

    def _build_prompt_with_context(self, message: str, history: TopicHistory) -> str:
        if len(history.messages) <= 1:
            return message
        context = history.format_for_prompt(max_messages=10)
        return (
            f"Previous conversation context:\n{context}\n\n"
            f"Current question/request:\n{message}"
        )

    def _build_command(self, message: str, history: TopicHistory) -> list[str]:
        prompt = self._build_prompt_with_context(message, history)
        cmd = [self.cli_path, "exec", "--skip-git-repo-check", "--ephemeral"]
        if self.model:
            cmd.extend(["-m", self.model])
        cmd.append(prompt)
        return cmd

    def _health_check_command(self) -> list[str]:
        cmd = [self.cli_path, "exec", "--skip-git-repo-check", "--ephemeral"]
        if self.model:
            cmd.extend(["-m", self.model])
        cmd.append("Reply with the single word: ok")
        return cmd

    def _auth_fix_hint(self) -> str:
        return "Run 'codex' in your terminal and follow the login prompts."

    def _install_hint(self) -> str:
        return "Run: npm install -g @openai/codex"

    def _parse_output(self, stdout: str, stderr: str) -> str:
        text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', stdout)
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")):
                continue
            lines.append(line)
        return "\n".join(lines).strip() or "(no output)"
```

- [ ] **Step 3: Update `sessions/gemini_session.py`**

```python
"""Gemini CLI session provider (OAuth via `gemini` CLI)."""

import json
import re
from sessions.base import CLISession
from history.store import HistoryStore, TopicHistory


class GeminiSession(CLISession):
    provider_name = "gemini"

    _AUTH_STRINGS = (
        "oauth",
        "not authenticated",
        "please log in",
        "authentication",
        "login required",
        "401",
        "unauthorized",
        "credentials",
    )

    def __init__(self, history_store: HistoryStore, model: str | None = None):
        super().__init__(history_store)
        self.model = model
        self.cli_path = self._get_cli_path("gemini")

    def _build_prompt_with_context(self, message: str, history: TopicHistory) -> str:
        if len(history.messages) <= 1:
            return message
        context = history.format_for_prompt(max_messages=10)
        return (
            f"Previous conversation context:\n{context}\n\n"
            f"Current question/request:\n{message}"
        )

    def _build_command(self, message: str, history: TopicHistory) -> list[str]:
        prompt = self._build_prompt_with_context(message, history)
        cmd = [self.cli_path, "-p", prompt, "--output-format", "json", "--yolo"]
        if self.model:
            cmd.extend(["-m", self.model])
        return cmd

    def _health_check_command(self) -> list[str]:
        cmd = [self.cli_path, "-p", "Reply with the single word: ok",
               "--output-format", "json", "--yolo"]
        if self.model:
            cmd.extend(["-m", self.model])
        return cmd

    def _auth_fix_hint(self) -> str:
        return "Run 'gemini auth login' in your terminal."

    def _install_hint(self) -> str:
        return "Run: npm install -g @google/gemini-cli"

    def _parse_output(self, stdout: str, stderr: str) -> str:
        text = stdout.strip()
        if text:
            try:
                data = json.loads(text)
                if isinstance(data, dict) and "response" in data:
                    return data["response"].strip() or "(no output)"
            except json.JSONDecodeError:
                pass

        text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', stdout)
        noise_prefixes = (
            "Loaded cached credentials",
            "Registering notification",
            "Server '",
            "Scheduling MCP",
            "Executing MCP",
            "MCP context refresh",
            "Skill ",
            "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
        )
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if any(stripped.startswith(p) for p in noise_prefixes):
                continue
            lines.append(line)
        return "\n".join(lines).strip() or "(no output)"
```

- [ ] **Step 4: Verify imports**

```bash
cd /Users/raylenzed/Documents/80_89_Github/app/mcp-multi-llm
python -c "
from history.store import HistoryStore
hs = HistoryStore()
from sessions.claude_session import ClaudeSession
from sessions.codex_session import CodexSession
from sessions.gemini_session import GeminiSession
print('ClaudeSession OK, available:', ClaudeSession(hs).available)
print('CodexSession OK, available:', CodexSession(hs).available)
print('GeminiSession OK, available:', GeminiSession(hs).available)
"
```

Expected: Each prints `OK` with `available: True` or `False` depending on whether the CLI is installed.

- [ ] **Step 5: Commit**

```bash
git add sessions/claude_session.py sessions/codex_session.py sessions/gemini_session.py
git commit -m "feat: implement _health_check_command(), _auth_fix_hint(), _install_hint() in CLI sessions"
```

---

### Task 3: Add `health_check()` and robustness to `OpenAICompatSession`

**Files:**
- Modify: `sessions/openai_compat_session.py`

- [ ] **Step 1: Rewrite `sessions/openai_compat_session.py`**

```python
"""OpenAI-compatible HTTP session provider (DeepSeek, Qianwen, etc.)."""

import asyncio
import logging
import time
import httpx
from history.store import HistoryStore, TopicHistory
from sessions.api_session import APISession
from sessions.base import HealthCheckResult

logger = logging.getLogger(__name__)


class OpenAICompatSession(APISession):
    """Session for any provider with an OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        history_store: HistoryStore,
        name: str,
        base_url: str,
        model: str,
        api_key: str,
        extra_body: dict | None = None,
    ):
        super().__init__(history_store, base_url=base_url, api_key=api_key)
        self.provider_name = name
        self.model = model
        self.extra_body = extra_body or {}

    def _build_messages(self, history: TopicHistory) -> list[dict]:
        messages = []
        for msg in history.messages[-10:]:
            role = "user" if msg.role == "user" else "assistant"
            messages.append({"role": role, "content": msg.content})
        return messages

    async def send(self, message: str, topic: str, timeout: int = 300) -> str:
        """Send a message to the OpenAI-compatible endpoint and return the reply."""
        if not self.available:
            return f"[{self.provider_name} unavailable — check API key or run health_check_providers()]"

        history = self.history_store.get_or_create(self.provider_name, topic)
        history.add("user", message)

        messages = self._build_messages(history)
        body = {"model": self.model, "messages": messages}
        if self.extra_body:
            body.update(self.extra_body)

        logger.info(f"[{self.provider_name}] POST /chat/completions model={self.model}")

        last_error: str = ""
        for attempt in range(3):
            try:
                response = await self._client.post(
                    "/chat/completions",
                    json=body,
                    timeout=timeout,
                )

                if response.status_code in (401, 403):
                    self.available = False
                    msg = (
                        f"[{self.provider_name}] HTTP {response.status_code}: invalid or expired API key. "
                        f"Update 'api_key' for '{self.provider_name}' in ~/.mcp-multi-llm/custom_providers.json"
                    )
                    logger.error(msg)
                    return f"Error: {msg}"

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "unknown")
                    return f"Error: [{self.provider_name}] Rate limited. Retry-After: {retry_after}s"

                response.raise_for_status()
                data = response.json()
                reply = data["choices"][0]["message"]["content"]
                history.add("assistant", reply)
                self.history_store.save(self.provider_name, topic)
                return reply

            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    error = f"[{self.provider_name}] HTTP {e.response.status_code}"
                    logger.error(error)
                    return f"Error: {error}"
                last_error = f"HTTP {e.response.status_code}"
                logger.warning(f"[{self.provider_name}] Server error on attempt {attempt + 1}: {last_error}")

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = str(e)
                logger.warning(f"[{self.provider_name}] Network error on attempt {attempt + 1}: {e}")

            except Exception as e:
                logger.exception(f"[{self.provider_name}] Unexpected error")
                return f"Error: [{self.provider_name}] {e}"

            if attempt < 2:
                await asyncio.sleep(1)

        error = f"[{self.provider_name}] Failed after 3 attempts: {last_error}"
        logger.error(error)
        return f"Error: {error}"

    async def health_check(self) -> HealthCheckResult:
        """Test the API with a minimal request."""
        start = time.monotonic()
        try:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                    "max_tokens": 10,
                },
                timeout=10.0,
            )
            latency_ms = int((time.monotonic() - start) * 1000)

            if response.status_code in (401, 403):
                self.available = False
                return HealthCheckResult(
                    provider=self.provider_name,
                    status="auth_error",
                    latency_ms=latency_ms,
                    error=f"HTTP {response.status_code}: invalid or expired API key",
                    fix_hint=f"Update 'api_key' for '{self.provider_name}' in ~/.mcp-multi-llm/custom_providers.json",
                )

            response.raise_for_status()
            return HealthCheckResult(
                provider=self.provider_name,
                status="ok",
                latency_ms=latency_ms,
                error=None,
                fix_hint=None,
            )

        except httpx.ConnectError as e:
            return HealthCheckResult(
                provider=self.provider_name,
                status="network_error",
                latency_ms=None,
                error=f"Connection failed: {e}",
                fix_hint=f"Check 'base_url' for '{self.provider_name}' in ~/.mcp-multi-llm/custom_providers.json",
            )
        except httpx.TimeoutException:
            return HealthCheckResult(
                provider=self.provider_name,
                status="timeout",
                latency_ms=10000,
                error="API did not respond within 10s",
                fix_hint=None,
            )
        except httpx.HTTPStatusError as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            return HealthCheckResult(
                provider=self.provider_name,
                status="error",
                latency_ms=latency_ms,
                error=f"HTTP {e.response.status_code}",
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
```

- [ ] **Step 2: Verify**

```bash
cd /Users/raylenzed/Documents/80_89_Github/app/mcp-multi-llm
python -c "from sessions.openai_compat_session import OpenAICompatSession; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add sessions/openai_compat_session.py
git commit -m "feat: add health_check(), retry logic, auth-disable, rate-limit message to OpenAICompatSession"
```

---

### Task 4: Add `health_check()` and robustness to `AnthropicCompatSession`

**Files:**
- Modify: `sessions/anthropic_compat_session.py`

- [ ] **Step 1: Rewrite `sessions/anthropic_compat_session.py`**

```python
"""Anthropic-protocol HTTP session provider."""

import asyncio
import logging
import time
import httpx
from history.store import HistoryStore, TopicHistory
from sessions.base import BaseSession, HealthCheckResult

logger = logging.getLogger(__name__)


class AnthropicCompatSession(BaseSession):
    """Session for providers using the Anthropic /v1/messages API format."""

    def __init__(
        self,
        history_store: HistoryStore,
        name: str,
        base_url: str,
        model: str,
        api_key: str,
        max_tokens: int = 8192,
    ):
        super().__init__(history_store)
        self.provider_name = name
        self.model = model
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=300.0,
        )

    def _build_messages(self, history: TopicHistory) -> list[dict]:
        messages = []
        for msg in history.messages[-10:]:
            role = "user" if msg.role == "user" else "assistant"
            messages.append({"role": role, "content": msg.content})
        return messages

    async def send(self, message: str, topic: str, timeout: int = 300) -> str:
        if not self.available:
            return f"[{self.provider_name} unavailable — check API key or run health_check_providers()]"

        history = self.history_store.get_or_create(self.provider_name, topic)
        history.add("user", message)

        messages = self._build_messages(history)
        logger.info(f"[{self.provider_name}] POST /v1/messages model={self.model}")

        last_error: str = ""
        for attempt in range(3):
            try:
                response = await self._client.post(
                    "/v1/messages",
                    json={
                        "model": self.model,
                        "max_tokens": self.max_tokens,
                        "messages": messages,
                    },
                    timeout=timeout,
                )

                if response.status_code in (401, 403):
                    self.available = False
                    msg = (
                        f"[{self.provider_name}] HTTP {response.status_code}: invalid or expired API key. "
                        f"Update 'api_key' for '{self.provider_name}' in ~/.mcp-multi-llm/custom_providers.json"
                    )
                    logger.error(msg)
                    return f"Error: {msg}"

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "unknown")
                    return f"Error: [{self.provider_name}] Rate limited. Retry-After: {retry_after}s"

                response.raise_for_status()
                data = response.json()
                reply = data["content"][0]["text"]
                history.add("assistant", reply)
                self.history_store.save(self.provider_name, topic)
                return reply

            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    error = f"[{self.provider_name}] HTTP {e.response.status_code}"
                    logger.error(error)
                    return f"Error: {error}"
                last_error = f"HTTP {e.response.status_code}"
                logger.warning(f"[{self.provider_name}] Server error on attempt {attempt + 1}: {last_error}")

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = str(e)
                logger.warning(f"[{self.provider_name}] Network error on attempt {attempt + 1}: {e}")

            except Exception as e:
                logger.exception(f"[{self.provider_name}] Unexpected error")
                return f"Error: [{self.provider_name}] {e}"

            if attempt < 2:
                await asyncio.sleep(1)

        error = f"[{self.provider_name}] Failed after 3 attempts: {last_error}"
        logger.error(error)
        return f"Error: {error}"

    async def health_check(self) -> HealthCheckResult:
        """Test the API with a minimal request."""
        start = time.monotonic()
        try:
            response = await self._client.post(
                "/v1/messages",
                json={
                    "model": self.model,
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                },
                timeout=10.0,
            )
            latency_ms = int((time.monotonic() - start) * 1000)

            if response.status_code in (401, 403):
                self.available = False
                return HealthCheckResult(
                    provider=self.provider_name,
                    status="auth_error",
                    latency_ms=latency_ms,
                    error=f"HTTP {response.status_code}: invalid or expired API key",
                    fix_hint=f"Update 'api_key' for '{self.provider_name}' in ~/.mcp-multi-llm/custom_providers.json",
                )

            response.raise_for_status()
            return HealthCheckResult(
                provider=self.provider_name,
                status="ok",
                latency_ms=latency_ms,
                error=None,
                fix_hint=None,
            )

        except httpx.ConnectError as e:
            return HealthCheckResult(
                provider=self.provider_name,
                status="network_error",
                latency_ms=None,
                error=f"Connection failed: {e}",
                fix_hint=f"Check 'base_url' for '{self.provider_name}' in ~/.mcp-multi-llm/custom_providers.json",
            )
        except httpx.TimeoutException:
            return HealthCheckResult(
                provider=self.provider_name,
                status="timeout",
                latency_ms=10000,
                error="API did not respond within 10s",
                fix_hint=None,
            )
        except httpx.HTTPStatusError as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            return HealthCheckResult(
                provider=self.provider_name,
                status="error",
                latency_ms=latency_ms,
                error=f"HTTP {e.response.status_code}",
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

    async def aclose(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 2: Verify**

```bash
cd /Users/raylenzed/Documents/80_89_Github/app/mcp-multi-llm
python -c "from sessions.anthropic_compat_session import AnthropicCompatSession; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add sessions/anthropic_compat_session.py
git commit -m "feat: add health_check(), retry logic, auth-disable to AnthropicCompatSession"
```

---

### Task 5: Add `health_check_providers()` and `dismiss_provider()` to `server.py`

**Files:**
- Modify: `server.py`

- [ ] **Step 1: Add imports and state variables**

After the existing imports in `server.py`, add:

```python
from sessions.base import BaseSession, HealthCheckResult
```

After the `history_store = HistoryStore()` line, add:

```python
_dismissed: set[str] = set()
_health_cache: dict[str, HealthCheckResult] = {}
```

- [ ] **Step 2: Update `list_available_providers()` to add tip**

Replace the existing `list_available_providers` function body:

```python
@mcp.tool()
async def list_available_providers() -> str:
    """List which LLM providers are available for consultation."""
    available = [name for name, s in _all_providers.items() if s.available and name not in _dismissed]
    unavailable = [name for name, s in _all_providers.items() if not s.available or name in _dismissed]
    lines = []
    if available:
        lines.append("Available: " + ", ".join(available))
    if unavailable:
        lines.append("Not available: " + ", ".join(unavailable))
    lines.append("Tip: call health_check_providers() to test real connectivity and authentication.")
    return "\n".join(lines) or "No providers available."
```

- [ ] **Step 3: Add `health_check_providers()` tool**

Add after `list_available_providers`:

```python
@mcp.tool()
async def health_check_providers(
    providers: list[str] | None = None,
    force: bool = False,
) -> str:
    """Test real connectivity and authentication for all (or specified) providers.

    Runs a lightweight real request to each provider in parallel. Results are cached
    for this session; pass force=True to re-test cached providers.

    Args:
        providers: List of provider names to test, or None to test all.
        force: Re-test even if cached results exist for this session.
    """
    # Determine which providers to target
    if providers is None:
        targets = dict(_all_providers)
    else:
        targets = {n: _all_providers[n] for n in providers if n in _all_providers}
        unknown = [n for n in providers if n not in _all_providers]
        if unknown:
            return f"Unknown providers: {', '.join(unknown)}. Available: {', '.join(_all_providers.keys())}"

    # Determine which need testing (not cached, or force)
    to_test = {n: s for n, s in targets.items() if force or n not in _health_cache}

    if to_test:
        results = await asyncio.gather(
            *[s.health_check() for s in to_test.values()],
            return_exceptions=True,
        )
        for name, result in zip(to_test.keys(), results):
            if isinstance(result, Exception):
                _health_cache[name] = HealthCheckResult(
                    provider=name,
                    status="error",
                    latency_ms=None,
                    error=str(result),
                    fix_hint=None,
                )
            else:
                _health_cache[name] = result
                if result.status != "ok":
                    _all_providers[name].available = False

    # Format report
    lines = ["## Provider Health Check\n"]
    ok_count = 0
    fail_count = 0

    for name in targets:
        r = _health_cache.get(name)
        if r is None:
            continue

        if name in _dismissed:
            lines.append(f"○ {name:<14} — dismissed for this session")
            continue

        if r.status == "ok":
            ok_count += 1
            latency = f"({r.latency_ms}ms)" if r.latency_ms is not None else ""
            lines.append(f"✓ {name:<14} — ok {latency}")
        else:
            fail_count += 1
            lines.append(f"✗ {name:<14} — {r.error or r.status}")
            if r.fix_hint:
                lines.append(f"  Fix: {r.fix_hint}")

    lines.append(f"\n{ok_count} available, {fail_count} unavailable.")
    if fail_count > 0:
        lines.append('Call dismiss_provider("name") to skip a provider for this session.')

    return "\n".join(lines)
```

- [ ] **Step 4: Add `dismiss_provider()` tool**

Add after `health_check_providers`:

```python
@mcp.tool()
async def dismiss_provider(name: str) -> str:
    """Skip a provider for this session. Excluded from group_discuss and future health checks.

    Args:
        name: The provider name to dismiss (e.g. "gemini", "codex").
    """
    if name not in _all_providers:
        return f"Unknown provider '{name}'. Known providers: {', '.join(_all_providers.keys())}"

    _dismissed.add(name)
    _all_providers[name].available = False
    return f"'{name}' dismissed for this session. It will be skipped in group_discuss."
```

- [ ] **Step 5: Verify server starts without errors**

```bash
cd /Users/raylenzed/Documents/80_89_Github/app/mcp-multi-llm
timeout 3 python server.py 2>&1 | head -20 || true
```

Expected: startup log lines, no import errors or tracebacks.

- [ ] **Step 6: Commit**

```bash
git add server.py
git commit -m "feat: add health_check_providers() and dismiss_provider() MCP tools, update list_available_providers() tip"
```

---

### Task 6: Write tests

**Files:**
- Create: `tests/test_health_check.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for provider health check functionality."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sessions.base import HealthCheckResult
from history.store import HistoryStore


def make_history_store():
    return HistoryStore()


# --- HealthCheckResult ---

def test_health_check_result_ok():
    r = HealthCheckResult(provider="claude", status="ok", latency_ms=500, error=None, fix_hint=None)
    assert r.status == "ok"
    assert r.latency_ms == 500
    assert r.error is None


def test_health_check_result_auth_error():
    r = HealthCheckResult(
        provider="gemini",
        status="auth_error",
        latency_ms=100,
        error="Authentication failed",
        fix_hint="Run 'gemini auth login'",
    )
    assert r.status == "auth_error"
    assert r.fix_hint == "Run 'gemini auth login'"


# --- CLISession: not_installed path ---

def test_cli_session_not_installed_returns_not_installed_status():
    from sessions.claude_session import ClaudeSession
    hs = make_history_store()
    with patch("shutil.which", return_value=None):
        session = ClaudeSession(hs)
    assert not session.available

    result = asyncio.get_event_loop().run_until_complete(session.health_check())
    assert result.status == "not_installed"
    assert result.fix_hint is not None


# --- CLISession: auth error detection ---

@pytest.mark.asyncio
async def test_cli_session_send_detects_auth_error():
    from sessions.claude_session import ClaudeSession
    hs = make_history_store()
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        session = ClaudeSession(hs)

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.communicate = AsyncMock(return_value=(
        b"",
        b"Error: not logged in. Please authenticate first.",
    ))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        response = await session.send("hello", "test-topic")

    assert "Authentication failed" in response
    assert not session.available


# --- OpenAICompatSession: health_check auth error ---

@pytest.mark.asyncio
async def test_openai_compat_health_check_auth_error():
    from sessions.openai_compat_session import OpenAICompatSession
    import httpx
    hs = make_history_store()
    session = OpenAICompatSession(
        hs, name="testprovider", base_url="http://localhost:9999",
        model="test-model", api_key="bad-key"
    )

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "401", request=MagicMock(), response=mock_response
    )

    with patch.object(session._client, "post", AsyncMock(return_value=mock_response)):
        result = await session.health_check()

    assert result.status == "auth_error"
    assert result.fix_hint is not None
    assert not session.available


# --- OpenAICompatSession: send() disables on 401 ---

@pytest.mark.asyncio
async def test_openai_compat_send_disables_on_401():
    from sessions.openai_compat_session import OpenAICompatSession
    hs = make_history_store()
    session = OpenAICompatSession(
        hs, name="testprovider", base_url="http://localhost:9999",
        model="test-model", api_key="bad-key"
    )

    mock_response = MagicMock()
    mock_response.status_code = 401

    with patch.object(session._client, "post", AsyncMock(return_value=mock_response)):
        result = await session.send("hello", "test")

    assert "401" in result or "invalid" in result.lower()
    assert not session.available


# --- OpenAICompatSession: send() skips when unavailable ---

@pytest.mark.asyncio
async def test_openai_compat_send_skips_when_unavailable():
    from sessions.openai_compat_session import OpenAICompatSession
    hs = make_history_store()
    session = OpenAICompatSession(
        hs, name="testprovider", base_url="http://localhost:9999",
        model="test-model", api_key="key"
    )
    session.available = False

    result = await session.send("hello", "test")
    assert "unavailable" in result
```

- [ ] **Step 2: Run the tests**

```bash
cd /Users/raylenzed/Documents/80_89_Github/app/mcp-multi-llm
uv run pytest tests/test_health_check.py -v
```

Expected: All tests pass (PASSED).

- [ ] **Step 3: Commit**

```bash
git add tests/test_health_check.py
git commit -m "test: add health check unit tests for HealthCheckResult, CLI auth detection, API auth/unavailable paths"
```

---

### Task 7: Update app-factory skill files

**Files:**
- Modify: `~/Github/app-factory/skills/new-app/SKILL.md` (line ~104)
- Modify: `~/Github/app-factory/skills/new-app-freeserver/SKILL.md` (line ~109)
- Modify: `~/Github/app-factory/skills/new-app-paidserver/SKILL.md` (line ~109)

- [ ] **Step 1: Update `new-app/SKILL.md` Phase 1c Step 0**

Find this block (around line 103-104):

```
**第零步：确认可用的 Agent**
调用 `list_available_providers`。如果所有外部模型都不可用 → 跳过本节，继续 1d，并在报告中注明"多模型评审未执行"。
```

Replace with:

```
**第零步：确认可用的 Agent**
调用 `health_check_providers()`。阅读返回的报告：
- ✓ 行表示该提供商可用，继续。
- ✗ 行表示该提供商不可用：询问用户"[provider] 不可用（[原因]）。Fix: [fix_hint]。现在修复，还是本次跳过？"
  - 选择跳过：调用 `dismiss_provider("[provider name]")`，继续使用其余可用提供商。
  - 选择修复：等待用户完成修复后，调用 `health_check_providers(["provider name"], force=true)` 重新验证。
- 如果所有外部模型都不可用 → 跳过本节，继续 1d，并在报告中注明"多模型评审未执行"。
```

- [ ] **Step 2: Update `new-app-freeserver/SKILL.md`**

Find (around line 109):

```
调用 `list_available_providers`，记录哪些可用。如果 Gemini 和 Codex 都不可用 → 跳过本节，继续 1e。
```

Replace with:

```
调用 `health_check_providers()`，阅读报告。对每个 ✗ 提供商询问用户是否跳过或修复（跳过则调用 `dismiss_provider("name")`，修复后调用 `health_check_providers(["name"], force=true)` 重新验证）。如果 Gemini 和 Codex 都不可用 → 跳过本节，继续 1e。
```

- [ ] **Step 3: Update `new-app-paidserver/SKILL.md`**

Apply the same replacement as Step 2 to `new-app-paidserver/SKILL.md` (same text at the same position).

- [ ] **Step 4: Verify the changes look correct**

```bash
grep -n "health_check_providers\|list_available_providers" \
  /Users/raylenzed/Github/app-factory/skills/new-app/SKILL.md \
  /Users/raylenzed/Github/app-factory/skills/new-app-freeserver/SKILL.md \
  /Users/raylenzed/Github/app-factory/skills/new-app-paidserver/SKILL.md
```

Expected: All three files show `health_check_providers`, none show `list_available_providers`.

- [ ] **Step 5: Commit (in app-factory repo)**

```bash
cd /Users/raylenzed/Github/app-factory
git add skills/new-app/SKILL.md skills/new-app-freeserver/SKILL.md skills/new-app-paidserver/SKILL.md
git commit -m "feat: replace list_available_providers with health_check_providers in Phase 1c roundtable"
```

---

### Task 8: Run full test suite and final verification

**Files:** None modified.

- [ ] **Step 1: Run all tests**

```bash
cd /Users/raylenzed/Documents/80_89_Github/app/mcp-multi-llm
uv run pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 2: Verify server starts cleanly**

```bash
cd /Users/raylenzed/Documents/80_89_Github/app/mcp-multi-llm
timeout 3 python server.py 2>&1 | head -30 || true
```

Expected: Startup logs only, no tracebacks or import errors.

- [ ] **Step 3: Smoke-test tool list**

```bash
cd /Users/raylenzed/Documents/80_89_Github/app/mcp-multi-llm
python -c "
import server
tools = [t for t in dir(server) if not t.startswith('_')]
print([t for t in ['health_check_providers', 'dismiss_provider', 'list_available_providers'] if hasattr(server, t)])
"
```

Expected: `['health_check_providers', 'dismiss_provider', 'list_available_providers']` (or similar confirmation the functions exist).

- [ ] **Step 4: Final commit tag**

```bash
cd /Users/raylenzed/Documents/80_89_Github/app/mcp-multi-llm
git log --oneline -8
```

Expected: 6 commits from this feature visible in log.
