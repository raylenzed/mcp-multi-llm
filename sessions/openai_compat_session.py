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
        language: str | None = None,
    ):
        super().__init__(history_store, base_url=base_url, api_key=api_key, language=language)
        self.provider_name = name
        self.model = model
        self.extra_body = extra_body or {}

    def _build_messages(self, history: TopicHistory) -> list[dict]:
        """Build OpenAI-format messages array from history (includes current message)."""
        messages = []
        instruction = self._language_instruction()
        if instruction:
            messages.append({"role": "system", "content": instruction})
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
                latency_ms=int((time.monotonic() - start) * 1000),
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
