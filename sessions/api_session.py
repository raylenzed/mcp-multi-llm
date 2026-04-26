"""Base class for HTTP-based (non-CLI) LLM providers."""

import httpx
import logging
from abc import abstractmethod
from history.store import HistoryStore
from sessions.base import BaseSession

logger = logging.getLogger(__name__)


def response_error_detail(response: httpx.Response, max_chars: int = 400) -> str:
    """Return a compact response-body snippet for debugging HTTP failures."""
    try:
        text = response.text.strip()
    except Exception:
        return ""

    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def http_status_fix_hint(provider_name: str, status_code: int) -> str | None:
    """Return a concise fix hint for common provider HTTP failures."""
    if status_code in (400, 404):
        return (
            f"Check 'base_url' and 'model' for '{provider_name}' "
            "in ~/.mcp-multi-llm/custom_providers.json"
        )
    if status_code in (401, 403):
        return f"Update 'api_key' for '{provider_name}' in ~/.mcp-multi-llm/custom_providers.json"
    if status_code == 429:
        return "The provider is rate limited; retry later or use another provider."
    if status_code >= 500:
        return "The provider endpoint returned a server error; retry later or switch providers."
    return None


class APISession(BaseSession):
    """Base for providers accessed via HTTP API (not CLI subprocess)."""

    def __init__(self, history_store: HistoryStore, base_url: str, api_key: str, language: str | None = None):
        super().__init__(history_store, language=language)
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=300.0,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    @abstractmethod
    async def send(self, message: str, topic: str, timeout: int = 300) -> str:
        ...
