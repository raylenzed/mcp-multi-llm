"""Base class for HTTP-based (non-CLI) LLM providers."""

import httpx
import logging
from abc import abstractmethod
from history.store import HistoryStore
from sessions.base import BaseSession

logger = logging.getLogger(__name__)


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
