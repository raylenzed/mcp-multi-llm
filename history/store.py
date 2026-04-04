"""Conversation history store for fallback context management."""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class Message:
    role: str  # "user" (claude asking) or "assistant" (llm responding)
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class TopicHistory:
    topic: str
    provider: str  # "codex" or "gemini"
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    project: str = "default"

    def add(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))

    def format_for_prompt(self, max_messages: int = 20) -> str:
        """Format recent history as context for the next prompt."""
        recent = self.messages[-max_messages:]
        lines = []
        for msg in recent:
            prefix = "You asked" if msg.role == "user" else "They replied"
            lines.append(f"[{prefix}]: {msg.content}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


class HistoryStore:
    """Persist conversation histories to disk."""

    def __init__(self, storage_dir: str | Path | None = None, project: str = "default"):
        if storage_dir is None:
            storage_dir = Path.home() / ".mcp-multi-llm" / "history"
        self.base_dir = Path(storage_dir)
        self.project = project
        self.storage_dir = self.base_dir / project
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._topics: dict[str, TopicHistory] = {}

    def _key(self, provider: str, topic: str) -> str:
        return f"{provider}::{topic}"

    def get_or_create(self, provider: str, topic: str) -> TopicHistory:
        key = self._key(provider, topic)
        if key not in self._topics:
            # Try loading from disk
            path = self.storage_dir / f"{provider}_{topic}.json"
            if path.exists():
                data = json.loads(path.read_text())
                history = TopicHistory(
                    topic=data["topic"],
                    provider=data["provider"],
                    messages=[Message(**m) for m in data["messages"]],
                    created_at=data["created_at"],
                    project=data.get("project", self.project),
                )
            else:
                history = TopicHistory(topic=topic, provider=provider, project=self.project)
            self._topics[key] = history
        return self._topics[key]

    def save(self, provider: str, topic: str):
        key = self._key(provider, topic)
        if key in self._topics:
            history = self._topics[key]
            path = self.storage_dir / f"{provider}_{topic}.json"
            path.write_text(json.dumps(history.to_dict(), ensure_ascii=False, indent=2))

    def list_topics(self, provider: str | None = None) -> list[str]:
        topics = []
        for key, history in self._topics.items():
            if provider is None or history.provider == provider:
                topics.append(f"{history.provider}::{history.topic}")
        # Also check disk
        for path in self.storage_dir.glob("*.json"):
            name = path.stem
            if provider is None or name.startswith(f"{provider}_"):
                disk_key = name.replace("_", "::", 1)
                if disk_key not in topics:
                    topics.append(disk_key)
        return topics

    def clear_topic(self, provider: str, topic: str):
        key = self._key(provider, topic)
        self._topics.pop(key, None)
        path = self.storage_dir / f"{provider}_{topic}.json"
        path.unlink(missing_ok=True)
