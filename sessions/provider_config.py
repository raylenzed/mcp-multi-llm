"""Load and validate custom provider configs from ~/.mcp-multi-llm/custom_providers.json."""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from sessions.web_search import resolve_web_search

logger = logging.getLogger(__name__)

_VALID_PROTOCOLS = {"openai", "anthropic"}
_VALID_NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")
_DEFAULT_CONFIG_PATH = Path.home() / ".mcp-multi-llm" / "custom_providers.json"


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key: str
    protocol: str = "openai"          # "openai" or "anthropic"
    extra_body: dict = field(default_factory=dict)  # merged into request body (openai protocol only)
    capabilities: list[str] = field(default_factory=list)  # e.g. ["web_search"]
    web_search: dict | None = None    # client-side search loop config (openai protocol only)


def load_custom_providers(path: Path | None = None) -> list[ProviderConfig]:
    """Load custom providers from a JSON config file.

    Returns an empty list if the file doesn't exist or has no valid entries.
    Invalid entries are skipped with a warning log.

    Supported fields per entry:
        name (str, required)       — identifier used as tool name suffix
        base_url (str, required)   — API base URL
        model (str, required)      — model name to pass in requests
        api_key (str, required)    — API key
        protocol (str, optional)   — "openai" (default) or "anthropic"
        extra_body (obj, optional) — extra fields merged into the request body (openai only)
        capabilities (list, optional) — declared abilities such as "web_search"
        web_search (str|obj, optional) — client-side search: a preset ("deepseek", "kimi")
                                         or {"preset"?, "endpoint", "query_field", ...}
    """
    if path is None:
        path = _DEFAULT_CONFIG_PATH

    if not path.exists():
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"[provider_config] Failed to read {path}: {e}")
        return []

    if not isinstance(raw, list):
        logger.error(f"[provider_config] {path} must contain a JSON array, got {type(raw).__name__}")
        return []

    providers = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            logger.warning(f"[provider_config] Entry {i}: not a JSON object — skipping")
            continue

        name = entry.get("name", "")
        base_url = entry.get("base_url", "")
        model = entry.get("model", "")
        api_key = entry.get("api_key", "")

        if not all([name, base_url, model, api_key]):
            logger.warning(f"[provider_config] Entry {i}: missing required field(s) — skipping")
            continue

        if not _VALID_NAME.match(name):
            logger.warning(f"[provider_config] Entry {i}: name '{name}' is not a valid identifier — skipping")
            continue

        protocol = entry.get("protocol", "openai")
        if protocol not in _VALID_PROTOCOLS:
            logger.warning(
                f"[provider_config] Entry {i}: unknown protocol '{protocol}', "
                f"must be one of {sorted(_VALID_PROTOCOLS)} — skipping"
            )
            continue

        extra_body = entry.get("extra_body", {})
        if not isinstance(extra_body, dict):
            logger.warning(f"[provider_config] Entry {i}: extra_body must be a JSON object — ignoring")
            extra_body = {}

        capabilities = entry.get("capabilities", [])
        if not isinstance(capabilities, list):
            logger.warning(f"[provider_config] Entry {i}: capabilities must be a JSON array — ignoring")
            capabilities = []
        normalized_capabilities = []
        for capability in capabilities:
            if not isinstance(capability, str):
                logger.warning(f"[provider_config] Entry {i}: non-string capability ignored")
                continue
            normalized = capability.strip().lower()
            if normalized and normalized not in normalized_capabilities:
                normalized_capabilities.append(normalized)

        web_search = None
        if "web_search" in entry:
            if protocol != "openai":
                logger.warning(f"[provider_config] Entry {i}: web_search needs the openai protocol — ignoring")
            else:
                web_search = resolve_web_search(entry["web_search"])
                if web_search is None:
                    logger.warning(f"[provider_config] Entry {i}: invalid web_search — ignoring")
                elif "web_search" not in normalized_capabilities:
                    normalized_capabilities.append("web_search")

        providers.append(ProviderConfig(
            name=name,
            base_url=base_url,
            model=model,
            api_key=api_key,
            protocol=protocol,
            extra_body=extra_body,
            capabilities=normalized_capabilities,
            web_search=web_search,
        ))

    # web_search.api_key_from: reuse another entry's key for the search API
    # (e.g. GLM chatting through DeepSeek's search) instead of copying the key.
    keys = {p.name.lower(): p.api_key for p in providers}
    for p in providers:
        source = (p.web_search or {}).pop("api_key_from", None)
        if source is None:
            continue
        if str(source).lower() in keys:
            p.web_search["api_key"] = keys[str(source).lower()]
        else:
            logger.warning(f"[provider_config] {p.name}: web_search.api_key_from '{source}' not found — ignoring web_search")
            p.web_search = None
            p.capabilities.remove("web_search")

    logger.info(f"[provider_config] Loaded {len(providers)} custom provider(s) from {path}")
    return providers
