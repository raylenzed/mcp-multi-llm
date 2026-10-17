"""Client-side web search for OpenAI-compatible providers.

Some providers (DeepSeek, Kimi) expose search as a separate REST API instead of a
server-side switch. The model asks for a search through a normal function tool,
the session runs it against that API, and feeds the results back as a tool message.
"""

import json
import logging

import httpx

logger = logging.getLogger(__name__)

PRESETS: dict[str, dict] = {
    "deepseek": {
        "endpoint": "https://api.deepseek.com/v1/tools/search",
        "query_field": "query",
        "results_field": "results",
    },
    "kimi": {
        "endpoint": "https://api.moonshot.cn/v1/tools/search",
        "query_field": "text_query",
        "results_field": "search_results",
    },
}

DEFAULT_COUNT = 5
DEFAULT_MAX_SEARCHES = 6
_SNIPPET_CHARS = 600

TOOL_NAME = "web_search"
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Search the live web. Use it whenever the task needs current facts, "
            "such as App Store competitors, ratings, prices, or user complaints."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms, at most 128 characters."},
            },
            "required": ["query"],
        },
    },
}


def resolve_web_search(raw) -> dict | None:
    """Normalize a provider's `web_search` config (a preset name or an object).

    Returns None when the config is missing or invalid.
    """
    if isinstance(raw, str):
        raw = {"preset": raw}
    if not isinstance(raw, dict):
        return None

    preset = str(raw.get("preset", "")).strip().lower()
    if preset and preset not in PRESETS:
        logger.warning(f"[web_search] Unknown preset '{preset}' — known: {sorted(PRESETS)}")
        return None

    cfg = dict(PRESETS.get(preset, {}))
    cfg.update({k: v for k, v in raw.items() if k != "preset"})
    if not isinstance(cfg.get("endpoint"), str) or not cfg["endpoint"].startswith("https://"):
        logger.warning("[web_search] 'endpoint' must be an https URL — ignoring web_search")
        return None

    cfg.setdefault("query_field", "query")
    cfg.setdefault("results_field", "results")
    for key, default in (("count", DEFAULT_COUNT), ("max_searches", DEFAULT_MAX_SEARCHES)):
        value = cfg.get(key, default)
        cfg[key] = value if isinstance(value, int) and value > 0 else default
    return cfg


def format_results(results: list, count: int) -> str:
    """Render search results as compact text for the model."""
    if not results:
        return "No results."
    lines = []
    for index, item in enumerate(results[:count], start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or item.get("link") or "").strip()
        date = str(item.get("date") or "").strip()
        body = str(item.get("summary") or item.get("snippet") or item.get("text") or item.get("content") or "")
        body = " ".join(body.split())[:_SNIPPET_CHARS]
        header = f"[{index}] {title}" + (f" ({date})" if date else "")
        lines.append(f"{header}\n{url}\n{body}".rstrip())
    return "\n\n".join(lines) or "No results."


async def run_search(client: httpx.AsyncClient, cfg: dict, arguments: str) -> str:
    """Run one search for a model tool call and return text for the tool message."""
    try:
        query = str(json.loads(arguments or "{}").get("query", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        query = ""
    if not query:
        return "Search error: empty query."

    headers = {"Authorization": f"Bearer {cfg['api_key']}"} if cfg.get("api_key") else None
    try:
        response = await client.post(
            cfg["endpoint"],
            json={cfg["query_field"]: query[:128], "count": cfg["count"]},
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        results = response.json().get(cfg["results_field"]) or []
    except Exception as e:
        logger.warning(f"[web_search] Search failed for '{query[:60]}': {e}")
        return f"Search error: {e}"

    logger.info(f"[web_search] '{query[:60]}' → {len(results)} result(s)")
    return format_results(results, cfg["count"])
