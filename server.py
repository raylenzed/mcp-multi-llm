"""MCP Multi-LLM Server — Any agent as host, others as consultants."""

import asyncio
import atexit
import logging
import os
from fastmcp import FastMCP

from history.store import HistoryStore
from sessions.base import BaseSession, HealthCheckResult
from sessions.claude_session import ClaudeSession
from sessions.codex_session import CodexSession
from sessions.gemini_session import GeminiSession
from sessions.openai_compat_session import OpenAICompatSession
from sessions.anthropic_compat_session import AnthropicCompatSession
from sessions.provider_config import load_custom_providers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Which CLI is hosting this MCP? Set via env var in each CLI's MCP config.
# e.g. MCP_HOST_PROVIDER=codex  → skip discuss_with_codex
_HOST = os.environ.get("MCP_HOST_PROVIDER", "").lower().strip()
if _HOST:
    logger.info(f"[server] Host provider: '{_HOST}' — its discuss tool will be omitted.")

# Language preference: e.g. MCP_LANGUAGE=zh → all providers respond in Chinese
_LANGUAGE = os.environ.get("MCP_LANGUAGE", "").strip() or None
if _LANGUAGE:
    logger.info(f"[server] Language preference: '{_LANGUAGE}'")

# Project scope: e.g. MCP_PROJECT=my-app → history stored under ~/.mcp-multi-llm/history/my-app/
_PROJECT = os.environ.get("MCP_PROJECT", "").strip()
if not _PROJECT:
    _PROJECT = os.path.basename(os.getcwd()) or "default"
logger.info(f"[server] Project scope: '{_PROJECT}'")

mcp = FastMCP(
    "multi-llm",
    instructions=(
        "Multi-LLM discussion server. Use these tools to consult with "
        "Claude (Anthropic), Codex (OpenAI), Gemini (Google), and any custom providers "
        "for second opinions, code review, architecture discussions, and collaborative "
        "problem-solving. Each conversation is scoped by a 'topic' to maintain context "
        "continuity. Use list_available_providers first to see which consultants are available."
    ),
)

history_store = HistoryStore(project=_PROJECT)

_dismissed: set[str] = set()
_health_cache: dict[str, "HealthCheckResult"] = {}


def _session_from_cfg(cfg) -> BaseSession:
    if cfg.protocol == "anthropic":
        return AnthropicCompatSession(
            history_store=history_store,
            name=cfg.name,
            base_url=cfg.base_url,
            model=cfg.model,
            api_key=cfg.api_key,
            language=_LANGUAGE,
        )
    return OpenAICompatSession(
        history_store=history_store,
        name=cfg.name,
        base_url=cfg.base_url,
        model=cfg.model,
        api_key=cfg.api_key,
        extra_body=cfg.extra_body,
        language=_LANGUAGE,
    )


# --- Load all providers from config ---
# claude/codex/gemini entries in custom_providers.json override the CLI fallback.
_all_cfgs = load_custom_providers()
_cfg_by_name = {cfg.name.lower(): cfg for cfg in _all_cfgs}
_custom_cfgs = [cfg for cfg in _all_cfgs if cfg.name.lower() not in ("claude", "codex", "gemini")]

claude: BaseSession = (
    _session_from_cfg(_cfg_by_name["claude"]) if "claude" in _cfg_by_name
    else ClaudeSession(history_store, language=_LANGUAGE)
)
codex: BaseSession = (
    _session_from_cfg(_cfg_by_name["codex"]) if "codex" in _cfg_by_name
    else CodexSession(history_store, language=_LANGUAGE)
)
gemini: BaseSession = (
    _session_from_cfg(_cfg_by_name["gemini"]) if "gemini" in _cfg_by_name
    else GeminiSession(history_store, language=_LANGUAGE)
)

_all_providers: dict[str, BaseSession] = {}
for _name, _session in [("claude", claude), ("codex", codex), ("gemini", gemini)]:
    if _name == _HOST:
        logger.info(f"[server] Skipping '{_name}' (host provider).")
        continue
    _all_providers[_name] = _session


def _make_discuss_tool(session: BaseSession):
    """Create a typed discuss tool function for a custom provider."""
    provider_name = session.provider_name

    async def discuss(message: str, topic: str = "general") -> str:
        return await session.send(message, topic)

    discuss.__name__ = f"discuss_with_{provider_name}"
    discuss.__qualname__ = f"discuss_with_{provider_name}"
    discuss.__doc__ = (
        f"Discuss with {provider_name}. Maintains conversation context per topic.\n\n"
        f"Args:\n"
        f"    message: What to ask or discuss.\n"
        f"    topic: Conversation topic for context continuity (e.g. 'auth-refactor', 'api-design')."
    )
    return discuss


for _cfg in _custom_cfgs:
    _session = _session_from_cfg(_cfg)
    _all_providers[_cfg.name] = _session
    mcp.tool()(_make_discuss_tool(_session))
    logger.info(f"[server] Registered custom provider: {_cfg.name}")


# --- Built-in tools ---

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
    if providers is None:
        targets = dict(_all_providers)
    else:
        targets = {n: _all_providers[n] for n in providers if n in _all_providers}
        unknown = [n for n in providers if n not in _all_providers]
        if unknown:
            return f"Unknown providers: {', '.join(unknown)}. Available: {', '.join(_all_providers.keys())}"

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


if _HOST != "claude":
    @mcp.tool()
    async def discuss_with_claude(message: str, topic: str = "general") -> str:
        """Discuss with Claude (Anthropic). Maintains conversation context per topic.

        Args:
            message: What to ask or discuss with Claude.
            topic: Conversation topic for context continuity (e.g. "auth-refactor", "api-design").
        """
        return await claude.send(message, topic)


if _HOST != "codex":
    @mcp.tool()
    async def discuss_with_codex(message: str, topic: str = "general") -> str:
        """Discuss with Codex (OpenAI). Maintains conversation context per topic.

        Args:
            message: What to ask or discuss with Codex.
            topic: Conversation topic for context continuity (e.g. "auth-refactor", "api-design").
        """
        return await codex.send(message, topic)


if _HOST != "gemini":
    @mcp.tool()
    async def discuss_with_gemini(message: str, topic: str = "general") -> str:
        """Discuss with Gemini (Google). Maintains conversation context per topic.

        Args:
            message: What to ask or discuss with Gemini.
            topic: Conversation topic for context continuity (e.g. "auth-refactor", "api-design").
        """
        return await gemini.send(message, topic)


@mcp.tool()
async def group_discuss(message: str, topic: str = "general") -> str:
    """Ask all available LLMs the same question in parallel. Skips providers not installed.

    Args:
        message: The question or topic to discuss with all available LLMs.
        topic: Conversation topic for context continuity.
    """
    active = {name: s for name, s in _all_providers.items() if s.available}
    if not active:
        return "No providers available. Install at least one CLI (claude, codex, gemini) or add custom providers."

    tasks = {name: s.send(message, topic) for name, s in active.items()}
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    sections = []
    for name, result in zip(tasks.keys(), results):
        resp = result if not isinstance(result, Exception) else f"Error: {result}"
        sections.append(f"=== {name} ===\n{resp}")

    return "\n\n".join(sections)


@mcp.tool()
async def list_discussions() -> str:
    """List all active discussion topics across providers."""
    topics = history_store.list_topics()
    if not topics:
        return "No active discussions."
    return "Active discussions:\n" + "\n".join(f"  - {t}" for t in topics)


@mcp.tool()
async def clear_discussion(topic: str, provider: str | None = None) -> str:
    """Clear conversation history for a topic.

    Args:
        topic: The topic to clear.
        provider: Optional — provider name, or None to clear for all providers.
    """
    providers = [provider] if provider else list(_all_providers.keys())
    for p in providers:
        history_store.clear_topic(p, topic)
    return f"Cleared discussion '{topic}' for {', '.join(providers)}."


def _shutdown_api_sessions():
    """Close httpx clients for API-based sessions on process exit."""
    loop = asyncio.new_event_loop()
    for session in _all_providers.values():
        if hasattr(session, "aclose"):
            try:
                loop.run_until_complete(session.aclose())
            except Exception:
                pass
    loop.close()


atexit.register(_shutdown_api_sessions)


if __name__ == "__main__":
    mcp.run()
