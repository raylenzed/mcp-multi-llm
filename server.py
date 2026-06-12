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
from app_factory_roundtable import run_app_factory_roundtable

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
        session = AnthropicCompatSession(
            history_store=history_store,
            name=cfg.name,
            base_url=cfg.base_url,
            model=cfg.model,
            api_key=cfg.api_key,
            language=_LANGUAGE,
        )
    else:
        session = OpenAICompatSession(
            history_store=history_store,
            name=cfg.name,
            base_url=cfg.base_url,
            model=cfg.model,
            api_key=cfg.api_key,
            extra_body=cfg.extra_body,
            language=_LANGUAGE,
        )
    session.capabilities = set(cfg.capabilities)
    return session


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


def _provider_names() -> str:
    return ", ".join(_all_providers.keys())


def _resolve_provider_name(name: str) -> str | None:
    """Resolve provider names case-insensitively to their registered key."""
    requested = name.strip().lower()
    for provider_name in _all_providers:
        if provider_name.lower() == requested:
            return provider_name
    return None


def _resolve_provider_targets(providers: list[str] | None) -> tuple[dict[str, BaseSession], list[str]]:
    if providers is None:
        return dict(_all_providers), []

    targets: dict[str, BaseSession] = {}
    unknown: list[str] = []
    for provider in providers:
        resolved = _resolve_provider_name(provider)
        if resolved is None:
            unknown.append(provider)
            continue
        targets.setdefault(resolved, _all_providers[resolved])
    return targets, unknown


def _history_provider_name(provider_name: str) -> str:
    return _all_providers[provider_name].provider_name


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
    targets, unknown = _resolve_provider_targets(providers)
    if unknown:
        return f"Unknown providers: {', '.join(unknown)}. Available: {_provider_names()}"

    to_test = {
        n: s for n, s in targets.items()
        if n not in _dismissed and (force or n not in _health_cache)
    }

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
                elif name not in _dismissed:
                    _all_providers[name].available = True

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
    provider_name = _resolve_provider_name(name)
    if provider_name is None:
        return f"Unknown provider '{name}'. Known providers: {_provider_names()}"

    _dismissed.add(provider_name)
    _all_providers[provider_name].available = False
    return f"'{provider_name}' dismissed for this session. It will be skipped in group_discuss."


@mcp.tool()
async def app_factory_roundtable(
    app_name: str,
    app_mode: str,
    pitch: str,
    data_brief: str,
    keywords: list[str] | None = None,
    min_participants: int = 2,
    max_participants: int = 5,
    force_health_check: bool = False,
) -> dict:
    """Run a two-round adversarial App Factory review (veto engine, not approval engine).

    The final gate is computed ONLY from classified, falsifiable challenges
    (HARD_DISQUALIFIER / PIVOTABLE_WEAKNESS / EVIDENCE_GAP / SOFT_NEGATIVE);
    approval votes are archived and carry zero signal. Same-meaning challenges
    are deduplicated across models (correlated judges count once). The returned
    `challenge_ledger` feeds the skill's Challenge Ledger directly.

    Args:
        app_name: Candidate app name.
        app_mode: One of "local", "freeserver", or "paidserver".
        pitch: Candidate app pitch without DATA BRIEF.
        data_brief: Local tool evidence from RespectASO/review_miner/reddit_miner.
        keywords: Optional App Store/search keywords for independent research.
        min_participants: Minimum desired participant count. Fewer returns partial.
        max_participants: Maximum providers to include.
        force_health_check: Re-test providers before selecting participants.
    """
    return await run_app_factory_roundtable(
        providers=_all_providers,
        dismissed=_dismissed,
        health_cache=_health_cache,
        app_name=app_name,
        app_mode=app_mode,
        pitch=pitch,
        data_brief=data_brief,
        keywords=keywords,
        min_participants=min_participants,
        max_participants=max_participants,
        force_health_check=force_health_check,
    )


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
    active = {name: s for name, s in _all_providers.items() if s.available and name not in _dismissed}
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
    if provider:
        provider_name = _resolve_provider_name(provider)
        if provider_name is None:
            return f"Unknown provider '{provider}'. Known providers: {_provider_names()}"
        providers = [provider_name]
    else:
        providers = list(_all_providers.keys())

    history_names = [_history_provider_name(p) for p in providers]
    for p in history_names:
        history_store.clear_topic(p, topic)
    return f"Cleared discussion '{topic}' for {', '.join(history_names)}."


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
