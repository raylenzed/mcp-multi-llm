# Provider Health Check & Robustness Design

**Date:** 2026-04-03  
**Status:** Approved

## Overview

Add real provider availability checking to mcp-multi-llm, so that before any multi-LLM workflow (e.g., app-factory's Phase 1c roundtable), Claude can test every provider with a real lightweight request, report failures with actionable fix instructions, and let the user decide to dismiss or fix unavailable providers — all within the normal MCP tool call flow.

---

## Problem

Currently:
- CLI providers (Claude, Codex, Gemini) only check if the binary exists in PATH (`shutil.which()`). A provider can be "installed" but unauthenticated (e.g., Gemini without OAuth) and only fail at actual use time.
- API providers (custom providers in `custom_providers.json`) always start as `available=True` — a bad API key or wrong base URL is only discovered when a real message is sent.
- `list_available_providers()` reports CLI binary presence, not actual connectivity.
- No way for the user to dismiss a provider for the session without editing config files.
- Error messages on failure are generic; no fix instructions are surfaced.

---

## Solution

### New MCP Tools

#### `health_check_providers(providers: list[str] | None = None, force: bool = False) -> str`

Tests each provider with a real lightweight request in parallel. Returns a structured report with status, latency, errors, and fix hints.

- If `providers` is None, tests all registered providers.
- Results are cached in-process for the session; subsequent calls return cached results unless `force=True` is passed.
- Timeout per provider: **30s for CLI**, **10s for API**.
- Test prompt: `"Reply with the single word: ok"` — minimal tokens, unambiguous success check.

**Return format (markdown for readability):**

```
## Provider Health Check

✓ claude       — ok (2.1s)
✓ moonshot     — ok (1.4s)
✗ gemini       — auth error: OAuth token not found
  Fix: run `gemini auth login` in your terminal
✗ codex        — not installed: binary not found in PATH
  Fix: run `npm install -g @openai/codex`
✗ deepseek     — HTTP 401: invalid API key
  Fix: update `api_key` in ~/.mcp-multi-llm/custom_providers.json

2 available, 3 unavailable.
Call dismiss_provider("name") to skip a provider for this session.
```

**Status codes (internal):**
- `ok` — responded correctly
- `not_installed` — binary not found (CLI only)
- `auth_error` — 401/403 or OAuth failure
- `network_error` — connection refused / DNS failure
- `timeout` — exceeded timeout
- `parse_error` — responded but output unparsable

#### `dismiss_provider(name: str) -> str`

Marks a provider as intentionally skipped for the current session. Sets `session.available = False` and adds to an in-memory `_dismissed` set so the provider is excluded from `group_discuss` and listed as "dismissed" in `list_available_providers()`.

Returns: `"gemini dismissed for this session."`

---

### Changes to Existing Tools

#### `list_available_providers()`

Keep as a fast check (no real requests). Add a footer note:

```
Tip: call health_check_providers() to test real connectivity and authentication.
```

---

### Robustness Improvements to Sessions

#### All API sessions (`OpenAICompatSession`, `AnthropicCompatSession`)

1. **Set `available = False` on auth errors**: if a `send()` call returns HTTP 401 or 403, mark the session unavailable and surface a clear error message instead of letting the next call fail again.

2. **Retry on transient errors**: up to 2 retries with 1s delay for `5xx` errors and network timeouts. Auth errors and `4xx` (except 429) are not retried.

3. **Rate limit handling**: on HTTP 429, include the `Retry-After` header value in the error message if present.

4. **Better error messages**: distinguish auth error / network error / rate limit / model not found in the returned string. Current code returns raw exception text.

#### CLI sessions (`ClaudeSession`, `CodexSession`, `GeminiSession`)

1. **Auth detection**: if stderr contains known auth failure strings (e.g., `"not logged in"`, `"OAuth"`, `"authentication"`, `"401"`), set `available = False` and return a clear message with fix hint — rather than returning garbled output.

2. **Retry on timeout**: one retry if the subprocess times out (some CLI tools have cold-start latency). Second timeout is final.

3. **Exit code check**: non-zero exit code with empty stdout is now an explicit error, not silently returned as empty string.

#### Health check implementation detail

Each session gets a new async method:

```python
async def health_check(self) -> HealthCheckResult:
    """Run a lightweight real request to verify connectivity and auth."""
    ...
```

`HealthCheckResult` is a dataclass:

```python
@dataclass
class HealthCheckResult:
    provider: str
    status: str          # ok | not_installed | auth_error | network_error | timeout | parse_error
    latency_ms: int | None
    error: str | None    # human-readable error detail
    fix_hint: str | None # actionable fix instruction
```

The `health_check_providers()` MCP tool calls `asyncio.gather()` over all session health checks, collects results, caches them, and formats the markdown report.

---

### app-factory Integration

In `skills/new-app/SKILL.md`, `skills/new-app-freeserver/SKILL.md`, and `skills/new-app-paidserver/SKILL.md` — **Phase 1c, Step 1**:

Replace:
```
list_available_providers()
```

With:
```
health_check_providers()
```

Add instruction after the call:
```
If any providers show ✗:
- Ask the user: "X is unavailable ([reason]). Fix: [fix_hint]. Skip it for now, or pause to fix?"
- If skip: call dismiss_provider("x"), continue with remaining providers.
- If fix: wait for user to complete the fix, then call health_check_providers(["x"]) to retest.
- If all providers are unavailable: abort Phase 1c and inform the user.
```

---

## Data Flow

```
new-app skill Phase 1c
    │
    ▼
health_check_providers()    ← parallel asyncio.gather()
    ├─ ClaudeSession.health_check()    → subprocess, 30s timeout
    ├─ GeminiSession.health_check()   → subprocess, 30s timeout
    ├─ CodexSession.health_check()    → subprocess, 30s timeout
    └─ OpenAICompatSession.health_check() → httpx POST, 10s timeout
    │
    ▼
Structured markdown report returned to Claude
    │
    ▼
Claude presents results, asks user about each ✗ provider
    │
    ├─ User: "skip gemini" → dismiss_provider("gemini")
    ├─ User: "I'll fix it" → user runs fix → health_check_providers(["gemini"])
    └─ All OK → proceed to group_discuss / discuss_with_*
```

---

## Files Changed

| File | Change |
|------|--------|
| `sessions/base.py` | Add `HealthCheckResult` dataclass; add abstract `health_check()` to `BaseSession`; add retry logic and auth detection to `CLISession.send()` |
| `sessions/api_session.py` | Add retry logic, rate limit handling, better error messages |
| `sessions/claude_session.py` | Implement `health_check()`; add auth string detection |
| `sessions/codex_session.py` | Implement `health_check()` |
| `sessions/gemini_session.py` | Implement `health_check()`; add auth string detection |
| `sessions/openai_compat_session.py` | Implement `health_check()`; add auth/rate-limit handling |
| `sessions/anthropic_compat_session.py` | Implement `health_check()`; add auth handling |
| `server.py` | Add `health_check_providers()` and `dismiss_provider()` tools; add `_dismissed` set; add result cache |
| `~/Github/app-factory/skills/new-app/SKILL.md` | Phase 1c: replace `list_available_providers()` with `health_check_providers()` + decision instructions |
| `~/Github/app-factory/skills/new-app-freeserver/SKILL.md` | Same |
| `~/Github/app-factory/skills/new-app-paidserver/SKILL.md` | Same |

---

## Out of Scope

- Persisting dismissed providers across sessions (session-only is sufficient)
- Auto-running health check at MCP server startup
- Modifying `custom_providers.json` from within Claude Code
- Health check for `list_discussions` / `clear_discussion` tools (no provider needed)
