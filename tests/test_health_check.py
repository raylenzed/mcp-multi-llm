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
    hs = make_history_store()
    session = OpenAICompatSession(
        hs, name="testprovider", base_url="http://localhost:9999",
        model="test-model", api_key="bad-key"
    )

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status = MagicMock()

    with patch.object(session._client, "post", AsyncMock(return_value=mock_response)):
        result = await session.health_check()

    assert result.status == "auth_error"
    assert result.fix_hint is not None
    assert not session.available


@pytest.mark.asyncio
async def test_openai_compat_health_check_http_error_includes_body_and_hint():
    from sessions.openai_compat_session import OpenAICompatSession
    hs = make_history_store()
    session = OpenAICompatSession(
        hs, name="testprovider", base_url="http://localhost:9999",
        model="missing-model", api_key="key"
    )

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = '{"error":"model not found"}'

    import httpx
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404", request=MagicMock(), response=mock_response
    )

    with patch.object(session._client, "post", AsyncMock(return_value=mock_response)):
        result = await session.health_check()

    assert result.status == "error"
    assert "model not found" in result.error
    assert "base_url" in result.fix_hint


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
