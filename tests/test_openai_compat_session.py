"""Tests for OpenAICompatSession — httpx-based OpenAI-compatible provider."""
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock
from history.store import HistoryStore
from sessions.openai_compat_session import OpenAICompatSession


def make_session(name="deepseek"):
    store = HistoryStore()
    return OpenAICompatSession(
        history_store=store,
        name=name,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        api_key="sk-test",
    )


def mock_response(content: str) -> MagicMock:
    """Build a fake httpx.Response with OpenAI-style JSON."""
    response = MagicMock(spec=httpx.Response)
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return response


@pytest.mark.asyncio
async def test_send_returns_model_content():
    """Session sends message and returns model reply."""
    session = make_session()
    with pytest.MonkeyPatch.context() as mp:
        _mock_response = mock_response("Hello from DeepSeek!")
        _mock_response.status_code = 200
        mock_post = AsyncMock(return_value=_mock_response)
        mp.setattr(session._client, "post", mock_post)
        result = await session.send("Hi", "test")
    assert result == "Hello from DeepSeek!"


@pytest.mark.asyncio
async def test_send_builds_correct_payload():
    """Session sends correct OpenAI-format payload."""
    session = make_session()
    with pytest.MonkeyPatch.context() as mp:
        mock_post = AsyncMock(return_value=mock_response("ok"))
        mp.setattr(session._client, "post", mock_post)
        await session.send("What is 2+2?", "math")

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs["json"]
    assert payload["model"] == "deepseek-chat"
    assert payload["messages"][-1] == {"role": "user", "content": "What is 2+2?"}


@pytest.mark.asyncio
async def test_send_http_error_returns_error_string():
    """HTTP errors are caught and returned as readable error strings."""
    session = make_session()
    with pytest.MonkeyPatch.context() as mp:
        mock_post = AsyncMock(side_effect=httpx.HTTPStatusError(
            "401", request=MagicMock(), response=MagicMock(status_code=401)
        ))
        mp.setattr(session._client, "post", mock_post)
        result = await session.send("Hi", "test")
    assert "deepseek" in result.lower() or "401" in result


@pytest.mark.asyncio
async def test_provider_name_set_correctly():
    """provider_name matches the name passed to constructor."""
    session = make_session("qianwen")
    assert session.provider_name == "qianwen"


@pytest.mark.asyncio
async def test_available_true_by_default():
    """API sessions are available by default (no CLI to find)."""
    session = make_session()
    assert session.available is True
