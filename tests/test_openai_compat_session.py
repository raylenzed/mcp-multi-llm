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


def json_response(payload: dict) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json.return_value = payload
    return response


def final_response(content: str) -> MagicMock:
    return json_response({"choices": [{"message": {"role": "assistant", "content": content}}]})


def tool_call_response(query: str) -> MagicMock:
    return json_response({"choices": [{"message": {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "web_search", "arguments": f'{{"query": "{query}"}}'}}],
    }}]})


def make_search_session(max_searches=6):
    from sessions.web_search import resolve_web_search
    store = HistoryStore()
    return OpenAICompatSession(
        history_store=store,
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        api_key="sk-test",
        web_search=resolve_web_search({"preset": "deepseek", "max_searches": max_searches}),
    )


@pytest.mark.asyncio
async def test_web_search_loop_runs_search_and_feeds_results_back():
    """A tool call triggers a search API request, and its results go back to the model."""
    session = make_search_session()
    search_results = json_response({"results": [
        {"title": "HiveMate", "url": "https://apps.apple.com/app/id1", "summary": "Beekeeping log"},
    ]})
    with pytest.MonkeyPatch.context() as mp:
        mock_post = AsyncMock(side_effect=[tool_call_response("hive app"), search_results, final_response("final")])
        mp.setattr(session._client, "post", mock_post)
        result = await session.send("Research hive apps", "research")

    assert result == "final"
    first, search, last = mock_post.call_args_list
    assert first.kwargs["json"]["tools"][0]["function"]["name"] == "web_search"
    assert search.args[0] == "https://api.deepseek.com/v1/tools/search"
    assert search.kwargs["json"]["query"] == "hive app"
    tool_message = last.kwargs["json"]["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "call_1"
    assert "HiveMate" in tool_message["content"]


@pytest.mark.asyncio
async def test_web_search_loop_stops_offering_tools_after_cap():
    """Once max_searches is spent, the next request has no tools and must answer."""
    session = make_search_session(max_searches=1)
    with pytest.MonkeyPatch.context() as mp:
        mock_post = AsyncMock(side_effect=[
            tool_call_response("q"), json_response({"results": []}), final_response("answer"),
        ])
        mp.setattr(session._client, "post", mock_post)
        result = await session.send("Research", "cap")

    assert result == "answer"
    assert "tools" not in mock_post.call_args_list[-1].kwargs["json"]


@pytest.mark.asyncio
async def test_no_tools_sent_without_web_search():
    """Plain providers keep the original single-request payload."""
    session = make_session()
    with pytest.MonkeyPatch.context() as mp:
        mock_post = AsyncMock(return_value=mock_response("ok"))
        mp.setattr(session._client, "post", mock_post)
        await session.send("hi", "plain")
    assert "tools" not in mock_post.call_args.kwargs["json"]
    assert mock_post.call_count == 1
