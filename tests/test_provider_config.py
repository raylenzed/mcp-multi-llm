"""Tests for custom provider config loading."""
import json
import pytest
from pathlib import Path
from sessions.provider_config import load_custom_providers, ProviderConfig


def write_config(tmp_path: Path, data) -> Path:
    p = tmp_path / "custom_providers.json"
    p.write_text(json.dumps(data))
    return p


def test_load_valid_config(tmp_path):
    """Valid config returns list of ProviderConfig objects."""
    cfg_path = write_config(tmp_path, [
        {"name": "deepseek", "base_url": "https://api.deepseek.com/v1",
         "model": "deepseek-chat", "api_key": "sk-123"},
    ])
    providers = load_custom_providers(cfg_path)
    assert len(providers) == 1
    assert providers[0].name == "deepseek"
    assert providers[0].model == "deepseek-chat"
    assert providers[0].api_key == "sk-123"
    assert providers[0].capabilities == []


def test_load_missing_file_returns_empty(tmp_path):
    """Missing config file returns empty list (not an error)."""
    providers = load_custom_providers(tmp_path / "nonexistent.json")
    assert providers == []


def test_invalid_name_is_skipped(tmp_path):
    """Entries with invalid names (spaces, special chars) are skipped with a warning."""
    cfg_path = write_config(tmp_path, [
        {"name": "bad name!", "base_url": "https://x.com", "model": "m", "api_key": "k"},
        {"name": "valid", "base_url": "https://x.com", "model": "m", "api_key": "k"},
    ])
    providers = load_custom_providers(cfg_path)
    assert len(providers) == 1
    assert providers[0].name == "valid"


def test_builtin_name_can_override_cli_fallback(tmp_path):
    """Builtin names may be configured to override the CLI fallback."""
    cfg_path = write_config(tmp_path, [
        {"name": "claude", "base_url": "https://x.com", "model": "m", "api_key": "k"},
    ])
    providers = load_custom_providers(cfg_path)
    assert len(providers) == 1
    assert providers[0].name == "claude"


def test_builtin_override_preserves_configured_case(tmp_path):
    """Server-side registration handles builtin names case-insensitively."""
    cfg_path = write_config(tmp_path, [
        {"name": "Claude", "base_url": "https://x.com", "model": "m", "api_key": "k"},
    ])
    providers = load_custom_providers(cfg_path)
    assert len(providers) == 1
    assert providers[0].name == "Claude"


def test_missing_required_field_is_skipped(tmp_path):
    """Entries missing required fields are skipped."""
    cfg_path = write_config(tmp_path, [
        {"name": "deepseek", "model": "deepseek-chat", "api_key": "sk-123"},  # no base_url
    ])
    providers = load_custom_providers(cfg_path)
    assert providers == []


def test_load_multiple_providers(tmp_path):
    """Multiple valid entries all load correctly."""
    cfg_path = write_config(tmp_path, [
        {"name": "deepseek", "base_url": "https://api.deepseek.com/v1",
         "model": "deepseek-chat", "api_key": "sk-a"},
        {"name": "qianwen", "base_url": "https://dashscope.aliyuncs.com/v1",
         "model": "qwen-plus", "api_key": "sk-b"},
    ])
    providers = load_custom_providers(cfg_path)
    assert len(providers) == 2
    assert {p.name for p in providers} == {"deepseek", "qianwen"}


def test_non_dict_entry_is_skipped(tmp_path):
    """Non-dict entries in the array are skipped without crashing."""
    cfg_path = write_config(tmp_path, ["not-a-dict", 42, None])
    providers = load_custom_providers(cfg_path)
    assert providers == []


def test_malformed_json_returns_empty(tmp_path):
    """Malformed JSON file returns empty list."""
    p = tmp_path / "custom_providers.json"
    p.write_text("this is not json")
    providers = load_custom_providers(p)
    assert providers == []


def test_capabilities_are_normalized(tmp_path):
    """Optional provider capabilities are normalized for orchestration."""
    cfg_path = write_config(tmp_path, [
        {
            "name": "searchy",
            "base_url": "https://x.com",
            "model": "m",
            "api_key": "k",
            "capabilities": ["web_search", " Web_Search ", 123, ""],
        },
    ])
    providers = load_custom_providers(cfg_path)
    assert len(providers) == 1
    assert providers[0].capabilities == ["web_search"]


def test_invalid_capabilities_are_ignored(tmp_path):
    """Invalid capability declarations degrade to no declared capabilities."""
    cfg_path = write_config(tmp_path, [
        {
            "name": "plain",
            "base_url": "https://x.com",
            "model": "m",
            "api_key": "k",
            "capabilities": "web_search",
        },
    ])
    providers = load_custom_providers(cfg_path)
    assert len(providers) == 1
    assert providers[0].capabilities == []


def test_web_search_preset_enables_capability(tmp_path):
    """A web_search preset resolves to a full config and implies the web_search capability."""
    cfg_path = write_config(tmp_path, [
        {"name": "deepseek", "base_url": "https://api.deepseek.com/v1",
         "model": "deepseek-chat", "api_key": "sk-1", "web_search": "deepseek"},
        {"name": "kimi", "base_url": "https://api.moonshot.cn/v1",
         "model": "kimi-k3", "api_key": "sk-2", "web_search": {"preset": "kimi", "count": 3}},
    ])
    deepseek, kimi = load_custom_providers(cfg_path)
    assert deepseek.web_search["endpoint"] == "https://api.deepseek.com/v1/tools/search"
    assert deepseek.web_search["query_field"] == "query"
    assert deepseek.capabilities == ["web_search"]
    assert kimi.web_search["query_field"] == "text_query"
    assert kimi.web_search["count"] == 3


def test_invalid_web_search_is_ignored(tmp_path):
    """Unknown presets and non-openai protocols do not enable search."""
    cfg_path = write_config(tmp_path, [
        {"name": "a", "base_url": "https://x.com", "model": "m", "api_key": "k", "web_search": "nope"},
        {"name": "b", "base_url": "https://x.com", "model": "m", "api_key": "k",
         "protocol": "anthropic", "web_search": "deepseek"},
        {"name": "c", "base_url": "https://x.com", "model": "m", "api_key": "k",
         "web_search": {"endpoint": "http://insecure.example/search"}},
    ])
    for provider in load_custom_providers(cfg_path):
        assert provider.web_search is None
        assert provider.capabilities == []


def test_web_search_api_key_from_reuses_other_entry_key(tmp_path):
    """api_key_from points the search API at another provider's key."""
    cfg_path = write_config(tmp_path, [
        {"name": "deepseek", "base_url": "https://api.deepseek.com/v1", "model": "m", "api_key": "sk-ds"},
        {"name": "glm", "base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "m", "api_key": "sk-glm",
         "web_search": {"preset": "deepseek", "api_key_from": "DeepSeek"}},
        {"name": "lost", "base_url": "https://x.com", "model": "m", "api_key": "k",
         "web_search": {"preset": "deepseek", "api_key_from": "missing"}},
    ])
    _, glm, lost = load_custom_providers(cfg_path)
    assert glm.web_search["api_key"] == "sk-ds"
    assert "api_key_from" not in glm.web_search
    assert lost.web_search is None
    assert lost.capabilities == []
