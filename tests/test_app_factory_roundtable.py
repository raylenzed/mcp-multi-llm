"""Tests for App Factory roundtable orchestration."""

import json

import pytest

from app_factory_roundtable import (
    aggregate_final_gate,
    parse_vote,
    run_app_factory_roundtable,
    select_participants,
)
from history.store import HistoryStore
from sessions.base import BaseSession, CLISession, HealthCheckResult


def ok_health(name: str) -> HealthCheckResult:
    return HealthCheckResult(provider=name, status="ok", latency_ms=1, error=None, fix_hint=None)


class FakeSession(BaseSession):
    def __init__(self, name: str, capabilities=None, verdict: str = "PROCEED"):
        super().__init__(HistoryStore())
        self.provider_name = name
        self.capabilities = set(capabilities or [])
        self.verdict = verdict
        self.prompts = []

    async def health_check(self) -> HealthCheckResult:
        return ok_health(self.provider_name)

    async def send(self, message: str, topic: str, timeout: int = 300) -> str:
        self.prompts.append((topic, message))
        payload = {
            "verdict": self.verdict,
            "reason": f"{self.provider_name} says {self.verdict}",
            "biggest_concern": "distribution proof",
        }
        if "independent research responsibility" in message or "second-round" in message:
            payload["independent_findings"] = {
                "competitors": "top App Store results have reviews",
                "keyword_demand": "keyword demand appears adequate",
                "pain_evidence": "review/forum complaints exist",
            }
            payload["data_consistency"] = {
                "competitors": "CONSISTENT",
                "keyword_demand": "CONSISTENT",
                "pain_evidence": "CONSISTENT",
            }
        return json.dumps(payload)


class FakeCLISession(CLISession):
    provider_name = "fake_cli"

    def __init__(self, name: str):
        super().__init__(HistoryStore())
        self.provider_name = name

    def _build_command(self, message, history):
        return ["fake"]

    def _parse_output(self, stdout, stderr):
        return stdout

    def _health_check_command(self):
        return ["fake", "--health"]

    async def health_check(self) -> HealthCheckResult:
        return ok_health(self.provider_name)

    async def send(self, message: str, topic: str, timeout: int = 300) -> str:
        return json.dumps({
            "verdict": "PROCEED",
            "reason": "cli proceed",
            "biggest_concern": "none",
        })


def test_parse_vote_accepts_json_and_fenced_json():
    raw = """```json
{"verdict":"PROCEED","reason":"solid","biggest_concern":"pricing"}
```"""
    vote = parse_vote("gemini", raw, research_capable=True)
    assert vote["verdict"] == "PROCEED"
    assert vote["reason"] == "solid"
    assert vote["biggest_concern"] == "pricing"


def test_parse_vote_falls_back_to_last_plain_text_verdict():
    vote = parse_vote("codex", "Could pivot, but final verdict: KILL", research_capable=True)
    assert vote["verdict"] == "KILL"


def test_parse_vote_marks_unparseable_as_unknown():
    vote = parse_vote("plain", "No clear answer", research_capable=False)
    assert vote["verdict"] == "UNKNOWN"


def test_select_participants_prioritizes_research_and_skips_dismissed():
    providers = {
        "plain": FakeSession("plain"),
        "searchy": FakeSession("searchy", capabilities={"web_search"}),
        "cli": FakeCLISession("cli"),
        "dismissed": FakeSession("dismissed", capabilities={"web_search"}),
    }
    health_cache = {name: ok_health(name) for name in providers}
    selected = select_participants(providers, {"dismissed"}, health_cache, max_participants=3)
    assert [p.name for p in selected] == ["cli", "searchy", "plain"]
    assert [p.research_capable for p in selected] == [True, True, False]


def test_aggregate_final_gate_requires_second_round_evidence():
    r1 = [{"verdict": "PROCEED", "counted": True}, {"verdict": "PROCEED", "counted": True}]
    assert aggregate_final_gate(r1, [], {"has_conflict": False}) == "NEEDS_EVIDENCE"


def test_aggregate_final_gate_kills_on_any_kill():
    r1 = [{"verdict": "PROCEED", "counted": True}, {"verdict": "KILL", "counted": True}]
    r2 = [{"verdict": "PROCEED", "counted": True}]
    assert aggregate_final_gate(r1, r2, {"has_conflict": False}) == "KILL"


@pytest.mark.asyncio
async def test_run_app_factory_roundtable_smoke_returns_schema():
    providers = {
        "searchy": FakeSession("searchy", capabilities={"web_search"}),
        "plain": FakeSession("plain"),
    }
    result = await run_app_factory_roundtable(
        providers=providers,
        dismissed=set(),
        health_cache={},
        app_name="Tiny Tool",
        app_mode="local",
        pitch="App: Tiny Tool\nTarget user: solo makers",
        data_brief="Keyword demand and review complaints exist.",
        keywords=["tiny tool"],
    )

    assert result["status"] == "completed"
    assert result["final_gate"] == "PROCEED_TO_EVIDENCE_CARD"
    assert result["participants"][0]["provider"] == "searchy"
    assert result["participants"][0]["participates_in_second_round"]
    assert not result["participants"][1]["participates_in_second_round"]
    assert result["rounds"]["round1_local_data"]["final_votes"]
    assert result["rounds"]["round2_independent_research"]["final_votes"]
    assert "多模型圆桌评审" in result["summary_markdown"]


@pytest.mark.asyncio
async def test_run_app_factory_roundtable_without_research_capable_is_partial():
    providers = {
        "plain": FakeSession("plain"),
        "other": FakeSession("other"),
    }
    result = await run_app_factory_roundtable(
        providers=providers,
        dismissed=set(),
        health_cache={},
        app_name="Tiny Tool",
        app_mode="local",
        pitch="App: Tiny Tool",
        data_brief="Some local evidence.",
        keywords=["tiny tool"],
    )

    assert result["status"] == "partial"
    assert result["final_gate"] == "NEEDS_EVIDENCE"
    assert result["rounds"]["round2_independent_research"]["skipped_reason"]
