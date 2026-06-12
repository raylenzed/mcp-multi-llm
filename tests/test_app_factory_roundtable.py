"""Tests for App Factory roundtable orchestration (veto-engine semantics)."""

import json

import pytest

from app_factory_roundtable import (
    aggregate_final_gate,
    dedupe_challenges,
    normalize_challenge,
    parse_vote,
    run_app_factory_roundtable,
    select_participants,
)
from history.store import HistoryStore
from sessions.base import BaseSession, CLISession, HealthCheckResult


def ok_health(name: str) -> HealthCheckResult:
    return HealthCheckResult(provider=name, status="ok", latency_ms=1, error=None, fix_hint=None)


def challenge(ctype="SOFT_NEGATIVE", claim="generic doubt", verification="check app store top-10"):
    return {"type": ctype, "claim": claim, "cheapest_verification": verification}


class FakeSession(BaseSession):
    def __init__(self, name: str, capabilities=None, verdict: str = "PROCEED", challenges=None):
        super().__init__(HistoryStore())
        self.provider_name = name
        self.capabilities = set(capabilities or [])
        self.verdict = verdict
        self.challenges = challenges if challenges is not None else []
        self.prompts = []

    async def health_check(self) -> HealthCheckResult:
        return ok_health(self.provider_name)

    async def send(self, message: str, topic: str, timeout: int = 300) -> str:
        self.prompts.append((topic, message))
        payload = {
            "verdict": self.verdict,
            "reason": f"{self.provider_name} says {self.verdict}",
            "biggest_concern": "",
            "challenges": self.challenges,
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
            "biggest_concern": "",
            "challenges": [],
        })


def vote_with(challenges, verdict="PROCEED", provider="m1", counted=True):
    return {
        "provider": provider,
        "verdict": verdict,
        "counted": counted,
        "challenges": challenges,
    }


def test_parse_vote_accepts_json_and_fenced_json():
    raw = """```json
{"verdict":"PROCEED","reason":"solid","biggest_concern":"pricing"}
```"""
    vote = parse_vote("gemini", raw, research_capable=True)
    assert vote["verdict"] == "PROCEED"
    assert vote["reason"] == "solid"
    assert vote["biggest_concern"] == "pricing"


def test_parse_vote_parses_classified_challenges():
    raw = json.dumps({
        "verdict": "PIVOT",
        "reason": "narrow it",
        "biggest_concern": "keyword too broad",
        "challenges": [
            {
                "type": "PIVOTABLE_WEAKNESS",
                "claim": "keyword 'recipe app' is red-ocean; narrow to sourdough",
                "cheapest_verification": "search App Store us 'recipe app' top-10 ratings",
            }
        ],
    })
    vote = parse_vote("gemini", raw, research_capable=True)
    assert vote["challenges"][0]["type"] == "PIVOTABLE_WEAKNESS"
    assert "sourdough" in vote["challenges"][0]["claim"]


def test_parse_vote_kill_without_challenges_becomes_evidence_gap():
    """旧格式 KILL（无分类硬伤）不再有处决权，折算为补证据质疑。"""
    vote = parse_vote("codex", "Could pivot, but final verdict: KILL", research_capable=True)
    assert vote["verdict"] == "KILL"
    assert vote["challenges"][0]["type"] == "EVIDENCE_GAP"


def test_parse_vote_marks_unparseable_as_unknown():
    vote = parse_vote("plain", "No clear answer", research_capable=False)
    assert vote["verdict"] == "UNKNOWN"


def test_normalize_challenge_downgrades_unverifiable_hard():
    normalized = normalize_challenge(
        {"type": "HARD_DISQUALIFIER", "claim": "this will fail", "cheapest_verification": ""}
    )
    assert normalized["type"] == "EVIDENCE_GAP"


def test_normalize_challenge_maps_aliases():
    normalized = normalize_challenge(
        {"type": "fatal", "claim": "Google answers it in 3s", "cheapest_verification": "google '1L 40:1'"}
    )
    assert normalized["type"] == "HARD_DISQUALIFIER"


def test_dedupe_merges_same_claim_across_models():
    votes = [
        vote_with([challenge("EVIDENCE_GAP", "no proof buyers search 'hive log' in german stores")], provider="a"),
        vote_with([challenge("EVIDENCE_GAP", "no proof buyers search 'hive log' in german stores today")], provider="b"),
        vote_with([challenge("SOFT_NEGATIVE", "completely different point about pricing tiers")], provider="c"),
    ]
    ledger = dedupe_challenges(votes)
    assert len(ledger) == 2
    merged = next(entry for entry in ledger if entry["type"] == "EVIDENCE_GAP")
    assert set(merged["sources"]) == {"a", "b"}


def test_gate_ignores_proceed_votes_requires_second_round():
    r1 = [vote_with([], provider="a"), vote_with([], provider="b")]
    assert aggregate_final_gate(r1, [], {"has_conflict": False}) == "NEEDS_EVIDENCE"


def test_gate_kills_only_on_verifiable_hard_disqualifier():
    hard = challenge("HARD_DISQUALIFIER", "free app 'X' fully covers core workflow", "open X, run workflow")
    r1 = [vote_with([], provider="a"), vote_with([hard], verdict="KILL", provider="b")]
    r2 = [vote_with([], provider="a")]
    assert aggregate_final_gate(r1, r2, {"has_conflict": False}) == "KILL"


def test_gate_does_not_kill_on_vote_without_hard_challenge():
    """KILL 票数本身没有处决权——这是与旧版的关键差异。"""
    r1 = [vote_with([], verdict="KILL", provider="a"), vote_with([], provider="b")]
    r2 = [vote_with([], provider="a")]
    assert aggregate_final_gate(r1, r2, {"has_conflict": False}) == "PROCEED_TO_EVIDENCE_CARD"


def test_gate_pivot_on_pivotable_weakness():
    weak = challenge("PIVOTABLE_WEAKNESS", "audience too broad, narrow to reef tanks")
    r1 = [vote_with([weak], provider="a")]
    r2 = [vote_with([], provider="a")]
    assert aggregate_final_gate(r1, r2, {"has_conflict": False}) == "PIVOT"


def test_gate_needs_evidence_on_gap_or_conflict():
    gap = challenge("EVIDENCE_GAP", "no data behind the 200 units/year claim")
    r1 = [vote_with([gap], provider="a")]
    r2 = [vote_with([], provider="a")]
    assert aggregate_final_gate(r1, r2, {"has_conflict": False}) == "NEEDS_EVIDENCE"
    assert aggregate_final_gate(
        [vote_with([], provider="a")], [vote_with([], provider="a")], {"has_conflict": True}
    ) == "NEEDS_EVIDENCE"


def test_gate_soft_negatives_never_block():
    soft = challenge("SOFT_NEGATIVE", "feels niche to me")
    r1 = [vote_with([soft], provider="a")]
    r2 = [vote_with([soft], provider="a")]
    assert aggregate_final_gate(r1, r2, {"has_conflict": False}) == "PROCEED_TO_EVIDENCE_CARD"


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


@pytest.mark.asyncio
async def test_run_app_factory_roundtable_smoke_returns_schema():
    providers = {
        "searchy": FakeSession(
            "searchy",
            capabilities={"web_search"},
            challenges=[challenge("SOFT_NEGATIVE", "small market feel", "")],
        ),
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
    assert result["challenge_ledger"]
    assert result["challenge_ledger"][0]["type"] == "SOFT_NEGATIVE"
    assert "质疑账本" in result["summary_markdown"]
    assert "否决器" in result["summary_markdown"]


@pytest.mark.asyncio
async def test_run_app_factory_roundtable_hard_disqualifier_kills_and_skips_round2():
    hard = challenge(
        "HARD_DISQUALIFIER",
        "Google answers the core calculation in 3 seconds",
        "google '1L 40:1 how much oil'",
    )
    providers = {
        "searchy": FakeSession("searchy", capabilities={"web_search"}, verdict="KILL", challenges=[hard]),
        "plain": FakeSession("plain"),
    }
    result = await run_app_factory_roundtable(
        providers=providers,
        dismissed=set(),
        health_cache={},
        app_name="Mix Helper",
        app_mode="local",
        pitch="App: Mix Helper",
        data_brief="Local data.",
        keywords=["fuel mix"],
    )

    assert result["final_gate"] == "KILL"
    assert result["rounds"]["round2_independent_research"]["initial_votes"] == []
    assert any(entry["type"] == "HARD_DISQUALIFIER" for entry in result["challenge_ledger"])


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
