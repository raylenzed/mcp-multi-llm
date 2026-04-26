"""App Factory roundtable orchestration for multi-model market review."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from sessions.base import BaseSession, CLISession, HealthCheckResult

APP_MODES = {"local", "freeserver", "paidserver"}
VERDICTS = {"PROCEED", "PIVOT", "KILL"}
FINAL_GATES = {"PROCEED_TO_EVIDENCE_CARD", "NEEDS_EVIDENCE", "KILL", "SKIPPED"}
CONSISTENCY_VALUES = {"CONSISTENT", "CONFLICT", "UNKNOWN"}

MODE_GUIDANCE = {
    "local": (
        "Pure local/offline iOS app. No server, no API calls, no subscription. "
        "One-time purchase $2.99-$4.99. China region may be allowed only if category is safe."
    ),
    "freeserver": (
        "iOS app with Cloudflare free-tier backend. Subscription allowed. "
        "No paid server cost. Overseas App Store only."
    ),
    "paidserver": (
        "iOS app with Hetzner CAX11 + Cloudflare Tunnel paid backend. "
        "Subscription and AI/API calls allowed. Overseas App Store only."
    ),
}


@dataclass
class Participant:
    name: str
    session: BaseSession
    health: HealthCheckResult | None
    research_capable: bool

    def to_dict(self, second_round: bool) -> dict[str, Any]:
        return {
            "provider": self.name,
            "health": health_to_dict(self.health),
            "research_capable": self.research_capable,
            "participates_in_second_round": second_round,
            "capabilities": sorted(get_capabilities(self.session)),
        }


def get_capabilities(session: BaseSession) -> set[str]:
    capabilities = getattr(session, "capabilities", set())
    if isinstance(capabilities, set):
        return {str(c).strip().lower() for c in capabilities if str(c).strip()}
    if isinstance(capabilities, list | tuple):
        return {str(c).strip().lower() for c in capabilities if str(c).strip()}
    return set()


def is_research_capable(session: BaseSession) -> bool:
    return isinstance(session, CLISession) or "web_search" in get_capabilities(session)


def health_to_dict(result: HealthCheckResult | None) -> dict[str, Any]:
    if result is None:
        return {
            "status": "unknown",
            "latency_ms": None,
            "error": None,
            "fix_hint": None,
        }
    return {
        "status": result.status,
        "latency_ms": result.latency_ms,
        "error": result.error,
        "fix_hint": result.fix_hint,
    }


async def ensure_health_checks(
    providers: dict[str, BaseSession],
    dismissed: set[str],
    health_cache: dict[str, HealthCheckResult],
    force: bool = False,
) -> None:
    """Run provider health checks and update the shared health cache."""
    to_test = {
        name: session
        for name, session in providers.items()
        if name not in dismissed and (force or name not in health_cache)
    }
    if not to_test:
        return

    results = await asyncio.gather(
        *[session.health_check() for session in to_test.values()],
        return_exceptions=True,
    )
    for name, result in zip(to_test.keys(), results):
        if isinstance(result, Exception):
            health_cache[name] = HealthCheckResult(
                provider=name,
                status="error",
                latency_ms=None,
                error=str(result),
                fix_hint=None,
            )
            providers[name].available = False
            continue

        health_cache[name] = result
        if result.status == "ok" and name not in dismissed:
            providers[name].available = True
        else:
            providers[name].available = False


def select_participants(
    providers: dict[str, BaseSession],
    dismissed: set[str],
    health_cache: dict[str, HealthCheckResult],
    max_participants: int,
) -> list[Participant]:
    """Select available providers, prioritizing research-capable participants."""
    builtin_rank = {"claude": 0, "codex": 1, "gemini": 2}
    candidates: list[Participant] = []
    for name, session in providers.items():
        if name in dismissed or not session.available:
            continue
        health = health_cache.get(name)
        if health is not None and health.status != "ok":
            continue
        candidates.append(
            Participant(
                name=name,
                session=session,
                health=health,
                research_capable=is_research_capable(session),
            )
        )

    candidates.sort(
        key=lambda p: (
            0 if p.research_capable else 1,
            builtin_rank.get(p.name.lower(), 100),
            p.name.lower(),
        )
    )
    return candidates[:max(1, max_participants)]


def normalize_verdict(value: Any) -> str:
    if not isinstance(value, str):
        return "UNKNOWN"
    verdict = value.strip().upper()
    return verdict if verdict in VERDICTS else "UNKNOWN"


def normalize_consistency(value: Any) -> str:
    if not isinstance(value, str):
        return "UNKNOWN"
    normalized = value.strip().upper()
    return normalized if normalized in CONSISTENCY_VALUES else "UNKNOWN"


def decode_first_json_object(text: str) -> dict[str, Any] | None:
    """Decode the first JSON object from raw or fenced model output."""
    candidates = [text]
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        candidates.insert(0, match.group(1))

    decoder = json.JSONDecoder()
    for candidate in candidates:
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                decoded, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                return decoded
    return None


def parse_vote(
    provider: str,
    response: str,
    research_capable: bool,
    counted: bool = True,
) -> dict[str, Any]:
    """Parse one provider response into a deterministic vote record."""
    parsed = decode_first_json_object(response)
    verdict = "UNKNOWN"
    reason = ""
    concern = ""
    data_consistency: dict[str, str] = {}
    independent_findings: dict[str, Any] = {}

    if parsed is not None:
        verdict = normalize_verdict(
            parsed.get("verdict")
            or parsed.get("final_verdict")
            or parsed.get("vote")
            or parsed.get("final_vote")
        )
        reason = str(
            parsed.get("reason")
            or parsed.get("one_line_reason")
            or parsed.get("rationale")
            or ""
        ).strip()
        concern = str(
            parsed.get("biggest_concern")
            or parsed.get("single_biggest_concern")
            or parsed.get("concern")
            or ""
        ).strip()
        raw_consistency = parsed.get("data_consistency")
        if isinstance(raw_consistency, dict):
            data_consistency = {
                str(key): normalize_consistency(value)
                for key, value in raw_consistency.items()
            }
        raw_findings = parsed.get("independent_findings")
        if isinstance(raw_findings, dict):
            independent_findings = raw_findings

    if verdict == "UNKNOWN":
        matches = re.findall(r"\b(PROCEED|PIVOT|KILL)\b", response.upper())
        if matches:
            verdict = matches[-1]

    if not concern:
        match = re.search(
            r"(?:biggest concern|single biggest concern|concern)\s*[:：]\s*(.+)",
            response,
            flags=re.IGNORECASE,
        )
        if match:
            concern = match.group(1).strip().splitlines()[0][:500]

    if not reason:
        reason = _first_nonempty_line(response)

    error = None
    stripped = response.strip()
    if stripped.startswith("Error:") or " unavailable " in stripped[:160]:
        error = stripped[:500]

    return {
        "provider": provider,
        "verdict": verdict,
        "reason": reason[:700],
        "biggest_concern": concern[:700],
        "research_capable": research_capable,
        "counted": counted,
        "data_consistency": data_consistency,
        "independent_findings": independent_findings,
        "parsed_json": parsed,
        "raw_response": response,
        "error": error,
    }


def aggregate_comparison(votes: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("competitors", "keyword_demand", "pain_evidence")
    rows: dict[str, str] = {}
    notes: dict[str, list[str]] = {metric: [] for metric in metrics}

    for metric in metrics:
        values = [
            vote.get("data_consistency", {}).get(metric, "UNKNOWN")
            for vote in votes
            if vote.get("counted")
        ]
        if any(value == "CONFLICT" for value in values):
            rows[metric] = "CONFLICT"
        elif values and values.count("CONSISTENT") >= max(1, len(values) // 2 + len(values) % 2):
            rows[metric] = "CONSISTENT"
        else:
            rows[metric] = "UNKNOWN"

    for vote in votes:
        findings = vote.get("independent_findings", {})
        if not isinstance(findings, dict):
            continue
        for metric in metrics:
            value = findings.get(metric)
            if value:
                notes[metric].append(f"{vote['provider']}: {str(value)[:500]}")

    return {
        "metrics": rows,
        "has_conflict": any(value == "CONFLICT" for value in rows.values()),
        "model_findings": notes,
    }


def aggregate_final_gate(
    round1_votes: list[dict[str, Any]],
    round2_votes: list[dict[str, Any]],
    comparison: dict[str, Any] | None = None,
) -> str:
    counted_r1 = [vote for vote in round1_votes if vote.get("counted")]
    counted_r2 = [vote for vote in round2_votes if vote.get("counted")]
    if not counted_r1:
        return "SKIPPED"

    if any(vote.get("verdict") == "KILL" for vote in counted_r1 + counted_r2):
        return "KILL"

    if not counted_r2:
        return "NEEDS_EVIDENCE"

    required_r1 = min(2, len(counted_r1))
    required_r2 = min(2, len(counted_r2))
    r1_proceed = sum(1 for vote in counted_r1 if vote.get("verdict") == "PROCEED")
    r2_proceed = sum(1 for vote in counted_r2 if vote.get("verdict") == "PROCEED")
    has_conflict = bool(comparison and comparison.get("has_conflict"))

    if r1_proceed >= required_r1 and r2_proceed >= required_r2 and not has_conflict:
        return "PROCEED_TO_EVIDENCE_CARD"
    return "NEEDS_EVIDENCE"


async def run_app_factory_roundtable(
    *,
    providers: dict[str, BaseSession],
    dismissed: set[str],
    health_cache: dict[str, HealthCheckResult],
    app_name: str,
    app_mode: str,
    pitch: str,
    data_brief: str,
    keywords: list[str] | None = None,
    min_participants: int = 2,
    max_participants: int = 5,
    force_health_check: bool = False,
) -> dict[str, Any]:
    """Run the App Factory two-round evaluation and return structured JSON."""
    if app_mode not in APP_MODES:
        return _skipped_result(app_name, f"Unsupported app_mode: {app_mode}")

    min_participants = max(1, min_participants)
    max_participants = max(min_participants, max_participants)

    await ensure_health_checks(providers, dismissed, health_cache, force=force_health_check)
    participants = select_participants(providers, dismissed, health_cache, max_participants)
    if not participants:
        return _skipped_result(app_name, "No healthy multi-LLM providers available.")

    topic_base = f"app-factory-{_slug(app_name)}"
    r1_prompt = _round1_prompt(app_name, app_mode, pitch, data_brief)
    r1_initial = await _collect_votes(participants, r1_prompt, f"{topic_base}-r1", counted=True)
    r1_discussion_prompt = _discussion_prompt(
        app_name=app_name,
        app_mode=app_mode,
        round_name="first-round local-data discussion",
        pitch=pitch,
        data_summary=data_brief,
        vote_summary=_votes_summary(r1_initial),
    )
    r1_discussion = await _collect_votes(
        participants,
        r1_discussion_prompt,
        f"{topic_base}-r1-discussion",
        counted=True,
    )
    r1_final = _prefer_parsed_votes(r1_discussion, r1_initial)

    r2_initial: list[dict[str, Any]] = []
    r2_discussion: list[dict[str, Any]] = []
    r2_final: list[dict[str, Any]] = []
    comparison = aggregate_comparison([])

    if any(vote.get("verdict") == "KILL" for vote in r1_final if vote.get("counted")):
        final_gate = "KILL"
    else:
        research_participants = [p for p in participants if p.research_capable]
        if research_participants:
            r2_prompt = _round2_prompt(app_name, app_mode, pitch, keywords or [])
            r2_initial = await _collect_votes(
                research_participants,
                r2_prompt,
                f"{topic_base}-r2",
                counted=True,
            )
            r2_discussion_prompt = _discussion_prompt(
                app_name=app_name,
                app_mode=app_mode,
                round_name="second-round independent-research discussion",
                pitch=pitch,
                data_summary=_votes_summary(r2_initial),
                vote_summary=_votes_summary(r2_initial),
            )
            r2_discussion = await _collect_votes(
                research_participants,
                r2_discussion_prompt,
                f"{topic_base}-r2-discussion",
                counted=True,
            )
            r2_final = _prefer_parsed_votes(r2_discussion, r2_initial)
        comparison = aggregate_comparison(r2_final or r2_initial)
        final_gate = aggregate_final_gate(r1_final, r2_final or r2_initial, comparison)

    second_round_names = {
        vote["provider"]
        for vote in (r2_final or r2_initial)
        if vote.get("counted")
    }
    participant_rows = [
        participant.to_dict(second_round=participant.name in second_round_names)
        for participant in participants
    ]
    status = _status(participants, min_participants, r1_final, r2_final or r2_initial, final_gate)
    summary = _summary_markdown(app_name, status, participant_rows, r1_final, r2_final or r2_initial, comparison, final_gate)

    return {
        "status": status,
        "participants": participant_rows,
        "rounds": {
            "round1_local_data": {
                "topic": f"{topic_base}-r1",
                "initial_votes": r1_initial,
                "discussion_votes": r1_discussion,
                "final_votes": r1_final,
            },
            "round2_independent_research": {
                "topic": f"{topic_base}-r2",
                "initial_votes": r2_initial,
                "discussion_votes": r2_discussion,
                "final_votes": r2_final or r2_initial,
                "skipped_reason": None if r2_initial else _round2_skip_reason(participants, final_gate),
            },
        },
        "comparison": comparison,
        "final_gate": final_gate if final_gate in FINAL_GATES else "NEEDS_EVIDENCE",
        "summary_markdown": summary,
    }


async def _collect_votes(
    participants: list[Participant],
    prompt: str,
    topic: str,
    counted: bool,
) -> list[dict[str, Any]]:
    tasks = [_ask(participant, prompt, topic) for participant in participants]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    votes = []
    for participant, response in zip(participants, responses):
        if isinstance(response, Exception):
            response_text = f"Error: {response}"
        else:
            response_text = response
        votes.append(
            parse_vote(
                participant.name,
                response_text,
                research_capable=participant.research_capable,
                counted=counted,
            )
        )
    return votes


async def _ask(participant: Participant, prompt: str, topic: str) -> str:
    try:
        return await participant.session.send(prompt, topic, timeout=600)
    except TypeError:
        return await participant.session.send(prompt, topic)


def _prefer_parsed_votes(
    preferred: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_provider = {vote["provider"]: vote for vote in fallback}
    final_votes = []
    for vote in preferred:
        if vote.get("verdict") != "UNKNOWN" and not vote.get("error"):
            final_votes.append(vote)
        else:
            final_votes.append(by_provider.get(vote["provider"], vote))
    return final_votes


def _round1_prompt(app_name: str, app_mode: str, pitch: str, data_brief: str) -> str:
    return f"""You are an Indie App Factory evaluator.

Context:
- App: {app_name}
- Mode: {app_mode}
- Constraints: {MODE_GUIDANCE[app_mode]}
- Goal: $300-$1500/year net. Break-even is small but real. This is not a VC startup.
- PROCEED only means worth continuing to evidence-card review. Never output BUILD_READY.

Evaluate the pitch and the real local tool data. Return ONLY one JSON object:
{{
  "verdict": "PROCEED|PIVOT|KILL",
  "reason": "one-line reason",
  "biggest_concern": "one sentence",
  "field_ratings": {{
    "data_quality": "SOLID|WEAK|FATAL",
    "pain": "SOLID|WEAK|FATAL",
    "repeat_use": "SOLID|WEAK|FATAL",
    "business_model": "SOLID|WEAK|FATAL",
    "break_even": "SOLID|WEAK|FATAL"
  }}
}}

=== PITCH ===
{pitch}
=== END PITCH ===

=== DATA BRIEF v1 - LOCAL TOOLS ===
{data_brief}
=== END DATA BRIEF v1 ===
"""


def _round2_prompt(app_name: str, app_mode: str, pitch: str, keywords: list[str]) -> str:
    keyword_text = ", ".join(keywords) if keywords else "(infer likely App Store keywords from the pitch)"
    return f"""You are an Indie App Factory evaluator with independent research responsibility.

Context:
- App: {app_name}
- Mode: {app_mode}
- Constraints: {MODE_GUIDANCE[app_mode]}
- Goal: $300-$1500/year net. Break-even is small but real. This is not a VC startup.
- Use your native web search/browsing tools before evaluating if your runtime provides them.
- Research keywords: {keyword_text}
- Do not use prior round conclusions. Do not output BUILD_READY.

Research before judging:
1. App Store competitors: top results, ratings, review counts, pricing.
2. Keyword demand: signs of real search intent or weak demand.
3. Pain evidence: Reddit/forums/reviews showing users struggle with this problem.
4. Paid evidence: paid competitors, paid templates, or clear professional willingness to pay.

Then return ONLY one JSON object:
{{
  "verdict": "PROCEED|PIVOT|KILL",
  "reason": "one-line reason",
  "biggest_concern": "one sentence",
  "independent_findings": {{
    "competitors": "what you found",
    "keyword_demand": "what you found",
    "pain_evidence": "what you found",
    "paid_evidence": "what you found"
  }},
  "data_consistency": {{
    "competitors": "CONSISTENT|CONFLICT|UNKNOWN",
    "keyword_demand": "CONSISTENT|CONFLICT|UNKNOWN",
    "pain_evidence": "CONSISTENT|CONFLICT|UNKNOWN"
  }},
  "must_get_right_first_30_days": "one concrete thing"
}}

=== PITCH ONLY ===
{pitch}
=== END PITCH ===
"""


def _discussion_prompt(
    *,
    app_name: str,
    app_mode: str,
    round_name: str,
    pitch: str,
    data_summary: str,
    vote_summary: str,
) -> str:
    return f"""You are in an App Factory roundtable discussion.

App: {app_name}
Mode: {app_mode}
Round: {round_name}
Constraints: {MODE_GUIDANCE[app_mode]}

Review the pitch, data, and other votes. Cast your final vote for this round.
Return ONLY one JSON object:
{{
  "verdict": "PROCEED|PIVOT|KILL",
  "reason": "one-line reason",
  "biggest_concern": "one sentence",
  "data_consistency": {{
    "competitors": "CONSISTENT|CONFLICT|UNKNOWN",
    "keyword_demand": "CONSISTENT|CONFLICT|UNKNOWN",
    "pain_evidence": "CONSISTENT|CONFLICT|UNKNOWN"
  }}
}}

=== PITCH ===
{pitch}
=== END PITCH ===

=== DATA / FINDINGS SUMMARY ===
{data_summary}
=== END DATA / FINDINGS SUMMARY ===

=== VOTE SUMMARY ===
{vote_summary}
=== END VOTE SUMMARY ===
"""


def _votes_summary(votes: list[dict[str, Any]]) -> str:
    if not votes:
        return "(no votes)"
    lines = []
    for vote in votes:
        line = f"- {vote['provider']}: {vote['verdict']} - {vote.get('reason') or '(no reason)'}"
        concern = vote.get("biggest_concern")
        if concern:
            line += f" Concern: {concern}"
        findings = vote.get("independent_findings")
        if findings:
            line += f" Findings: {json.dumps(findings, ensure_ascii=False)[:900]}"
        lines.append(line)
    return "\n".join(lines)


def _summary_markdown(
    app_name: str,
    status: str,
    participants: list[dict[str, Any]],
    round1_votes: list[dict[str, Any]],
    round2_votes: list[dict[str, Any]],
    comparison: dict[str, Any],
    final_gate: str,
) -> str:
    lines = [
        f"### 多模型圆桌评审：{app_name}",
        "",
        f"- 状态：`{status}`",
        f"- 最终闸门：`{final_gate}`",
        f"- 参与模型：{', '.join(p['provider'] for p in participants) or '无'}",
        "",
        "第一轮（本地工具数据）：",
    ]
    lines.extend(_summary_vote_lines(round1_votes))
    lines.append("")
    lines.append("第二轮（模型独立自采）：")
    if round2_votes:
        lines.extend(_summary_vote_lines(round2_votes))
    else:
        lines.append("- 未执行：没有可计入独立自采的模型，或第一轮已 KILL。")
    lines.append("")
    lines.append("数据一致性：")
    for metric, value in comparison.get("metrics", {}).items():
        lines.append(f"- {metric}: `{value}`")
    lines.append("")
    lines.append(_gate_explanation(final_gate))
    return "\n".join(lines)


def _summary_vote_lines(votes: list[dict[str, Any]]) -> list[str]:
    if not votes:
        return ["- 无可用投票。"]
    return [
        f"- {vote['provider']}: `{vote['verdict']}` - {vote.get('reason') or '未给出原因'}"
        for vote in votes
    ]


def _gate_explanation(final_gate: str) -> str:
    if final_gate == "PROCEED_TO_EVIDENCE_CARD":
        return "结论：可以进入证据卡与最终风控；这不是 BUILD_READY。"
    if final_gate == "KILL":
        return "结论：圆桌出现明确 KILL，应立即换题。"
    if final_gate == "SKIPPED":
        return "结论：多模型评审未执行，按 skill 降级规则继续。"
    return "结论：证据或一致性不足，需要补证据、调整假设或重跑评审。"


def _status(
    participants: list[Participant],
    min_participants: int,
    round1_votes: list[dict[str, Any]],
    round2_votes: list[dict[str, Any]],
    final_gate: str,
) -> str:
    if not participants:
        return "skipped"
    if len(participants) < min_participants:
        return "partial"
    if final_gate != "KILL" and not round2_votes:
        return "partial"
    all_votes = round1_votes + round2_votes
    if any(vote.get("error") or vote.get("verdict") == "UNKNOWN" for vote in all_votes):
        return "partial"
    return "completed"


def _round2_skip_reason(participants: list[Participant], final_gate: str) -> str:
    if final_gate == "KILL":
        return "First round produced KILL, so independent research was not needed."
    if not any(participant.research_capable for participant in participants):
        return "No selected provider has CLI/native web search or declared web_search capability."
    return "Second round did not produce counted votes."


def _skipped_result(app_name: str, reason: str) -> dict[str, Any]:
    summary = (
        f"### 多模型圆桌评审：{app_name}\n\n"
        "- 状态：`skipped`\n"
        "- 最终闸门：`SKIPPED`\n"
        f"- 原因：{reason}\n\n"
        "结论：按 skill 降级规则继续，不阻塞主流程。"
    )
    return {
        "status": "skipped",
        "participants": [],
        "rounds": {
            "round1_local_data": {"initial_votes": [], "discussion_votes": [], "final_votes": []},
            "round2_independent_research": {"initial_votes": [], "discussion_votes": [], "final_votes": [], "skipped_reason": reason},
        },
        "comparison": aggregate_comparison([]),
        "final_gate": "SKIPPED",
        "summary_markdown": summary,
    }


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:700]
    return ""


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return slug or "untitled"
