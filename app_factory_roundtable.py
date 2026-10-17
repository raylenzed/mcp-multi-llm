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
FINAL_GATES = {"PROCEED_TO_EVIDENCE_CARD", "NEEDS_EVIDENCE", "PIVOT", "KILL", "SKIPPED"}
CONSISTENCY_VALUES = {"CONSISTENT", "CONFLICT", "UNKNOWN"}

# 否决器语义：最终闸门只由分类质疑（challenges）决定，赞成票仅存档。
# 与 app-factory Challenge Protocol 同一套分类法。
CHALLENGE_TYPES = ("HARD_DISQUALIFIER", "PIVOTABLE_WEAKNESS", "EVIDENCE_GAP", "SOFT_NEGATIVE")
_CHALLENGE_ALIASES = {
    "HARD_DISQUALIFIER": "HARD_DISQUALIFIER",
    "HARD": "HARD_DISQUALIFIER",
    "FATAL": "HARD_DISQUALIFIER",
    "DISQUALIFIER": "HARD_DISQUALIFIER",
    "PIVOTABLE_WEAKNESS": "PIVOTABLE_WEAKNESS",
    "PIVOTABLE": "PIVOTABLE_WEAKNESS",
    "PIVOT": "PIVOTABLE_WEAKNESS",
    "WEAKNESS": "PIVOTABLE_WEAKNESS",
    "EVIDENCE_GAP": "EVIDENCE_GAP",
    "GAP": "EVIDENCE_GAP",
    "MISSING_EVIDENCE": "EVIDENCE_GAP",
    "NEEDS_EVIDENCE": "EVIDENCE_GAP",
    "SOFT_NEGATIVE": "SOFT_NEGATIVE",
    "SOFT": "SOFT_NEGATIVE",
    "CONCERN": "SOFT_NEGATIVE",
}

CALIBRATION_CONTEXT = """Account calibration facts (real shipped-app outcomes; treat as priors):
- Historical prediction optimism: median 7.35x across 8 shipped apps (predicted vs actual sales).
- Unanimous panel approval predicted nothing: HiveLog passed 5/5 yet sold 4.7x below forecast; ReefLog passed 4/4, still 3.5x below. Your approval adds ZERO information. Only specific, falsifiable flaws you find add information.
- Winners: serious-hobby logbooks with niche jargon keywords at $4.99+ (HiveLog, ReefLog). Zero-rating apps reached App Store Top 3 purely via exact title match -- ranking is NOT the bottleneck; keyword search volume is the revenue ceiling.
- Losers: generic calculators/counters/timers on red-ocean keywords (TileCalc, ProofTime, KnitCount: 0-1 lifetime sales). These categories carry a 10x risk multiplier.

Hard operating constraints (a plan that violates one is a HARD_DISQUALIFIER):
- Distribution is App Store organic search ONLY. The operator cannot post on Reddit/forums/social media (posts get removed instantly). Any plan that depends on marketing posts is invalid.
- Zero probe budget: no paid ads, no paid validation per topic. Validation = free tools + shipping itself.
- Default monetization: free download + one-time IAP unlock ($3.99-$6.99), or paid upfront $2.99-$4.99. Apple net rate 0.85 (Small Business Program)."""

CHALLENGE_SCHEMA_GUIDE = """Challenge types (use exactly these):
- HARD_DISQUALIFIER: independently checkable fatal fact (Google answers the core need in 3 seconds; a free top app fully covers the core workflow; no exact search phrase a buyer would type; App Review policy blocks the category; backend cost cannot break even; plan requires marketing posts). MUST include cheapest_verification or it will be downgraded.
- PIVOTABLE_WEAKNESS: real flaw fixable by narrowing the user, scene, or keyword.
- EVIDENCE_GAP: a load-bearing claim in the pitch/data with no evidence behind it.
- SOFT_NEGATIVE: taste / market-feel doubts (archived, never blocks)."""

MODE_GUIDANCE = {
    "local": (
        "Pure local/offline iOS app. No server, no API calls. "
        "Default: free download + one-time IAP unlock $3.99-$6.99 (or paid upfront $2.99-$4.99). "
        "China region may be allowed only if category is safe."
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
    challenges: list[dict[str, Any]] = []

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
        raw_challenges = parsed.get("challenges")
        if isinstance(raw_challenges, list):
            for item in raw_challenges:
                normalized = normalize_challenge(item)
                if normalized:
                    challenges.append(normalized)

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

    # 旧格式兜底：KILL 意图不丢失，但无分类硬伤时只折算为"补证据"质疑，
    # 不允许凭一句话处决（与 skill 层规则一致，现在由 MCP 本身强制）。
    if not challenges and verdict == "KILL":
        basis = concern or reason
        if basis:
            challenges.append(
                {"type": "EVIDENCE_GAP", "claim": basis[:500], "cheapest_verification": ""}
            )
    elif not challenges and concern:
        challenges.append(
            {"type": "SOFT_NEGATIVE", "claim": concern[:500], "cheapest_verification": ""}
        )

    error = None
    stripped = response.strip()
    if stripped.startswith("Error:") or " unavailable " in stripped[:160]:
        error = stripped[:500]

    return {
        "provider": provider,
        "verdict": verdict,
        "reason": reason[:700],
        "biggest_concern": concern[:700],
        "challenges": challenges,
        "research_capable": research_capable,
        "counted": counted,
        "data_consistency": data_consistency,
        "independent_findings": independent_findings,
        "parsed_json": parsed,
        "raw_response": response,
        "error": error,
    }


def normalize_challenge(raw: Any) -> dict[str, Any] | None:
    """把模型输出的质疑统一成 {type, claim, cheapest_verification}。

    护栏：HARD_DISQUALIFIER 必须带 cheapest_verification，否则降级为 EVIDENCE_GAP——
    不可验证的"硬伤"没有处决权。
    """
    if isinstance(raw, str):
        claim = raw.strip()
        if not claim:
            return None
        return {"type": "SOFT_NEGATIVE", "claim": claim[:500], "cheapest_verification": ""}
    if not isinstance(raw, dict):
        return None
    claim = str(raw.get("claim") or raw.get("description") or raw.get("issue") or "").strip()
    if not claim:
        return None
    raw_type = str(raw.get("type") or raw.get("category") or "").strip().upper().replace(" ", "_")
    challenge_type = _CHALLENGE_ALIASES.get(raw_type, "SOFT_NEGATIVE")
    verification = str(
        raw.get("cheapest_verification") or raw.get("verification") or raw.get("how_to_verify") or ""
    ).strip()
    if challenge_type == "HARD_DISQUALIFIER" and not verification:
        challenge_type = "EVIDENCE_GAP"
    return {
        "type": challenge_type,
        "claim": claim[:500],
        "cheapest_verification": verification[:300],
    }


def _claim_tokens(text: str) -> set[str]:
    """英文/数字按词切分；中文没有空格，按相邻两字（bigram）切分，否则整句成一个 token、去重失效。"""
    tokens: set[str] = set()
    for run in re.findall(r"[a-z0-9]+|[一-鿿]+", text.lower()):
        if run[0] < "一" or len(run) == 1:
            tokens.add(run)
        else:
            tokens.update(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


def dedupe_challenges(votes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """跨模型合并同义质疑：5 个模型说同一句话 = 1 条质疑（相关评委去相关）。"""
    type_order = {name: index for index, name in enumerate(CHALLENGE_TYPES)}
    ledger: list[dict[str, Any]] = []
    for vote in votes:
        if not vote.get("counted"):
            continue
        for challenge in vote.get("challenges", []):
            tokens = _claim_tokens(challenge["claim"])
            merged = False
            for entry in ledger:
                if entry["type"] != challenge["type"]:
                    continue
                entry_tokens = _claim_tokens(entry["claim"])
                union = tokens | entry_tokens
                if union and len(tokens & entry_tokens) / len(union) >= 0.55:
                    if vote["provider"] not in entry["sources"]:
                        entry["sources"].append(vote["provider"])
                    if not entry["cheapest_verification"] and challenge["cheapest_verification"]:
                        entry["cheapest_verification"] = challenge["cheapest_verification"]
                    merged = True
                    break
            if not merged:
                ledger.append({**challenge, "sources": [vote["provider"]]})
    ledger.sort(key=lambda entry: (type_order.get(entry["type"], 9), -len(entry["sources"])))
    return ledger


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
    challenge_ledger: list[dict[str, Any]] | None = None,
) -> str:
    """否决器语义：闸门只由分类质疑决定，PROCEED 票数不计入任何信号。

    校准依据：全票通过从未预测成功（HiveLog 5/5 PASS 仍乐观 4.7×），
    但单条可验证硬伤是真信号——共识无信息量，缺陷有。
    """
    counted_r1 = [vote for vote in round1_votes if vote.get("counted")]
    counted_r2 = [vote for vote in round2_votes if vote.get("counted")]
    if not counted_r1:
        return "SKIPPED"

    ledger = (
        challenge_ledger
        if challenge_ledger is not None
        else dedupe_challenges(counted_r1 + counted_r2)
    )
    types = {entry["type"] for entry in ledger}

    if "HARD_DISQUALIFIER" in types:
        return "KILL"
    if not counted_r2:
        return "NEEDS_EVIDENCE"
    if comparison and comparison.get("has_conflict"):
        return "NEEDS_EVIDENCE"
    if "PIVOTABLE_WEAKNESS" in types:
        return "PIVOT"
    if "EVIDENCE_GAP" in types:
        return "NEEDS_EVIDENCE"
    # 反方未找到硬伤；SOFT_NEGATIVE 只存档不拦路。这不构成需求证据。
    return "PROCEED_TO_EVIDENCE_CARD"


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

    r1_hard = any(
        challenge.get("type") == "HARD_DISQUALIFIER"
        for vote in r1_final
        if vote.get("counted")
        for challenge in vote.get("challenges", [])
    )
    if r1_hard:
        challenge_ledger = dedupe_challenges(r1_final)
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
        challenge_ledger = dedupe_challenges(r1_final + (r2_final or r2_initial))
        final_gate = aggregate_final_gate(
            r1_final, r2_final or r2_initial, comparison, challenge_ledger
        )

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
    summary = _summary_markdown(
        app_name,
        status,
        participant_rows,
        r1_final,
        r2_final or r2_initial,
        comparison,
        final_gate,
        challenge_ledger,
    )

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
        "challenge_ledger": challenge_ledger,
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
    return f"""You are an adversarial reviewer in an Indie App Factory red-team. Your job is to find specific, falsifiable flaws -- NOT to judge whether the idea "sounds good". Approval carries zero weight in the gate; only classified challenges do.

Context:
- App: {app_name}
- Mode: {app_mode}
- Constraints: {MODE_GUIDANCE[app_mode]}
- Goal: $300-$1500/year net. Break-even is small but real. This is not a VC startup.
- PROCEED only means "I could not find a disqualifying flaw". Never output BUILD_READY.

{CALIBRATION_CONTEXT}

{CHALLENGE_SCHEMA_GUIDE}

Falsification checklist -- attempt every line against the pitch and the data:
1. Google-3-seconds: does a plain search fully answer the core need?
2. Free coverage: does a free top competitor already cover the core workflow?
3. Search entry: is there an exact phrase a buyer types? Is its Top-10 beatable (weak titles / <100 ratings / stale apps)?
4. Repeat use: what does the user concretely lose in week 2 if they stop using it?
5. Demand math: divide the pitch's estimate by the 7.35x optimism prior (10x for calculator/counter/timer) -- does it still break even?
6. Compliance / App Review category risk. For server modes: is the backend truly necessary and affordable?

Return ONLY one JSON object:
{{
  "verdict": "PROCEED|PIVOT|KILL",
  "reason": "one-line reason (archived only, not a gate signal)",
  "biggest_concern": "one sentence",
  "challenges": [
    {{
      "type": "HARD_DISQUALIFIER|PIVOTABLE_WEAKNESS|EVIDENCE_GAP|SOFT_NEGATIVE",
      "claim": "specific falsifiable statement -- name the exact keyword/competitor/policy",
      "cheapest_verification": "one $0 action that proves or kills this claim"
    }}
  ],
  "field_ratings": {{
    "data_quality": "SOLID|WEAK|FATAL",
    "pain": "SOLID|WEAK|FATAL",
    "repeat_use": "SOLID|WEAK|FATAL",
    "business_model": "SOLID|WEAK|FATAL",
    "break_even": "SOLID|WEAK|FATAL"
  }}
}}

An empty challenges list asserts the topic survived every checklist line. Vague doubts ("competitive market", "demand unclear") belong in SOFT_NEGATIVE or EVIDENCE_GAP, never in HARD_DISQUALIFIER.

=== PITCH ===
{pitch}
=== END PITCH ===

=== DATA BRIEF v1 - LOCAL TOOLS ===
{data_brief}
=== END DATA BRIEF v1 ===
"""


def _round2_prompt(app_name: str, app_mode: str, pitch: str, keywords: list[str]) -> str:
    keyword_text = ", ".join(keywords) if keywords else "(infer likely App Store keywords from the pitch)"
    return f"""You are an adversarial reviewer with independent research responsibility in an Indie App Factory red-team. Verify with your own tools; your approval carries zero weight -- only classified, falsifiable challenges do.

Context:
- App: {app_name}
- Mode: {app_mode}
- Constraints: {MODE_GUIDANCE[app_mode]}
- Goal: $300-$1500/year net. Break-even is small but real. This is not a VC startup.
- Use your native web search/browsing tools before judging if your runtime provides them.
- Research keywords: {keyword_text}
- Do not rely on prior round conclusions. Never output BUILD_READY.

{CALIBRATION_CONTEXT}

{CHALLENGE_SCHEMA_GUIDE}

Research to attempt before judging (each line is a chance to find a challenge):
1. App Store competitors for the keywords: titles, rating counts, pricing, staleness. Does a free app already own the exact phrase?
2. Keyword demand: any sign buyers actually type these phrases? If the only strong phrases are red-ocean, that is a challenge.
3. Pain evidence: forums/reviews showing real workarounds -- or their absence.
4. Paid evidence: paid competitors, paid templates, professional willingness to pay.

Then return ONLY one JSON object:
{{
  "verdict": "PROCEED|PIVOT|KILL",
  "reason": "one-line reason (archived only, not a gate signal)",
  "biggest_concern": "one sentence",
  "challenges": [
    {{
      "type": "HARD_DISQUALIFIER|PIVOTABLE_WEAKNESS|EVIDENCE_GAP|SOFT_NEGATIVE",
      "claim": "specific falsifiable statement backed by what you found",
      "cheapest_verification": "one $0 action that proves or kills this claim"
    }}
  ],
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
    return f"""You are in an App Factory red-team roundtable discussion.

App: {app_name}
Mode: {app_mode}
Round: {round_name}
Constraints: {MODE_GUIDANCE[app_mode]}

Review the pitch, data, and the other reviewers' challenges. Your job is NOT to converge with the group -- it is to stress-test the surviving claims:
- Escalate a challenge (with a concrete cheapest_verification) if others under-rated it.
- Withdraw or downgrade your own challenge if the data refutes it.
- Add any flaw everyone missed. Duplicate challenges are merged across reviewers, so repeating others adds nothing.

{CHALLENGE_SCHEMA_GUIDE}

Return ONLY one JSON object:
{{
  "verdict": "PROCEED|PIVOT|KILL",
  "reason": "one-line reason (archived only, not a gate signal)",
  "biggest_concern": "one sentence",
  "challenges": [
    {{
      "type": "HARD_DISQUALIFIER|PIVOTABLE_WEAKNESS|EVIDENCE_GAP|SOFT_NEGATIVE",
      "claim": "specific falsifiable statement",
      "cheapest_verification": "one $0 action that proves or kills this claim"
    }}
  ],
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

=== CHALLENGES / VOTES SO FAR ===
{vote_summary}
=== END CHALLENGES / VOTES SO FAR ===
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
        for challenge in (vote.get("challenges") or [])[:4]:
            line += (
                f"\n  * [{challenge['type']}] {challenge['claim']}"
                + (f" (verify: {challenge['cheapest_verification']})" if challenge.get("cheapest_verification") else "")
            )
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
    challenge_ledger: list[dict[str, Any]] | None = None,
) -> str:
    lines = [
        f"### 多模型圆桌评审（否决器模式）：{app_name}",
        "",
        f"- 状态：`{status}`",
        f"- 最终闸门：`{final_gate}`",
        f"- 参与模型：{', '.join(p['provider'] for p in participants) or '无'}",
        "",
        "质疑账本（跨模型去重；唯一的闸门信号）：",
    ]
    if challenge_ledger:
        for entry in challenge_ledger:
            line = f"- [{entry['type']}] {entry['claim']}（来源：{', '.join(entry['sources'])}）"
            if entry.get("cheapest_verification"):
                line += f"｜验证：{entry['cheapest_verification']}"
            lines.append(line)
    else:
        lines.append("- 反方未提出可分类的质疑。")
    lines.append("")
    lines.append("第一轮投票（仅存档，不构成证据——共识无信息量）：")
    lines.extend(_summary_vote_lines(round1_votes))
    lines.append("")
    lines.append("第二轮投票（仅存档，不构成证据）：")
    if round2_votes:
        lines.extend(_summary_vote_lines(round2_votes))
    else:
        lines.append("- 未执行：没有可计入独立自采的模型，或第一轮已出现硬伤。")
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
        return (
            "结论：反方未找到硬伤 → 可进入证据卡与最终风控。"
            "注意：这不构成需求证据，更不是 BUILD_READY；需求只由证据包 + 校准账本判定。"
        )
    if final_gate == "PIVOT":
        return "结论：存在可修复弱点（PIVOTABLE_WEAKNESS），限一次缩窄人群/场景/关键词后重评。"
    if final_gate == "KILL":
        return (
            "结论：出现可验证的 HARD_DISQUALIFIER。"
            "先执行质疑账本中的 cheapest_verification 复核，确认后立即换题、回候选池递补。"
        )
    if final_gate == "SKIPPED":
        return "结论：多模型评审未执行，按 skill 降级规则继续。"
    return "结论：存在 EVIDENCE_GAP 或数据冲突，限一次补证据后重评；仍弱则 KILL_SOFT。"


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
        return "First round produced a verifiable HARD_DISQUALIFIER, so independent research was not needed."
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
        "challenge_ledger": [],
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
