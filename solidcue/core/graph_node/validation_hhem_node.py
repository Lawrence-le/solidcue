from __future__ import annotations

import os
import re
import time
from typing import Any

from solidcue.agents.configs.loader import load_agent
from solidcue.core.execution.provider_resolver import get_provider_for_role
from solidcue.services.hhem_service import get_hhem_model, HHEM_MODEL_ID
from solidcue.core.state.schema import AgentState
from solidcue.core.utils.metrics import build_metric, build_metric_state_delta, timed_generate
from solidcue.prompts.validation_hhem_prompt import build_hhem_verify_messages

"""
Validation HHEM Node - Function Overview
----------------------------------------

_positive_int_env:
Read positive integer env override with safe defaulting.

_split_claims:
Split hypothesis text into bounded claim list for scoring.

_chunk_premise:
Split long premise into bounded chunks for retrieval/scoring.

_tokens:
Tokenize text for lexical-overlap chunk ranking.

_select_relevant_chunks:
Pick top relevant premise chunks for a specific claim.

_build_verifier_premise:
Assemble compact verifier premise from failed claims and selected chunks.

_score_groundedness:
Run HHEM claim-level groundedness scoring and aggregate metrics.

_build_premise:
Construct item-scoped validation premise from handoff payloads.

_llm_verify_failures:
Optional reviewer-model verification for uncertain/failed claims.

validation_hhem_node:
Main entrypoint. Phases:
1) Build premise + claims
2) Score groundedness with HHEM
3) Optionally verify failures with reviewer LLM
4) Emit pass/fail validation report + metrics
"""

SCORE_THRESHOLD = 0.5

PREMISE_CHUNK_CHARS = 1200


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


MAX_CLAIMS = _positive_int_env("SOLIDCUE_HHEM_MAX_CLAIMS", 40)
MAX_CLAIM_CHARS = _positive_int_env("SOLIDCUE_HHEM_MAX_CLAIM_CHARS", 500)
MAX_LLM_PREMISE_CHUNKS_PER_CLAIM = _positive_int_env("SOLIDCUE_HHEM_MAX_CHUNKS", 8)
MAX_LLM_VERIFY_CLAIMS = _positive_int_env("SOLIDCUE_HHEM_MAX_LLM_VERIFY_CLAIMS", 8)
HHEM_BATCH_SIZE = _positive_int_env("SOLIDCUE_HHEM_BATCH_SIZE", 128)
LLM_VERIFY_PREMISE_CHARS = _positive_int_env("SOLIDCUE_HHEM_LLM_PREMISE_CHARS", 6000)
LLM_VERIFY_MAX_TOKENS = _positive_int_env("SOLIDCUE_HHEM_LLM_VERIFY_MAX_TOKENS", 300)
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.\-]{1,}")
_PREMISE_MAX_CHARS = 12000
_PREMISE_ENTRY_MAX_CHARS = 3000
_TEXT_FIELD_CANDIDATES = ("content", "text", "body", "markdown")


def _split_claims(text: str) -> list[str]:
    lines = re.split(r"\n+", text)
    claims = []
    for line in lines:
        raw_line = line.strip().lstrip("•-–—*#").strip()
        if not raw_line or len(raw_line) <= 10:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", raw_line)
        claims.extend(s.strip() for s in sentences if len(s.strip()) > 10)
    return claims[:MAX_CLAIMS]


def _chunk_premise(premise: str) -> list[str]:
    if len(premise) <= PREMISE_CHUNK_CHARS:
        return [premise]
    chunks = []
    paragraphs = re.split(r"\n{2,}", premise)
    current = ""
    for para in paragraphs:
        if len(para) > PREMISE_CHUNK_CHARS:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), PREMISE_CHUNK_CHARS):
                chunk = para[i : i + PREMISE_CHUNK_CHARS].strip()
                if chunk:
                    chunks.append(chunk)
            continue
        if len(current) + len(para) + 2 > PREMISE_CHUNK_CHARS:
            if current:
                chunks.append(current)
            current = para
        else:
            current = current + "\n\n" + para if current else para
    if current:
        chunks.append(current)
    return chunks or [premise[:PREMISE_CHUNK_CHARS]]


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(text or "")}


def _select_relevant_chunks(claim: str, premise_chunks: list[str]) -> list[str]:
    if len(premise_chunks) <= MAX_LLM_PREMISE_CHUNKS_PER_CLAIM:
        return premise_chunks

    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return premise_chunks[:MAX_LLM_PREMISE_CHUNKS_PER_CLAIM]

    scored_chunks = []
    for idx, chunk in enumerate(premise_chunks):
        overlap = len(claim_tokens & _tokens(chunk))
        scored_chunks.append((overlap, idx, chunk))

    selected = sorted(scored_chunks, key=lambda item: (-item[0], item[1]))[:MAX_LLM_PREMISE_CHUNKS_PER_CLAIM]
    return [chunk for _, _, chunk in sorted(selected, key=lambda item: item[1])]


def _build_verifier_premise(premise: str, failed_claims: list[dict[str, Any]]) -> str:
    chunks = _chunk_premise(premise)
    selected: list[str] = []
    seen: set[str] = set()
    for failure in failed_claims[:MAX_LLM_VERIFY_CLAIMS]:
        claim = str(failure.get("claim") or "")
        for chunk in _select_relevant_chunks(claim, chunks):
            if chunk in seen:
                continue
            selected.append(chunk)
            seen.add(chunk)
    relevant_premise = "\n\n".join(selected) if selected else premise
    return relevant_premise[:LLM_VERIFY_PREMISE_CHARS]


def _score_groundedness(premise: str, hypothesis: str) -> tuple[float, list[dict[str, Any]], dict[str, int]]:
    model = get_hhem_model()
    claims = _split_claims(hypothesis)

    if not claims:
        chunks = _chunk_premise(premise)
        scores = model.predict([[chunks[0], hypothesis[:500]]])
        return float(scores[0]), [], {"hhem_chunk_count": len(chunks), "hhem_pair_count": 1}

    premise_chunks = _chunk_premise(premise)

    pairs: list[list[str]] = []
    pair_claim_indexes: list[int] = []
    for claim_idx, claim in enumerate(claims):
        trimmed_claim = claim[:MAX_CLAIM_CHARS]
        for chunk in premise_chunks:
            pairs.append([chunk, trimmed_claim])
            pair_claim_indexes.append(claim_idx)

    all_scores: list[float] = []
    for i in range(0, len(pairs), HHEM_BATCH_SIZE):
        batch_scores = model.predict(pairs[i : i + HHEM_BATCH_SIZE])
        all_scores.extend(float(s) for s in batch_scores)

    best_scores_by_claim = [0.0] * len(claims)
    for score, claim_idx in zip(all_scores, pair_claim_indexes):
        if score > best_scores_by_claim[claim_idx]:
            best_scores_by_claim[claim_idx] = score

    claim_scores = [
        {"claim": claim, "score": best_scores_by_claim[claim_idx]}
        for claim_idx, claim in enumerate(claims)
    ]

    min_score = min(c["score"] for c in claim_scores)
    return min_score, claim_scores, {
        "hhem_chunk_count": len(premise_chunks),
        "hhem_pair_count": len(pairs),
        "hhem_chunks_per_claim": len(premise_chunks),
    }


def _build_premise(state: AgentState) -> str:
    def current_task_item_key() -> str | None:
        task_plan = state.get("task_plan")
        current_task_id = state.get("current_task")
        if not isinstance(task_plan, list) or not current_task_id:
            return None
        current_task = next((t for t in task_plan if isinstance(t, dict) and t.get("id") == current_task_id), None)
        if not isinstance(current_task, dict):
            return None
        context = current_task.get("context")
        if not isinstance(context, dict):
            return None
        item_key = context.get("item_key")
        if not isinstance(item_key, str):
            return None
        cleaned = item_key.strip()
        return cleaned or None

    def handoff_for_item(handoff: dict[str, Any], item_key: str | None) -> dict[str, Any]:
        if not item_key:
            return handoff
        suffix = f"::{item_key}"
        scoped: dict[str, Any] = {}
        for key, value in handoff.items():
            if isinstance(key, str) and key.endswith(suffix):
                scoped[key[: -len(suffix)]] = value
        return scoped or handoff

    def extract_text(item: Any) -> str:
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            for key in _TEXT_FIELD_CANDIDATES:
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for nested in item.values():
                text = extract_text(nested)
                if text:
                    return text
        if isinstance(item, list):
            for nested in item:
                text = extract_text(nested)
                if text:
                    return text
        return ""

    handoff = state.get("handoff")
    if isinstance(handoff, dict) and handoff:
        item_key = current_task_item_key()
        handoff_view = handoff_for_item(handoff, item_key)
        collected: list[str] = []
        total_chars = 0
        for source_key, value in handoff_view.items():
            if not isinstance(source_key, str):
                continue
            if "base64" in source_key.casefold():
                continue
            text = extract_text(value)
            if not text:
                continue
            if len(text) > _PREMISE_ENTRY_MAX_CHARS:
                text = text[:_PREMISE_ENTRY_MAX_CHARS].rstrip() + "… [truncated]"
            remaining = _PREMISE_MAX_CHARS - total_chars
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining].rstrip() + "… [truncated]"
            collected.append(text)
            total_chars += len(text)
        if collected:
            return "\n\n".join(collected)

    return ""


def _llm_verify_failures(
    state: AgentState, failed_claims: list[dict[str, Any]], premise: str
) -> tuple[list[str], str, dict[str, Any]]:
    import json

    agent_key = state.get("agent_key")
    if not isinstance(agent_key, str) or not agent_key:
        return [c["claim"] for c in failed_claims], "No agent configured for LLM verification.", {}

    try:
        agent = load_agent(agent_key)
        provider = get_provider_for_role(agent, "reviewer")
    except Exception:
        return [c["claim"] for c in failed_claims], "Could not load reviewer provider.", {}

    claims_to_verify = failed_claims[:MAX_LLM_VERIFY_CLAIMS]
    claims_text = "\n".join(f'- "{c["claim"]}" (score: {c["score"]:.2f})' for c in claims_to_verify)
    messages = build_hhem_verify_messages(claims_text)

    raw_output, metric = timed_generate(provider, messages, max_tokens=LLM_VERIFY_MAX_TOKENS)
    raw = str(raw_output or "").strip()

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            real_failures = parsed.get("real_failures", [])
            reason = parsed.get("reason", "")
            if isinstance(real_failures, list):
                return real_failures, reason, metric
    except (json.JSONDecodeError, AttributeError):
        pass

    return [c["claim"] for c in failed_claims], "LLM verification failed to parse.", metric


def validation_hhem_node(state: AgentState) -> dict[str, Any]:
    """
    Scores draft groundedness using a heavily optimized short-circuiting local HHEM matrix,
    falling back to an LLM verification layer for anomalies.
    """
    draft_output = state.get("synthesis_draft")

    if not isinstance(draft_output, str) or not draft_output.strip():
        return {
            "failure_type": "bad_synthesis",
            "validation_report": {"reason": "Draft output is empty.", "score": 0.0},
            **build_metric_state_delta("validation_hhem", "metric_validation_hhem", {}),
        }

    premise = _build_premise(state)
    if not premise.strip():
        return {
            "failure_type": None,
            "validation_report": {"reason": "No context to check groundedness against; skipping.", "score": 1.0},
            **build_metric_state_delta("validation_hhem", "metric_validation_hhem", {}),
        }

    hhem_start = time.perf_counter()
    min_score, claim_scores, hhem_stats = _score_groundedness(premise, draft_output.strip())
    hhem_elapsed = time.perf_counter() - hhem_start

    failed_claims = [c for c in claim_scores if c["score"] < SCORE_THRESHOLD]

    if not failed_claims:
        metric = build_metric(
            {
                "estimated_total": 0,
                "message_count": len(claim_scores),
                "hhem_claims_scored": len(claim_scores),
                "hhem_claims_failed": 0,
                **hhem_stats,
                "hhem_time_s": hhem_elapsed,
                "method": "optimized_hhem_inference",
            },
            hhem_elapsed,
            HHEM_MODEL_ID,
        )
        return {
            "failure_type": None,
            "validation_report": {"reason": f"All claims grounded (min score {min_score:.2f}).", "score": min_score},
            **build_metric_state_delta("validation_hhem", "metric_validation_hhem", metric),
        }

    real_failures, llm_reason, llm_metric = _llm_verify_failures(state, failed_claims, premise)
    llm_time = float(llm_metric.get("time_s") or 0.0) if isinstance(llm_metric, dict) else 0.0
    llm_tokens = llm_metric.get("tokens") if isinstance(llm_metric, dict) else {}
    total_time = hhem_elapsed + llm_time

    metric = build_metric(
        {
            "prompt_tokens": int((llm_tokens or {}).get("prompt_tokens") or 0),
            "completion_tokens": int((llm_tokens or {}).get("completion_tokens") or 0),
            "total_tokens": int((llm_tokens or {}).get("total_tokens") or 0),
            "cached_tokens": int((llm_tokens or {}).get("cached_tokens") or 0),
            "estimated_total": int((llm_tokens or {}).get("estimated_total") or 0),
            "estimated_system": int((llm_tokens or {}).get("estimated_system") or 0),
            "estimated_user": int((llm_tokens or {}).get("estimated_user") or 0),
            "estimated_assistant": int((llm_tokens or {}).get("estimated_assistant") or 0),
            "estimated_tool": 0,
            "estimated_other": 0,
            "message_count": int((llm_tokens or {}).get("message_count") or 0),
            "llm_call_count": 1,
            "hhem_claims_scored": len(claim_scores),
            "hhem_claims_failed": len(failed_claims),
            **hhem_stats,
            "hhem_time_s": hhem_elapsed,
            "method": "optimized_hhem_inference_plus_llm_verify",
        },
        total_time,
        str(llm_metric.get("model") or HHEM_MODEL_ID) if isinstance(llm_metric, dict) else HHEM_MODEL_ID,
    )

    if not real_failures:
        return {
            "failure_type": None,
            "validation_report": {
                "reason": f"HHEM flagged {len(failed_claims)} claims but LLM confirmed all are metadata/formatting.",
                "score": min_score,
            },
            **build_metric_state_delta("validation_hhem", "metric_validation_hhem", metric),
        }

    failed_reasons = "; ".join(f'"{c[:60]}"' for c in real_failures[:3])
    return {
        "failure_type": "bad_synthesis",
        "validation_report": {
            "reason": f"Ungrounded claims confirmed by LLM: {failed_reasons}. {llm_reason}",
            "score": min_score,
        },
        **build_metric_state_delta("validation_hhem", "metric_validation_hhem", metric),
    }
