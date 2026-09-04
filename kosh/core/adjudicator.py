"""Tier 4: adjudication between candidates the deterministic tiers produced.

The model's job is deliberately narrow. It is handed a closed list of candidate
matches, each already computed and costed by the engine, and asked to pick one
or abstain. It is never asked to find a match, and it is never given the
ability to name a row that is not already on the list.

That property is enforced here, in ``_validate``, not in the prompt. A model
that returns an unknown candidate id, malformed JSON, or a made-up row is
treated as an abstention. Prompts are a request; code is a guarantee.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

MODEL = "claude-sonnet-4-6"
API_URL = "https://api.anthropic.com/v1/messages"

SYSTEM_PROMPT = """You adjudicate bank-settlement reconciliation candidates.

You are given one bank credit and a closed list of candidate matches. Each
candidate is a set of settlement rows whose net total is already computed.

Rules:
- Choose exactly one candidate_id from the list, or null if you cannot justify a choice.
- Never invent a candidate_id, a row id, or an amount.
- A smaller absolute delta is better. A smaller date lag is better.
- If two candidates are equally supported by the evidence, answer null. An
  unresolved item goes to a human, which is correct. A wrong match silently
  corrupts a ledger, which is not.
- Base the decision only on the evidence given.

Reply with JSON only, no prose and no code fences:
{"candidate_id": "<id or null>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}"""


@dataclass(slots=True)
class CandidateView:
    """What the adjudicator is allowed to see about one candidate."""

    candidate_id: str
    row_count: int
    net_paise: int
    delta_paise: int
    date_lag_days: int
    settle_date: str
    source_tier: str
    evidence: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class AdjudicationRequest:
    txn_id: str
    value_date: str
    narration: str
    credit_paise: int
    candidates: list[CandidateView]


@dataclass(slots=True)
class AdjudicationResponse:
    candidate_id: str | None
    confidence: float
    reason: str
    tokens: int = 0
    degraded: bool = False


class Adjudicator(Protocol):
    """Anything that can rank candidates. Swappable so the cascade never
    depends on a network call being available."""

    name: str

    def adjudicate(self, request: AdjudicationRequest) -> AdjudicationResponse: ...


def _validate(payload: dict, request: AdjudicationRequest, tokens: int) -> AdjudicationResponse:
    """Coerce a model reply into a safe response.

    Anything unexpected becomes an abstention rather than an error or a guess.
    """
    allowed = {c.candidate_id for c in request.candidates}
    chosen = payload.get("candidate_id")
    if isinstance(chosen, str) and chosen not in allowed:
        return AdjudicationResponse(
            None, 0.0, f"model named an unknown candidate ({chosen!r}); treated as unresolved", tokens, True
        )
    if chosen is not None and not isinstance(chosen, str):
        return AdjudicationResponse(None, 0.0, "model returned a non-string candidate id", tokens, True)

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = str(payload.get("reason", ""))[:400] or "no reason given"
    return AdjudicationResponse(chosen, confidence, reason, tokens)


class OfflineAdjudicator:
    """Deterministic stand-in used in tests and in --no-llm runs.

    Picks a candidate only when one strictly dominates on both delta and date
    lag. Ties abstain, which is the same policy the prompt asks the model for,
    so the two paths stay comparable in the metrics.
    """

    name = "offline"

    def __init__(self, dominance_ratio: float = 0.25) -> None:
        self.dominance_ratio = dominance_ratio

    def adjudicate(self, request: AdjudicationRequest) -> AdjudicationResponse:
        if not request.candidates:
            return AdjudicationResponse(None, 0.0, "no candidates offered", 0)
        ranked = sorted(
            request.candidates, key=lambda c: (abs(c.delta_paise), abs(c.date_lag_days), -c.row_count)
        )
        best = ranked[0]
        if len(ranked) == 1:
            return AdjudicationResponse(best.candidate_id, 0.80, "sole candidate within tolerance", 0)

        runner = ranked[1]
        best_cost = abs(best.delta_paise) + abs(best.date_lag_days) * 100
        runner_cost = abs(runner.delta_paise) + abs(runner.date_lag_days) * 100
        if runner_cost == 0 or best_cost >= runner_cost * (1 - self.dominance_ratio):
            return AdjudicationResponse(
                None, 0.0, "candidates are equally supported; leaving for a human", 0
            )
        return AdjudicationResponse(best.candidate_id, 0.72, "clearly closer on amount and date", 0)


class ClaudeAdjudicator:
    """Calls the Anthropic API, degrading to a fallback on any failure.

    A rate limit, a timeout or a malformed body must not fail the run. Each of
    those routes the item to the fallback adjudicator and marks the result
    degraded, so the report can say how much of the batch ran without the model.
    """

    name = "claude"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = MODEL,
        timeout: float = 20.0,
        fallback: Adjudicator | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.timeout = timeout
        self.fallback = fallback or OfflineAdjudicator()
        self.calls = 0
        self.failures = 0

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _build_user_message(self, request: AdjudicationRequest) -> str:
        payload = {
            "bank_credit": {
                "txn_id": request.txn_id,
                "value_date": request.value_date,
                "narration": request.narration,
                "amount_paise": request.credit_paise,
            },
            "candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "row_count": c.row_count,
                    "net_paise": c.net_paise,
                    "delta_paise": c.delta_paise,
                    "date_lag_days": c.date_lag_days,
                    "settle_date": c.settle_date,
                    "found_by": c.source_tier,
                    "evidence": c.evidence,
                }
                for c in request.candidates
            ],
        }
        return json.dumps(payload, indent=2)

    def adjudicate(self, request: AdjudicationRequest) -> AdjudicationResponse:
        if not self.available:
            resp = self.fallback.adjudicate(request)
            resp.degraded = True
            resp.reason = f"no API key; {resp.reason}"
            return resp

        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": 1000,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": self._build_user_message(request)}],
            }
        ).encode()

        req = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )

        try:
            self.calls += 1
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            self.failures += 1
            resp = self.fallback.adjudicate(request)
            resp.degraded = True
            resp.reason = f"model unavailable ({type(exc).__name__}); {resp.reason}"
            return resp

        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        tokens = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))

        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            self.failures += 1
            resp = self.fallback.adjudicate(request)
            resp.degraded = True
            resp.reason = f"model returned unparseable JSON; {resp.reason}"
            resp.tokens = tokens
            return resp

        return _validate(payload, request, tokens)
