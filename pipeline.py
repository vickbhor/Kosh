"""End-to-end run orchestration.

Wraps parse, match, drift-detect, and audit into one call, and adds the two
things a demo needs that a library does not: deliberate fault injection, and a
plain-language decomposition of any matched credit.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kosh.core.adjudicator import (
    Adjudicator,
    AdjudicationRequest,
    AdjudicationResponse,
    ClaudeAdjudicator,
    OfflineAdjudicator,
)
from kosh.core.fee_drift import detect_fee_drift
from kosh.core.ledger import Ledger
from kosh.core.models import EXCEPTION_ACTIONS, RowKind, Tier, format_inr
from kosh.core.normalize import ParseResult, load_directory
from kosh.eval.harness import Stopwatch
from kosh.matchers.cascade import Cascade, ReconcileConfig, ReconcileRun


@dataclass(slots=True)
class ChaosConfig:
    """Faults to inject on purpose.

    Every one of these happens in production. Triggering them deliberately is
    the only way to know the system degrades instead of breaking.
    """

    adjudicator_failure_rate: float = 0.0

    truncate_bank_fraction: float = 0.0

    @property
    def active(self) -> bool:
        return self.adjudicator_failure_rate > 0 or self.truncate_bank_fraction > 0


class FlakyAdjudicator:
    """Wraps an adjudicator and fails a share of calls.

    Failure returns a degraded abstention, never an exception, which is the
    behaviour the cascade is written to expect from an unreliable dependency.
    """

    name = "flaky"

    def __init__(self, inner: Adjudicator, failure_rate: float) -> None:
        self.inner = inner
        self.failure_rate = failure_rate
        self._n = 0

    def adjudicate(self, request: AdjudicationRequest) -> AdjudicationResponse:
        self._n += 1

        every = max(1, round(1 / self.failure_rate)) if self.failure_rate else 0
        if every and self._n % every == 1 % every:
            return AdjudicationResponse(
                None, 0.0, "injected fault: adjudicator unreachable; left for a human", 0, degraded=True
            )
        return self.inner.adjudicate(request)


@dataclass
class RunReport:
    run_id: str
    reused_existing: bool
    parse: ParseResult
    run: ReconcileRun
    drift: dict
    wall_seconds: float
    chain_tip: str
    config: dict = field(default_factory=dict)

    def summary(self) -> dict:
        matched = self.run.matched
        total_credit = sum(1 for r in self.run.results)
        return {
            "run_id": self.run_id,
            "reused_existing": self.reused_existing,
            "bank_lines": total_credit,
            "settlement_rows": len(self.parse.settlement_rows),
            "quarantined": len(self.parse.quarantined),
            "matched": len(matched),
            "exceptions": len(self.run.exceptions),
            "tier_counts": self.run.tier_counts,
            "escalated": self.run.escalated,
            "llm_calls": self.run.llm_calls,
            "llm_tokens": self.run.llm_tokens,
            "degraded_calls": self.run.degraded_calls,
            "deterministic_resolved": self.run.deterministic_resolved,
            "wall_seconds": round(self.wall_seconds, 3),
            "chain_tip": self.chain_tip,
            "drift": self.drift,
        }


def _rehydrate(ledger: Ledger, run_id: str) -> ReconcileRun:
    """Rebuild a completed run's results from its audit entries."""
    import json as _json

    from kosh.core.models import ExceptionType, MatchResult

    run = ReconcileRun()
    summary = ledger.run_summary(run_id) or {}
    for entry in ledger.entries(run_id, limit=200_000):
        if entry["kind"] not in ("match", "exception"):
            continue
        payload = _json.loads(entry["payload_json"])
        exc = payload.get("exception_type")
        run.results.append(
            MatchResult(
                txn_id=entry["subject"],
                tier=Tier(payload["tier"]),
                row_ids=tuple(payload.get("rows", ())),
                delta_paise=payload.get("delta_paise", 0),
                confidence=payload.get("confidence", 0.0),
                reason=payload.get("reason", ""),
                exception_type=ExceptionType(exc) if exc else None,
                candidates_considered=payload.get("candidates_considered", 0),
            )
        )
    run.tier_counts = summary.get("tier_counts", {})
    run.escalated = summary.get("escalated", 0)
    run.llm_calls = summary.get("llm_calls", 0)
    run.llm_tokens = summary.get("llm_tokens", 0)
    run.degraded_calls = summary.get("degraded_calls", 0)
    return run


def build_adjudicator(enable_llm: bool, chaos: ChaosConfig | None = None) -> Adjudicator:
    """Pick an adjudicator. Falls back cleanly when no key is configured."""
    base: Adjudicator = ClaudeAdjudicator() if enable_llm else OfflineAdjudicator()
    if isinstance(base, ClaudeAdjudicator) and not base.available:
        base = OfflineAdjudicator()
    if chaos and chaos.adjudicator_failure_rate > 0:
        return FlakyAdjudicator(base, chaos.adjudicator_failure_rate)
    return base


def reconcile_directory(
    data_dir: Path,
    *,
    config: ReconcileConfig | None = None,
    ledger: Ledger | None = None,
    enable_llm: bool = True,
    chaos: ChaosConfig | None = None,
    contracted_bps: int = 200,
) -> RunReport:
    """Parse, reconcile, detect drift, and write the audit chain."""
    config = config or ReconcileConfig()
    chaos = chaos or ChaosConfig()
    owns_ledger = ledger is None
    ledger = ledger or Ledger(data_dir / "kosh.db")

    parse = load_directory(data_dir)

    if chaos.truncate_bank_fraction > 0:
        keep = int(len(parse.bank_txns) * (1 - chaos.truncate_bank_fraction))
        parse.bank_txns = parse.bank_txns[:keep]

    from kosh.core.ledger import fingerprint_files

    fingerprint = fingerprint_files(
        [data_dir / "settlement_report.csv", data_dir / "bank_statement.csv", data_dir / "orders.csv"]
    )

    import hashlib as _hashlib

    variant = json.dumps(
        {
            "config": asdict(config),
            "llm": enable_llm,
            "chaos": [chaos.adjudicator_failure_rate, chaos.truncate_bank_fraction],
        },
        sort_keys=True,
        default=str,
    )
    fingerprint = _hashlib.sha256(f"{fingerprint}:{variant}".encode()).hexdigest()

    run_id = uuid.uuid4().hex[:12]
    prior = ledger.start_run(run_id, fingerprint, asdict(config))
    if prior:

        report = RunReport(
            run_id=prior,
            reused_existing=True,
            parse=parse,
            run=_rehydrate(ledger, prior),
            drift=(ledger.run_summary(prior) or {}).get("drift", {}),
            wall_seconds=0.0,
            chain_tip=(ledger.run_summary(prior) or {}).get("chain_tip", ""),
            config=asdict(config),
        )
        if owns_ledger:
            ledger.close()
        return report

    adjudicator = build_adjudicator(enable_llm and config.enable_adjudication, chaos)
    cascade = Cascade(parse.settlement_rows, config, adjudicator)

    with Stopwatch() as sw:
        run = cascade.run(parse.bank_txns)

    drift = detect_fee_drift(parse.settlement_rows, contracted_bps=contracted_bps).to_dict()

    records: list[tuple[str, str, dict]] = []
    for q in parse.quarantined:
        records.append(("quarantine", f"{q.source}:{q.line_no}", {"reason": q.reason, "raw": q.raw[:200]}))
    for r in run.results:
        records.append(
            (
                "match" if r.matched else "exception",
                r.txn_id,
                {
                    "tier": r.tier.value,
                    "rows": list(r.row_ids),
                    "delta_paise": r.delta_paise,
                    "confidence": round(r.confidence, 3),
                    "reason": r.reason,
                    "exception_type": r.exception_type.value if r.exception_type else None,
                    "candidates_considered": r.candidates_considered,
                },
            )
        )
    chain_tip = ledger.append_many(run_id, records)

    report = RunReport(
        run_id=run_id,
        reused_existing=False,
        parse=parse,
        run=run,
        drift=drift,
        wall_seconds=sw.seconds,
        chain_tip=chain_tip,
        config=asdict(config),
    )
    ledger.finish_run(run_id, report.summary())
    if owns_ledger:
        ledger.close()
    return report


def explain(report: RunReport, txn_id: str) -> dict:
    """Decompose one matched credit into the arithmetic behind it.

    The engine supplies and verifies every number here. A model may later be
    asked to turn this dict into a sentence, but it is never asked to produce
    the figures, so the explanation cannot drift from the ledger.
    """
    result = next((r for r in report.run.results if r.txn_id == txn_id), None)
    if result is None:
        return {"error": f"no such bank line: {txn_id}"}

    txn = next((t for t in report.parse.bank_txns if t.txn_id == txn_id), None)
    rows = {r.row_id: r for r in report.parse.settlement_rows}
    members = [rows[r] for r in result.row_ids if r in rows]

    if not result.matched:

        from datetime import timedelta

        claimed = {r for m in report.run.results if m.matched for r in m.row_ids}
        lo = (txn.value_date - timedelta(days=3)) if txn else None
        hi = (txn.value_date + timedelta(days=3)) if txn else None
        nearby = [
            r
            for r in report.parse.settlement_rows
            if lo and lo <= r.settled_on <= hi and r.row_id not in claimed
        ]
        return {
            "txn_id": txn_id,
            "matched": False,
            "tier": result.tier.value,
            "reason": result.reason,
            "exception_type": result.exception_type.value if result.exception_type else None,
            "suggested_action": EXCEPTION_ACTIONS.get(result.exception_type, "")
            if result.exception_type
            else "",
            "bank_credit_paise": txn.signed_paise if txn else 0,
            "narration": txn.narration if txn else "",
            "diagnostics": {
                "candidates_considered": result.candidates_considered,
                "unclaimed_rows_near_this_date": len(nearby),
                "unclaimed_net_paise": sum(r.net_paise for r in nearby),
                "total_unclaimed_rows": len(report.parse.settlement_rows) - len(claimed),
            },
        }

    payments = [r for r in members if r.kind is RowKind.PAYMENT]
    refunds = [r for r in members if r.kind is RowKind.REFUND]
    adjustments = [r for r in members if r.kind is RowKind.ADJUSTMENT]

    gross = sum(r.gross_paise for r in payments)
    fee = sum(r.fee_paise for r in payments)
    tax = sum(r.tax_paise for r in payments)
    refunded = sum(abs(r.gross_paise) for r in refunds)
    adjusted = sum(r.net_paise for r in adjustments)
    computed = gross - fee - tax - refunded + adjusted

    return {
        "txn_id": txn_id,
        "matched": result.matched,
        "tier": result.tier.value,
        "reason": result.reason,
        "exception_type": result.exception_type.value if result.exception_type else None,
        "suggested_action": EXCEPTION_ACTIONS.get(result.exception_type, "") if result.exception_type else "",
        "bank_credit_paise": txn.credit_paise if txn else 0,
        "components": {
            "payments": {"count": len(payments), "gross_paise": gross},
            "platform_fee_paise": -fee,
            "gst_on_fee_paise": -tax,
            "refunds": {"count": len(refunds), "paise": -refunded},
            "adjustments": {"count": len(adjustments), "paise": adjusted},
        },
        "computed_net_paise": computed,
        "residual_paise": (txn.credit_paise if txn else 0) - computed,
        "narrative": (
            f"{format_inr(txn.credit_paise if txn else 0)} = {len(payments)} payments "
            f"({format_inr(gross)}) - platform fee ({format_inr(fee)}) "
            f"- GST on fee ({format_inr(tax)})"
            + (f" - {len(refunds)} refunds ({format_inr(refunded)})" if refunds else "")
            + (f" + adjustments ({format_inr(adjusted)})" if adjustments else "")
        ),
    }


def ai_budget(report: RunReport, cost_per_1k_inr: float = 0.65) -> dict:
    """What the run cost, and what it would have cost with the model switched off.

    The point of showing this is that the second number is close to the first.
    Most of the work never needed a model, and the panel says so out loud.
    """
    total = len(report.run.results)
    deterministic = report.run.deterministic_resolved
    return {
        "bank_lines": total,
        "resolved_without_ai": deterministic,
        "resolved_without_ai_pct": round(100 * deterministic / total, 1) if total else 0.0,
        "escalated_to_model": report.run.escalated,
        "model_calls": report.run.llm_calls,
        "degraded_calls": report.run.degraded_calls,
        "tokens": report.run.llm_tokens,
        "estimated_cost_inr": round(report.run.llm_tokens / 1000 * cost_per_1k_inr, 4),
        "cost_per_1k_rows_inr": round(
            (report.run.llm_tokens / 1000 * cost_per_1k_inr) / max(len(report.parse.settlement_rows), 1) * 1000,
            4,
        ),
    }
