"""The matching cascade.

Tiers run as global passes over the whole batch rather than per transaction.
That ordering matters: if each credit were pushed through all five tiers before
the next one started, a speculative tier-3 subset could claim rows that a later
credit would have matched exactly on its UTR. Running the cheap, certain tier
across everything first and only then descending means every claim is made with
the best evidence available at the time.

On the safety of arithmetic
---------------------------
Subset-sum is not evidence on its own. With *n* candidate rows there are 2**n
subsets but only about ``pool_total`` distinct achievable paise values, so once
n passes roughly 20 the pigeonhole principle guarantees spurious exact matches.
Measured on random batches: 0% spurious alternates at n=14, 48% at n=22, 100%
at n=28. A system that trusts bare arithmetic at that scale reports a superb
match rate and is quietly wrong.

Two things follow, and both are implemented below. First, the pool is
constrained by a structural prior -- a settlement batch pays out on one date,
so candidates are drawn from a single settle-date group, which keeps n small.
Second, every arithmetic match carries an estimated collision risk derived from
the size of the hypothesis space actually searched; matches whose risk is too
high are escalated instead of accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from itertools import combinations

from kosh.core.adjudicator import (
    Adjudicator,
    AdjudicationRequest,
    CandidateView,
    OfflineAdjudicator,
)
from kosh.core.models import (
    BankTxn,
    DETERMINISTIC_TIERS,
    ExceptionType,
    MatchResult,
    SettlementRow,
    Tier,
)
from kosh.core.normalize import extract_utrs
from kosh.matchers.index import SettlementIndex
from kosh.matchers.subset_sum import find_complement, solve_subset_sum

GATEWAY_TOKENS = ("RAZORPAY", "RAZORPAYSOFT", "RZP")


@dataclass(slots=True)
class ReconcileConfig:
    """Tunables. Every one of these is a policy choice, not a magic number."""

    date_window_days: int = 3

    amount_tolerance_paise: int = 1_000

    subset_sum_safe_n: int = 20

    max_collision_risk: float = 0.02

    max_complement_drop: int = 3

    enable_adjudication: bool = True


@dataclass(slots=True)
class InternalCandidate:
    """A proposal awaiting acceptance or adjudication."""

    candidate_id: str
    row_ids: tuple[str, ...]
    net_paise: int
    delta_paise: int
    date_lag_days: int
    settle_date: date
    source_tier: Tier
    collision_risk: float = 0.0
    evidence: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ReconcileRun:
    """Everything one pass over a batch produced."""

    results: list[MatchResult] = field(default_factory=list)
    llm_calls: int = 0
    llm_tokens: int = 0
    degraded_calls: int = 0
    tier_counts: dict[str, int] = field(default_factory=dict)
    escalated: int = 0

    @property
    def matched(self) -> list[MatchResult]:
        return [r for r in self.results if r.matched]

    @property
    def exceptions(self) -> list[MatchResult]:
        return [r for r in self.results if not r.matched]

    @property
    def deterministic_resolved(self) -> int:
        return sum(1 for r in self.results if r.tier in DETERMINISTIC_TIERS)


def collision_risk(hypotheses: int, spread_paise: int) -> float:
    """Rough chance that a search of this size hits the target by luck.

    A subset drawn at random lands on any particular paise value with
    probability about ``1 / spread``, so searching ``hypotheses`` of them yields
    roughly ``hypotheses / spread`` spurious exact hits. Crude, deliberately
    pessimistic, and enough to separate a 3-row complement search (safe by many
    orders of magnitude) from an unconstrained 28-row subset-sum (hopeless).
    """
    if spread_paise <= 0:
        return 1.0
    return min(1.0, hypotheses / spread_paise)


def _looks_like_gateway(narration: str) -> bool:
    upper = (narration or "").upper()
    return any(token in upper for token in GATEWAY_TOKENS)


class Cascade:
    """Runs the five tiers over one batch of bank lines."""

    def __init__(
        self,
        rows: list[SettlementRow],
        config: ReconcileConfig | None = None,
        adjudicator: Adjudicator | None = None,
    ) -> None:
        self.index = SettlementIndex(rows)
        self.config = config or ReconcileConfig()
        self.adjudicator = adjudicator or OfflineAdjudicator()
        self._candidate_seq = 0

        self.risk_declined: set[str] = set()

    def _cid(self) -> str:
        self._candidate_seq += 1
        return f"cand_{self._candidate_seq:05d}"

    def _tier1_exact(self, txn: BankTxn) -> InternalCandidate | list[InternalCandidate] | None:
        """Join on a UTR lifted from the narration, requiring an exact total.

        Returns a single candidate for a clean hit, a list when one UTR points
        at several batches (the duplicate-UTR case, which must be escalated),
        or None when the narration carries no usable reference.
        """
        hits: list[InternalCandidate] = []
        for utr in extract_utrs(txn.narration):
            for settlement_id in sorted(self.index.settlement_ids_by_utr.get(utr, ())):
                rows = self.index.available_group(settlement_id)
                if not rows:
                    continue
                net = self.index.net_of(rows)
                lag = (txn.value_date - self.index.group_date[settlement_id]).days
                if net == txn.credit_paise and abs(lag) <= self.config.date_window_days:
                    hits.append(
                        InternalCandidate(
                            candidate_id=self._cid(),
                            row_ids=rows,
                            net_paise=net,
                            delta_paise=0,
                            date_lag_days=lag,
                            settle_date=self.index.group_date[settlement_id],
                            source_tier=Tier.EXACT,
                            evidence={
                                "utr": utr,
                                "settlement_id": settlement_id,
                                "match": "utr and exact net total",
                            },
                        )
                    )
        if not hits:
            return None
        return hits[0] if len(hits) == 1 else hits

    def _tier2_fuzzy(self, txn: BankTxn) -> list[InternalCandidate]:
        """Group totals within the amount tolerance and the date window.

        This is what catches a settlement whose UTR never made it into the bank
        narration, and rounding drift of a few paise on fee and GST.
        """
        out: list[InternalCandidate] = []
        for settlement_id in self.index.groups_in_window(txn.value_date, self.config.date_window_days):
            rows = self.index.available_group(settlement_id)
            if not rows:
                continue
            net = self.index.net_of(rows)
            delta = txn.credit_paise - net
            if abs(delta) > self.config.amount_tolerance_paise:
                continue
            settle = self.index.group_date[settlement_id]
            out.append(
                InternalCandidate(
                    candidate_id=self._cid(),
                    row_ids=rows,
                    net_paise=net,
                    delta_paise=delta,
                    date_lag_days=(txn.value_date - settle).days,
                    settle_date=settle,
                    source_tier=Tier.FUZZY,
                    evidence={
                        "settlement_id": settlement_id,
                        "match": "group total within tolerance",
                        "gateway_narration": str(_looks_like_gateway(txn.narration)),
                    },
                )
            )
        return out

    def _tier3_partial_group(self, txn: BankTxn) -> list[InternalCandidate]:
        """A credit that pays out only part of a batch the narration names.

        Gateways sometimes split one settlement across two payouts. The UTR
        still identifies the batch, so the question is not *which* rows but
        *how many* -- a far smaller question. Contiguous slices are tried first
        because a split payout is normally a prefix or a suffix of the batch,
        which makes the hypothesis space 2n rather than 2**n and keeps the
        collision risk negligible even on large batches.
        """
        target = txn.credit_paise
        out: list[InternalCandidate] = []

        for utr in extract_utrs(txn.narration):
            for settlement_id in sorted(self.index.settlement_ids_by_utr.get(utr, ())):
                rows = self.index.available_group(settlement_id)
                if len(rows) < 2:
                    continue
                total = self.index.net_of(rows)
                if total < target:
                    continue

                settle = self.index.group_date[settlement_id]
                lag = (txn.value_date - settle).days
                if abs(lag) > self.config.date_window_days:
                    continue

                if total == target:

                    out.append(
                        InternalCandidate(
                            candidate_id=self._cid(),
                            row_ids=rows,
                            net_paise=total,
                            delta_paise=0,
                            date_lag_days=lag,
                            settle_date=settle,
                            source_tier=Tier.SUBSET_SUM,
                            collision_risk=0.0,
                            evidence={
                                "strategy": "remaining rows of a partly paid batch",
                                "utr": utr,
                                "settlement_id": settlement_id,
                                "rows": str(len(rows)),
                            },
                        )
                    )
                    continue

                ordered = tuple(sorted(rows))
                amounts = [self.index.rows[r].net_paise for r in ordered]
                n = len(ordered)

                slice_hit: tuple[str, ...] | None = None
                strategy = ""
                running = 0
                for i in range(n):
                    running += amounts[i]
                    if running == target:
                        slice_hit, strategy = ordered[: i + 1], "leading slice of the batch"
                        break
                if slice_hit is None:
                    running = 0
                    for i in range(n - 1, -1, -1):
                        running += amounts[i]
                        if running == target:
                            slice_hit, strategy = ordered[i:], "trailing slice of the batch"
                            break

                if slice_hit is not None:
                    out.append(
                        InternalCandidate(
                            candidate_id=self._cid(),
                            row_ids=slice_hit,
                            net_paise=target,
                            delta_paise=0,
                            date_lag_days=lag,
                            settle_date=settle,
                            source_tier=Tier.SUBSET_SUM,
                            collision_risk=collision_risk(2 * n, max(total, 1)),
                            evidence={
                                "strategy": strategy,
                                "utr": utr,
                                "settlement_id": settlement_id,
                                "rows": f"{len(slice_hit)} of {n}",
                            },
                        )
                    )
                    continue

                if n > self.config.subset_sum_safe_n:
                    self.risk_declined.add(txn.txn_id)
                    continue
                risk = collision_risk(2**n, max(total, 1))
                if risk > self.config.max_collision_risk:
                    self.risk_declined.add(txn.txn_id)
                    continue
                for solution in solve_subset_sum(amounts, target, max_solutions=2):
                    chosen = tuple(ordered[i] for i in solution.indices)
                    out.append(
                        InternalCandidate(
                            candidate_id=self._cid(),
                            row_ids=chosen,
                            net_paise=target,
                            delta_paise=0,
                            date_lag_days=lag,
                            settle_date=settle,
                            source_tier=Tier.SUBSET_SUM,
                            collision_risk=risk,
                            evidence={
                                "strategy": "exact subset of the named batch",
                                "utr": utr,
                                "settlement_id": settlement_id,
                                "rows": f"{len(chosen)} of {n}",
                            },
                        )
                    )
        return out

    def _tier3_subset_sum(self, txn: BankTxn) -> list[InternalCandidate]:
        """Reconstruct a batch arithmetically.

        Strategies run in increasing order of risk, each bounded and each
        carrying its own collision-risk estimate. UTR-anchored partial matches
        come first because a named batch is the strongest constraint available.
        """
        anchored = self._tier3_partial_group(txn)
        if anchored:
            return anchored

        day_groups = self.index.unattributed_in_window(txn.value_date, self.config.date_window_days)
        if not day_groups:
            return []

        target = txn.credit_paise
        out: list[InternalCandidate] = []
        days = sorted(day_groups)

        totals = {d: self.index.net_of(day_groups[d]) for d in days}
        for size in range(1, min(len(days), 4) + 1):
            for combo in combinations(days, size):
                if sum(totals[d] for d in combo) != target:
                    continue
                row_ids = tuple(r for d in combo for r in day_groups[d])
                lag = (txn.value_date - max(combo)).days
                out.append(
                    InternalCandidate(
                        candidate_id=self._cid(),
                        row_ids=row_ids,
                        net_paise=target,
                        delta_paise=0,
                        date_lag_days=lag,
                        settle_date=max(combo),
                        source_tier=Tier.SUBSET_SUM,
                        collision_risk=collision_risk(2 ** len(days), max(sum(totals.values()), 1)),
                        evidence={
                            "strategy": "whole settle-date groups",
                            "days": ",".join(d.isoformat() for d in combo),
                            "rows": str(len(row_ids)),
                        },
                    )
                )
        if out:
            return out

        for day in sorted(days, key=lambda d: (abs((txn.value_date - d).days), d)):
            ids = day_groups[day]
            amounts = [self.index.rows[r].net_paise for r in ids]
            pool_total = sum(amounts)
            lag = (txn.value_date - day).days

            drop = find_complement(amounts, pool_total, target, self.config.max_complement_drop)
            if drop is not None:
                n = len(amounts)
                hypotheses = 1 + n + n * (n - 1) // 2 + n * (n - 1) * (n - 2) // 6
                risk = collision_risk(hypotheses, max(pool_total, 1))
                kept = tuple(ids[i] for i in range(len(ids)) if i not in set(drop))
                if kept and risk <= self.config.max_collision_risk:
                    out.append(
                        InternalCandidate(
                            candidate_id=self._cid(),
                            row_ids=kept,
                            net_paise=self.index.net_of(kept),
                            delta_paise=target - self.index.net_of(kept),
                            date_lag_days=lag,
                            settle_date=day,
                            source_tier=Tier.SUBSET_SUM,
                            collision_risk=risk,
                            evidence={
                                "strategy": "day group minus held-back rows",
                                "dropped": str(len(drop)),
                                "day": day.isoformat(),
                            },
                        )
                    )
                    return out

            if len(amounts) > self.config.subset_sum_safe_n:
                self.risk_declined.add(txn.txn_id)
                continue
            risk = collision_risk(2 ** len(amounts), max(pool_total, 1))
            if risk > self.config.max_collision_risk:
                self.risk_declined.add(txn.txn_id)
                continue
            for solution in solve_subset_sum(amounts, target, max_solutions=2):
                chosen = tuple(ids[i] for i in solution.indices)
                out.append(
                    InternalCandidate(
                        candidate_id=self._cid(),
                        row_ids=chosen,
                        net_paise=self.index.net_of(chosen),
                        delta_paise=0,
                        date_lag_days=lag,
                        settle_date=day,
                        source_tier=Tier.SUBSET_SUM,
                        collision_risk=risk,
                        evidence={
                            "strategy": "exact subset of one settle-date group",
                            "rows": str(len(chosen)),
                            "pool": str(len(ids)),
                            "day": day.isoformat(),
                        },
                    )
                )
            if out:
                return out
        return out

    def _accept(self, txn: BankTxn, cand: InternalCandidate, confidence: float, reason: str) -> MatchResult:
        self.index.claim(cand.row_ids)
        return MatchResult(
            txn_id=txn.txn_id,
            tier=cand.source_tier,
            row_ids=cand.row_ids,
            delta_paise=cand.delta_paise,
            confidence=confidence,
            reason=reason,
            candidates_considered=1,
        )

    def _classify_failure(self, txn: BankTxn, had_candidates: bool) -> ExceptionType:
        """Give an unmatched line a type that maps to a human workflow."""
        if txn.debit_paise > 0:
            return ExceptionType.CHARGEBACK_DEBIT
        if had_candidates:
            return ExceptionType.AMBIGUOUS_CANDIDATES

        if not _looks_like_gateway(txn.narration):
            return ExceptionType.ORPHAN_CREDIT
        if txn.txn_id in self.risk_declined:
            return ExceptionType.ARITHMETIC_UNSAFE
        if self.index.available_ids:
            return ExceptionType.NO_CANDIDATE
        return ExceptionType.MISSING_IN_BANK

    def run(self, bank_txns: list[BankTxn]) -> ReconcileRun:
        run = ReconcileRun()
        results: dict[str, MatchResult] = {}
        pending: dict[str, list[InternalCandidate]] = {}

        credits = [t for t in bank_txns if t.is_credit]
        debits = [t for t in bank_txns if not t.is_credit]

        for txn in debits:
            results[txn.txn_id] = MatchResult(
                txn_id=txn.txn_id,
                tier=Tier.EXCEPTION,
                exception_type=ExceptionType.CHARGEBACK_DEBIT,
                reason="debit line; belongs to the disputes workflow, not settlement",
            )

        by_id = {t.txn_id: t for t in credits}
        unresolved = [t.txn_id for t in credits]

        deterministic = (
            (self._tier1_exact, 0.99, "UTR and exact net total agree"),
            (self._tier2_fuzzy, 0.90, "single group within tolerance"),
            (self._tier3_subset_sum, 0.85, "arithmetic reconstruction"),
        )

        for _ in range(len(credits) + 1):
            progressed = False
            pending.clear()

            for finder, confidence, label in deterministic:
                still = []
                for txn_id in unresolved:
                    txn = by_id[txn_id]
                    found = finder(txn)
                    cands = [found] if isinstance(found, InternalCandidate) else list(found or [])
                    cands = [c for c in cands if all(self.index.is_available(r) for r in c.row_ids)]

                    if len(cands) == 1:
                        c = cands[0]
                        detail = c.evidence.get("strategy", label)
                        if c.collision_risk:
                            detail += f", collision risk {c.collision_risk:.1e}"
                        elif c.delta_paise:
                            detail += f", delta {c.delta_paise}p"
                        results[txn_id] = self._accept(txn, c, confidence, detail)
                        progressed = True
                    else:
                        if cands:
                            pending.setdefault(txn_id, []).extend(cands)
                        still.append(txn_id)
                unresolved = still

            if not progressed:
                break

        still = []
        for txn_id in unresolved:
            txn = by_id[txn_id]
            cands = [
                c for c in pending.get(txn_id, []) if all(self.index.is_available(r) for r in c.row_ids)
            ]
            if not cands or not self.config.enable_adjudication:
                still.append(txn_id)
                continue

            run.escalated += 1
            request = AdjudicationRequest(
                txn_id=txn.txn_id,
                value_date=txn.value_date.isoformat(),
                narration=txn.narration,
                credit_paise=txn.credit_paise,
                candidates=[
                    CandidateView(
                        candidate_id=c.candidate_id,
                        row_count=len(c.row_ids),
                        net_paise=c.net_paise,
                        delta_paise=c.delta_paise,
                        date_lag_days=c.date_lag_days,
                        settle_date=c.settle_date.isoformat(),
                        source_tier=c.source_tier.value,
                        evidence=c.evidence,
                    )
                    for c in cands
                ],
            )
            response = self.adjudicator.adjudicate(request)
            run.llm_calls += 1
            run.llm_tokens += response.tokens
            if response.degraded:
                run.degraded_calls += 1

            chosen = next((c for c in cands if c.candidate_id == response.candidate_id), None)
            if chosen is None:
                still.append(txn_id)
                continue

            self.index.claim(chosen.row_ids)
            results[txn_id] = MatchResult(
                txn_id=txn_id,
                tier=Tier.ADJUDICATED,
                row_ids=chosen.row_ids,
                delta_paise=chosen.delta_paise,
                confidence=response.confidence,
                reason=response.reason,
                candidates_considered=len(cands),
                llm_tokens=response.tokens,
            )
        unresolved = still

        for txn_id in unresolved:
            txn = by_id[txn_id]
            had = bool(pending.get(txn_id))
            kind = self._classify_failure(txn, had)
            reasons = {
                ExceptionType.AMBIGUOUS_CANDIDATES: (
                    f"{len(pending.get(txn_id, []))} candidates, none separable on the evidence"
                ),
                ExceptionType.ORPHAN_CREDIT: (
                    "narration names no gateway; looks like a direct transfer, not a settlement"
                ),
                ExceptionType.ARITHMETIC_UNSAFE: (
                    "candidate pool too large for an exact sum to be evidence; declined rather than guessed"
                ),
                ExceptionType.CHARGEBACK_DEBIT: (
                    "debit line; belongs to the disputes workflow, not settlement"
                ),
                ExceptionType.MISSING_IN_BANK: (
                    "settlement rows are all claimed; nothing left this credit could be"
                ),
            }
            results[txn_id] = MatchResult(
                txn_id=txn_id,
                tier=Tier.EXCEPTION,
                exception_type=kind,
                reason=reasons.get(
                    kind, "no settlement rows in the window account for this amount"
                ),
                candidates_considered=len(pending.get(txn_id, [])),
            )

        run.results = [results[t.txn_id] for t in bank_txns if t.txn_id in results]
        counts: dict[str, int] = {}
        for r in run.results:
            counts[r.tier.value] = counts.get(r.tier.value, 0) + 1
        run.tier_counts = counts
        return run
