"""Scoring a reconciliation run against ground truth.

The headline number here is not the match rate. It is the false-match rate: the
share of accepted matches that are wrong. In a ledger a missed match costs an
analyst ten minutes, while a wrong match silently misstates the books, so the
two failures are not interchangeable and are never averaged together.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from kosh.core.models import DETERMINISTIC_TIERS, MatchResult, Tier
from kosh.eval.generator import TruthEntry
from kosh.matchers.cascade import ReconcileRun

COST_PER_1K_TOKENS_INR = 0.65


@dataclass
class ScenarioScore:
    scenario: str
    total: int = 0
    correct: int = 0
    wrong: int = 0
    missed: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class Scorecard:
    """Everything a reviewer needs to decide whether to trust the run."""

    bank_lines: int = 0
    matched: int = 0
    correct_matches: int = 0
    wrong_matches: int = 0
    missed_matches: int = 0
    correct_exceptions: int = 0
    correctly_typed_exceptions: int = 0
    typed_exception_total: int = 0
    exceptions: int = 0
    quarantined_rows: int = 0

    tier_counts: dict[str, int] = field(default_factory=dict)
    exception_counts: dict[str, int] = field(default_factory=dict)
    by_scenario: dict[str, ScenarioScore] = field(default_factory=dict)

    llm_calls: int = 0
    llm_tokens: int = 0
    degraded_calls: int = 0
    escalated: int = 0

    wall_seconds: float = 0.0
    settlement_rows: int = 0

    wrong_examples: list[dict] = field(default_factory=list)

    @property
    def auto_match_rate(self) -> float:
        return self.matched / self.bank_lines if self.bank_lines else 0.0

    @property
    def precision(self) -> float:
        """Share of accepted matches that are right."""
        return self.correct_matches / self.matched if self.matched else 0.0

    @property
    def false_match_rate(self) -> float:
        return self.wrong_matches / self.matched if self.matched else 0.0

    @property
    def recall(self) -> float:
        expected = self.correct_matches + self.wrong_matches + self.missed_matches
        return self.correct_matches / expected if expected else 0.0

    @property
    def exception_precision(self) -> float:
        """Share of raised exceptions that genuinely had no correct match."""
        return self.correct_exceptions / self.exceptions if self.exceptions else 0.0

    @property
    def exception_type_accuracy(self) -> float:
        """Of exceptions that ground truth labelled, how many got the right type.

        Raising an exception is only half the job. An operator routes on the
        type, so a chargeback filed as an orphan credit still wastes their day.
        """
        if not self.typed_exception_total:
            return 0.0
        return self.correctly_typed_exceptions / self.typed_exception_total

    @property
    def deterministic_share(self) -> float:
        total = sum(self.tier_counts.get(t.value, 0) for t in DETERMINISTIC_TIERS)
        return total / self.bank_lines if self.bank_lines else 0.0

    @property
    def rows_per_second(self) -> float:
        return self.settlement_rows / self.wall_seconds if self.wall_seconds else 0.0

    @property
    def llm_cost_inr(self) -> float:
        return self.llm_tokens / 1000 * COST_PER_1K_TOKENS_INR

    def to_dict(self) -> dict:
        d = asdict(self)
        d["by_scenario"] = {k: asdict(v) for k, v in self.by_scenario.items()}
        d["derived"] = {
            "auto_match_rate": round(self.auto_match_rate, 4),
            "precision": round(self.precision, 4),
            "false_match_rate": round(self.false_match_rate, 4),
            "recall": round(self.recall, 4),
            "exception_precision": round(self.exception_precision, 4),
            "exception_type_accuracy": round(self.exception_type_accuracy, 4),
            "deterministic_share": round(self.deterministic_share, 4),
            "rows_per_second": round(self.rows_per_second, 1),
            "llm_cost_inr": round(self.llm_cost_inr, 4),
        }
        return d


def score(
    run: ReconcileRun,
    truth: dict[str, TruthEntry],
    *,
    settlement_rows: int,
    quarantined: int = 0,
    wall_seconds: float = 0.0,
) -> Scorecard:
    card = Scorecard(
        bank_lines=len(run.results),
        tier_counts=dict(run.tier_counts),
        llm_calls=run.llm_calls,
        llm_tokens=run.llm_tokens,
        degraded_calls=run.degraded_calls,
        escalated=run.escalated,
        wall_seconds=wall_seconds,
        settlement_rows=settlement_rows,
        quarantined_rows=quarantined,
    )

    for result in run.results:
        entry = truth.get(result.txn_id)
        scenario = entry.scenario if entry else "unlabelled"
        bucket = card.by_scenario.setdefault(scenario, ScenarioScore(scenario))
        bucket.total += 1

        expected = set(entry.expected_row_ids) if entry else set()
        should_match = bool(expected)

        if result.matched:
            card.matched += 1
            if set(result.row_ids) == expected:
                card.correct_matches += 1
                bucket.correct += 1
            else:
                card.wrong_matches += 1
                bucket.wrong += 1
                if len(card.wrong_examples) < 12:
                    card.wrong_examples.append(
                        {
                            "txn_id": result.txn_id,
                            "scenario": scenario,
                            "tier": result.tier.value,
                            "expected_rows": len(expected),
                            "matched_rows": len(result.row_ids),
                            "overlap": len(expected & set(result.row_ids)),
                            "reason": result.reason,
                        }
                    )
        else:
            card.exceptions += 1
            key = result.exception_type.value if result.exception_type else "untyped"
            card.exception_counts[key] = card.exception_counts.get(key, 0) + 1
            if should_match:
                card.missed_matches += 1
                bucket.missed += 1
            else:
                card.correct_exceptions += 1
                bucket.correct += 1
                if entry and entry.expect_exception:
                    card.typed_exception_total += 1
                    if key == entry.expect_exception:
                        card.correctly_typed_exceptions += 1

    return card


class Stopwatch:
    """Tiny context manager so throughput is measured, not estimated."""

    def __enter__(self) -> "Stopwatch":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self.start


def render(card: Scorecard, title: str = "Reconciliation scorecard") -> str:
    """Plain-text report for the CLI and for pasting into a submission."""
    lines = [
        "",
        f"  {title}",
        "  " + "-" * 62,
        f"  bank lines                {card.bank_lines:>10,}",
        f"  settlement rows           {card.settlement_rows:>10,}",
        f"  quarantined rows          {card.quarantined_rows:>10,}",
        "",
        f"  auto-match rate           {card.auto_match_rate:>10.1%}",
        f"  precision                 {card.precision:>10.2%}",
        f"  FALSE-MATCH RATE          {card.false_match_rate:>10.2%}",
        f"  recall                    {card.recall:>10.2%}",
        f"  exception precision       {card.exception_precision:>10.2%}",
        f"  exception type accuracy   {card.exception_type_accuracy:>10.2%}",
        "",
        f"  resolved without AI       {card.deterministic_share:>10.1%}",
        f"  escalated to adjudicator  {card.escalated:>10,}",
        f"  adjudicator calls         {card.llm_calls:>10,}",
        f"  degraded calls            {card.degraded_calls:>10,}",
        f"  tokens                    {card.llm_tokens:>10,}",
        f"  estimated cost            {'INR ' + format(card.llm_cost_inr, '.2f'):>10}",
        "",
        f"  wall time                 {card.wall_seconds:>10.2f}s",
        f"  throughput                {card.rows_per_second:>10,.0f} rows/s",
        "",
        "  by tier",
    ]
    for tier in Tier:
        n = card.tier_counts.get(tier.value, 0)
        if n:
            lines.append(f"    {tier.value:<24}{n:>8,}")
    if card.exception_counts:
        lines.append("")
        lines.append("  exceptions by type")
        for key, n in sorted(card.exception_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {key:<24}{n:>8,}")
    lines.append("")
    lines.append("  by scenario")
    for name, s in sorted(card.by_scenario.items()):
        flag = "  <-- wrong" if s.wrong else ""
        lines.append(
            f"    {name:<24}{s.correct:>4}/{s.total:<4} ok  {s.wrong:>3} wrong  {s.missed:>3} missed{flag}"
        )
    if card.wrong_examples:
        lines.append("")
        lines.append("  false matches")
        for w in card.wrong_examples:
            lines.append(
                f"    {w['txn_id']:<14} {w['scenario']:<22} {w['tier']:<18} "
                f"expected {w['expected_rows']:>3} rows, took {w['matched_rows']:>3}, overlap {w['overlap']:>3}"
            )
    lines.append("")
    return "\n".join(lines)
