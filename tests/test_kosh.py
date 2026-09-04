"""Tests for the reconciliation engine.

The suite is organised around the properties that have to hold rather than
around the modules. The most important ones are negative: the engine must not
invent a match, must not trust arithmetic it cannot justify, and must not let a
model name a row it was never shown.
"""

from __future__ import annotations

from datetime import date

import pytest

from kosh.core.adjudicator import (
    AdjudicationRequest,
    CandidateView,
    OfflineAdjudicator,
    _validate,
)
from kosh.core.fee_drift import detect_fee_drift
from kosh.core.ledger import Ledger, fingerprint_files
from kosh.core.models import (
    BankTxn,
    ExceptionType,
    RowKind,
    SettlementRow,
    Tier,
    format_inr,
    to_paise,
)
from kosh.core.normalize import extract_utrs, parse_date
from kosh.eval.generator import DatasetGenerator
from kosh.eval.harness import score
from kosh.matchers.cascade import Cascade, ReconcileConfig, collision_risk
from kosh.matchers.subset_sum import find_complement, solve_subset_sum


class TestMoney:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("1234.35", 123435),
            ("1,23,456.78", 12345678),
            ("0.01", 1),
            ("-45.60", -4560),
            ("100", 10000),
            ("100.5", 10050),
            ("\u20b9 2,000.00", 200000),
            ("", 0),
        ],
    )
    def test_parses_without_float(self, text, expected):
        assert to_paise(text) == expected

    def test_float_path_would_have_been_wrong(self):
        """The reason parsing goes via the decimal string, made explicit."""
        assert round(1234.35 * 100) == 123435
        assert int(8.70 * 100) == 869
        assert to_paise("8.70") == 870

    @pytest.mark.parametrize(
        "paise,expected",
        [(12345678, "\u20b91,23,456.78"), (100, "\u20b91.00"), (-4560, "-\u20b945.60")],
    )
    def test_indian_grouping(self, paise, expected):
        assert format_inr(paise) == expected

    def test_net_of_refund_is_negative(self):
        row = SettlementRow("r1", RowKind.REFUND, None, 5000, 0, 0, date(2026, 4, 1))
        assert row.net_paise == -5000

    def test_net_of_payment_deducts_fee_and_tax(self):
        row = SettlementRow("r1", RowKind.PAYMENT, "o1", 100000, 2000, 360, date(2026, 4, 1))
        assert row.net_paise == 97640


class TestParsing:
    @pytest.mark.parametrize(
        "text", ["2026-04-01", "01-04-2026", "01/04/2026", "2026/04/01", "01-Apr-2026"]
    )
    def test_accepts_common_bank_formats(self, text):
        assert parse_date(text) == date(2026, 4, 1)

    def test_rejects_impossible_date(self):
        with pytest.raises(ValueError):
            parse_date("31-02-2026")

    def test_extracts_twelve_and_sixteen_digit_references(self):
        assert extract_utrs("NEFT-123456789012-RAZORPAY") == ["123456789012"]
        assert extract_utrs("RTGS 1234567890123456 RZP") == ["1234567890123456"]

    def test_ignores_short_numbers(self):
        """Pincodes and account fragments must not be mistaken for UTRs."""
        assert extract_utrs("BRANCH 250001 ACCT 98765") == []


class TestSubsetSum:
    def test_finds_exact_subset(self):
        sols = solve_subset_sum([100, 200, 300, 400], 700, max_solutions=1)
        assert sols and sum([100, 200, 300, 400][i] for i in sols[0].indices) == 700

    def test_returns_nothing_when_impossible(self):
        assert solve_subset_sum([100, 200], 55) == []

    def test_handles_refunds_as_negatives(self):
        amounts = [50000, 30000, 20000, -10000]
        sols = solve_subset_sum(amounts, 90000, max_solutions=1)
        assert sols and sols[0].total(amounts) == 90000

    def test_detects_a_genuine_tie(self):
        """Two valid answers must both be reported so the caller can refuse."""
        sols = solve_subset_sum([1000, 2000, 3000], 3000, max_solutions=2)
        assert len(sols) == 2

    def test_refuses_targets_beyond_the_bound(self):
        assert solve_subset_sum([1, 2, 3], 10**12) == []

    def test_complement_finds_the_dropped_rows(self):
        amounts = [500, 700, 900, 1100, 1900]
        total = sum(amounts)
        assert find_complement(amounts, total, total - 700 - 1900) == (1, 4)

    def test_complement_answers_are_not_always_unique(self):
        """Elimination collides too, just far less often than full subset-sum.

        900+1100 and 700+1300 both sum to 2000, so asking "which rows were held
        back" can have several right answers. The complement search is trusted
        anyway because its hypothesis space is about n**3/6 rather than 2**n,
        which is what keeps its collision risk orders of magnitude lower.
        """
        amounts = [500, 700, 900, 1100, 1300]
        total = sum(amounts)
        drop = find_complement(amounts, total, total - 2000)
        assert drop is not None
        assert sum(amounts[i] for i in drop) == 2000

    def test_complement_returns_empty_tuple_when_nothing_dropped(self):
        amounts = [10, 20]
        assert find_complement(amounts, 30, 30) == ()


class TestCollisionRisk:
    def test_small_search_space_is_safe(self):

        assert collision_risk(4525, 10_000_000) < 1e-3

    def test_unconstrained_subset_sum_is_not(self):
        assert collision_risk(2**28, 10_000_000) == 1.0

    def test_spurious_matches_become_certain_as_pools_grow(self):
        """The measured property the whole tier-3 design is built around."""
        import random

        def alternates(n: int, trials: int = 20) -> int:
            rng = random.Random(4)
            hits = 0
            for _ in range(trials):
                amounts = [rng.randint(9900, 1_500_000) for _ in range(n)]
                target = sum(amounts[: n // 2])
                if len(solve_subset_sum(amounts, target, max_solutions=2)) > 1:
                    hits += 1
            return hits

        assert alternates(12) == 0
        assert alternates(28) >= 15


def _request(n: int = 2) -> AdjudicationRequest:
    return AdjudicationRequest(
        txn_id="bank_1",
        value_date="2026-04-03",
        narration="NEFT-123456789012-RAZORPAY",
        credit_paise=500000,
        candidates=[
            CandidateView(f"cand_{i}", 5, 500000, i * 10, 0, "2026-04-01", "tier3_subset_sum")
            for i in range(n)
        ],
    )


class TestAdjudicatorSafety:
    def test_unknown_candidate_id_is_treated_as_abstention(self):
        """The guarantee that the model cannot invent a match, enforced in code."""
        resp = _validate({"candidate_id": "cand_does_not_exist", "confidence": 0.99}, _request(), 0)
        assert resp.candidate_id is None
        assert resp.degraded

    def test_non_string_candidate_is_rejected(self):
        resp = _validate({"candidate_id": 7}, _request(), 0)
        assert resp.candidate_id is None

    def test_confidence_is_clamped(self):
        resp = _validate({"candidate_id": "cand_0", "confidence": 5.0}, _request(), 0)
        assert resp.confidence == 1.0

    def test_garbage_confidence_does_not_raise(self):
        resp = _validate({"candidate_id": "cand_0", "confidence": "very"}, _request(), 0)
        assert resp.confidence == 0.0

    def test_valid_choice_passes_through(self):
        resp = _validate({"candidate_id": "cand_1", "confidence": 0.8, "reason": "closer"}, _request(), 12)
        assert resp.candidate_id == "cand_1"
        assert resp.tokens == 12

    def test_offline_adjudicator_abstains_on_a_tie(self):
        req = _request(2)
        req.candidates[0].delta_paise = 0
        req.candidates[1].delta_paise = 0
        assert OfflineAdjudicator().adjudicate(req).candidate_id is None

    def test_offline_adjudicator_picks_a_clear_winner(self):
        req = _request(2)
        req.candidates[0].delta_paise = 0
        req.candidates[1].delta_paise = 90000
        assert OfflineAdjudicator().adjudicate(req).candidate_id == "cand_0"


def _batch(n: int, day: date, settlement_id: str | None, utr: str | None, start: int = 0):
    rows = []
    for i in range(n):
        gross = 100000 + i * 1000
        fee = gross * 200 // 10000
        tax = fee * 1800 // 10000
        rows.append(
            SettlementRow(
                row_id=f"pay_{start + i:04d}",
                kind=RowKind.PAYMENT,
                order_id=f"o{start + i}",
                gross_paise=gross,
                fee_paise=fee,
                tax_paise=tax,
                settled_on=day,
                settlement_id=settlement_id,
                utr=utr,
            )
        )
    return rows


class TestCascade:
    def test_exact_utr_match(self):
        rows = _batch(5, date(2026, 4, 1), "setl_1", "123456789012")
        total = sum(r.net_paise for r in rows)
        txn = BankTxn("b1", date(2026, 4, 1), "NEFT-123456789012-RAZORPAY", credit_paise=total)
        run = Cascade(rows).run([txn])
        assert run.results[0].tier is Tier.EXACT
        assert set(run.results[0].row_ids) == {r.row_id for r in rows}

    def test_matches_without_a_utr_in_the_narration(self):
        rows = _batch(5, date(2026, 4, 1), "setl_1", "123456789012")
        total = sum(r.net_paise for r in rows)
        txn = BankTxn("b1", date(2026, 4, 2), "FUND TRF FROM RAZORPAY SOFTWARE", credit_paise=total)
        assert Cascade(rows).run([txn]).results[0].tier is Tier.FUZZY

    def test_rebuilds_a_batch_with_no_settlement_id(self):
        rows = _batch(5, date(2026, 4, 1), None, None)
        total = sum(r.net_paise for r in rows)
        txn = BankTxn("b1", date(2026, 4, 1), "CR RAZORPAY SETTLEMENT", credit_paise=total)
        assert Cascade(rows).run([txn]).results[0].tier is Tier.SUBSET_SUM

    def test_split_payout_resolves_across_iterations(self):
        """Needs the fixpoint: the second half only matches once the first claims."""
        rows = _batch(6, date(2026, 4, 1), "setl_1", "123456789012")
        first = sum(r.net_paise for r in rows[:3])
        second = sum(r.net_paise for r in rows[3:])
        txns = [
            BankTxn("b1", date(2026, 4, 1), "NEFT-123456789012-RAZORPAY PART1", credit_paise=first),
            BankTxn("b2", date(2026, 4, 2), "NEFT-123456789012-RAZORPAY PART2", credit_paise=second),
        ]
        run = Cascade(rows).run(txns)
        assert all(r.matched for r in run.results)
        claimed = [set(r.row_ids) for r in run.results]
        assert claimed[0].isdisjoint(claimed[1]), "a row was claimed twice"

    def test_a_row_is_never_claimed_by_two_credits(self):
        rows = _batch(20, date(2026, 4, 1), None, None)
        total = sum(r.net_paise for r in rows)
        txns = [
            BankTxn("b1", date(2026, 4, 1), "CR RAZORPAY", credit_paise=total),
            BankTxn("b2", date(2026, 4, 1), "CR RAZORPAY", credit_paise=total),
        ]
        run = Cascade(rows).run(txns)
        seen: set[str] = set()
        for r in run.results:
            assert seen.isdisjoint(r.row_ids)
            seen.update(r.row_ids)

    def test_orphan_credit_is_typed(self):
        rows = _batch(3, date(2026, 4, 1), "setl_1", "123456789012")
        txn = BankTxn("b1", date(2026, 4, 1), "IMPS 999888777666 R VERMA PERSONAL", credit_paise=777)
        result = Cascade(rows).run([txn]).results[0]
        assert result.exception_type is ExceptionType.ORPHAN_CREDIT

    def test_debit_is_routed_to_disputes(self):
        rows = _batch(3, date(2026, 4, 1), "setl_1", "123456789012")
        txn = BankTxn("b1", date(2026, 4, 1), "CHARGEBACK DEBIT RAZORPAY", debit_paise=5000)
        result = Cascade(rows).run([txn]).results[0]
        assert result.exception_type is ExceptionType.CHARGEBACK_DEBIT

    def test_declines_rather_than_guesses_on_a_large_pool(self):
        """Above the safe pool size, an exact sum is not evidence."""
        rows = _batch(30, date(2026, 4, 1), None, None)
        target = sum(r.net_paise for r in rows[:14])
        txn = BankTxn("b1", date(2026, 4, 1), "CR RAZORPAY SETTLEMENT", credit_paise=target)
        result = Cascade(rows, ReconcileConfig(subset_sum_safe_n=20)).run([txn]).results[0]
        assert not result.matched
        assert result.exception_type is ExceptionType.ARITHMETIC_UNSAFE

    def test_duplicate_utr_does_not_produce_a_confident_match(self):
        a = _batch(4, date(2026, 4, 1), "setl_1", "123456789012", start=0)
        b = _batch(4, date(2026, 4, 1), "setl_2", "123456789012", start=100)
        total = sum(r.net_paise for r in a)
        txn = BankTxn("b1", date(2026, 4, 1), "NEFT-123456789012-RAZORPAY", credit_paise=total)
        result = Cascade(a + b).run([txn]).results[0]

        assert result.tier is not Tier.EXACT


class TestLedger:
    def test_chain_is_intact_when_untouched(self, tmp_path):
        ledger = Ledger(tmp_path / "l.db")
        ledger.start_run("run1", "fp1", {})
        ledger.append_many("run1", [("match", f"b{i}", {"i": i}) for i in range(10)])
        assert ledger.verify_chain() == []

    def test_tampering_is_detected(self, tmp_path):
        ledger = Ledger(tmp_path / "l.db")
        ledger.start_run("run1", "fp1", {})
        ledger.append_many("run1", [("match", f"b{i}", {"amount": i}) for i in range(10)])
        ledger.conn.execute("UPDATE entries SET payload_json = ? WHERE seq = 5", ('{"amount": 999}',))
        ledger.conn.commit()
        breaks = ledger.verify_chain()
        assert breaks and breaks[0].seq == 5

    def test_identical_inputs_are_not_reprocessed(self, tmp_path):
        ledger = Ledger(tmp_path / "l.db")
        assert ledger.start_run("run1", "fp1", {}) is None
        assert ledger.start_run("run2", "fp1", {}) == "run1"

    def test_fingerprint_ignores_filename(self, tmp_path):
        a, b = tmp_path / "bank.csv", tmp_path / "bank (1).csv"
        a.write_text("same")
        b.write_text("same")
        assert fingerprint_files([a]) == fingerprint_files([b])


class TestFeeDrift:
    def test_flags_an_overcharged_payment(self):
        gross = 1_000_000
        overfee = gross * 250 // 10000
        row = SettlementRow(
            "r1", RowKind.PAYMENT, "o1", gross, overfee, overfee * 1800 // 10000, date(2026, 4, 1)
        )
        report = detect_fee_drift([row], contracted_bps=200)
        assert report.findings and report.total_excess_paise > 0

    def test_leaves_contracted_rate_alone(self):
        gross = 1_000_000
        fee = gross * 200 // 10000
        row = SettlementRow("r1", RowKind.PAYMENT, "o1", gross, fee, fee * 1800 // 10000, date(2026, 4, 1))
        assert detect_fee_drift([row], contracted_bps=200).findings == []

    def test_ignores_refunds(self):
        row = SettlementRow("r1", RowKind.REFUND, "o1", 5000, 0, 0, date(2026, 4, 1))
        assert detect_fee_drift([row]).rows_checked == 0


class TestEndToEnd:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_never_produces_a_false_match(self, seed):
        """The property that matters most, checked on every seed."""
        ds = DatasetGenerator(seed=seed, n_batches=60).generate()
        run = Cascade(ds.settlement_rows, ReconcileConfig(), OfflineAdjudicator()).run(ds.bank_txns)
        card = score(run, ds.truth, settlement_rows=len(ds.settlement_rows))
        assert card.wrong_matches == 0, card.wrong_examples

    def test_clean_data_reconciles_completely(self):
        ds = DatasetGenerator(seed=3, n_batches=40, difficulty=0.0).generate()
        run = Cascade(ds.settlement_rows, ReconcileConfig(), OfflineAdjudicator()).run(ds.bank_txns)
        card = score(run, ds.truth, settlement_rows=len(ds.settlement_rows))
        assert card.auto_match_rate == 1.0
        assert card.false_match_rate == 0.0

    def test_recall_stays_high_on_hard_data(self):
        ds = DatasetGenerator(seed=5, n_batches=100).generate()
        run = Cascade(ds.settlement_rows, ReconcileConfig(), OfflineAdjudicator()).run(ds.bank_txns)
        card = score(run, ds.truth, settlement_rows=len(ds.settlement_rows))
        assert card.recall > 0.90

    def test_disabling_the_model_still_reconciles(self):
        """The deterministic floor. Most of the work never needed a model."""
        ds = DatasetGenerator(seed=6, n_batches=80).generate()
        config = ReconcileConfig(enable_adjudication=False)
        run = Cascade(ds.settlement_rows, config, OfflineAdjudicator()).run(ds.bank_txns)
        card = score(run, ds.truth, settlement_rows=len(ds.settlement_rows))
        assert card.llm_calls == 0
        assert card.auto_match_rate > 0.85
        assert card.false_match_rate == 0.0
