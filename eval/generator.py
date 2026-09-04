"""Synthetic settlement data with ground truth.

This module exists so accuracy claims are checkable. Every bank line it emits
carries a label saying which settlement rows *should* match it and which
scenario produced it, which is what lets the harness compute a real
false-match rate instead of eyeballing a demo.

The scenarios are drawn from failures that actually happen in Indian gateway
settlement: T+2 lag, UTRs missing from narration, batches split across two
credits, direct bank transfers that never went through the gateway, and MDR
being charged above the contracted rate.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from enum import Enum
from pathlib import Path

from kosh.core.models import BankTxn, Order, RowKind, SettlementRow

CONTRACTED_MDR_BPS = 200
GST_ON_FEE_BPS = 1800


class Scenario(str, Enum):
    """What was done to a settlement batch to make it hard."""

    CLEAN = "clean"
    UTR_MISSING = "utr_missing"
    SETTLEMENT_ID_MISSING = "settlement_id_missing"
    SPLIT_CREDIT = "split_credit"
    ORPHAN_CREDIT = "orphan_credit"
    MISSING_IN_BANK = "missing_in_bank"
    DUPLICATE_UTR = "duplicate_utr"
    ROUNDING_DRIFT = "rounding_drift"
    AMBIGUOUS = "ambiguous"
    CHARGEBACK = "chargeback"
    FEE_DRIFT = "fee_drift"

DEFAULT_WEIGHTS: dict[Scenario, float] = {
    Scenario.CLEAN: 0.44,
    Scenario.UTR_MISSING: 0.11,
    Scenario.SETTLEMENT_ID_MISSING: 0.11,
    Scenario.SPLIT_CREDIT: 0.05,
    Scenario.ORPHAN_CREDIT: 0.05,
    Scenario.MISSING_IN_BANK: 0.04,
    Scenario.DUPLICATE_UTR: 0.03,
    Scenario.ROUNDING_DRIFT: 0.05,
    Scenario.AMBIGUOUS: 0.03,
    Scenario.CHARGEBACK: 0.04,
    Scenario.FEE_DRIFT: 0.05,
}

NARRATION_WITH_UTR = [
    "NEFT-{utr}-RAZORPAY SOFTWARE PVT LTD-SETTLEMENT",
    "UPI/{utr}/RAZORPAY/SETTLE/NA",
    "IMPS {utr} RAZORPAYSOFT COLLECTION",
    "RTGS CR {utr} RAZORPAY SOFTWARE PRIVATE LIMITED",
    "NEFT CR-HDFC0000060-RAZORPAY SOFTWARE-{utr}",
]

NARRATION_NO_UTR = [
    "NEFT CR-RAZORPAY SOFTWARE PRIVATE LIMITED-SETTLEMENT",
    "FUND TRF FROM RAZORPAY SOFTWARE PVT LTD",
    "CR-CONSOLIDATED SETTLEMENT-GATEWAY",
]

ORPHAN_NARRATIONS = [
    "NEFT-{utr}-SHREE BALAJI TRADERS-INVOICE PAYMENT",
    "IMPS {utr} R VERMA PERSONAL TRANSFER",
    "UPI/{utr}/DIRECT/CUSTOMER PAYMENT",
]

METHODS = ["upi", "card", "netbanking", "wallet", "upi", "upi", "card"]


@dataclass
class TruthEntry:
    """Ground truth for one bank line."""

    txn_id: str
    scenario: str
    expected_row_ids: list[str] = field(default_factory=list)
    expect_exception: str | None = None
    note: str = ""


@dataclass
class Dataset:
    orders: list[Order]
    settlement_rows: list[SettlementRow]
    bank_txns: list[BankTxn]
    truth: dict[str, TruthEntry]
    malformed_settlement_lines: list[str] = field(default_factory=list)
    malformed_bank_lines: list[str] = field(default_factory=list)
    fee_drift_row_ids: list[str] = field(default_factory=list)


class DatasetGenerator:
    """Builds a labelled reconciliation dataset.

    ``difficulty`` scales how often a batch gets a hard scenario. At 0.0 every
    batch is clean, which is the sanity floor the test suite asserts against; at
    1.0 the weights above apply in full. The harness uses the spread between
    those two runs to show the cascade is doing work rather than getting lucky.
    """

    def __init__(
        self,
        seed: int = 7,
        n_batches: int = 40,
        start: date | None = None,
        difficulty: float = 1.0,
        weights: dict[Scenario, float] | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.n_batches = n_batches
        self.start = start or date(2026, 4, 1)
        self.difficulty = max(0.0, min(1.0, difficulty))
        self.weights = weights or DEFAULT_WEIGHTS
        self._seq = 0

    def _next(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}_{self._seq:06d}"

    def _utr(self) -> str:
        return f"{self.rng.randint(10**11, 10**12 - 1)}"

    def _pick_scenario(self) -> Scenario:
        if self.rng.random() > self.difficulty:
            return Scenario.CLEAN
        names = list(self.weights)
        return self.rng.choices(names, weights=[self.weights[n] for n in names], k=1)[0]

    def _make_payment(self, captured_on: date, settlement_id: str | None, utr: str | None, inflate_fee: bool) -> tuple[SettlementRow, Order]:
        """One captured payment plus the order behind it.

        Fee is computed from basis points on integers so the fee, the GST on the
        fee, and the net all stay exact. ``inflate_fee`` models the gateway
        charging above the contracted MDR, which the drift detector later finds.
        """
        gross = self.rng.randint(9900, 1_500_000)
        bps = CONTRACTED_MDR_BPS + (self.rng.randint(25, 60) if inflate_fee else 0)
        fee = gross * bps // 10_000
        tax = fee * GST_ON_FEE_BPS // 10_000
        order = Order(
            order_id=self._next("order"),
            placed_on=captured_on,
            gross_paise=gross,
            customer_id=f"cust_{self.rng.randint(1000, 9999)}",
        )
        row = SettlementRow(
            row_id=self._next("pay"),
            kind=RowKind.PAYMENT,
            order_id=order.order_id,
            gross_paise=gross,
            fee_paise=fee,
            tax_paise=tax,
            settled_on=captured_on + timedelta(days=2),
            settlement_id=settlement_id,
            method=self.rng.choice(METHODS),
            utr=utr,
        )
        return row, order

    def _make_refund(self, captured_on: date, settlement_id: str | None, utr: str | None, against: SettlementRow) -> SettlementRow:
        portion = self.rng.choice([1.0, 0.5, 0.25])
        amount = int(against.gross_paise * portion) // 100 * 100
        return SettlementRow(
            row_id=self._next("rfnd"),
            kind=RowKind.REFUND,
            order_id=against.order_id,
            gross_paise=max(amount, 100),
            fee_paise=0,
            tax_paise=0,
            settled_on=captured_on + timedelta(days=2),
            settlement_id=settlement_id,
            method=against.method,
            utr=utr,
        )

    def generate(self) -> Dataset:
        orders: list[Order] = []
        rows: list[SettlementRow] = []
        bank: list[BankTxn] = []
        truth: dict[str, TruthEntry] = {}
        drift_ids: list[str] = []
        used_utrs: list[str] = []

        for batch_i in range(self.n_batches):
            scenario = self._pick_scenario()
            captured_on = self.start + timedelta(days=batch_i)
            settled_on = captured_on + timedelta(days=2)
            settlement_id = self._next("setl")
            hide_attribution = scenario in (Scenario.SETTLEMENT_ID_MISSING, Scenario.AMBIGUOUS)
            stored_settlement_id = None if hide_attribution else settlement_id
            inflate = scenario is Scenario.FEE_DRIFT

            utr = self._utr()
            if scenario is Scenario.DUPLICATE_UTR and used_utrs:
                utr = self.rng.choice(used_utrs)
            used_utrs.append(utr)
            stored_utr = None if hide_attribution else utr

            batch_rows: list[SettlementRow] = []
            for _ in range(self.rng.randint(6, 28)):
                row, order = self._make_payment(captured_on, stored_settlement_id, stored_utr, inflate)
                batch_rows.append(row)
                orders.append(order)
                if inflate:
                    drift_ids.append(row.row_id)

            if self.rng.random() < 0.35 and batch_rows:
                target = self.rng.choice(batch_rows)
                batch_rows.append(self._make_refund(captured_on, stored_settlement_id, stored_utr, target))

            rows.extend(batch_rows)
            net_total = sum(r.net_paise for r in batch_rows)
            row_ids = [r.row_id for r in batch_rows]

            if scenario is Scenario.MISSING_IN_BANK:
                continue

            if scenario is Scenario.UTR_MISSING:
                narration = self.rng.choice(NARRATION_NO_UTR)
            else:
                narration = self.rng.choice(NARRATION_WITH_UTR).format(utr=utr)

            if scenario is Scenario.SPLIT_CREDIT and len(batch_rows) >= 4:
                self._emit_split(bank, truth, batch_rows, settled_on, utr, scenario)
            elif scenario is Scenario.ROUNDING_DRIFT:
                txn_id = self._next("bank")
                drift = self.rng.choice([-7, -3, -1, 1, 4, 9])
                bank.append(BankTxn(txn_id, settled_on, narration, credit_paise=net_total + drift))
                truth[txn_id] = TruthEntry(txn_id, scenario.value, row_ids, note=f"credit off by {drift}p")
            elif scenario is Scenario.AMBIGUOUS:
                self._emit_ambiguous(rows, bank, truth, batch_rows, settled_on, narration, scenario)
            else:
                txn_id = self._next("bank")
                lag = self.rng.choice([0, 0, 0, 1])
                bank.append(
                    BankTxn(txn_id, settled_on + timedelta(days=lag), narration, credit_paise=net_total)
                )
                truth[txn_id] = TruthEntry(txn_id, scenario.value, row_ids)

            if scenario is Scenario.ORPHAN_CREDIT:
                txn_id = self._next("bank")
                bank.append(
                    BankTxn(
                        txn_id,
                        settled_on,
                        self.rng.choice(ORPHAN_NARRATIONS).format(utr=self._utr()),
                        credit_paise=self.rng.randint(50_000, 900_000),
                    )
                )
                truth[txn_id] = TruthEntry(
                    txn_id, scenario.value, [], expect_exception="orphan_credit",
                    note="direct transfer, never went through the gateway",
                )

            if scenario is Scenario.CHARGEBACK:
                txn_id = self._next("bank")
                bank.append(
                    BankTxn(
                        txn_id,
                        settled_on,
                        f"CHARGEBACK DEBIT {self._utr()} RAZORPAY DISPUTE",
                        debit_paise=self.rng.randint(20_000, 400_000),
                    )
                )
                truth[txn_id] = TruthEntry(
                    txn_id, scenario.value, [], expect_exception="chargeback_debit"
                )

        dataset = Dataset(
            orders=orders,
            settlement_rows=rows,
            bank_txns=sorted(bank, key=lambda t: (t.value_date, t.txn_id)),
            truth=truth,
            fee_drift_row_ids=drift_ids,
        )
        self._inject_malformed(dataset)
        return dataset

    def _emit_split(self, bank, truth, batch_rows, settled_on, utr, scenario) -> None:
        """One settlement paid out as two bank credits on consecutive days."""
        half = len(batch_rows) // 2
        first, second = batch_rows[:half], batch_rows[half:]
        for idx, part in enumerate((first, second)):
            txn_id = self._next("bank")
            bank.append(
                BankTxn(
                    txn_id,
                    settled_on + timedelta(days=idx),
                    f"NEFT-{utr}-RAZORPAY SOFTWARE PVT LTD-SETTLEMENT PART{idx + 1}",
                    credit_paise=sum(r.net_paise for r in part),
                )
            )
            truth[txn_id] = TruthEntry(
                txn_id, scenario.value, [r.row_id for r in part], note=f"part {idx + 1} of 2"
            )

    def _emit_ambiguous(self, all_rows, bank, truth, batch_rows, settled_on, narration, scenario) -> None:
        """Construct a genuine tie: a decoy set with an identical net total.

        Two ordinary payments are drawn at random, then an adjustment line
        (fee-free, so its net equals its gross) closes the gap to the exact
        total. Both subsets are now arithmetically valid answers, so no amount
        of deterministic cleverness can separate them. This is the case the
        adjudicator exists for, and the case where refusing to guess is right.
        """
        net_total = sum(r.net_paise for r in batch_rows)
        decoy: list[SettlementRow] = []
        accumulated = 0

        for _ in range(self.rng.randint(5, 8)):
            gross = self.rng.randint(max(net_total // 20, 10_000), max(net_total // 9, 20_000))
            fee = gross * CONTRACTED_MDR_BPS // 10_000
            tax = fee * GST_ON_FEE_BPS // 10_000
            row = SettlementRow(
                row_id=self._next("pay"),
                kind=RowKind.PAYMENT,
                order_id=None,
                gross_paise=gross,
                fee_paise=fee,
                tax_paise=tax,
                settled_on=settled_on,
                settlement_id=None,
                method="card",
            )
            decoy.append(row)
            accumulated += row.net_paise

        remainder = net_total - accumulated
        decoy.append(
            SettlementRow(
                row_id=self._next("adjs"),
                kind=RowKind.ADJUSTMENT,
                order_id=None,
                gross_paise=remainder,
                fee_paise=0,
                tax_paise=0,
                settled_on=settled_on,
                settlement_id=None,
                method="na",
            )
        )
        all_rows.extend(decoy)

        txn_id = self._next("bank")
        bank.append(BankTxn(txn_id, settled_on, narration, credit_paise=net_total))
        truth[txn_id] = TruthEntry(
            txn_id,
            scenario.value,
            [r.row_id for r in batch_rows],
            note="a decoy subset sums to the same total",
        )

    def _inject_malformed(self, dataset: Dataset) -> None:
        """Rows that break the parser, appended as raw text.

        These never appear in ground truth: the correct behaviour is to
        quarantine them and keep the run going, not to match them.
        """
        dataset.malformed_settlement_lines = [
            "pay_BAD001,payment,order_x,not-a-number,120.00,21.60,2026-04-05,setl_000001,upi,321456789012",
            "pay_BAD002,payment,order_y,4500.00,90.00,16.20,31-02-2026,setl_000001,card,321456789013",
            "pay_BAD003,payment",
        ]
        dataset.malformed_bank_lines = [
            "bank_BAD001,2026-13-45,GARBAGE NARRATION,1000.00,0.00",
            ",,,,",
        ]


def write_dataset(dataset: Dataset, out_dir: Path) -> Path:
    """Write CSVs plus ground truth, with malformed rows spliced into the CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def rupees(paise: int) -> str:
        return f"{paise // 100}.{abs(paise) % 100:02d}"

    lines = ["row_id,kind,order_id,gross,fee,tax,settled_on,settlement_id,method,utr"]
    for r in dataset.settlement_rows:
        lines.append(
            ",".join(
                [
                    r.row_id,
                    r.kind.value,
                    r.order_id or "",
                    rupees(r.gross_paise),
                    rupees(r.fee_paise),
                    rupees(r.tax_paise),
                    r.settled_on.isoformat(),
                    r.settlement_id or "",
                    r.method,
                    r.utr or "",
                ]
            )
        )
    lines.extend(dataset.malformed_settlement_lines)
    (out_dir / "settlement_report.csv").write_text("\n".join(lines) + "\n")

    lines = ["txn_id,value_date,narration,credit,debit"]
    for t in dataset.bank_txns:
        lines.append(
            ",".join(
                [
                    t.txn_id,
                    t.value_date.isoformat(),
                    '"' + t.narration.replace('"', "'") + '"',
                    rupees(t.credit_paise),
                    rupees(t.debit_paise),
                ]
            )
        )
    lines.extend(dataset.malformed_bank_lines)
    (out_dir / "bank_statement.csv").write_text("\n".join(lines) + "\n")

    lines = ["order_id,placed_on,gross,customer_id"]
    for o in dataset.orders:
        lines.append(",".join([o.order_id, o.placed_on.isoformat(), rupees(o.gross_paise), o.customer_id]))
    (out_dir / "orders.csv").write_text("\n".join(lines) + "\n")

    (out_dir / "ground_truth.json").write_text(
        json.dumps(
            {
                "entries": [asdict(e) for e in dataset.truth.values()],
                "fee_drift_row_ids": dataset.fee_drift_row_ids,
            },
            indent=2,
        )
    )
    return out_dir
