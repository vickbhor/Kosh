"""Parsing and canonicalisation of the three input files.

Design rule: a bad row must never abort a run. Finance files arrive truncated,
double-encoded, and with dates in three formats in the same column. Each row is
parsed independently and failures are quarantined with the reason and the raw
text attached, so the operator can fix the source while the other 1,999 rows
still reconcile.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from kosh.core.models import BankTxn, Order, RowKind, SettlementRow, to_paise

UTR_PATTERN = re.compile(r"(?<!\d)(\d{12}|\d{16})(?!\d)")

DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%b-%Y", "%m/%d/%Y")


@dataclass(slots=True)
class Quarantined:
    """A row that could not be parsed, kept with enough context to fix it."""

    source: str
    line_no: int
    raw: str
    reason: str


@dataclass(slots=True)
class ParseResult:
    orders: list[Order]
    settlement_rows: list[SettlementRow]
    bank_txns: list[BankTxn]
    quarantined: list[Quarantined]

    @property
    def total_rows(self) -> int:
        return len(self.settlement_rows) + len(self.bank_txns) + len(self.orders)


def parse_date(value: str) -> date:
    """Accept the date formats that show up in Indian bank exports."""
    text = (value or "").strip()
    if not text:
        raise ValueError("empty date")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unparseable date: {text!r}")


def extract_utrs(narration: str) -> list[str]:
    """Pull every candidate bank reference out of a free-text narration.

    Returns all matches rather than the first: some narrations carry both the
    settlement UTR and an unrelated IFSC-adjacent number, and letting the
    matcher test each one is cheaper than guessing here.
    """
    return UTR_PATTERN.findall(narration or "")


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[tuple[int, str]]]:
    """Read a CSV, returning parsed dicts and rows with the wrong column count."""
    text = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return [], []
    width = len(header)
    good: list[dict[str, str]] = []
    bad: list[tuple[int, str]] = []
    for line_no, row in enumerate(reader, start=2):
        if not any(cell.strip() for cell in row):
            bad.append((line_no, ",".join(row)))
            continue
        if len(row) != width:
            bad.append((line_no, ",".join(row)))
            continue
        good.append(dict(zip(header, row)))
    return good, bad


def load_settlement_report(path: Path) -> tuple[list[SettlementRow], list[Quarantined]]:
    rows, bad = _read_rows(path)
    out: list[SettlementRow] = []
    quarantine = [Quarantined("settlement_report", n, raw, "wrong column count") for n, raw in bad]
    for i, r in enumerate(rows, start=2):
        try:
            gross = to_paise(r["gross"])
            if gross == 0 and r["gross"].strip() not in {"0", "0.00"}:
                raise ValueError(f"unparseable amount: {r['gross']!r}")
            out.append(
                SettlementRow(
                    row_id=r["row_id"].strip(),
                    kind=RowKind(r["kind"].strip()),
                    order_id=r["order_id"].strip() or None,
                    gross_paise=gross,
                    fee_paise=to_paise(r["fee"]),
                    tax_paise=to_paise(r["tax"]),
                    settled_on=parse_date(r["settled_on"]),
                    settlement_id=r["settlement_id"].strip() or None,
                    method=r["method"].strip() or "unknown",
                    utr=(r.get("utr") or "").strip() or None,
                )
            )
        except (KeyError, ValueError) as exc:
            quarantine.append(Quarantined("settlement_report", i, ",".join(r.values()), str(exc)))
    return out, quarantine


def load_bank_statement(path: Path) -> tuple[list[BankTxn], list[Quarantined]]:
    rows, bad = _read_rows(path)
    out: list[BankTxn] = []
    quarantine = [Quarantined("bank_statement", n, raw, "wrong column count") for n, raw in bad]
    for i, r in enumerate(rows, start=2):
        try:
            txn_id = r["txn_id"].strip()
            if not txn_id:
                raise ValueError("missing txn_id")
            credit, debit = to_paise(r["credit"]), to_paise(r["debit"])
            if credit and debit:
                raise ValueError("row has both credit and debit")
            out.append(
                BankTxn(
                    txn_id=txn_id,
                    value_date=parse_date(r["value_date"]),
                    narration=r["narration"].strip(),
                    credit_paise=credit,
                    debit_paise=debit,
                )
            )
        except (KeyError, ValueError) as exc:
            quarantine.append(Quarantined("bank_statement", i, ",".join(r.values()), str(exc)))
    return out, quarantine


def load_orders(path: Path) -> tuple[list[Order], list[Quarantined]]:
    if not path.exists():
        return [], []
    rows, bad = _read_rows(path)
    out: list[Order] = []
    quarantine = [Quarantined("orders", n, raw, "wrong column count") for n, raw in bad]
    for i, r in enumerate(rows, start=2):
        try:
            out.append(
                Order(
                    order_id=r["order_id"].strip(),
                    placed_on=parse_date(r["placed_on"]),
                    gross_paise=to_paise(r["gross"]),
                    customer_id=r["customer_id"].strip(),
                )
            )
        except (KeyError, ValueError) as exc:
            quarantine.append(Quarantined("orders", i, ",".join(r.values()), str(exc)))
    return out, quarantine


def load_directory(path: Path) -> ParseResult:
    """Load a whole dataset directory, collecting quarantine across all files."""
    rows, q1 = load_settlement_report(path / "settlement_report.csv")
    bank, q2 = load_bank_statement(path / "bank_statement.csv")
    orders, q3 = load_orders(path / "orders.csv")
    return ParseResult(orders=orders, settlement_rows=rows, bank_txns=bank, quarantined=q1 + q2 + q3)
