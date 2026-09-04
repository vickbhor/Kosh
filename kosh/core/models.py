"""Domain model for settlement reconciliation.

Money rule: every amount in this system is an ``int`` of paise. Rupee floats
appear only at the CSV boundary (parsed in ``normalize``) and at the display
boundary (``format_inr``). A float never touches a comparison, a sum, or a
stored value -- 0.1 + 0.2 != 0.3 is not an acceptable property for a ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class RowKind(str, Enum):
    """What a settlement report line represents."""

    PAYMENT = "payment"
    REFUND = "refund"
    ADJUSTMENT = "adjustment"


class Tier(str, Enum):
    """Which layer of the cascade resolved a match.

    Ordered cheapest to most expensive. ``EXCEPTION`` means nothing resolved it
    and a human must look.
    """

    EXACT = "tier1_exact"
    FUZZY = "tier2_fuzzy"
    SUBSET_SUM = "tier3_subset_sum"
    ADJUDICATED = "tier4_adjudicated"
    EXCEPTION = "tier5_exception"

DETERMINISTIC_TIERS = frozenset({Tier.EXACT, Tier.FUZZY, Tier.SUBSET_SUM})


class ExceptionType(str, Enum):
    """Typed failure reasons.

    An untyped 'could not match' pile is not actionable. Each of these maps to a
    different human workflow, so the queue is grouped by type in the UI.
    """

    ORPHAN_CREDIT = "orphan_credit"
    MISSING_IN_BANK = "missing_in_bank"
    AMOUNT_MISMATCH = "amount_mismatch"
    AMBIGUOUS_CANDIDATES = "ambiguous_candidates"
    DUPLICATE_UTR = "duplicate_utr"
    MALFORMED_ROW = "malformed_row"
    CHARGEBACK_DEBIT = "chargeback_debit"
    NO_CANDIDATE = "no_candidate"
    ARITHMETIC_UNSAFE = "arithmetic_unsafe"

EXCEPTION_ACTIONS: dict[ExceptionType, str] = {
    ExceptionType.ORPHAN_CREDIT: "Trace credit to a non-gateway source (direct transfer, refund reversal).",
    ExceptionType.MISSING_IN_BANK: "Settlement reported but not yet credited. Re-run after the next bank feed.",
    ExceptionType.AMOUNT_MISMATCH: "Candidate found but amounts differ beyond tolerance. Check fees and refunds.",
    ExceptionType.AMBIGUOUS_CANDIDATES: "Several equally valid matches. Needs a human decision.",
    ExceptionType.DUPLICATE_UTR: "Two settlements share one UTR. Confirm with the gateway.",
    ExceptionType.MALFORMED_ROW: "Row failed parsing and was quarantined. Fix the source file.",
    ExceptionType.CHARGEBACK_DEBIT: "Debit line, not a settlement credit. Route to the disputes workflow.",
    ExceptionType.NO_CANDIDATE: "No settlement rows in the window come close. Widen the window or check the report.",
    ExceptionType.ARITHMETIC_UNSAFE: "Pool too large for an exact sum to be evidence. Supply settlement ids or narrow the date window.",
}


@dataclass(frozen=True, slots=True)
class SettlementRow:
    """One line of the gateway settlement report.

    ``net_paise`` is what the merchant actually receives for this line: gross
    less fee less tax for a payment, and negative for a refund. Bank credits are
    matched against sums of this field, never against gross.
    """

    row_id: str
    kind: RowKind
    order_id: str | None
    gross_paise: int
    fee_paise: int
    tax_paise: int
    settled_on: date
    settlement_id: str | None = None
    method: str = "upi"

    utr: str | None = None

    @property
    def net_paise(self) -> int:
        if self.kind is RowKind.REFUND:
            return -abs(self.gross_paise)
        return self.gross_paise - self.fee_paise - self.tax_paise


@dataclass(frozen=True, slots=True)
class BankTxn:
    """One line of the bank statement.

    Exactly one of credit/debit is non-zero. ``narration`` is the free-text
    field the UTR has to be dug out of, and is deliberately messy.
    """

    txn_id: str
    value_date: date
    narration: str
    credit_paise: int = 0
    debit_paise: int = 0

    @property
    def is_credit(self) -> bool:
        return self.credit_paise > 0

    @property
    def signed_paise(self) -> int:
        return self.credit_paise - self.debit_paise


@dataclass(frozen=True, slots=True)
class Order:
    """Merchant-side order record."""

    order_id: str
    placed_on: date
    gross_paise: int
    customer_id: str


@dataclass(slots=True)
class Candidate:
    """A proposed match between a bank line and a set of settlement rows.

    Carries its own evidence so the adjudicator, the audit ledger and the UI all
    read from one structure rather than each re-deriving why a match was made.
    """

    row_ids: tuple[str, ...]
    net_paise: int
    delta_paise: int
    date_lag_days: int
    score: float
    evidence: dict[str, str] = field(default_factory=dict)

    @property
    def settlement_ids(self) -> tuple[str, ...]:
        return tuple(sorted({e for e in self.evidence.get("settlement_ids", "").split(",") if e}))


@dataclass(slots=True)
class MatchResult:
    """Outcome for a single bank line after the cascade has run."""

    txn_id: str
    tier: Tier
    row_ids: tuple[str, ...] = ()
    delta_paise: int = 0
    confidence: float = 0.0
    reason: str = ""
    exception_type: ExceptionType | None = None
    candidates_considered: int = 0
    llm_tokens: int = 0

    @property
    def matched(self) -> bool:
        return self.tier is not Tier.EXCEPTION


def to_paise(rupees: str | float | int) -> int:
    """Parse a rupee value into integer paise without going through binary float.

    ``float`` cannot represent 1234.35 exactly, so rounding it introduces a
    silent paise error that compounds across a 2,000-row batch. Parsing the
    decimal string directly avoids the whole class of bug.
    """
    text = str(rupees).strip().replace(",", "").replace("\u20b9", "")
    if not text or text in {"-", "nan", "None"}:
        return 0
    negative = text.startswith("-")
    text = text.lstrip("+-")
    if "." in text:
        whole, _, frac = text.partition(".")
        frac = (frac + "00")[:2]
    else:
        whole, frac = text, "00"
    value = int(whole or 0) * 100 + int(frac or 0)
    return -value if negative else value


def format_inr(paise: int) -> str:
    """Render paise as Indian-grouped rupees (1,23,456.78)."""
    sign = "-" if paise < 0 else ""
    paise = abs(paise)
    whole, frac = divmod(paise, 100)
    digits = str(whole)
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        digits = ",".join(parts + [tail])
    return f"{sign}\u20b9{digits}.{frac:02d}"
