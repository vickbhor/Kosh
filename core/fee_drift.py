"""Finding money the merchant was overcharged.

Reconciliation answers "does this credit tie out". It does not answer "should
the credit have been larger". Gateways charge a contracted MDR plus GST on the
fee, and a mis-set rate on one payment method leaks quietly for months because
every individual line looks plausible.

This runs on the same rows the cascade already parsed, so it costs nothing
extra, and it is deliberately conservative: a line is only flagged when its
effective rate exceeds the contract by more than a tolerance that already
absorbs integer rounding on both the fee and the GST.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from kosh.core.models import RowKind, SettlementRow

DEFAULT_TOLERANCE_BPS = 4


@dataclass(slots=True)
class DriftFinding:
    row_id: str
    method: str
    gross_paise: int
    charged_paise: int
    expected_paise: int
    effective_bps: int

    @property
    def excess_paise(self) -> int:
        return self.charged_paise - self.expected_paise


@dataclass(slots=True)
class DriftReport:
    contracted_bps: int
    gst_bps: int
    tolerance_bps: int
    findings: list[DriftFinding] = field(default_factory=list)
    rows_checked: int = 0

    @property
    def total_excess_paise(self) -> int:
        return sum(f.excess_paise for f in self.findings)

    @property
    def affected_methods(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for f in self.findings:
            out[f.method] += f.excess_paise
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def to_dict(self) -> dict:
        return {
            "contracted_bps": self.contracted_bps,
            "gst_bps": self.gst_bps,
            "rows_checked": self.rows_checked,
            "flagged": len(self.findings),
            "total_excess_paise": self.total_excess_paise,
            "by_method": self.affected_methods,
            "sample": [
                {
                    "row_id": f.row_id,
                    "method": f.method,
                    "gross_paise": f.gross_paise,
                    "charged_paise": f.charged_paise,
                    "expected_paise": f.expected_paise,
                    "excess_paise": f.excess_paise,
                    "effective_bps": f.effective_bps,
                }
                for f in sorted(self.findings, key=lambda x: -x.excess_paise)[:20]
            ],
        }


def detect_fee_drift(
    rows: list[SettlementRow],
    contracted_bps: int = 200,
    gst_bps: int = 1800,
    tolerance_bps: int = DEFAULT_TOLERANCE_BPS,
) -> DriftReport:
    """Flag payments charged above the contracted rate.

    ``effective_bps`` is computed from fee plus GST against gross, so a gateway
    that keeps the headline MDR but inflates the tax line is caught too.
    """
    report = DriftReport(contracted_bps=contracted_bps, gst_bps=gst_bps, tolerance_bps=tolerance_bps)
    ceiling = contracted_bps + tolerance_bps

    for row in rows:
        if row.kind is not RowKind.PAYMENT or row.gross_paise <= 0:
            continue
        report.rows_checked += 1

        charged = row.fee_paise + row.tax_paise
        expected_fee = row.gross_paise * contracted_bps // 10_000
        expected = expected_fee + expected_fee * gst_bps // 10_000

        fee_bps = row.fee_paise * 10_000 // row.gross_paise
        if fee_bps <= ceiling or charged <= expected:
            continue

        report.findings.append(
            DriftFinding(
                row_id=row.row_id,
                method=row.method,
                gross_paise=row.gross_paise,
                charged_paise=charged,
                expected_paise=expected,
                effective_bps=fee_bps,
            )
        )
    return report
