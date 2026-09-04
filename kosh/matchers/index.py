"""Lookup structures over the settlement report, plus claim tracking.

A settlement row belongs to at most one bank credit. Once a tier claims a row
it leaves the available pool, which is what stops a cheap low-confidence match
from stealing rows a later, higher-confidence match needed.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

from kosh.core.models import SettlementRow


class SettlementIndex:
    """Indexed, claimable view of the settlement report."""

    def __init__(self, rows: list[SettlementRow]) -> None:
        self.rows: dict[str, SettlementRow] = {r.row_id: r for r in rows}
        self._available: set[str] = set(self.rows)

        self.group_rows: dict[str, list[str]] = defaultdict(list)
        self.settlement_ids_by_utr: dict[str, set[str]] = defaultdict(set)
        self.unattributed_by_date: dict[date, list[str]] = defaultdict(list)
        self.group_date: dict[str, date] = {}

        for r in rows:
            if r.settlement_id:
                self.group_rows[r.settlement_id].append(r.row_id)
                self.group_date.setdefault(r.settlement_id, r.settled_on)
                if r.utr:
                    self.settlement_ids_by_utr[r.utr].add(r.settlement_id)
            else:
                self.unattributed_by_date[r.settled_on].append(r.row_id)

    @property
    def available_ids(self) -> set[str]:
        return self._available

    def is_available(self, row_id: str) -> bool:
        return row_id in self._available

    def claim(self, row_ids: tuple[str, ...] | list[str]) -> None:
        self._available.difference_update(row_ids)

    def release(self, row_ids: tuple[str, ...] | list[str]) -> None:
        """Undo a claim. Used when a run is rolled back mid-batch."""
        self._available.update(r for r in row_ids if r in self.rows)

    def net_of(self, row_ids: tuple[str, ...] | list[str]) -> int:
        return sum(self.rows[r].net_paise for r in row_ids)

    def available_group(self, settlement_id: str) -> tuple[str, ...]:
        """Rows of a settlement batch that nothing has claimed yet."""
        return tuple(r for r in self.group_rows.get(settlement_id, ()) if r in self._available)

    def groups_in_window(self, centre: date, window_days: int) -> list[str]:
        """Settlement batches settling within +/- window_days of a date.

        Bank value dates drift from the gateway's settled_on by a day or two,
        so the window is what keeps an otherwise exact amount match honest.
        """
        lo, hi = centre - timedelta(days=window_days), centre + timedelta(days=window_days)
        return [sid for sid, d in self.group_date.items() if lo <= d <= hi and self.available_group(sid)]

    def unattributed_in_window(self, centre: date, window_days: int) -> dict[date, list[str]]:
        """Available rows with no settlement id, grouped by settle date.

        Grouped rather than flattened on purpose: a batch settles on one day, so
        the day grouping is a structural prior that keeps the subset-sum pool
        small enough for its answers to mean something.
        """
        out: dict[date, list[str]] = {}
        for offset in range(-window_days, window_days + 1):
            day = centre + timedelta(days=offset)
            ids = [r for r in self.unattributed_by_date.get(day, ()) if r in self._available]
            if ids:
                out[day] = ids
        return out
