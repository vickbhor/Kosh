"""Exact subset-sum over settlement rows.

A bank credit is rarely one payment. It is the net of a batch, and when the
gateway drops the settlement id there is nothing to group on but the arithmetic
itself: find the set of rows whose net amounts sum to the credit exactly.

Implementation notes
--------------------
The DP is a bitset held in a single Python ``int``. Bit *k* is set when *k*
paise is reachable, and folding in one amount is ``m |= m << a`` -- one
C-level shift over the whole table rather than a Python loop over every state.
Masking back to ``target`` bits after each step keeps both memory and the shift
cost bounded.

Refunds are negative, which a bitset cannot represent, so negative rows are
handled by enumerating their subsets and solving the positive remainder for
each. Real batches carry very few refund lines, and the count is capped anyway.

Everything here is *bounded*. If a pool is too large or a target too big, the
solver returns nothing rather than spinning -- an unresolved row that reaches a
human beats a run that never finishes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

MAX_TARGET_PAISE = 100_000_000

MAX_POSITIVE_ROWS = 64

MAX_NEGATIVE_ROWS = 14


@dataclass(frozen=True, slots=True)
class SubsetSolution:
    """One exact-sum subset, as indices into the amounts list given."""

    indices: tuple[int, ...]

    def total(self, amounts: list[int]) -> int:
        return sum(amounts[i] for i in self.indices)


def _solve_positive(
    amounts: list[int],
    indices: list[int],
    target: int,
    forbidden: frozenset[int],
) -> tuple[int, ...] | None:
    """Find one subset of non-negative amounts summing exactly to target.

    ``forbidden`` names indices that may not be used, which is how the caller
    asks for a *different* solution than one it already has.
    """
    if target < 0:
        return None
    if target == 0:
        return ()

    usable = [i for i in indices if i not in forbidden and 0 < amounts[i] <= target]
    if not usable or sum(amounts[i] for i in usable) < target:
        return None

    limit_mask = (1 << (target + 1)) - 1

    masks: list[int] = [1]
    reach = 1
    for i in usable:
        reach = (reach | (reach << amounts[i])) & limit_mask
        masks.append(reach)
        if (reach >> target) & 1:

            usable = usable[: len(masks) - 1]
            break

    if not (masks[-1] >> target) & 1:
        return None

    chosen: list[int] = []
    remaining = target
    for k in range(len(masks) - 1, 0, -1):
        idx = usable[k - 1]
        if (masks[k - 1] >> remaining) & 1:
            continue
        chosen.append(idx)
        remaining -= amounts[idx]
    return tuple(sorted(chosen))


def solve_subset_sum(
    amounts: list[int],
    target: int,
    max_solutions: int = 2,
) -> list[SubsetSolution]:
    """Return up to ``max_solutions`` distinct exact-sum subsets.

    Finding a second solution matters as much as finding the first: two valid
    answers means the arithmetic cannot decide, and the caller must escalate
    rather than pick. Additional solutions are sought by forbidding one member
    of the first solution at a time, which is cheap and enough to prove a tie.
    """
    if not amounts or target <= 0 or target > MAX_TARGET_PAISE:
        return []

    positives = [i for i, a in enumerate(amounts) if a > 0]
    negatives = [i for i, a in enumerate(amounts) if a < 0]
    if len(positives) > MAX_POSITIVE_ROWS or len(negatives) > MAX_NEGATIVE_ROWS:
        return []

    solutions: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    negative_subsets: list[tuple[int, ...]] = [()]
    for size in range(1, len(negatives) + 1):
        negative_subsets.extend(combinations(negatives, size))

    for neg in negative_subsets:
        offset = -sum(amounts[i] for i in neg)
        sub_target = target + offset
        if sub_target > MAX_TARGET_PAISE:
            continue

        forbidden: frozenset[int] = frozenset()
        base = _solve_positive(amounts, positives, sub_target, forbidden)
        if base is None:
            continue

        candidate = tuple(sorted(base + neg))
        if candidate not in seen:
            seen.add(candidate)
            solutions.append(candidate)
        if len(solutions) >= max_solutions:
            break

        for excluded in base[: min(len(base), 8)]:
            alt = _solve_positive(amounts, positives, sub_target, frozenset({excluded}))
            if alt is None:
                continue
            candidate = tuple(sorted(alt + neg))
            if candidate not in seen:
                seen.add(candidate)
                solutions.append(candidate)
            if len(solutions) >= max_solutions:
                break
        if len(solutions) >= max_solutions:
            break

    return [SubsetSolution(s) for s in solutions[:max_solutions]]


def find_complement(amounts: list[int], pool_total: int, target: int, max_drop: int = 3) -> tuple[int, ...] | None:
    """Match by elimination: which few rows have to be *removed* to hit target.

    When a credit is nearly the whole pool -- a batch minus one held-back
    payment -- searching for the small excluded set is far cheaper than
    searching for the large included one. Tries drops of size 1, then 2, then 3.
    """
    deficit = pool_total - target
    if deficit < 0:
        return None
    if deficit == 0:
        return ()

    n = len(amounts)
    if n == 0:
        return None

    for i in range(n):
        if amounts[i] == deficit:
            return (i,)
    if max_drop >= 2:
        by_value: dict[int, int] = {}
        for i in range(n):
            need = deficit - amounts[i]
            if need in by_value:
                return tuple(sorted((by_value[need], i)))
            by_value[amounts[i]] = i
    if max_drop >= 3 and n <= 120:
        for i, j in combinations(range(n), 2):
            need = deficit - amounts[i] - amounts[j]
            for k in range(j + 1, n):
                if amounts[k] == need:
                    return (i, j, k)
    return None
