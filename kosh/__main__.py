"""Command line entry point.

    python -m kosh generate --batches 120
    python -m kosh run --data data/demo
    python -m kosh sweep --seeds 12          # the honest one
    python -m kosh explain --data data/demo --txn bank_000042
    python -m kosh chaos --data data/demo
    python -m kosh verify --data data/demo
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from kosh.core.adjudicator import OfflineAdjudicator
from kosh.core.ledger import Ledger
from kosh.core.models import format_inr
from kosh.eval.generator import DatasetGenerator, write_dataset
from kosh.eval.harness import Stopwatch, render, score
from kosh.matchers.cascade import Cascade, ReconcileConfig
from kosh.pipeline import ChaosConfig, ai_budget, explain, reconcile_directory


def cmd_generate(args) -> int:
    ds = DatasetGenerator(seed=args.seed, n_batches=args.batches, difficulty=args.difficulty).generate()
    out = write_dataset(ds, Path(args.out))
    print(f"wrote {len(ds.settlement_rows):,} settlement rows and {len(ds.bank_txns):,} bank lines to {out}")
    print(f"ground truth for {len(ds.truth):,} bank lines, {len(ds.fee_drift_row_ids):,} rows carry fee drift")
    return 0


def _scored_run(seed: int, batches: int, difficulty: float, config: ReconcileConfig):
    ds = DatasetGenerator(seed=seed, n_batches=batches, difficulty=difficulty).generate()
    cascade = Cascade(ds.settlement_rows, config, OfflineAdjudicator())
    with Stopwatch() as sw:
        run = cascade.run(ds.bank_txns)
    return score(run, ds.truth, settlement_rows=len(ds.settlement_rows), wall_seconds=sw.seconds)


def cmd_run(args) -> int:
    data = Path(args.data)
    config = ReconcileConfig(enable_adjudication=not args.no_llm)
    report = reconcile_directory(data, config=config, enable_llm=not args.no_llm)
    summary = report.summary()

    if report.reused_existing:
        print(f"identical inputs already reconciled as run {report.run_id}; nothing reposted")
        print(json.dumps(summary, indent=2, default=str))
        return 0

    print(json.dumps(summary, indent=2, default=str))
    print("\n  AI budget")
    for k, v in ai_budget(report).items():
        print(f"    {k:<28}{v}")

    drift = report.drift
    if drift.get("flagged"):
        print(
            f"\n  fee drift: {drift['flagged']:,} of {drift['rows_checked']:,} payments charged above "
            f"contract, {format_inr(drift['total_excess_paise'])} recoverable"
        )
        for method, excess in list(drift.get("by_method", {}).items())[:5]:
            print(f"    {method:<16}{format_inr(excess)}")
    return 0


def cmd_sweep(args) -> int:
    """Score many seeds. One good run is an anecdote; a distribution is evidence."""
    config = ReconcileConfig()
    cards = []
    print(f"  running {args.seeds} seeds x {args.batches} batches\n")
    header = f"  {'seed':>5}  {'lines':>6}  {'match':>7}  {'precis':>7}  {'FALSE':>7}  {'recall':>7}  {'rows/s':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for seed in range(args.seeds):
        card = _scored_run(seed, args.batches, args.difficulty, config)
        cards.append(card)
        print(
            f"  {seed:>5}  {card.bank_lines:>6,}  {card.auto_match_rate:>6.1%}  "
            f"{card.precision:>6.1%}  {card.false_match_rate:>6.2%}  {card.recall:>6.1%}  "
            f"{card.rows_per_second:>9,.0f}"
        )

    def agg(fn):
        vals = [fn(c) for c in cards]
        return min(vals), statistics.mean(vals), max(vals)

    print()
    for label, fn, fmt in (
        ("auto-match rate", lambda c: c.auto_match_rate, ".1%"),
        ("precision", lambda c: c.precision, ".2%"),
        ("false-match rate", lambda c: c.false_match_rate, ".2%"),
        ("recall", lambda c: c.recall, ".2%"),
        ("exception type accuracy", lambda c: c.exception_type_accuracy, ".1%"),
    ):
        lo, mean, hi = agg(fn)
        print(f"  {label:<26} min {lo:{fmt}}   mean {mean:{fmt}}   max {hi:{fmt}}")

    worst = max(cards, key=lambda c: c.false_match_rate)
    total_wrong = sum(c.wrong_matches for c in cards)
    total_matched = sum(c.matched for c in cards)
    print(
        f"\n  across all seeds: {total_wrong:,} wrong matches out of {total_matched:,} accepted "
        f"({total_wrong / max(total_matched, 1):.3%})"
    )
    if worst.wrong_examples:
        print("\n  worst seed's false matches")
        for w in worst.wrong_examples[:5]:
            print(f"    {w['txn_id']:<14} {w['scenario']:<22} {w['tier']}")
    return 0


def cmd_detail(args) -> int:
    card = _scored_run(args.seed, args.batches, args.difficulty, ReconcileConfig())
    print(render(card, f"Scorecard (seed {args.seed}, {args.batches} batches)"))
    return 0


def cmd_explain(args) -> int:
    report = reconcile_directory(Path(args.data), enable_llm=False)
    print(json.dumps(explain(report, args.txn), indent=2, default=str))
    return 0


def cmd_chaos(args) -> int:
    """Break things on purpose and show the run survives."""
    data = Path(args.data)
    print("  baseline")
    base = reconcile_directory(data, enable_llm=False)
    b = base.summary()
    print(f"    matched {b['matched']}/{b['bank_lines']}  quarantined {b['quarantined']}  chain {b['chain_tip'][:16]}")

    print("\n  same files uploaded again")
    again = reconcile_directory(data, enable_llm=False)
    print(f"    reused_existing={again.reused_existing}  run_id={'same' if again.run_id == base.run_id else 'NEW'}")

    print("\n  adjudicator failing half its calls")
    flaky = reconcile_directory(
        data, enable_llm=True, chaos=ChaosConfig(adjudicator_failure_rate=0.5)
    )
    f = flaky.summary()
    print(
        f"    matched {f['matched']}/{f['bank_lines']}  escalations {f['escalated']}  "
        f"degraded calls {f['degraded_calls']}  run completed"
    )

    print("\n  bank statement truncated by 30%")
    trunc = reconcile_directory(data, enable_llm=False, chaos=ChaosConfig(truncate_bank_fraction=0.3))
    t = trunc.summary()
    print(f"    matched {t['matched']}/{t['bank_lines']}  run completed on partial input")

    ledger = Ledger(data / "kosh.db")
    breaks = ledger.verify_chain()
    ledger.close()
    print(f"\n  audit chain: {'intact' if not breaks else str(len(breaks)) + ' broken links'}")
    return 0


def cmd_verify(args) -> int:
    ledger = Ledger(Path(args.data) / "kosh.db")
    breaks = ledger.verify_chain()
    n = len(ledger.entries(limit=100_000))
    ledger.close()
    if not breaks:
        print(f"  audit chain intact across {n:,} entries")
        return 0
    print(f"  chain broken at {len(breaks)} point(s); first at seq {breaks[0].seq}")
    return 1


def cmd_serve(args) -> int:
    from kosh.api.server import serve

    serve(args.data, args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kosh", description="Settlement reconciliation")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--batches", type=int, default=120)
        p.add_argument("--difficulty", type=float, default=1.0)
        return p

    g = common(sub.add_parser("generate", help="write a labelled synthetic dataset"))
    g.add_argument("--seed", type=int, default=11)
    g.add_argument("--out", default="data/demo")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="reconcile a data directory")
    r.add_argument("--data", default="data/demo")
    r.add_argument("--no-llm", action="store_true", help="deterministic floor, no model calls")
    r.set_defaults(func=cmd_run)

    s = common(sub.add_parser("sweep", help="score many seeds"))
    s.add_argument("--seeds", type=int, default=12)
    s.set_defaults(func=cmd_sweep)

    d = common(sub.add_parser("detail", help="full scorecard for one seed"))
    d.add_argument("--seed", type=int, default=11)
    d.set_defaults(func=cmd_detail)

    e = sub.add_parser("explain", help="decompose one bank line")
    e.add_argument("--data", default="data/demo")
    e.add_argument("--txn", required=True)
    e.set_defaults(func=cmd_explain)

    c = sub.add_parser("chaos", help="inject faults and show graceful degradation")
    c.add_argument("--data", default="data/demo")
    c.set_defaults(func=cmd_chaos)

    v = sub.add_parser("verify", help="check the audit chain")
    v.add_argument("--data", default="data/demo")
    v.set_defaults(func=cmd_verify)

    sv = sub.add_parser("serve", help="run the dashboard")
    sv.add_argument("--data", default="data/demo")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
