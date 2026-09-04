"""HTTP layer.

Deliberately built on ``http.server`` rather than a framework. A reconciliation
tool that handles settlement files should carry as little third-party code as
it can justify, and the whole API is six read-only routes. Zero runtime
dependencies also means ``python -m kosh serve`` works on a fresh machine with
no install step, which matters more during a demo than routing sugar does.
"""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from kosh.core.ledger import Ledger
from kosh.core.models import EXCEPTION_ACTIONS, format_inr
from kosh.matchers.cascade import ReconcileConfig
from kosh.pipeline import ChaosConfig, ai_budget, explain, reconcile_directory

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


def _load_scorecard(data_dir: Path, report) -> dict | None:
    """Score the run if the directory carries ground-truth labels.

    Real merchant data has no labels, so accuracy is only measurable on a
    generated set. The dashboard says so rather than showing a made-up figure:
    an unverifiable number on a finance screen is worse than an absent one.
    """
    truth_path = data_dir / "ground_truth.json"
    if not truth_path.exists():
        return None
    from kosh.eval.generator import TruthEntry
    from kosh.eval.harness import score

    raw = json.loads(truth_path.read_text())
    truth = {e["txn_id"]: TruthEntry(**e) for e in raw.get("entries", [])}
    card = score(
        report.run,
        truth,
        settlement_rows=len(report.parse.settlement_rows),
        quarantined=len(report.parse.quarantined),
        wall_seconds=report.wall_seconds,
    )
    return card.to_dict()


def _serialise(report, config_label: str, scorecard: dict | None = None) -> dict:
    """Flatten a run into everything the dashboard needs, in one payload."""
    txns = {t.txn_id: t for t in report.parse.bank_txns}
    lines = []
    for r in report.run.results:
        txn = txns.get(r.txn_id)
        amount = txn.signed_paise if txn else 0
        lines.append(
            {
                "txn_id": r.txn_id,
                "value_date": txn.value_date.isoformat() if txn else "",
                "narration": txn.narration if txn else "",
                "amount_paise": amount,
                "amount": format_inr(amount),
                "tier": r.tier.value,
                "matched": r.matched,
                "rows": len(r.row_ids),
                "delta_paise": r.delta_paise,
                "confidence": round(r.confidence, 2),
                "reason": r.reason,
                "exception_type": r.exception_type.value if r.exception_type else None,
                "action": EXCEPTION_ACTIONS.get(r.exception_type, "") if r.exception_type else "",
            }
        )

    drift = report.drift or {}
    return {
        "config": config_label,
        "scorecard": scorecard,
        "summary": report.summary(),
        "budget": ai_budget(report),
        "lines": lines,
        "quarantined": [
            {"source": q.source, "line": q.line_no, "reason": q.reason, "raw": q.raw[:120]}
            for q in report.parse.quarantined
        ],
        "drift": {
            **drift,
            "total_excess": format_inr(drift.get("total_excess_paise", 0)),
            "by_method_display": {
                k: format_inr(v) for k, v in list(drift.get("by_method", {}).items())[:6]
            },
        },
        "totals": {
            "credits": format_inr(sum(t.credit_paise for t in report.parse.bank_txns)),
            "settlement_rows": len(report.parse.settlement_rows),
        },
    }


@lru_cache(maxsize=8)
def _cached_run(data_dir: str, adjudicate: bool, failure_rate: float, truncate: float) -> dict:
    chaos = ChaosConfig(adjudicator_failure_rate=failure_rate, truncate_bank_fraction=truncate)
    report = reconcile_directory(
        Path(data_dir),
        config=ReconcileConfig(enable_adjudication=adjudicate),
        enable_llm=adjudicate,
        chaos=chaos,
    )
    label = "with adjudicator" if adjudicate else "deterministic only"
    if chaos.active:
        label += f" | chaos: fail {failure_rate:.0%}, truncate {truncate:.0%}"
    return _serialise(report, label, _load_scorecard(Path(data_dir), report))


class Handler(BaseHTTPRequestHandler):
    data_dir = "data/demo"

    def log_message(self, *args) -> None:
        pass

    def _send(self, payload: dict | list, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._send({"error": "not found"}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        route = url.path
        params = dict(
            p.split("=", 1) for p in url.query.split("&") if "=" in p
        )

        if route in ("/", "/index.html"):
            self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            return

        if route.startswith("/vendor/") or route.endswith((".js", ".css", ".svg", ".woff2")):
            candidate = (WEB_DIR / route.lstrip("/")).resolve()
            if not str(candidate).startswith(str(WEB_DIR.resolve())):
                self._send({"error": "forbidden"}, 403)
                return
            types = {
                ".js": "text/javascript",
                ".css": "text/css",
                ".svg": "image/svg+xml",
                ".woff2": "font/woff2",
            }
            self._send_file(candidate, types.get(candidate.suffix, "application/octet-stream"))
            return

        if route == "/api/run":
            adjudicate = params.get("ai", "1") != "0"
            fail = float(params.get("fail", 0) or 0)
            trunc = float(params.get("truncate", 0) or 0)
            self._send(_cached_run(self.data_dir, adjudicate, fail, trunc))
            return

        if route.startswith("/api/explain/"):
            txn_id = route.rsplit("/", 1)[-1]
            report = reconcile_directory(Path(self.data_dir), enable_llm=False)
            self._send(explain(report, txn_id))
            return

        if route == "/api/verify":
            ledger = Ledger(Path(self.data_dir) / "kosh.db")
            breaks = ledger.verify_chain()
            count = len(ledger.entries(limit=200_000))
            ledger.close()
            self._send(
                {
                    "intact": not breaks,
                    "entries": count,
                    "breaks": [{"seq": b.seq} for b in breaks],
                }
            )
            return

        self._send({"error": "not found"}, 404)


def serve(data_dir: str = "data/demo", port: int = 8000) -> None:
    Handler.data_dir = data_dir
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"  kosh serving {data_dir} on http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
