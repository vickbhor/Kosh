# Kosh

**Settlement reconciliation that refuses to guess.**

> **0 wrong matches out of 1,422 accepted, across 12 independent seeds.**
> 89.7% of bank lines matched automatically. The model was called **once in 134 lines**.

Built for the Razorpay AI Buildathon — **Track 04, AI Finance Controller**.

| | |
|---|---|
| **What it does** | Matches bank credits back to a payment gateway's settlement report |
| **Input** | Three CSVs: settlement report, bank statement, order ledger |
| **Output** | A reconciled ledger, a typed exception queue, and a fee-overcharge report |
| **Stack** | Python 3.11+, standard library only. Browser dashboard with three.js |
| **Install** | None. `python -m pytest -q` then `python -m kosh serve` |
| **Size** | 3,683 lines of Python, 64 tests, ~1.5s to run them |

---

## Table of contents

1. [The problem, in plain English](#1-the-problem-in-plain-english)
2. [Quick start](#2-quick-start)
3. [Glossary](#3-glossary)
4. [What the numbers mean](#4-what-the-numbers-mean)
5. [The finding that shaped the design](#5-the-finding-that-shaped-the-design)
6. [How the matching works](#6-how-the-matching-works)
7. [Where the AI is, and where it deliberately isn't](#7-where-the-ai-is-and-where-it-deliberately-isnt)
8. [When things go wrong](#8-when-things-go-wrong)
9. [Beyond reconciliation](#9-beyond-reconciliation)
10. [Every command](#10-every-command)
11. [Reading the code](#11-reading-the-code)
12. [Troubleshooting](#12-troubleshooting)
13. [What is not built](#13-what-is-not-built)

---

## 1. The problem, in plain English

A merchant sells things online. Customers pay through a gateway. Every couple of
days the gateway sends the merchant's money to their bank in one lump.

The merchant's accountant now has a problem. The bank says **one credit of
₹1,84,971.98 arrived on 7 April**. What was it for?

It was not one sale. It was:

```
24 payments, gross                        ₹2,00,683.88
platform fee                                −₹4,013.57
GST on that fee                               −₹722.33
1 refund                                   −₹10,976.00
                                          ─────────────
bank credit                               ₹1,84,971.98
```

Now do that for 134 bank lines against 2,083 settlement rows, where:

- the payments settled **two days** before the money arrived
- the reference number linking them (the UTR) is buried in bank text like
  `RTGS CR 681404506058 RAZORPAY SOFTWARE PRIVATE LIMITED` — when it is there at all
- some settlements arrive **split across two credits** on different days
- some bank credits are **not from the gateway at all** — a customer paid directly
- some rows in the files are **malformed** and will not parse

Finance teams do this in Excel, by hand, every month. Kosh does it in 0.03
seconds and, crucially, **tells you which lines it could not resolve and why**
instead of forcing a match.

---

## 2. Quick start

Requires Python 3.11 or newer. Nothing to install, no API key needed.

```bash
git clone https://github.com/<you>/kosh.git
cd kosh
pip install pytest
```

Then, in order:

**Run the tests.** Confirms the engine works on your machine.

```bash
python -m pytest -q
# 64 passed in 1.6s
```

**Generate a dataset.** Creates fake-but-realistic CSVs *with ground-truth
labels*, which is what makes the accuracy numbers checkable.

```bash
python -m kosh generate --batches 120
# wrote 2,083 settlement rows and 134 bank lines to data/demo
# ground truth for 134 bank lines, 88 rows carry fee drift
```

**Check the accuracy.** Twelve independent datasets, scored against their labels.

```bash
python -m kosh sweep --seeds 12
# across all seeds: 0 wrong matches out of 1,422 accepted (0.000%)
```

**Open the dashboard.**

```bash
python -m kosh serve
# http://localhost:8000
```

> On Windows use `python`. On macOS and Linux you may need `python3`, and
> `make test`, `make sweep`, `make serve` wrap the same commands.

---

## 3. Glossary

Payments jargon, in one line each. Skip if you know it.

| Term | Meaning |
|---|---|
| **Settlement** | The gateway paying a merchant. One settlement covers many payments |
| **Settlement batch** | The set of payments paid out together in one settlement |
| **UTR** | Unique Transaction Reference — a 12- or 16-digit bank reference number |
| **Narration** | The free-text description on a bank statement line |
| **MDR** | Merchant Discount Rate — the gateway's commission, e.g. 2% |
| **GST on fee** | 18% tax charged on the gateway's fee, not on the sale |
| **T+2** | Money arrives two days after the payment |
| **Paise** | 1/100 of a rupee. All money here is stored as whole paise |
| **Net amount** | What the merchant actually receives: gross − fee − GST |
| **Chargeback** | A customer disputes a payment; money is taken back |
| **Reconciliation** | Proving each bank credit equals a known set of payments |

---

## 4. What the numbers mean

Twelve seeds, 120 settlement batches each, ~2,000 settlement rows per seed:

| metric | min | mean | max | what it means |
|---|---|---|---|---|
| auto-match rate | 87.8% | 89.7% | 91.7% | lines matched without a human |
| precision | 100% | 100% | 100% | of the matches made, how many are right |
| **false-match rate** | **0.00%** | **0.00%** | **0.00%** | of the matches made, how many are **wrong** |
| recall | 94.6% | 96.7% | 98.4% | of matchable lines, how many were found |
| exception type accuracy | 100% | 100% | 100% | unmatched lines given the right failure type |

### Why false-match rate is the headline

Two failures, very different costs:

- **A missed match** — the system says "I don't know", an analyst spends ten
  minutes, the books stay correct.
- **A wrong match** — the system confidently ties a credit to the wrong
  payments. The books are now wrong and **nobody finds out**, because the
  reconciliation report says everything balanced.

Averaging those into one "accuracy" number hides the second one. So this system
optimises for never making the second mistake, and accepts more of the first.

Reproduce any of this with `python -m kosh sweep --seeds 12`.

---

## 5. The finding that shaped the design

The obvious way to match a batch with no reference number: find the set of rows
whose amounts add up to the bank credit. That's the classic **subset-sum**
problem.

**It is silently wrong at scale**, and this repo measures it.

With *n* candidate rows there are 2ⁿ possible subsets, but only about
`pool_total` distinct achievable paise values. Once *n* passes roughly 20, there
are more subsets than possible totals, so by the pigeonhole principle **subsets
other than the true one will hit your target exactly**.

Measured on random batches of realistic ticket sizes:

| pool size | runs where a *wrong* subset also summed exactly right |
|---|---|
| 10 rows | 0% |
| 14 rows | 0% |
| 18 rows | 2% |
| 22 rows | 48% |
| 28 rows | **100%** |

A system that trusts bare arithmetic at settlement scale reports a superb match
rate and is quietly, confidently wrong. Pinned in the test suite as
`test_spurious_matches_become_certain_as_pools_grow`, so it cannot regress.

**Two things follow, both implemented:**

**Constrain the pool before doing arithmetic.** A settlement batch pays out on
one date, so candidates come from a single settle-date group rather than the
whole window. That keeps *n* small enough for an exact sum to actually mean
something.

**Attach a collision risk to every arithmetic match.** Each strategy reports how
large a search it ran and the odds of a lucky hit:

| strategy | hypotheses searched | risk |
|---|---|---|
| whole settle-date groups | ~2⁷ | ~1 in 10 million |
| contiguous slice of a named batch | 2n | ~1 in 300,000 |
| drop 1–3 rows (elimination) | ~n³/6 | ~1 in 2,000 |
| unconstrained 28-row subset-sum | 2²⁸ | **certain** |

Matches above the risk ceiling are **declined** and typed `arithmetic_unsafe` —
which tells the operator the engine refused, rather than implying nothing was
there. You can see the risk on every matched row in the dashboard, e.g.
`whole settle-date groups, collision risk 1.4e-07`.

---

## 6. How the matching works

Five tiers, cheapest and most certain first.

### Tier 1 — Exact (≈50% of lines)

Pull the UTR out of the bank narration, look it up in the settlement report,
check the totals agree exactly.

```
bank_000179  ₹1,84,971.98  "RTGS CR 681404506058 RAZORPAY SOFTWARE PRIVATE LIMITED"
             → UTR 681404506058 → settlement batch of 25 rows → total matches exactly
```

### Tier 2 — Fuzzy (≈25%)

The narration has no UTR at all (`FUND TRF FROM RAZORPAY SOFTWARE PVT LTD`). Fall
back to: does any settlement batch in a ±3 day window total this amount, within a
small tolerance for rounding on fee and GST?

### Tier 3 — Arithmetic reconstruction (≈14%)

The gateway dropped the settlement id, so there is nothing to group on but the
arithmetic. Four bounded strategies, in increasing order of risk:

1. **Whole settle-date groups** — do all rows settling on one date sum to this?
2. **Contiguous slice** — the UTR names the batch, but only part was paid out.
   Try leading and trailing slices (2n hypotheses, not 2ⁿ).
3. **Elimination** — which 1–3 rows were held back? Far cheaper than asking which
   many rows were included.
4. **Full subset-sum** — only where the pool is small enough to trust.

The solver holds its DP table as a bitset inside a single Python integer, so
folding in an amount is `m |= m << a` — one C-level shift over the whole table.

### Tier 4 — Adjudicated (**under 1%**)

Deterministic tiers found two candidates and can't separate them. Only now does
a language model get involved. See section 7.

### Tier 5 — Typed exception (≈11%)

Eight failure types, each carrying the action it implies:

| type | meaning | what the operator does |
|---|---|---|
| `orphan_credit` | narration names no gateway | trace a direct transfer |
| `chargeback_debit` | money going out, not in | route to disputes |
| `arithmetic_unsafe` | pool too big to trust a sum | supply settlement ids |
| `ambiguous_candidates` | two equally valid answers | make a human decision |
| `missing_in_bank` | settled but not yet credited | re-run after the next feed |
| `amount_mismatch` | close, but outside tolerance | check fees and refunds |
| `duplicate_utr` | two batches, one reference | confirm with the gateway |
| `malformed_row` | failed to parse | fix the source file |

### One important detail: the cascade repeats until nothing changes

Claiming rows changes what's available, which can turn a previously ambiguous
line into an exact one. The classic case is a **split payout**: one settlement
paid as two bank credits. The second credit matches its batch exactly — but only
after the first credit has taken its share. Running each tier once misses every
one of these. Running to a fixpoint catches them all.

---

## 7. Where the AI is, and where it deliberately isn't

**Tiers 1–3 use no model at all.** They are deterministic, cost nothing but CPU,
and resolve **88.8%** of lines on the demo dataset.

The model only sees lines the arithmetic genuinely could not separate. And when
it does, it is handed a **closed list** of candidates the engine already computed
and costed. It picks one, or abstains. **It is never asked to find a match.**

```json
{"candidate_id": "cand_00042", "confidence": 0.81, "reason": "..."}
```

If it returns a candidate id that wasn't on the list, malformed JSON, or times
out, that is treated as an abstention and the line goes to a human.

**That guarantee lives in `adjudicator._validate` — in code, not in the prompt.**
A prompt is a request; code is a guarantee. Four tests cover it.

The dashboard has an **Adjudicator off** toggle that reruns the entire batch with
no model. It still resolves 88.8%. That is the point.

---

## 8. When things go wrong

```bash
python -m kosh chaos
```

```
  baseline
    matched 119/134  quarantined 5  chain fae83a9d2bb7
  same files uploaded again
    reused_existing=True  run_id=same
  adjudicator failing half its calls
    matched 119/134  escalations 1  degraded calls 1  run completed
  bank statement truncated by 30%
    matched 83/93  run completed on partial input
  audit chain: intact
```

| what breaks | what happens |
|---|---|
| A row won't parse | Quarantined with the error and raw text; the run continues |
| The model times out or 429s | Degrades to an abstention; the run never raises |
| Someone uploads the same file twice | Deduplicated by content hash; nothing reposted |
| The bank file is truncated | Completes on what arrived |

### Tamper-evident audit log

Every match, exception and quarantine is appended to a SQLite log where **each
record stores the hash of the record before it**. Edit any historical row and
every hash after it becomes invalid.

Try it:

```bash
sqlite3 data/demo/kosh.db "UPDATE entries SET payload_json='{}' WHERE seq=5;"
python -m kosh verify
# chain broken at 1 point(s); first at seq 5
```

About twenty lines of code that turn "trust our output" into "check our output".

---

## 9. Beyond reconciliation

**Fee drift — finding money.** The same rows, checked against the contracted MDR.
On the demo dataset, **88 payments were charged above contract**. Reconciliation
tells you the books tie out; this tells you the gateway owes you money.

**Explain any line.** Click a credit and get the arithmetic, with the residual
shown so you can see it actually balances. Click an *unmatched* line and get a
diagnosis instead — why it was refused, how many candidates were considered, and
what was unclaimed nearby.

**The cascade floor.** A three.js view where every bank line rests on the slab of
the tier that resolved it, cube height scaled to the amount. It shows the one
thing a table cannot: a table says the adjudicator fired once, the floor shows
how alone that cube is. Degrades to the tables without WebGL or under
`prefers-reduced-motion`.

**Queue-driven ledger.** Exception types are cards; selecting one filters the
ledger to those lines, because an operator reading a queue immediately wants the
lines behind it.

---

## 10. Every command

| command | what it does |
|---|---|
| `python -m kosh generate` | Write a labelled synthetic dataset to `data/demo` |
| `python -m kosh run` | Reconcile it; print the summary, AI budget and fee drift |
| `python -m kosh sweep --seeds 12` | Score 12 independent datasets — the honest numbers |
| `python -m kosh detail --seed 11` | Full scorecard for one dataset, broken down by scenario |
| `python -m kosh serve` | Dashboard on port 8000 |
| `python -m kosh chaos` | Inject faults and show graceful degradation |
| `python -m kosh verify` | Check the audit chain |
| `python -m kosh explain --txn bank_000179` | Decompose one bank line |

Useful flags: `--batches N` (dataset size), `--difficulty 0.0` (all-clean data),
`--no-llm` (deterministic floor), `--port`, `--data`.

Docker:

```bash
docker build -t kosh . && docker run -p 8000:8000 kosh
```

Optional — enable the real adjudicator. Everything works without it:

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # Windows: set ANTHROPIC_API_KEY=...
```

---

## 11. Reading the code

```
kosh/
  core/
    models.py        domain types; money as integer paise
    normalize.py     CSV parsing, UTR extraction, quarantine
    adjudicator.py   tier 4: the closed-list model call and its validation
    ledger.py        hash-chained audit log, idempotency
    fee_drift.py     MDR overcharge detection
  matchers/
    index.py         indexed, claimable view of the settlement report
    subset_sum.py    bitset DP, complement search, bounded
    cascade.py       the five tiers and the fixpoint loop
  eval/
    generator.py     synthetic data WITH ground-truth labels
    harness.py       scoring, the scorecard
  api/server.py      six read-only routes, stdlib http.server
  pipeline.py        parse → match → drift → audit, plus chaos injection
  __main__.py        CLI
web/
  index.html         dashboard
  cascade3d.js       the three.js cascade floor
  vendor/            three.js r161, Archivo, JetBrains Mono
tests/test_kosh.py   64 tests, organised by property not by module
```

**Suggested reading order for a reviewer:**

1. `matchers/cascade.py` — the module docstring explains the whole design
2. `matchers/subset_sum.py` — why the solver is bounded
3. `core/adjudicator.py` — `_validate`, the "cannot invent a match" guarantee
4. `eval/generator.py` — every accuracy claim traces back here
5. `tests/test_kosh.py` — the properties that must hold

**Money handling.** Every amount is an integer of paise. Floats appear only at
the CSV boundary and the display boundary. `int(8.70 * 100) == 869` is a test in
this repo, because that class of bug is not acceptable in a ledger — the parser
goes via the decimal string to avoid it entirely.

---

## 12. Troubleshooting

**The dashboard shows old results after I changed something.**
Delete `data/` and regenerate. Runs are deduplicated by a fingerprint of the
input files plus config, so an unchanged input returns the stored run from the
audit ledger. That is correct behaviour, and annoying during development.

```bash
rm -rf data && python -m kosh generate --batches 120
```

**The page looks unstyled, or the 3D view is missing.**
Hard-refresh: `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (macOS). The fonts,
CSS and JS cache aggressively.

**`python: command not found`.** Try `python3`. On Windows, reinstall Python with
"Add to PATH" ticked.

**Tests fail on import.** Run from the repository root, not from inside `kosh/`.

**Port 8000 is in use.** `python -m kosh serve --port 8123`.

**`llm_tokens: 0` and `estimated_cost_inr: 0.0`.** No `ANTHROPIC_API_KEY` is set,
so the offline adjudicator ran. Expected, and everything else works.

**`4 adjudicated · 0` in the legend.** Not a bug. One line escalated, the
adjudicator abstained on a genuine tie, and it went to the queue. Zero lines were
decided by a model.

---

## 13. What is not built

Honest scope, since the brief asks what broke.

- **Order-to-payment reconciliation** is modelled in the data but the cascade
  only reconciles bank-to-settlement. That is the harder and more valuable axis;
  the other is a straightforward join.
- **No live Razorpay API integration.** Everything runs on generated data,
  because that is the only way to have ground truth. The parser targets the
  settlement report column layout, so swapping in a real export is a loader
  change.
- **Adjudicator accuracy is not separately measured.** It fires on roughly 1% of
  lines — too few for a meaningful precision figure. The offline heuristic used
  in tests applies the same abstain-on-ties policy, so the two paths are
  comparable but not equivalent.
- **The collision-risk estimate is crude** — a deliberately pessimistic
  uniform-distribution approximation. Good enough to separate safe strategies
  from unsafe ones by orders of magnitude, not to be quoted as a probability.
- **Single-currency, single-gateway.** Multi-currency would need FX rates on the
  settlement date and a rounding policy per pair.

---

## License

MIT. Vendored assets keep their own licences: three.js (MIT), Archivo (OFL),
JetBrains Mono (OFL) — see `web/vendor/`.
