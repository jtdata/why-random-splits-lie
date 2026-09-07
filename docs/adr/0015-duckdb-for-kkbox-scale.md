# ADR-0015: KKBox's read/clean/panel-construction layer uses DuckDB, not pandas

- **Status:** Accepted
- **Date:** 2026-08-28
- **Deciders:** JT Moeller

## Context

`docs/standards/coding.md` already states the rule this project has never
needed until now: "Anything larger than comfortable in pandas goes through
DuckDB over Parquet. SQL for set-based work, Python for row-wise logic that
SQL would obscure." Online Retail II's `transactions.csv`-equivalent (the
Excel workbook) is ~1M rows and every notebook through `06` used pandas
directly. `duckdb`/`pyarrow` have been project dependencies since the
start (`pyproject.toml`) but genuinely unused. KKBox's `transactions.csv`
is 21.5M rows / 1.73GB, and `03`'s per-origin construction pattern
(`build_asof_panel` loops over `origins` in Python, each iteration doing a
full-table pandas filter+groupby) was tried against it directly and
measured, not assumed to be too slow: an `origins JOIN transactions ON
transaction_date < as_of` pattern (even bounded to a 500-day window per
origin) did not finish in several minutes and was killed. This ADR records
both the read/clean decision and what was learned building the panel-
construction query on top of it.

## Decision

`kkbox_io.py`'s `load_kkbox_transactions` never reads the raw CSV into
pandas: DuckDB's `read_csv_auto` streams directly into a `COPY ... TO
parquet` cleaning pass (measured: a full scan of the real file takes single-
digit seconds; the full clean-and-cache pass takes ~20s), and only the
cached, already-cleaned Parquet is ever read into a pandas `DataFrame`. The
cleaning rules themselves are one SQL query (`_clean_query`), shared
unchanged between that real-file path and this module's own tests (run
against a tiny in-memory frame registered with DuckDB), one source of
truth for the rules, not duplicated logic that could drift between the
tested and production paths.

`kkbox_features.py`'s `build_kkbox_asof_panel` computes eligibility,
features, and the label for every origin in 2-3 vectorized DuckDB SQL
passes across the whole origin set at once, not a per-origin Python loop.
The eligibility/feature pass specifically uses an `ASOF JOIN` (DuckDB's
purpose-built operator for "find the nearest prior matching row per group")
rather than a general inequality join, after the inequality-join version
was measured to not scale (see Alternatives below). The `ASOF JOIN` version
completes in ~37s for the full ~20-origin, 2.36M-customer join, against a
version of the same computation that did not finish in several minutes.

## Alternatives considered

| Option | Why it was plausible | Why it was rejected |
|---|---|---|
| Read `transactions.csv` with pandas `read_csv`, same as `io.py` does for the Excel source | Keeps every dataset's loader using the same library | Untested but implausible to be competitive with a ~20s DuckDB pass at 21.5M rows/1.73GB, parsing cost alone in pandas is materially higher, and this is precisely the case `coding.md`'s existing rule was written for, not a new judgment call |
| `origins JOIN transactions ON transaction_date < as_of`, bounded to a fixed window before each `as_of` (e.g. 500 days) | A natural first attempt, an inequality join is the obvious SQL translation of "every transaction before this origin," and the 500-day bound looked like it should keep the join small | Measured directly: did not finish in several minutes against the real file and was killed. DuckDB's optimizer does not push an inequality-join bound down the way an equi-join gets hash-joined. The practical cost is closer to `len(origins) x len(transactions)` before filtering narrows it, not the pruned cost the bound was intended to produce |
| `ROW_NUMBER() OVER (PARTITION BY as_of, customer_id ORDER BY transaction_date DESC)` on top of the same inequality join, to get "most recent transaction" | The natural SQL idiom for "latest row per group" | Inherits the same join-cost problem, the window function runs after the expensive join has already been computed, so it doesn't fix the bottleneck, only adds to the cost on top of it |
| `ASOF JOIN` between an explicit `origins x distinct(customer_id)` grid and `transactions` | Purpose-built for exactly this "nearest prior row per group" query shape | This is the option that was kept, included here to record that it was verified empirically (~37s, measured against the real file), not assumed to be faster because it sounds like the right tool |

## Consequences

**Good:** `load_kkbox_transactions`/`build_kkbox_asof_panel` are tractable
in a notebook cell (seconds to tens of seconds, not minutes-to-hours) at
KKBox's real scale, verified by direct measurement rather than estimated.
This is also the first real exercise of `coding.md`'s DuckDB rule and the
first real evidence for *why* the rule exists in this project, not just an
assertion in a standards document.

**Bad:** `kkbox_features.py`'s SQL is materially harder to read and modify
than `splits.py`'s pandas/Python equivalent, a future contributor changing
the eligibility window or feature set needs to understand `ASOF JOIN`
semantics and DuckDB's date-interval syntax, not just pandas. This is an
accepted cost of the scale, not a stylistic preference.

**Revisit if:** a future KKBox-scale dataset needs eligibility/feature
logic complex enough that the SQL becomes substantially harder to verify by
hand than the value it provides. At that point, consider whether a hybrid
(DuckDB for the initial narrowing pass, pandas for the final small-population
computation) reads more clearly without sacrificing the performance this
ADR is about.

## Addendum (2026-08-28): `ASOF JOIN` needed an explicit tie-break

Adversarial review of `07_generalisation` found a real defect in the `ASOF
JOIN` this ADR chose, not in the choice of `ASOF JOIN` itself: DuckDB gives
no tie-break guarantee when multiple rows share a join key, and 255,595
`(customer_id, transaction_date)` groups in the real data do (94.6% of them
with a *different* `membership_expire_date` per tied row). Reproduced
directly, two in-process calls to `build_kkbox_asof_panel` on identical
cached input picked different eligible populations and different labels for
~0.8% of the panel, not merely different feature values. Fixed by adding
`kkbox_features._dedupe_for_asof_join`, which resolves same-date ties with
an explicit, deterministic rule (largest `membership_expire_date` wins, a
stable sort so any further tie resolves the same way every run) before the
`ASOF JOIN` ever sees the data. This does not change the decision above
(`ASOF JOIN` against a deduplicated table is still the fast, correct
operator for this query shape) it closes a gap in how it was applied.
Regression-tested in
`tests/test_kkbox_features.py::test_current_membership_state_is_deterministic_and_prefers_the_later_expiry_on_a_tied_date`.
