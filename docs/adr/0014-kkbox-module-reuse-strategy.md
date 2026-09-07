# ADR-0014: KKBox reuses the backtest/evaluation/calibration/hazard engines unchanged, rebuilds panel construction fresh, and generalizes one hazard function in place

- **Status:** Accepted
- **Date:** 2026-08-28
- **Deciders:** JT Moeller

## Context

`07_generalisation` scales this project's protocol to KKBox, a dataset with
a genuinely different schema and a genuinely different churn mechanism
(contractual expiry, not inferred purchase gaps) from Online Retail II.
Before writing any KKBox-specific code, every candidate function in
`churnval.windows`/`splits`/`features`/`evaluation`/`calibration`/`hazard`/
`naive_baseline` was read to determine which are reusable unchanged and
which hardcode Online-Retail-specific assumptions. This is a real decision
with three genuinely different resolutions applied to different functions,
not one uniform policy — worth recording so a later reader isn't left to
guess why the same notebook reuses some functions untouched, rewrites
others from scratch, and modifies exactly one in place.

## Decision

**Reused unchanged, zero KKBox-specific code:** `windows.py` in full
(`Window`, `rolling_origins`, `mask_feature_events`, `mask_label_events` —
the latter two already take `time_col` as a parameter); `splits.
rolling_origin_backtest`, `fit_at_origin`, `frozen_model_backtest`; all of
`evaluation.py`; all of `calibration.py`; `hazard.survival_from_hazards`,
`hazard_curve`, `predict_survival`, `hazard_rolling_origin_backtest`,
`fit_hazard_at_origin`. Confirmed reusable by inspection (they only ever
touch `panel["as_of"]`/`panel["customer_id"]`/`panel["churned"]`/
caller-supplied `feature_cols`, or raw `y_true`/`y_prob` arrays) and by the
fact that `tests/test_splits.py`/`tests/test_hazard.py` already exercised
them with a generic `"signal"` fixture column, not retail RFM names, before
this notebook existed.

**Rebuilt fresh, no shared code with the Online Retail II versions:**
`kkbox_features.py`'s `build_kkbox_asof_panel` (the `splits.eligible_customers`
+ `splits.build_asof_panel` + `features.asof_features` analogue) and
`kkbox_naive_features` (the `naive_baseline.naive_features` analogue).
These don't just use different column names — they encode a genuinely
different *criterion*: KKBox eligibility is expiry-based, not
activity-based (ADR-0013); KKBox features are subscription-state values
(days until expiry, plan price, auto-renew, cancellation counts), not RFM.
Parameterizing the Online Retail versions into something generic enough for
both would need `feature_fn`/`eligible_fn`/`time_col`/`entity_col` all
injected — harder to read than two separate implementations, and it would
mean touching functions every one of `03`–`06`'s notebooks depends on for
their *exact current* behavior.

**Generalized in place, one parameter added:** `hazard.first_event_period`
(and `build_person_period_panel`, which forwards to it) gained a required
keyword-only `time_col: str` parameter (no default — matches
`mask_feature_events`/`mask_label_events`'s own no-default convention in
`windows.py`). Confirmed via `grep` before touching it: exactly one
internal call site, two test call sites, and one notebook call site
(`06_hazard_framing.py`) needed updating — a small, mechanical, low-risk
change, not a rewrite. `first_event_period` hardcoded exactly one literal
(`"invoice_date"`) inside an otherwise fully generic body that already
called a `time_col`-parameterized helper (`mask_label_events`) — a
fundamentally smaller case than `build_asof_panel`'s, where the *logic*,
not just a column name, differs between datasets.

After making this change, `tests/test_hazard.py` was re-run and
`06_hazard_framing.ipynb` was re-executed fresh via `papermill`; every
numeric output was diffed against the prior run and found identical
(0 of the notebook's cell outputs changed) — confirmed, not assumed.

## Alternatives considered

| Option | Why it was plausible | Why it was rejected |
|---|---|---|
| Parameterize `splits.eligible_customers`/`build_asof_panel` in place (add `time_col`/`feature_fn`/`eligible_fn` arguments) rather than writing `kkbox_features.py` fresh | Avoids "two versions of similar-looking logic" | The two datasets' eligibility and feature logic aren't the same computation with different column names — they're different computations. Injecting enough parameters to cover both would make the shared function harder to read than either dataset-specific version alone, for no reuse benefit beyond avoiding two file headers |
| Write a fully separate `kkbox_hazard.py` rather than adding one parameter to `hazard.first_event_period` | Keeps `hazard.py` untouched, mirroring the "don't touch `splits.py`" decision above for consistency | `first_event_period`'s hardcode is a single literal in an otherwise dataset-agnostic function, not a different criterion — parameterizing it is strictly smaller-scope and lower-risk than the alternative of duplicating the entire (already-correct, already-tested) discrete-time hazard machinery a second time |
| Treat `naive_baseline.py`'s "nothing here is reused past notebook 02" precedent as blanket guidance against ever sharing code between a correct and a naive-comparison panel, and duplicate `build_kkbox_asof_panel`'s logic for the naive KKBox side too | Consistency with an established pattern | That precedent is about adversarial independence between *deliberately-wrong* and *correct* code, so a reader can trust the wrong one is wrong on its own terms — not a general rule against reuse. It doesn't apply between the same, correct protocol applied to two datasets, which is what this ADR is actually about |

## Consequences

**Good:** the reused engines (backtest, evaluation, calibration, hazard) now
have a real, executed proof that their genericity claim was correct, not
just an assumption — the same code that validated Online Retail II's
protocol validates KKBox's without modification. The one function that did
need a change was changed by the smallest possible edit, with a direct
before/after regression check on the notebook that already depended on it.

**Bad:** `kkbox_features.py` duplicates the *shape* of `splits.py`'s
eligibility/panel-construction pattern (a `build_X_asof_panel` function,
an eligibility helper, a label helper) without sharing any code with it —
a later reader adding a third dataset will likely repeat this shape a third
time rather than inheriting an abstraction, since no such abstraction was
built. That's an accepted cost, not an oversight: building the abstraction
now, from two data points, would be guessing at what a third dataset's
requirements look like.

**Revisit if:** a third dataset is added to this project and its
eligibility/feature/label logic turns out to share real structure with
KKBox's (not just Online Retail II's) — at that point, a shared abstraction
has two real examples to be built from rather than one guessed at from a
single case.
