# ADR-0007: The Online Retail II naive panel uses a 90-day horizon and a 365-day, >=1-purchase eligibility lookback

- **Status:** Accepted
- **Date:** 2026-08-25
- **Deciders:** JT Moeller

## Context

`01_naive_baseline`'s first implementation defined the naive churn label as
"no purchase in the final `N` days of the whole dataset," anchored to
`transactions["invoice_date"].max()`. `naive_features`'s `recency_days` uses
the same reference date. The two are therefore not merely correlated
(`churned == (recency_days > N)` is a mathematical identity) and the naive
model scored a literal ROC AUC of 1.0, not just an inflated one. That is a
different failure mode than the "inflated but plausible" number the project
argument needs, so the label was rebuilt to be forward-looking, using
`churnval.windows.rolling_origins` and `mask_label_events` to define, at each
`as_of` origin, whether a customer purchased again during that origin's
label window. The three mistakes the notebook is meant to demonstrate (full-
history features ignoring `as_of`, a row-level random split that lets one
customer appear in both train and test, and no gap) now live in the panel
and split construction, not in the label's definition.

Rebuilding the label around real scoring occasions forced two further
choices specific to this dataset: how wide the label window should be, and
which customers are eligible to be scored at a given origin at all.

Online Retail II is characterized in [ADR-0003](0003-dataset-selection.md)
as non-contractual churn with a largely wholesale-cadence, UK-based
customer base — purchases are lumpy and irregular, not a monthly-cadence
consumer subscription. The project-wide defaults in `config.py`
(`DEFAULT_HORIZON_DAYS = 30`, `DEFAULT_GAP_DAYS = 7`) are explicitly
documented there as "defaults, not constraints: notebooks that vary them do
so explicitly and say why", they remain what later notebooks use for KKBox,
which is a monthly-billing subscription product where a 30-day horizon is
the natural unit.

## Decision

For the Online Retail II naive panel only, in `churnval.naive_baseline`:

1. `HORIZON_DAYS = 90`. A 30-day label window would call most genuinely-
   active repeat customers "churned" simply for not reordering within a
   calendar month, given this dataset's irregular purchase intervals. 90
   days is a better match for realistic reorder cadence.
2. `ELIGIBILITY_LOOKBACK_DAYS = 365`, with the rule that a customer needs at
   least one purchase in the 365 days before an origin's `as_of` to be
   scored at that origin (`eligible_customers`). This is the maturity check:
   a customer with nothing in the lookback window has no recent activity for
   the naive full-history features to summarize, and would not be a
   realistic scoring candidate.

## Alternatives considered

| Option | Why it was plausible | Why it was rejected |
|---|---|---|
| Keep `HORIZON_DAYS = 30` (the project-wide default) | Consistency with `config.py`; one fewer constant to explain | `config.py` itself says defaults are meant to be varied with a stated reason, and this dataset's purchase cadence is exactly that reason: a 30-day window would mislabel most active wholesale-style customers as churned |
| Eligibility requires >=2 purchases in the lookback window, not >=1 | More strongly indicates an established repeat-purchase relationship; screens out true one-time buyers who were never really "at risk" of churning in a repeat-purchase sense | Shrinks the eligible population and excludes legitimate first-time-repeat candidates from ever being scored; keeping the naive baseline's population close to the full customer base was preferred for this notebook. Revisit when `03_temporal_protocol` or `06_hazard_framing` needs a more established cohort |

## Consequences

**Good:** the naive label is no longer a mathematical identity with the
recency feature, so `01_naive_baseline`'s inflated result is a genuine
(if still wrong) offline estimate rather than a tautology. The three
intended mistakes (full-history features, row-level random split, no gap)
are now what the notebook's ablation and diagnosis in `02_leakage_diagnosis`
have to explain. The eligibility rule gives every origin a real, checkable
maturity condition instead of silently scoring customers with no history.

**Bad:** `HORIZON_DAYS` and `ELIGIBILITY_LOOKBACK_DAYS` are now dataset-
specific constants that live in `naive_baseline.py` rather than `config.py`,
so a reader has to know to look there rather than in the one place the
project usually documents window constants. The `>=1`-purchase eligibility
rule is permissive enough that a customer who bought exactly once, 364 days
before an origin, is scored, a weak basis for a recency/frequency feature.

**Revisit if:** `03_temporal_protocol` or `06_hazard_framing` need a cohort
with an established purchase pattern (in which case switch eligibility to
`>=2` purchases and record why), or if KKBox's contractual cadence turns out
to also need a horizon other than the 30-day project default, in which case
these per-dataset constants should move into a `config.py` structure keyed by
dataset rather than living ad hoc in each dataset's module.
