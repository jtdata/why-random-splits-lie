# ADR-0013: KKBox eligibility is expiry-based; HORIZON_DAYS/GAP_DAYS need no override; the label does not reproduce KKBox's own churn definition byte-for-byte

- **Status:** Accepted
- **Date:** 2026-08-28
- **Deciders:** JT Moeller

## Context

`07_generalisation` scales this project's protocol to KKBox, contractual
subscription churn, chosen alongside Online Retail II specifically to test
whether the argument holds outside non-contractual retail (ADR-0003). Three
related design questions had to be answered before `build_kkbox_asof_panel`
could be written, each with a real, measured alternative, the same way
ADR-0007 settled Online Retail II's naive-panel definition.

**Eligibility.** Online Retail II's `eligible_customers` (`splits.py`) is
activity-based: at least one purchase in the 365 days before `as_of`. Ported
directly to KKBox transactions, this rule was measured (not estimated)
against the real data at four sample dates:

| `as_of` | 365-day activity-lookback eligible customers |
|---|---|
| 2015-08-01 | 1,053,245 |
| 2016-02-01 | 1,539,682 |
| 2016-08-01 | 1,569,324 |
| 2017-01-01 | 1,743,970 |

Over a million customers per origin, most of whom aren't due for a renewal
decision for months, not a meaningful population to ask "will this
customer churn *at this origin*" about, and expensive to score at that
size across ~20 origins.

**`HORIZON_DAYS`/`GAP_DAYS`.** Online Retail II needed a dataset-specific
override (`naive_baseline.HORIZON_DAYS = 90`, ADR-0007) because its
non-contractual purchase cadence didn't fit `config.py`'s
`DEFAULT_HORIZON_DAYS = 30` default. `config.py`'s own comment on that
constant already earmarks it for "the value later notebooks reach for
first when scoring KKBox, whose monthly billing cadence is the case this
default was written for", written before this notebook existed, and
confirmed still accurate once KKBox's real `payment_plan_days` distribution
was checked: 88% of transactions are on a 30-day plan, 3.6% on 31-day,
2.7% on a 7-day trial, with a long tail at 90/100/180/195/410 days.

**Label definition.** KKBox's own Kaggle-competition rule defines churn as:
no valid new subscription within 30 days of *that member's own*
`membership_expire_date`. That's anchored per-customer to their own expiry
date, with no shared `as_of` or separate operational-gap concept at all,
structurally different from this project's `Window(as_of, gap_days,
horizon_days)`, which scores every eligible customer against one shared
`as_of` per origin.

## Decision

**Eligibility:** a customer is eligible at an origin if their current
membership's expiry falls in `[as_of - ELIGIBILITY_LOOKBACK_DAYS, as_of +
gap_days)`, `ELIGIBILITY_LOOKBACK_DAYS = 45`, wider than KKBox's own
30-day grace convention specifically to catch the non-30/31-day
`payment_plan_days` tail (a 410-day-plan customer whose membership lapsed
40 days ago is still a real renewal-decision candidate; a 30-day rule would
have already given up on them). Measured directly against the real 20-origin
backtest this rule produces: population per origin ranges 199,626-424,255
(mean ~312K), fluctuating with signup seasonality rather than growing
monotonically, bounded and tractable, unlike the activity-based
alternative's 1-1.7M.

**`HORIZON_DAYS`/`GAP_DAYS`:** no KKBox override. `HORIZON_DAYS =
config.DEFAULT_HORIZON_DAYS = 30`, `GAP_DAYS = config.DEFAULT_GAP_DAYS = 7`,
 both already-existing project defaults, used as-is, a deliberate contrast
with ADR-0007's retail override. `gap_days + horizon_days = 37` days spans
more than one 30-day origin step but at most two, so the maturity purge
(ADR-0009's rule, unchanged) drops exactly the two most recent origins
before any test origin, smaller than Online Retail II's three-origin
purge.

**Label:** `churned = 0` iff a non-cancellation transaction (`is_cancel =
0`) lands in `[label_start, label_end)`, a cancellation transaction inside
the label window counts as *stronger* evidence of churn, not a renewal.
This does **not** reproduce KKBox's own 30-days-after-own-expiry rule.
Stated plainly rather than glossed over: reproducing that rule exactly
would mean abandoning `rolling_origins`/`mask_label_events`, i.e. not
reusing the shared `as_of`/gap/horizon machinery this notebook exists to
demonstrate transfers to a second dataset. The two definitions are close in
spirit (both give roughly a month's grace after a subscription lapses) but
not identical, and this notebook's finding is about this project's own
protocol applied to KKBox, not a reproduction of the Kaggle competition's
scoring rule.

## Alternatives considered

| Option | Why it was plausible | Why it was rejected |
|---|---|---|
| Activity-based eligibility (≥1 transaction in N days before `as_of`), matching Online Retail II's rule | Consistency with the established pattern; simpler to explain as "the same rule, different dataset" | Measured directly: scores 1.05M-1.74M ever-present customers per origin, most not due for a renewal decision for months, wrong population for a monthly-cadence contractual product, and far more expensive to score across ~20 origins for no benefit |
| Reproduce KKBox's own 30-days-after-own-expiry churn definition exactly | Matches the Kaggle competition's own, well-established rule; avoids the appearance of an invented definition | Requires abandoning the shared `as_of`/`gap_days`/`horizon_days` structure this entire project's machinery (and every reused function in ADR-0014) is built around, not reusing the protocol is a bigger cost than the definitional gap between "this project's honest label" and "KKBox's competition label," which are close in spirit anyway |
| `ELIGIBILITY_LOOKBACK_DAYS = 30`, matching KKBox's own grace period exactly | Simpler, one fewer arbitrary-seeming constant to justify | The real `payment_plan_days` distribution has a genuine tail beyond 30/31 days (90/100/180/195/410-day plans); a strict 30-day rule would silently exclude legitimate longer-cadence subscribers who are still due for a renewal decision, just on a slower clock |

## Consequences

**Good:** the eligible population is bounded and directly interpretable
("who is actually due for a renewal decision around now"), tractable to
score at KKBox's scale (~312K/origin, not ~1.4M), and the label's departure
from KKBox's own definition is stated explicitly rather than left for a
reader to discover by comparing the two rules themselves.

**Bad:** this notebook's headline numbers are not directly comparable to
the Kaggle competition's own leaderboard scores (different label, different
population, different evaluation protocol entirely). `train.csv`'s
official `is_churn` is used only as an informal, heavily-caveated cross-
check, never as ground truth for this project's own numbers.

**Revisit if:** a later notebook wants to report a number directly
comparable to the KKBox competition's own leaderboard. That would need a
second, competition-faithful label built specifically for that comparison,
not a change to this protocol's own label.
