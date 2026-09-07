# ADR-0012: The discrete-time hazard model uses 30-day periods

- **Status:** Accepted
- **Date:** 2026-08-26
- **Deciders:** JT Moeller

## Context

`06_hazard_framing` reframes the 90-day churn label as a discrete-time
survival problem: instead of one binary outcome per customer per origin, a
customer's follow-up is split into periods, and the model estimates a
hazard (probability of returning) for each period, conditional on having
not returned yet. This needs a period width (how many days wide each
discretization step is) chosen before any of `churnval.hazard`'s
functions could be written, since it fixes the shape of the person-period
expansion (`build_person_period_panel`) and the size of the training data
the hazard model sees at every origin.

The horizon is fixed at 90 days (ADR-0007, dataset-specific). A period
width has to divide it evenly (`build_person_period_panel` and
`first_event_period` both assert this) and trades off two things directly
against each other: a narrower period gives finer temporal resolution
(more granular "when") at the cost of a larger person-period expansion (a
7-day period would produce roughly 13 periods and a person-period panel
several times the size of a 30-day period's), and a wider period gives
coarser resolution with a smaller, cheaper expansion.

## Decision

Periods are 30 days wide: 3 periods per 90-day horizon
(`churnval.hazard.PERIOD_DAYS = 30`). This matches
`config.DEFAULT_STEP_DAYS`, the spacing this project already uses between
rolling-origin scoring occasions, and the monthly-ish cadence ADR-0007
already established as the right granularity for this dataset's
non-contractual, largely-wholesale purchase pattern. The three features
this project uses (`recency_days`, `frequency`, `monetary`) carry no
information at a finer time resolution than that (they are RFM
aggregates, not event-sequence features) so a narrower period would
multiply the row count and training cost without giving the model
anything sharper to learn from.

## Alternatives considered

| Option | Why it was plausible | Why it was rejected |
|---|---|---|
| Weekly periods (7 days, ~13 periods) | Finer temporal resolution — could show exactly which week risk peaks in | The RFM features have no signal at weekly resolution to justify it; the person-period expansion would grow roughly 4x for no corresponding gain in what the model could learn, and `06`'s own results confirm this after the fact — the 30-day hazard curve is already only mildly non-flat (0.194/0.193/0.171), suggesting a finer grain would mostly add noise, not structure |
| A single period (equivalent to the binary classifier) | Not really an alternative — this is what `03` already does | Defeats the purpose of this notebook, which exists specifically to check whether period-level resolution reveals anything the binary label can't |
| Unequal-width periods (e.g., a short first period to isolate "immediate" returners, wider periods after) | Some prior literature front-loads the earliest window, on the theory early risk behaves differently | No prior evidence in this dataset that early risk is qualitatively different rather than just quantitatively similar (`06`'s own hazard curve shows periods 1 and 2 nearly equal, 0.194 vs. 0.193) before ever building it; adds a second, harder-to-justify parameter (where to place the boundary) for a benefit not yet demonstrated to exist |

## Consequences

**Good:** 3 periods keeps the person-period expansion small (`06` measured
2.52x the original panel's row count, not an order of magnitude larger),
keeps per-origin LightGBM fits cheap, and reuses an already-established
project cadence (`DEFAULT_STEP_DAYS`) rather than introducing a new,
unrelated time constant.

**Bad:** 3 periods is coarse enough that the hazard curve can only say
"risk is roughly flat, tapering slightly in the final third" — it cannot
resolve anything sharper (a spike in week 2, say) even if one existed.

**Revisit if:** the KKBox scale-up (`07`) has a shorter natural horizon or
a higher-frequency feature set (e.g., login events, session-level
activity) where finer-grained hazard resolution would have something real
to learn from — re-derive the period width for that dataset rather than
reusing 30 days by default.
