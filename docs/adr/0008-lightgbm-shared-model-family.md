# ADR-0008: LightGBM is the single model family reused across 01, 03, and 04

- **Status:** Accepted
- **Date:** 2026-08-25
- **Deciders:** JT Moeller

## Context

`01_naive_baseline` fits the naive panel with `LGBMClassifier(random_state=
SEED, verbosity=-1)` on three RFM features (`recency_days`, `frequency`,
`monetary`). This project's headline result, produced in `04_the_gap`
(per `docs/timeline.md`: "How big is the lie?"), is a single number, the
difference between the random-split (wrong) result from `01` and the
temporal-split (correct) result `03_temporal_protocol` will produce. If `01`
used one model family and `03`/`04` used a different one, the measured gap
would be confounded: part of it could be attributable to a change of model
rather than to the validation-design fix (random split, `as_of`-blind
features, no gap vs. temporal split, `as_of`-safe features, a real gap).
The project's argument is about validation methodology, not modelling
choice, so the model has to be held constant for the headline number to mean
what it claims to mean.

## Decision

Use `LGBMClassifier` with `churnval.config.SEED` as `random_state` everywhere
a classifier is fit across `01_naive_baseline`, `03_temporal_protocol`, and
`04_the_gap`. `06_hazard_framing`'s discrete-time hazard model is a separate
comparison, not a replacement for this classifier.

## Alternatives considered

| Option | Why it was plausible | Why it was rejected |
|---|---|---|
| Logistic regression | Directly interpretable coefficients — would make the tautological closing section's "`recency_days` *is* the label" point almost visually obvious via a coefficient near the threshold boundary, on top of the `single_feature_auc` check already used | Weaker baseline for the multi-feature, as-of-safe protocol `03` builds and for the hazard-model comparison in `06`; this project wants one model family end to end for the isolation argument above, not the most interpretable model for `01` in isolation |
| Random forest | A comparably strong, similarly non-linear tree ensemble | Slower to tune, and less directly aligned with the calibration work in `05_calibration`, where LightGBM's probability outputs are the more common pairing. Not a hard blocker either way, but no reason to prefer it over LightGBM |
| A different model per notebook, chosen for whatever fits each notebook's narrower purpose best | Each notebook could use its "best" model on its own terms | Directly reintroduces the confounding problem this ADR exists to avoid — `04_the_gap`'s headline number would be unable to distinguish "the split was fixed" from "the model also changed" |

## Consequences

**Good:** the gap measured in `04_the_gap` is attributable to validation
design alone, which is the entire point of this repo.

**Bad:** ties the whole project's headline claim to one library's specific
behaviour (LightGBM's handling of missing values, monotonic splits, tree
randomness) rather than demonstrating the finding is model-agnostic. A
reader could reasonably ask "does this gap appear with logistic regression
too?" and this repo would not yet have an answer.

**Revisit if:** a later notebook (e.g. `07_generalisation`, or explicit
reviewer feedback) specifically wants to demonstrate the gap is not an
artifact of one model family — in which case add a secondary model (e.g.
logistic regression) as a robustness check alongside LightGBM, not as its
replacement.
