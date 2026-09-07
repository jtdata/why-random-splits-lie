# ADR-0009: The rolling-origin backtest uses an expanding training window with a label-maturity purge

- **Status:** Accepted
- **Date:** 2026-08-25
- **Deciders:** JT Moeller

## Context

`03_temporal_protocol` builds the corrected validation protocol:
as-of-safe features, a real gap, and a walk-forward backtest across
origins, retraining at each one, per `docs/standards/validation.md`'s
requirement for "several origins, retraining forward at each." Two related
questions had to be answered for how `rolling_origin_backtest`
(`src/churnval/splits.py`) selects training data at each origin, and the
second surfaced a real bug during this notebook's own construction.

**Window shape.** Online Retail II, under ADR-0007's 90-day horizon and
30-day step, produces only 10 origins total. Any design that discards older
origins from training shrinks an already-small dataset further.

**Label maturity.** The first implementation trained on every origin with
index `< i` for test origin `i`, with no check on whether that origin's own
label window had closed by the time of the test origin's `as_of`. With
`HORIZON_DAYS = 90` (ADR-0007) and `GAP_DAYS = 7`
(`config.DEFAULT_GAP_DAYS`), a label does not resolve until 97 days after
its own origin — more than three of this dataset's 30-day origin steps
(`3 × 30 = 90 < 97 ≤ 4 × 30 = 120`). That means the three most recent
origins before *any* test origin still had open, unresolved label windows
at that simulated point in time, every time. Training on them uses real,
correctly-computed labels that would not actually have been knowable yet
in a live deployment. This was verified directly (`origins[j].label_end`
against `origins[i].as_of` for every origin pair, not assumed) and
confirmed to inflate the pooled backtest ROC AUC from 0.765 (fixed) to
0.788 (bug) — an inflation the same rough size as several of the leaks
`01`/`02` diagnosed, found this time in the notebook meant to be correct.

## Decision

`rolling_origin_backtest` uses an **expanding** training window (every
origin strictly before the test origin is a training candidate, not just
the most recent `K`), filtered by a **label-maturity purge**: a training
origin is only used if `origin.label_end <= test_origin.as_of`. A test
origin left with zero mature training origins is skipped entirely rather
than trained on immature labels. For this dataset's constants, this always
purges exactly the three most recent origins before any test origin — a
property of `HORIZON_DAYS + GAP_DAYS` relative to `step_days`, not a fixed
number baked into the code.

## Alternatives considered

| Option | Why it was plausible | Why it was rejected |
|---|---|---|
| Fixed/rolling lookback window (train only on the most recent `K` origins) | Better simulates "only recent behaviour matters," if that were true, and is cheaper to retrain | No evidence in this dataset that purchase patterns go stale within the ~8-month backtest span, and it would shrink an already-small set of origins further |
| No maturity check (the original, buggy implementation) | Simpler; more origins usable as both training and test | Trains on labels that would not have been knowable yet at the simulated `as_of`, a real leak, not a simplification, in the one notebook whose entire purpose is being the correct baseline |

## Consequences

**Good:** the backtest's training data at every point is what would
actually have been available in a live deployment, not what an offline
dataset happens to already know. The purge mechanism (`label_end` vs.
`as_of`) generalizes to any horizon/gap/step combination rather than
hardcoding "skip 3."

**Bad:** the notebook loses one tested origin (six instead of seven), and
the first tested origin trains on only one prior origin's worth of data,
the least mature, most data-starved point in the backtest, visible in the
per-origin chart as the softest ROC AUC.

**Revisit if:** `HORIZON_DAYS`, `GAP_DAYS`, or `DEFAULT_STEP_DAYS` change
for this dataset (the purge count of 3 is a consequence of the current
values, not a constant) or when the KKBox scale-up in later notebooks
needs to re-derive its own purge count for that dataset's window sizes.
