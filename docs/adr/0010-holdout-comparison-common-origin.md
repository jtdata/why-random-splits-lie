# ADR-0010: The naive-vs-correct holdout comparison uses one common origin, one common population, and isolates training/serving skew rather than "holdout cost"

- **Status:** Accepted
- **Date:** 2026-08-26
- **Deciders:** JT Moeller

## Context

`04_the_gap` originally added a holdout comparison to check whether each of
the three reported numbers (naive, no-gap, with-gap) survives scoring on an
origin never used in training. The naive arm was rebuilt by mirroring
`01`'s own "look-forward holdout" construction: train on every origin
before the naive panel's last one, score on that last origin.

Review of that construction found three real problems, all of which
happened to favor the naive arm:

1. **The naive holdout was still leaked.** `naive_features`' `recency_days`
   is measured to `transactions["invoice_date"].max()` (the dataset's own
   end date) for every row, regardless of that row's `as_of`. At the naive
   panel's last origin, that reference date sits *later than the end of
   that origin's own label window*, not just later than `as_of`. The
   feature partially encodes whether the label-defining purchase already
   happened. A holdout *split* does nothing to fix a feature that ignores
   `as_of` in the first place, the number looked like a clean forward
   test but was not one.
2. **The three arms trained on different data by accident, not by design.**
   The naive holdout pooled every earlier origin with no maturity check;
   the two temporal arms went through `rolling_origin_backtest`, which
   purges label-immature origins (ADR-0009). The comparison read as "how
   much does a real holdout cost each approach," which implicitly assumed
   the three models were trained comparably. They weren't, and the
   difference wasn't stated.
3. **The three arms weren't even scored on the same date.** The naive and
   no-gap origin sequences share `gap_days=0` and so share a last `as_of`;
   the with-gap sequence's maturity purge stops it one step earlier, so
   `per_origin_withgap.iloc[-1]` was evaluated at an earlier date than the
   other two arms.

None of these three problems is subtle once named, and all three bias the
comparison in the same direction: they make the naive model's holdout
number look better than it is.

## Decision

Rebuild the comparison around **one common holdout origin and one common
eligible population**, varying only what is under test, and reframe the
question from "how much does holdout cost" to "can the naive model
actually be served, and if you serve it honestly, how does it compare":

- The common `as_of` is `origins_withgap[-1].as_of` (the latest date the
  correctly-trained protocol can test at all) since it is the most
  constrained of the candidate origins by the maturity purge.
- The eligible population is computed once at that `as_of`
  (`churnval.splits.eligible_customers`) and asserted equal to the customer
  set `03`'s own with-gap panel independently produced for the same origin,
  checked, not assumed, since `naive_baseline.eligible_customers` and
  `splits.eligible_customers` are deliberately separate implementations.
- **Arm A — naive trained, naive features at scoring.** Reproduces what
  `01` would report at this date. Kept and shown, but labelled explicitly
  as impossible to serve, not as a legitimate result: it requires a feature
  computed from data that does not exist yet at scoring time.
- **Arm B — naive trained, as-of features at serving.** The same fitted
  model as A, scored with `churnval.features.asof_features` computed
  strictly before the common `as_of`, the features an actual deployment
  would have. This is the honest number for "what if we shipped the
  naively-trained model," and the gap between A and B is training/serving
  skew, named and computed directly.
- **Arm C — correctly trained, as-of features.** `03`'s own protocol,
  reused directly from the with-gap backtest's last row at the same
  origin, not refit.

The training-set difference between A/B (naive convention, no purge) and C
(purge-respecting) is kept, not eliminated. It is the thing being
compared, "naively trained" vs. "correctly trained," now stated as the
variable rather than left as an unstated confound.

## Alternatives considered

| Option | Why it was plausible | Why it was rejected |
|---|---|---|
| Fix the naive holdout's feature leak (compute `recency_days` to the holdout `as_of` instead of `T_end`) and keep the three-arm "holdout cost" framing | Smaller change; keeps the original chart | Recomputing "as-of-correct" features for the naive arm stops being a demonstration of mistake #1 at all.  At that point it is no longer the naive model, it is a re-labelled correct one, and the three-arm comparison still has problems 2 and 3 |
| Report only arm C, drop the naive holdout comparison entirely | Avoids the leak and the mismatch questions altogether | Throws away the actual finding review surfaced: that the naive model's number cannot be served, which is a sharper and more useful result than "the naive number is inflated," and one this project's audience benefits from seeing demonstrated, not just asserted |

## Consequences

**Good:** A, B, and C are scored on the identical rows against the
identical label, so the only thing that varies between B and C is training
regime, and the only thing that varies between A and B is which feature
table serves the same fitted model. The training/serving skew number (A
minus B) is a real, isolated quantity for the first time, and the
`leakage-audit` skill now has a mechanical check
(any feature whose reference date is later than the `as_of` of the row it
describes) that would have caught the arm-A leak on its own.

**Bad:** the comparison loses the earlier version's headline claim ("a
real holdout costs the naive model 4x what it costs the corrected ones").
That claim depended on the very construction this ADR replaces, and does
not survive the fix. What replaces it is narrower: the naive model cannot
be honestly served at all, and served honestly it is not obviously better
than the correct one, at one single origin rather than across a spread of
holdouts.

**Revisit if:** a later notebook wants a *spread* of training/serving-skew
estimates rather than one origin's worth. That would need arm A/B rebuilt
at every origin the naive sequence supports, not just the last, and a
restated common-population argument at each one.
