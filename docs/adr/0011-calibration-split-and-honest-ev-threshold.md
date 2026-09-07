# ADR-0011: Calibration uses a three-way temporal split, and the EV-optimal threshold is derived from stated costs, never chosen by searching evaluation outcomes

- **Status:** Accepted
- **Date:** 2026-08-26
- **Deciders:** JT Moeller

## Context

`05_calibration` checks whether the corrected protocol's (`03`) probabilities
are calibrated, and what miscalibration costs in an actual retention
decision. Two design questions had to be settled, and the second was only
settled correctly after an adversarial review caught a real defect in the
first attempt.

**The split.** `docs/standards/validation.md` requires a calibrator to be fit
on a held-out slice strictly later than the classifier's own training data.
Online Retail II's with-gap protocol has only 6 testable origins after
`03`'s maturity purge (ADR-0009), and 5 of those are needed to give the
classifier a reasonable amount of training data at all, leaving exactly one
origin-pair to spare for "fit the calibrator" and "evaluate it" as two
*separate* later origins.

**The EV threshold — the defect.** The first version of the
expected-value section swept a grid of threshold values
(`churnval.evaluation.sweep_expected_value`) over each probability variant
(uncalibrated, Platt, isotonic), picked each variant's threshold by
`argmax` over expected value *computed on the evaluation slice's own
labels*, and reported that number as "the EV-optimal decision." An
adversarial review of this notebook caught the defect: this is the same
in-sample-selection bias a random train/test split produces — searching
`N` options against the outcomes being scored, then reporting the best one
found — applied to a threshold search instead of a train/test split. It
produced a materially wrong headline: the peeked-at numbers said the
uncalibrated model made the *better* business decision ($1,169 vs. $1,111
for Platt), which would have undercut this project's own argument for
calibrating at all. Recomputed honestly (below), the ranking reverses
completely.

## Decision

**Split:** the calibration origin is the second-to-last testable origin
(2011-07-29); the evaluation origin is the last (2011-08-28, the same
origin `04`'s A/B/C comparison uses). The classifier is trained once, via
the new `churnval.splits.fit_at_origin`, on every origin whose label window
closes before the calibration origin — and is *not* retrained through the
calibration origin before scoring the evaluation origin. Folding the
calibration origin into training would leave nothing later than training
to fit a calibrator on; not retraining costs one origin's worth of
training data relative to what `04`'s arm C used for the same evaluation
origin (disclosed in the notebook, not hidden).

**EV threshold:** the reported expected value is computed at a threshold
derived *analytically* from the three stated cost assumptions
(`offer_cost / (save_rate * saved_margin)`, the standard Bayes
break-even rule), decided before `evaluation_y_true` is touched at all.
`sweep_expected_value`'s grid is kept only as a descriptive chart ("what EV
would result at every threshold"), explicitly labeled as not the source of
the reported number, with the biased peeked-at numbers shown alongside it
for contrast rather than deleted — the wrong answer, and why it's wrong, is
part of the notebook's argument now, the same way `01`'s wrong panel is
part of this project's argument.

## Alternatives considered

| Option | Why it was plausible | Why it was rejected |
|---|---|---|
| Keep the argmax-over-evaluation-labels threshold | Simpler; "sweep and take the best" is the first thing most tutorials do | It's biased in exactly the way this project exists to catch elsewhere — an adversarial review caught it, and its number was wrong enough to invert the notebook's own conclusion |
| Sweep on the calibration slice instead, pick that argmax, evaluate on the evaluation slice | Avoids peeking at the evaluation labels specifically | Still peeks at *some* labels to choose the threshold, and the calibration slice is small enough (n=4,317) that its own argmax would carry real sampling noise; the cost-derived threshold needs no data at all and is the theoretically correct rule if probabilities are calibrated, which is exactly the property being tested |
| A K-fold or rolling calibration comparison across more than one origin-pair | Would give a variance estimate, matching the rigor `03`/`04` apply to ranking metrics | Only one origin-pair is available after reserving 5 origins for training; not fixable within this dataset's window sizes (see ADR-0009's constants). Recorded as a disclosed limitation, not solved here — the KKBox scale-up in `07` should repeat this comparison with more origins to spare |

## Consequences

**Good:** the reported EV comparison is now unbiased and threshold-agnostic
— it doesn't depend on which of 19 grid points happened to look best on the
evaluation slice, and it directly tests the property that actually matters
for a probability threshold rule: whether `p` means what it claims to mean.
The result reverses cleanly and is arguably a *better* demonstration of why
calibration matters for decisions (a large EV gap, in the intuitive
direction) than the biased version ever showed.

**Bad:** the calibration/EV comparison rests on a single origin-pair with
97.1% customer overlap between the calibration and evaluation slices — the
finding shows the calibration holds for the same population a month later,
not that it generalizes to new customers, and carries no variance estimate
across regimes the way `03`/`04`'s ranking metrics do.

**Revisit if:** the KKBox scale-up (`07`) has enough testable origins to
spare more than one pair for calibration vs. evaluation — repeat this
comparison across several origin-pairs there rather than accepting the
single-pair limitation as permanent.
