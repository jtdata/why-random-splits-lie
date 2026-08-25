# Validation standards

This is the substantive standard of the portfolio — the others are hygiene.

## The default

Any model trained on data with a time axis is validated by time. A random split
requires a written justification in the notebook stating why the data has no
temporal structure that matters. "It performed similarly" is not a
justification.

## Required elements of a temporal design

1. **A timeline diagram.** Observation window, feature as-of date, gap,
   prediction window, label window. Drawn before any code is written. If the
   diagram cannot be drawn, the problem is not yet defined.
2. **An explicit gap** between the feature as-of date and the start of the
   label window. The gap is at least the operational lead time — the delay
   between scoring and acting. With no gap, features computed near the event
   encode the event.
3. **Features snapshotted as-of.** Every aggregation is computed over data
   available strictly before the as-of timestamp. Enforced in code, not by
   convention.
4. **Rolling-origin backtest.** Several origins, retraining forward at each.
   A single split is one draw and cannot show variance across regimes.
5. **A maturity check.** For each feature, confirm the value would have been
   populated at scoring time, not merely that the column exists.

## Metrics

- Report a **ranking** metric and a **calibration** metric together. ROC AUC
  alone is not an acceptable result. Pair it with PR AUC where the positive
  class is rare, and always with Brier score plus a reliability curve.
- Report the **uncorrected** metric alongside the corrected one whenever a
  validation flaw is fixed. The gap is the finding; hiding it hides the point.
- Where a decision follows from the prediction, report the expected value at
  the chosen threshold, with the cost assumptions stated.

## Calibration

- Probabilities used for a decision must be calibrated. Fit the calibrator on a
  held-out slice that is *later* than the training data, never on training folds
  that precede it, or the calibration itself leaks.
- Compare isotonic and Platt, and state which was chosen and why. Isotonic needs
  more data; Platt assumes a shape.
- A better-ranked but poorly-calibrated model is not automatically preferred.
  Show the decision consequence.

## Labels

- State whether the recorded date is the event date or the observation date.
- Entities whose outcome window has not closed are **censored**, not negatives.
  Either exclude them or model them as censored — never silently label them 0.
- Where "when" matters as well as "whether", fit a hazard model alongside the
  binary classifier and compare what each gets wrong.

## The audit

Before any result is reported, run the `leakage-audit` skill over the feature
set and the split. Record its findings in the notebook, including the ones that
came back clean.
