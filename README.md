# Why random splits lie about churn

A controlled demonstration that the standard offline evaluation of churn models
overstates their real performance, and a worked protocol for doing it properly.

> **Status:** in progress. Results below are placeholders until the notebooks
> run end to end.

## The claim

Take a churn dataset with real event timestamps. Build a model the way most
tutorials do — random train/test split, features computed over the full
history. Report the AUC. Then rebuild the same model with a temporal split, an
explicit gap between the feature as-of date and the label window, and a
rolling-origin backtest.

The difference between those two numbers is not a modelling improvement. It is
a measurement of how much the first number was lying.

| Evaluation | ROC AUC | PR AUC | Brier |
|---|---|---|---|
| Random split (the wrong way) | TBD | TBD | TBD |
| Temporal split, no gap | TBD | TBD | TBD |
| Temporal split with gap, rolling origin | TBD | TBD | TBD |

## Why it matters

A model selected on an inflated offline metric is selected on the wrong
criterion. Worse, a *ranking* metric says nothing about whether the predicted
probabilities are usable for a retention decision — so this repo also covers
calibration and the expected-value threshold that follows from it.

## Datasets

Two panels, chosen because they have real event timestamps:

- **KKBox WSDM Churn Challenge** — contractual subscription churn, with
  membership expiry, renewals and cancellations.
- **Online Retail II (UCI)** — non-contractual churn, where the event is never
  observed and has to be inferred.

Showing the same failure in both settings is what makes this a method rather
than a case study. See [ADR-0003](docs/adr/0003-dataset-selection.md) for why
the popular Telco churn dataset cannot be used for this.

## Reproducing

```powershell
uv sync
uv run churnval fetch retail
.\make.ps1 nb
```

## Contents

- `notebooks/` — the argument, in order. Start at `00_problem_definition`.
- `src/churnval/` — the implementation.
- `docs/standards/` — the engineering and validation rules this repo follows.
- `docs/adr/` — why things are the way they are.

## Licence

MIT for the code. The datasets are covered by their own terms and are **not**
redistributed here — the fetch script downloads them from source.
