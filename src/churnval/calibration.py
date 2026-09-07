"""Recalibrating a classifier's probabilities, and measuring whether it worked.

`docs/standards/validation.md` requires calibration on a held-out slice that
is *later* than the training data, never on training folds that precede it --
fitting a calibrator on the same rows the classifier trained on lets the
calibrator memorize the classifier's own overconfidence rather than correct
it. This module only implements the fitting and the measurement; which rows
play which temporal role is a notebook-level decision (`05_calibration`
builds the three-way split), not this module's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def fit_platt(raw_prob: np.ndarray, y_true: np.ndarray) -> LogisticRegression:
    """Platt scaling (Platt, 1999): a sigmoid refit of the classifier's output.

    Classically a logistic regression on an SVM's raw margin; applied here
    directly to the base classifier's own probability output instead, since
    it already outputs a probability rather than a margin -- the common
    generalization when the base model isn't an SVM. A single-feature
    logistic regression can only stretch or compress the probability scale
    monotonically, so it corrects systematic over/under-confidence but
    cannot fix a non-monotonic miscalibration pattern; that's what isotonic
    regression is for.

    Args:
        raw_prob: the classifier's uncalibrated predicted probabilities, on
            the slice reserved for calibration.
        y_true: binary outcomes on that same slice.

    Returns:
        A fitted `LogisticRegression` taking `raw_prob.reshape(-1, 1)` and
        returning calibrated probabilities via `.predict_proba(...)[:, 1]`.
    """
    model = LogisticRegression()
    model.fit(np.asarray(raw_prob).reshape(-1, 1), np.asarray(y_true))
    return model


def fit_isotonic(raw_prob: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """Isotonic regression calibration (Zadrozny & Elkan, 2002).

    A non-parametric, monotonic step function fit directly to
    (raw probability, outcome) pairs -- more flexible than Platt's sigmoid,
    at the cost of needing more calibration data to avoid overfitting the
    step boundaries to noise.

    Args:
        raw_prob: the classifier's uncalibrated predicted probabilities, on
            the slice reserved for calibration.
        y_true: binary outcomes on that same slice.

    Returns:
        A fitted `IsotonicRegression`; call `.predict(raw_prob)` to get
        calibrated probabilities. `out_of_bounds="clip"` so a probability
        outside the calibration slice's own range doesn't extrapolate.
    """
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(np.asarray(raw_prob), np.asarray(y_true))
    return model


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Bin predictions and compare mean predicted probability to observed rate.

    The data underlying a reliability curve: a well-calibrated model has
    `observed_rate` close to `mean_predicted` in every bin. Equal-width bins
    over `[0, 1]`, not equal-count, so a bin's width is comparable across
    charts even as the data underneath it changes.

    Args:
        y_true: binary outcomes.
        y_prob: predicted probabilities in `[0, 1]`.
        n_bins: number of equal-width bins spanning `[0, 1]`.

    Returns:
        One row per *non-empty* bin: `bin_lower`, `bin_upper`, `n`,
        `mean_predicted`, `observed_rate`. Empty bins are dropped rather
        than shown as a misleading zero.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    frame = pd.DataFrame({"y_true": np.asarray(y_true), "y_prob": np.asarray(y_prob)})
    # `labels=False` returns the integer bin index rather than a pandas
    # `Interval`, whose `.left`/`.right` pandas nudges by a small epsilon
    # under `include_lowest=True` -- indexing `edges` directly keeps the
    # reported bounds exactly the values passed in.
    frame["bin_idx"] = pd.cut(frame["y_prob"], bins=edges, labels=False, include_lowest=True)

    grouped = frame.groupby("bin_idx", observed=True)
    table = grouped.agg(
        n=("y_true", "size"), mean_predicted=("y_prob", "mean"), observed_rate=("y_true", "mean")
    )
    bin_idx = table.index.to_numpy().astype(int)
    table["bin_lower"] = edges[bin_idx]
    table["bin_upper"] = edges[bin_idx + 1]
    return table.reset_index(drop=True)[
        ["bin_lower", "bin_upper", "n", "mean_predicted", "observed_rate"]
    ]


def brier_decomposition(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """The Murphy (1973) three-term decomposition of the Brier score.

    `brier = uncertainty - resolution + reliability`:

    * `uncertainty = p * (1 - p)` (`p` the overall base rate) -- how hard the
      problem is regardless of any model, the score a model that always
      predicts the base rate would get.
    * `resolution` -- how much predicted probabilities separate genuinely
      different observed rates across bins. Higher is better: it means the
      model's bins have observed rates that differ from the base rate.
    * `reliability` -- how far each bin's mean predicted probability sits
      from that bin's own observed rate. Lower is better; zero is perfect
      calibration.

    This identity is exact for the *binned* Brier score -- the score you'd
    get if every prediction in a bin were replaced by that bin's mean
    predicted probability -- not for the raw, per-instance Brier score
    `churnval.evaluation.score` reports, which uses each row's own
    unrounded probability. The two converge as `n_bins` grows and each bin's
    internal spread shrinks; they are not asserted equal here, and reporting
    both side by side (this function's `brier_binned` and `score`'s `brier`)
    is deliberate, not an inconsistency to reconcile.

    Args:
        y_true: binary outcomes.
        y_prob: predicted probabilities in `[0, 1]`.
        n_bins: number of equal-width bins spanning `[0, 1]`, matching
            `reliability_table`.

    Returns:
        `reliability`, `resolution`, `uncertainty`, and `brier_binned`
        (`= uncertainty - resolution + reliability`, by construction).
    """
    table = reliability_table(y_true, y_prob, n_bins=n_bins)
    n_total = int(table["n"].sum())
    p_bar = float(np.asarray(y_true).mean())

    weights = table["n"] / n_total
    reliability = float((weights * (table["mean_predicted"] - table["observed_rate"]) ** 2).sum())
    resolution = float((weights * (table["observed_rate"] - p_bar) ** 2).sum())
    uncertainty = p_bar * (1 - p_bar)

    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_binned": uncertainty - resolution + reliability,
    }
