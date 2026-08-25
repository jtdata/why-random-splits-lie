"""The reporting bundle.

The standard for this project is that a ranking metric is never reported alone.
Every result carries a discrimination metric, a calibration metric, and the
prevalence it was measured against — because AUC without prevalence is not
interpretable and a ranking without calibration is not a decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class Result:
    label: str
    n: int
    prevalence: float
    roc_auc: float
    pr_auc: float
    brier: float

    def as_row(self) -> dict[str, float | str | int]:
        return asdict(self)


def score(y_true: np.ndarray, y_prob: np.ndarray, label: str) -> Result:
    """Compute the full reporting bundle for one set of predictions.

    Args:
        y_true: binary outcomes.
        y_prob: predicted probabilities, not scores — Brier is meaningless on
            an uncalibrated decision function.
        label: how this result is described in the comparison table.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob must be probabilities in [0, 1]; got a raw score")

    return Result(
        label=label,
        n=int(y_true.size),
        prevalence=float(y_true.mean()),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
    )


def expected_value(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    *,
    offer_cost: float,
    saved_margin: float,
    save_rate: float,
) -> float:
    """Expected value of acting on everyone above ``threshold``.

    This is what a ranking metric cannot tell you. The assumptions are explicit
    arguments precisely so that they end up stated in the notebook rather than
    buried.

    Args:
        offer_cost: cost of extending a retention offer, incurred for every
            targeted entity regardless of outcome.
        saved_margin: margin retained when an offer prevents a churn.
        save_rate: fraction of genuinely-churning targeted entities the offer
            actually saves. This is the number nobody measures and everybody
            assumes.
    """
    if not 0.0 <= save_rate <= 1.0:
        raise ValueError("save_rate must be in [0, 1]")
    targeted = np.asarray(y_prob) >= threshold
    n_targeted = int(targeted.sum())
    true_churners_targeted = int(np.asarray(y_true)[targeted].sum())
    return true_churners_targeted * save_rate * saved_margin - n_targeted * offer_cost
