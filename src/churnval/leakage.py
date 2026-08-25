"""Reusable checks for feature-label leakage.

These are fast guards, not the full diagnosis -- `02_leakage_diagnosis`
builds the richer adversarial-validation and ablation story. This module
exists so a minimal check ("does any single feature alone nearly determine
the label?") can run identically from a notebook's audit cell and from a
pytest regression test, rather than being reimplemented in each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def single_feature_auc(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """ROC AUC of each feature used alone as a ranking score.

    A feature can point either direction (a *higher* value could mean *more*
    or *less* likely to churn), so this reports `max(auc, 1 - auc)` per
    feature -- direction doesn't matter for deciding whether a feature is
    suspiciously close to encoding the label outright.

    Args:
        X: candidate features, one column per feature, no missing values.
        y: binary label, aligned to X by position.

    Returns:
        Series indexed by feature name, descending.
    """
    y = np.asarray(y)
    aucs = {column: roc_auc_score(y, X[column]) for column in X.columns}
    return pd.Series({k: max(v, 1 - v) for k, v in aucs.items()}).sort_values(ascending=False)


def assert_no_dominant_single_feature(
    X: pd.DataFrame, y: pd.Series, *, threshold: float = 0.99
) -> pd.Series:
    """Raise if any single feature alone nearly determines the label.

    A univariate AUC this high almost always means the feature measures the
    same underlying fact as the label -- a tautology or near-tautology (see
    `01_naive_baseline`'s `tautological_label` demonstration) -- not that
    the feature is a legitimately strong predictor. A coarse guard, not a
    substitute for the ablation study in `02_leakage_diagnosis`.

    Args:
        X: candidate features.
        y: binary label.
        threshold: univariate AUC above this raises.

    Returns:
        The per-feature AUC series (see `single_feature_auc`), so a caller
        that catches the exception can still inspect what triggered it.

    Raises:
        ValueError: at least one feature exceeds `threshold` alone.
    """
    aucs = single_feature_auc(X, y)
    offenders = aucs[aucs > threshold]
    if not offenders.empty:
        raise ValueError(
            f"feature(s) alone achieve ROC AUC > {threshold}, almost certainly "
            f"a tautology rather than a real signal: {offenders.to_dict()}"
        )
    return aucs
