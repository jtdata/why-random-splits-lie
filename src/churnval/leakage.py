"""Reusable checks for feature-label leakage.

`single_feature_auc` and `assert_no_dominant_single_feature` are fast guards
used from `01_naive_baseline`. `adversarial_validation_auc` and
`ablation_auc` are the richer diagnostics `02_leakage_diagnosis` builds and
uses -- kept here rather than as one-off notebook code so the same function
runs identically from a notebook cell and from a pytest regression test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


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


def adversarial_validation_auc(
    X_a: pd.DataFrame,
    X_b: pd.DataFrame,
    *,
    seed: int,
    test_size: float = 0.3,
    groups: pd.Series | None = None,
) -> tuple[float, pd.Series]:
    """AUC of a classifier trained to tell rows of `X_a` from rows of `X_b`.

    An AUC materially above 0.5 means the two groups are separable on these
    features -- evidence of drift or leakage, depending on what A and B are.
    An AUC near 0.5 means they are *not* separable on these features, which
    is not automatically "no leak": it only rules out a leak that shows up
    as a distributional difference in this feature set between these two
    groups. `02_leakage_diagnosis` uses this same function on two different
    pairs of groups and gets two different answers, which is the point.

    Args:
        X_a: features from group A (e.g. train rows, or naively-computed
            features).
        X_b: features from group B (e.g. test rows, or as-of-correctly
            computed features), with the same columns as `X_a`.
        seed: passed to both the split and the classifier.
        test_size: fraction of the combined rows held out to score the AUC.
        groups: pass this when `X_a` and `X_b` are *matched* -- row `i` of
            `X_a` and row `i` of `X_b` describe the same underlying entity,
            just measured two ways (e.g. the same (customer, as_of) row's
            naive vs. as-of-correct features). Give the same length as
            `X_a`/`X_b`; it is reused for both, since a matched row shares
            its entity across the pair. Without this, the classifier's own
            internal split can put one version of an entity in its internal
            train fold and the other version of the *same* entity in its
            internal test fold, letting it partly recognize the entity
            (e.g. "this customer has large values in general") rather than
            the actual difference between the two computations, inflating
            the AUC. With `groups`, both versions of an entity always land
            on the same side of the classifier's internal split.

    Returns:
        `(auc, feature_importances)` -- importances indexed like the shared
        columns of `X_a`/`X_b`, from the fitted classifier, descending.
    """
    combined = pd.concat([X_a, X_b], ignore_index=True)
    labels = np.array([0] * len(X_a) + [1] * len(X_b))

    if groups is None:
        X_train, X_test, y_train, y_test = train_test_split(
            combined, labels, test_size=test_size, random_state=seed, stratify=labels
        )
    else:
        combined_groups = pd.concat([groups, groups], ignore_index=True)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(combined, labels, groups=combined_groups))
        X_train, X_test = combined.iloc[train_idx], combined.iloc[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

    model = LGBMClassifier(random_state=seed, verbosity=-1)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    importances = pd.Series(model.feature_importances_, index=combined.columns).sort_values(
        ascending=False
    )
    return float(auc), importances


def ablation_auc(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    seed: int,
) -> pd.Series:
    """Drop-one-feature ablation: refit with each feature removed in turn.

    A feature whose removal collapses ROC AUC toward chance is either the
    whole signal or the whole leak. A feature whose removal costs almost
    nothing is redundant with the others, not necessarily innocent -- see
    `single_feature_auc` for the complementary univariate view.

    Args:
        X_train: training features.
        y_train: training labels.
        X_test: held-out features, same columns as `X_train`.
        y_test: held-out labels.
        seed: passed to every refit.

    Returns:
        Series indexed `"all features"` then `"drop <column>"` for each
        column of `X_train`, in that order -- ROC AUC on `X_test`/`y_test`.
    """
    results = {}
    baseline = LGBMClassifier(random_state=seed, verbosity=-1)
    baseline.fit(X_train, y_train)
    results["all features"] = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])

    for column in X_train.columns:
        remaining = [c for c in X_train.columns if c != column]
        model = LGBMClassifier(random_state=seed, verbosity=-1)
        model.fit(X_train[remaining], y_train)
        results[f"drop {column}"] = roc_auc_score(
            y_test, model.predict_proba(X_test[remaining])[:, 1]
        )

    return pd.Series(results)
