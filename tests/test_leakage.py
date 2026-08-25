"""Tests for the leakage-diagnosis tools.

`single_feature_auc`/`assert_no_dominant_single_feature` expected AUCs are
hand-computed: with 4 rows split 2 negatives / 2 positives, there are
2x2=4 negative-positive pairs, and ROC AUC is the fraction of those pairs
the feature ranks correctly (negative < positive).

`adversarial_validation_auc` and `ablation_auc` wrap a fitted LightGBM
model, so an exact decimal can't be hand-derived the same way. Instead each
test constructs synthetic data with a ground truth provable from its
construction (two groups drawn from the same distribution are not
separable by any classifier in expectation; two groups with non-overlapping
support on a feature are perfectly separable by a single split) and asserts
a threshold consistent with that ground truth, not a value read back from
the function itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from churnval.leakage import (
    ablation_auc,
    adversarial_validation_auc,
    assert_no_dominant_single_feature,
    single_feature_auc,
)

Y = [0, 0, 1, 1]


@pytest.fixture
def features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            # every negative (1, 2) ranks below every positive (3, 4):
            # 4/4 pairs correct -> AUC = 1.0
            "perfect_ascending": [1, 2, 3, 4],
            # the same relationship, inverted: 0/4 pairs correct in the raw
            # direction (AUC = 0.0), but max(auc, 1 - auc) = 1.0
            "perfect_descending": [4, 3, 2, 1],
            # pairs: (1,2)ok, (1,4)ok, (3,2)wrong, (3,4)ok -> 3/4 -> AUC=0.75
            "partially_predictive": [1, 3, 2, 4],
        }
    )


def test_single_feature_auc_takes_the_stronger_direction(features):
    aucs = single_feature_auc(features, Y)
    assert aucs["perfect_ascending"] == pytest.approx(1.0)
    assert aucs["perfect_descending"] == pytest.approx(1.0)
    assert aucs["partially_predictive"] == pytest.approx(0.75)


def test_assert_no_dominant_single_feature_raises_on_the_perfect_features(features):
    with pytest.raises(ValueError, match="perfect_ascending"):
        assert_no_dominant_single_feature(features, Y, threshold=0.99)


def test_assert_no_dominant_single_feature_passes_below_threshold():
    weak = pd.DataFrame({"partially_predictive": [1, 3, 2, 4]})
    aucs = assert_no_dominant_single_feature(weak, Y, threshold=0.99)
    assert aucs["partially_predictive"] == pytest.approx(0.75)


def test_adversarial_validation_auc_is_near_chance_for_identically_distributed_groups():
    # Both groups are iid draws from the same distribution, so no
    # classifier can separate them better than chance in expectation.
    rng = np.random.default_rng(0)
    columns = {"x1": rng.normal(size=400), "x2": rng.normal(size=400)}
    X_a = pd.DataFrame({k: v[:200] for k, v in columns.items()})
    X_b = pd.DataFrame({k: v[200:] for k, v in columns.items()})

    auc, importances = adversarial_validation_auc(X_a, X_b, seed=0)

    assert 0.35 < auc < 0.65
    assert set(importances.index) == {"x1", "x2"}


def test_adversarial_validation_auc_with_groups_stays_near_chance_with_no_signal():
    # Matched pairs (same entity, same value in both groups) with no
    # separating signal at all -- grouping must not conjure signal out of
    # nothing.
    rng = np.random.default_rng(0)
    entity_value = rng.normal(size=250)
    X_a = pd.DataFrame({"x1": entity_value})
    X_b = pd.DataFrame({"x1": entity_value.copy()})
    entity_id = pd.Series(range(250))

    auc, _ = adversarial_validation_auc(X_a, X_b, seed=0, groups=entity_id)

    assert 0.3 < auc < 0.7


def test_adversarial_validation_auc_with_groups_still_detects_real_separation():
    # A genuinely separable signal (non-overlapping support on x1) that does
    # not depend on entity identity -- grouping must not destroy it.
    rng = np.random.default_rng(0)
    X_a = pd.DataFrame({"x1": rng.uniform(0, 1, size=200)})
    X_b = pd.DataFrame({"x1": rng.uniform(10, 11, size=200)})
    entity_id = pd.Series(range(200))

    auc, _ = adversarial_validation_auc(X_a, X_b, seed=0, groups=entity_id)

    assert auc > 0.95


def test_adversarial_validation_auc_is_near_one_for_separable_groups():
    # x1's support does not overlap between the two groups -- a single split
    # on x1 perfectly separates them.
    rng = np.random.default_rng(0)
    X_a = pd.DataFrame({"x1": rng.uniform(0, 1, size=200), "x2": rng.normal(size=200)})
    X_b = pd.DataFrame({"x1": rng.uniform(10, 11, size=200), "x2": rng.normal(size=200)})

    auc, importances = adversarial_validation_auc(X_a, X_b, seed=0)

    assert auc > 0.95
    assert importances["x1"] > importances["x2"]


def test_ablation_auc_collapses_only_when_the_signal_feature_is_dropped():
    # `signal` alone determines y; `noise1`/`noise2` are pure noise. Dropping
    # `signal` should collapse AUC toward chance; dropping a noise column
    # should not move it.
    rng = np.random.default_rng(0)
    n = 600
    signal = rng.uniform(size=n)
    y = (signal > np.median(signal)).astype(int)
    X = pd.DataFrame(
        {
            "signal": signal,
            "noise1": rng.normal(size=n),
            "noise2": rng.normal(size=n),
        }
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )

    result = ablation_auc(X_train, y_train, X_test, y_test, seed=0)

    assert result["all features"] > 0.95
    assert result["drop signal"] < 0.65
    assert result["drop noise1"] > 0.95
    assert result["drop noise2"] > 0.95
