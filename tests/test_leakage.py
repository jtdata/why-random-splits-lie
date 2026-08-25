"""Tests for the single-feature leakage guard.

Expected AUCs are hand-computed: with 4 rows split 2 negatives / 2 positives,
there are 2x2=4 negative-positive pairs, and ROC AUC is the fraction of those
pairs the feature ranks correctly (negative < positive).
"""

from __future__ import annotations

import pandas as pd
import pytest

from churnval.leakage import assert_no_dominant_single_feature, single_feature_auc

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
