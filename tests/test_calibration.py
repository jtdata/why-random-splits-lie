import numpy as np
import pytest

from churnval.calibration import brier_decomposition, fit_isotonic, fit_platt, reliability_table

# Two equal-width bins over [0, 1]: (0.0-0.5] and (0.5-1.0].
# Bin 1: probs 0.1, 0.2, 0.3, 0.4; outcomes 0, 0, 1, 1 -> mean_predicted 0.25, observed_rate 0.5
# Bin 2: probs 0.6, 0.7, 0.8, 0.9; outcomes 0, 1, 1, 1 -> mean_predicted 0.75, observed_rate 0.75
Y_TRUE = np.array([0, 0, 1, 1, 0, 1, 1, 1])
Y_PROB = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])


def test_reliability_table_bins_and_rates_match_hand_computation():
    table = reliability_table(Y_TRUE, Y_PROB, n_bins=2)

    assert list(table["n"]) == [4, 4]
    assert table["mean_predicted"].to_numpy() == pytest.approx([0.25, 0.75])
    assert table["observed_rate"].to_numpy() == pytest.approx([0.5, 0.75])
    assert table["bin_lower"].to_numpy() == pytest.approx([0.0, 0.5])
    assert table["bin_upper"].to_numpy() == pytest.approx([0.5, 1.0])


def test_reliability_table_drops_empty_bins():
    # All predictions sit in the top half of [0, 1] -- the bottom half's
    # bins must not appear as spurious n=0 rows.
    y_prob = np.array([0.6, 0.7, 0.8, 0.9])
    y_true = np.array([0, 1, 1, 1])
    table = reliability_table(y_true, y_prob, n_bins=10)
    assert (table["n"] > 0).all()
    assert table["bin_lower"].min() >= 0.5


def test_brier_decomposition_matches_hand_computation():
    # p_bar = 5/8 = 0.625; uncertainty = 0.625 * 0.375 = 0.234375
    # reliability = 0.5*(0.25-0.5)^2 + 0.5*(0.75-0.75)^2 = 0.03125
    # resolution  = 0.5*(0.5-0.625)^2 + 0.5*(0.75-0.625)^2 = 0.015625
    # brier_binned = 0.234375 - 0.015625 + 0.03125 = 0.25
    result = brier_decomposition(Y_TRUE, Y_PROB, n_bins=2)
    assert result["uncertainty"] == pytest.approx(0.234375)
    assert result["reliability"] == pytest.approx(0.03125)
    assert result["resolution"] == pytest.approx(0.015625)
    assert result["brier_binned"] == pytest.approx(0.25)


def test_brier_decomposition_identity_holds_regardless_of_binning():
    # uncertainty - resolution + reliability == brier_binned is an algebraic
    # identity by construction -- true for any n_bins, not just the
    # hand-checked case above.
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0, 1, size=200)
    y_true = rng.integers(0, 2, size=200)
    for n_bins in (4, 5, 20):
        result = brier_decomposition(y_true, y_prob, n_bins=n_bins)
        expected = result["uncertainty"] - result["resolution"] + result["reliability"]
        assert result["brier_binned"] == pytest.approx(expected)


def test_fit_isotonic_matches_hand_computed_pool_adjacent_violators():
    # x = [0.1, 0.2, 0.3], y = [1, 0, 1]: y violates monotonicity between the
    # first two points (1 > 0), so pool-adjacent-violators averages them to
    # 0.5 each; the third point stays at 1.0. Worked by hand, not read back
    # from the fitted model.
    raw_prob = np.array([0.1, 0.2, 0.3])
    y_true = np.array([1, 0, 1])
    model = fit_isotonic(raw_prob, y_true)
    assert model.predict(raw_prob) == pytest.approx([0.5, 0.5, 1.0])


def test_fit_isotonic_is_monotonic_non_decreasing():
    rng = np.random.default_rng(1)
    raw_prob = np.sort(rng.uniform(0, 1, size=50))
    y_true = rng.integers(0, 2, size=50)
    model = fit_isotonic(raw_prob, y_true)
    calibrated = model.predict(raw_prob)
    assert np.all(np.diff(calibrated) >= -1e-12)


def test_fit_platt_is_monotonic_and_bounded():
    # A clearly positively-correlated fixture: platt scaling is a
    # single-feature logistic regression, so a positive relationship in the
    # data must fit a positive coefficient, which makes predict_proba
    # non-decreasing in raw_prob by construction.
    raw_prob = np.linspace(0.05, 0.95, 40)
    y_true = (raw_prob > 0.5).astype(int)
    model = fit_platt(raw_prob, y_true)

    assert model.coef_[0, 0] > 0
    calibrated = model.predict_proba(raw_prob.reshape(-1, 1))[:, 1]
    assert np.all(np.diff(calibrated) >= -1e-12)
    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0
