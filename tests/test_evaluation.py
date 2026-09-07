import numpy as np
import pytest

from churnval.evaluation import expected_value, score, sweep_expected_value

# Perfect separation: both positives (indices 1, 2) rank above both
# negatives (indices 0, 3) -> roc_auc = 1.0 and pr_auc (average precision)
# = 1.0 exactly, regardless of the specific probability values.
Y_TRUE = np.array([0, 1, 1, 0])
Y_PROB = np.array([0.2, 0.8, 0.6, 0.4])


def test_score_matches_hand_computation():
    # brier = mean((prob - y)^2) = (0.04 + 0.04 + 0.16 + 0.16) / 4 = 0.10
    result = score(Y_TRUE, Y_PROB, label="test")
    assert result.label == "test"
    assert result.n == 4
    assert result.prevalence == pytest.approx(0.5)
    assert result.roc_auc == pytest.approx(1.0)
    assert result.pr_auc == pytest.approx(1.0)
    assert result.brier == pytest.approx(0.10)


def test_score_as_row_is_a_plain_dict():
    row = score(Y_TRUE, Y_PROB, label="test").as_row()
    assert row == {
        "label": "test",
        "n": 4,
        "prevalence": pytest.approx(0.5),
        "roc_auc": pytest.approx(1.0),
        "pr_auc": pytest.approx(1.0),
        "brier": pytest.approx(0.10),
    }


def test_score_rejects_raw_scores_outside_probability_range():
    with pytest.raises(ValueError, match="probabilities"):
        score(Y_TRUE, np.array([0.2, 1.8, 0.6, 0.4]), label="bad")


def test_expected_value_matches_hand_computation():
    # threshold=0.5 targets indices 1, 2 (probs 0.8, 0.6); both are true
    # churners: EV = 2 * 0.5 * 10.0 - 2 * 1.0 = 8.0
    ev = expected_value(
        Y_TRUE, Y_PROB, threshold=0.5, offer_cost=1.0, saved_margin=10.0, save_rate=0.5
    )
    assert ev == pytest.approx(8.0)


def test_expected_value_targets_everyone_at_threshold_zero():
    # threshold=0.0 targets all 4 rows; 2 are true churners:
    # EV = 2 * 0.5 * 10.0 - 4 * 1.0 = 6.0
    ev = expected_value(
        Y_TRUE, Y_PROB, threshold=0.0, offer_cost=1.0, saved_margin=10.0, save_rate=0.5
    )
    assert ev == pytest.approx(6.0)


def test_expected_value_rejects_save_rate_outside_unit_interval():
    with pytest.raises(ValueError, match="save_rate"):
        expected_value(
            Y_TRUE, Y_PROB, threshold=0.5, offer_cost=1.0, saved_margin=10.0, save_rate=1.5
        )


def test_sweep_expected_value_matches_hand_computation_at_each_threshold():
    # threshold=0.9: nothing clears the bar (max prob is 0.8) -> EV = 0.0
    sweep = sweep_expected_value(
        Y_TRUE,
        Y_PROB,
        thresholds=[0.9, 0.0, 0.5],
        offer_cost=1.0,
        saved_margin=10.0,
        save_rate=0.5,
    )
    assert list(sweep["threshold"]) == [0.0, 0.5, 0.9]
    assert list(sweep["n_targeted"]) == [4, 2, 0]
    assert sweep["expected_value"].to_numpy() == pytest.approx([6.0, 8.0, 0.0])
