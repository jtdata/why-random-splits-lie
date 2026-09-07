"""Tests for the discrete-time hazard framing.

`first_event_period`/`build_person_period_panel`/`survival_from_hazards` are
tested against hand-computed expected values. `predict_survival` is tested
against a stub model with a known, hand-computable hazard function, isolating
the aggregation logic from any real model fit.
`hazard_rolling_origin_backtest` wraps a fitted LightGBM model, so it is
tested structurally, the same approach `test_splits.py` uses for
`rolling_origin_backtest`.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from churnval.hazard import (
    build_person_period_panel,
    first_event_period,
    fit_hazard_at_origin,
    hazard_curve,
    hazard_rolling_origin_backtest,
    predict_survival,
    survival_from_hazards,
)
from churnval.windows import Window

# as_of=2011-01-01, gap=0, horizon=90 -> label window [2011-01-01, 2011-04-01).
# period_days=30 -> period 1: [Jan 1, Jan 31), period 2: [Jan 31, Mar 2),
# period 3: [Mar 2, Apr 1).
#
# Customer 501: purchase 2011-01-10 -- day 9 into the window -> period 1.
# Customer 502: purchase 2011-02-10 -- day 40 into the window -> period 2.
# Customer 503: no purchase in the window at all -- censored.
WINDOW = Window(as_of=pd.Timestamp("2011-01-01"), gap_days=0, horizon_days=90)


@pytest.fixture
def transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [501, 502],
            "invoice_date": pd.to_datetime(["2011-01-10", "2011-02-10"]),
        }
    )


@pytest.fixture
def panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [501, 502, 503],
            "as_of": WINDOW.as_of,
            "signal": [1.0, 2.0, 3.0],
            "churned": [0, 0, 1],
        }
    )


def test_first_event_period_matches_hand_computation(transactions):
    result = first_event_period(transactions, WINDOW, time_col="invoice_date", period_days=30)
    assert result.to_dict() == {501: 1, 502: 2}
    assert 503 not in result.index


def test_first_event_period_rejects_horizon_not_a_multiple_of_period():
    bad_window = Window(as_of=pd.Timestamp("2011-01-01"), gap_days=0, horizon_days=100)
    with pytest.raises(ValueError, match="multiple"):
        first_event_period(
            pd.DataFrame(columns=["customer_id", "invoice_date"]),
            bad_window,
            time_col="invoice_date",
            period_days=30,
        )


def test_first_event_period_rejects_a_time_col_that_does_not_exist():
    # Regression test: time_col has no default, so a caller pointing it at a
    # dataset that doesn't have "invoice_date" must get a real error, not a
    # silent fallback to some retail-specific column name.
    transactions = pd.DataFrame(
        {
            "customer_id": [1],
            "event_date": [pd.Timestamp("2011-01-10")],
        }
    )
    with pytest.raises(KeyError):
        first_event_period(transactions, WINDOW, time_col="invoice_date", period_days=30)


def test_first_event_period_works_with_a_non_retail_time_col_name():
    transactions = pd.DataFrame(
        {
            "customer_id": [501],
            "event_date": [pd.Timestamp("2011-01-10")],
        }
    )
    result = first_event_period(transactions, WINDOW, time_col="event_date", period_days=30)
    assert result.to_dict() == {501: 1}


def test_build_person_period_panel_matches_hand_computation(transactions, panel):
    result = build_person_period_panel(
        transactions, [WINDOW], panel, feature_cols=["signal"], time_col="invoice_date"
    )

    rows = {(r.customer_id, r.period): r.event for r in result.itertuples()}
    assert rows == {
        (501, 1): 1,
        (502, 1): 0,
        (502, 2): 1,
        (503, 1): 0,
        (503, 2): 0,
        (503, 3): 0,
    }
    # Features carry over unchanged across every period row for a customer.
    assert (result.loc[result["customer_id"] == 503, "signal"] == 3.0).all()


def test_survival_from_hazards_matches_hand_computation():
    hazard_frame = pd.DataFrame(
        {
            "customer_id": [1, 1, 1, 2, 2],
            "period": [1, 2, 3, 1, 2],
            "hazard": [0.1, 0.2, 0.3, 0.5, 0.5],
        }
    )
    result = survival_from_hazards(hazard_frame)
    # customer 1: (1-0.1)*(1-0.2)*(1-0.3) = 0.9*0.8*0.7 = 0.504
    # customer 2: (1-0.5)*(1-0.5) = 0.25
    assert result.loc[1] == pytest.approx(0.504)
    assert result.loc[2] == pytest.approx(0.25)


class _StubHazardModel:
    """hazard = 0.1 * period, independent of any other feature."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        hazard = 0.1 * X["period"].to_numpy()
        return np.column_stack([1 - hazard, hazard])


def test_hazard_curve_matches_hand_computation_with_stub_model():
    panel_slice = pd.DataFrame({"customer_id": [1, 2], "signal": [5.0, 9.0]})
    result = hazard_curve(_StubHazardModel(), panel_slice, feature_cols=["signal"], n_periods=3)
    rows = {(r.customer_id, r.period): r.hazard for r in result.itertuples()}
    # hazard = 0.1 * period, independent of "signal" -- both customers get
    # the same three period hazards.
    for customer_id in (1, 2):
        assert rows[(customer_id, 1)] == pytest.approx(0.1)
        assert rows[(customer_id, 2)] == pytest.approx(0.2)
        assert rows[(customer_id, 3)] == pytest.approx(0.3)


def test_predict_survival_matches_hand_computation_with_stub_model():
    panel_slice = pd.DataFrame({"customer_id": [1], "signal": [5.0]})
    result = predict_survival(_StubHazardModel(), panel_slice, feature_cols=["signal"], n_periods=3)
    # hazards 0.1, 0.2, 0.3 -> survival = 0.9*0.8*0.7 = 0.504, same as above.
    assert result.loc[1] == pytest.approx(0.504)


def test_hazard_rolling_origin_backtest_tests_only_mature_origins_and_scores_well():
    # Origin A's label window closes well before origin B's as_of, so B is
    # the only testable origin at min_training_origins=1, trained on A.
    origin_a = Window(as_of=pd.Timestamp("2011-01-01"), gap_days=0, horizon_days=90)
    origin_b = Window(
        as_of=pd.Timestamp("2011-01-01") + timedelta(days=100), gap_days=0, horizon_days=90
    )
    origins = [origin_a, origin_b]

    rng = np.random.default_rng(0)
    panel_rows = []
    transaction_rows = []
    for i, window in enumerate(origins):
        signal = rng.uniform(size=100)
        will_return = signal > 0.5
        for k in range(100):
            customer_id = f"{i}-{k}"
            panel_rows.append(
                {
                    "customer_id": customer_id,
                    "as_of": window.as_of,
                    "signal": signal[k],
                    "churned": int(not will_return[k]),
                }
            )
            if will_return[k]:
                # Returns 5 days into the window -- period 1.
                transaction_rows.append(
                    {"customer_id": customer_id, "invoice_date": window.as_of + timedelta(days=5)}
                )
    panel = pd.DataFrame(panel_rows)
    transactions = pd.DataFrame(transaction_rows)

    person_period_panel = build_person_period_panel(
        transactions, origins, panel, feature_cols=["signal"], time_col="invoice_date"
    )
    per_origin, predictions = hazard_rolling_origin_backtest(
        person_period_panel, panel, origins, feature_cols=["signal"], min_training_origins=1, seed=0
    )

    assert list(per_origin["as_of"]) == [origin_b.as_of]
    assert per_origin["n"].iloc[0] == 100
    assert per_origin["roc_auc"].iloc[0] > 0.9
    assert len(predictions) == 100


def test_fit_hazard_at_origin_matches_hazard_rolling_origin_backtests_own_purge_and_predictions():
    # Three origins: A and B are both mature relative to C (A's label
    # window closes day 90, B's closes day 190, both well before C's
    # as_of on day 250), so fit_hazard_at_origin(target_index=2) should
    # train on both and reproduce hazard_rolling_origin_backtest's own
    # predictions for C exactly.
    base = pd.Timestamp("2011-01-01")
    origin_a = Window(as_of=base, gap_days=0, horizon_days=90)
    origin_b = Window(as_of=base + timedelta(days=100), gap_days=0, horizon_days=90)
    origin_c = Window(as_of=base + timedelta(days=250), gap_days=0, horizon_days=90)
    origins = [origin_a, origin_b, origin_c]

    rng = np.random.default_rng(0)
    panel_rows = []
    transaction_rows = []
    for i, window in enumerate(origins):
        signal = rng.uniform(size=50)
        will_return = signal > 0.5
        for k in range(50):
            customer_id = f"{i}-{k}"
            panel_rows.append(
                {
                    "customer_id": customer_id,
                    "as_of": window.as_of,
                    "signal": signal[k],
                    "churned": int(not will_return[k]),
                }
            )
            if will_return[k]:
                transaction_rows.append(
                    {"customer_id": customer_id, "invoice_date": window.as_of + timedelta(days=5)}
                )
    panel = pd.DataFrame(panel_rows)
    transactions = pd.DataFrame(transaction_rows)
    person_period_panel = build_person_period_panel(
        transactions, origins, panel, feature_cols=["signal"], time_col="invoice_date"
    )

    _, predictions = hazard_rolling_origin_backtest(
        person_period_panel, panel, origins, feature_cols=["signal"], min_training_origins=1, seed=0
    )
    model, mature_train_windows = fit_hazard_at_origin(
        person_period_panel, origins, target_index=2, feature_cols=["signal"], seed=0
    )

    assert [w.as_of for w in mature_train_windows] == [origin_a.as_of, origin_b.as_of]

    target_panel = panel[panel["as_of"] == origin_c.as_of].reset_index(drop=True)
    reproduced = predict_survival(model, target_panel, feature_cols=["signal"], n_periods=3)
    reference = predictions[predictions["as_of"] == origin_c.as_of].set_index("customer_id")[
        "y_prob"
    ]
    assert set(reproduced.index) == set(reference.index)
    aligned_reference = reference.reindex(reproduced.index)
    assert reproduced.to_numpy() == pytest.approx(aligned_reference.to_numpy())
