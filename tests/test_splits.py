"""Tests for the corrected panel and rolling-origin backtest.

`eligible_customers`/`build_asof_panel` expected values are hand-computed
against small fixtures. `rolling_origin_backtest` wraps a fitted LightGBM
model, so it's tested structurally (correct origins tested, correct row
counts, correct prediction coverage) plus one hand-reasoned ground truth
(a feature that perfectly determines the label should score near-perfect
AUC), not exact decimals -- the same approach used for the model-wrapping
functions in `churnval.leakage`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnval.splits import (
    build_asof_panel,
    eligible_customers,
    fit_at_origin,
    frozen_model_backtest,
    rolling_origin_backtest,
)
from churnval.windows import Window

# Fixture for eligibility and panel construction. as_of = 2011-03-01,
# gap=0, horizon=30 -> label window [2011-03-01, 2011-03-31). Default
# 365-day eligibility lookback -> window [2010-03-01, 2011-03-01).
#
# Customer 501: purchase 2011-01-15 (eligible; before as_of) and
#   2011-03-10 (inside the label window) -> active (churned=0).
# Customer 502: purchase 2011-01-20 (eligible), nothing in the label
#   window -> churned=1.
# Customer 503: purchase 2009-06-01 only -- more than 365 days before
#   as_of -- never eligible, despite also having a purchase inside the
#   label window (2011-03-05), which must not rescue it.


@pytest.fixture
def transactions() -> pd.DataFrame:
    rows = [
        ("A", 501, "2011-01-15", 1, 10.0),
        ("B", 501, "2011-03-10", 1, 10.0),
        ("C", 502, "2011-01-20", 1, 20.0),
        ("D", 503, "2009-06-01", 1, 1.0),
        ("E", 503, "2011-03-05", 1, 1.0),
    ]
    df = pd.DataFrame(rows, columns=["invoice", "customer_id", "invoice_date", "quantity", "price"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["revenue"] = df["quantity"] * df["price"]
    return df


def test_eligible_customers_uses_the_lookback_window(transactions):
    as_of = pd.Timestamp("2011-03-01")
    assert set(eligible_customers(transactions, as_of, lookback_days=365)) == {501, 502}


def test_build_asof_panel_excludes_ineligible_customers_and_labels_the_rest(transactions):
    window = Window(as_of=pd.Timestamp("2011-03-01"), gap_days=0, horizon_days=30)
    panel = build_asof_panel(transactions, [window])

    assert set(panel["customer_id"]) == {501, 502}
    assert panel.set_index("customer_id")["churned"].to_dict() == {501: 0, 502: 1}
    # customer 501's second purchase (2011-03-10) is inside the label
    # window, not before as_of, so it must not count toward their features.
    row_501 = panel.set_index("customer_id").loc[501]
    assert row_501["frequency"] == 1
    assert row_501["monetary"] == 10.0


def test_build_asof_panel_recomputes_features_per_origin(transactions):
    # Unlike the naive panel, the same customer's features differ across
    # origins once a new purchase occurs strictly before the later as_of.
    early = Window(as_of=pd.Timestamp("2011-01-20"), gap_days=0, horizon_days=30)
    late = Window(as_of=pd.Timestamp("2011-03-15"), gap_days=0, horizon_days=30)
    panel = build_asof_panel(transactions, [early, late])

    rows_501 = panel.loc[panel["customer_id"] == 501].set_index("as_of")
    # At the early origin, only the 2011-01-15 purchase is visible.
    assert rows_501.loc[early.as_of, "frequency"] == 1
    # At the late origin, the 2011-03-10 purchase is also visible.
    assert rows_501.loc[late.as_of, "frequency"] == 2


def test_rolling_origin_backtest_tests_only_origins_past_the_minimum():
    rng = np.random.default_rng(0)
    origins = [
        Window(as_of=pd.Timestamp("2011-01-01"), gap_days=0, horizon_days=30),
        Window(as_of=pd.Timestamp("2011-02-01"), gap_days=0, horizon_days=30),
        Window(as_of=pd.Timestamp("2011-03-01"), gap_days=0, horizon_days=30),
    ]
    frames = []
    for window in origins:
        signal = rng.uniform(size=100)
        frames.append(
            pd.DataFrame(
                {
                    "customer_id": range(100),
                    "as_of": window.as_of,
                    "signal": signal,
                    "churned": (signal > 0.5).astype(int),
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)

    per_origin, predictions = rolling_origin_backtest(
        panel, origins, feature_cols=["signal"], min_training_origins=1, seed=0
    )

    assert list(per_origin["as_of"]) == [origins[1].as_of, origins[2].as_of]
    assert per_origin["n"].tolist() == [100, 100]
    assert (per_origin["roc_auc"] > 0.9).all()
    assert len(predictions) == 200
    assert set(predictions["as_of"].unique()) == {origins[1].as_of, origins[2].as_of}


def test_rolling_origin_backtest_skips_origins_with_no_mature_training_data():
    # Origin A's label window (100-day horizon) does not close before
    # origin B's as_of (only 30 days later) -- B has no mature training
    # origin and must be skipped entirely. By origin C (150 days after A),
    # A's label window (closes on day 100) has closed, so C is testable,
    # trained on both A and B (B's label window closes on day 130, also
    # before C's as_of on day 150).
    origins = [
        Window(as_of=pd.Timestamp("2011-01-01"), gap_days=0, horizon_days=100),
        Window(as_of=pd.Timestamp("2011-01-31"), gap_days=0, horizon_days=100),
        Window(as_of=pd.Timestamp("2011-05-31"), gap_days=0, horizon_days=100),
    ]
    rng = np.random.default_rng(0)
    frames = []
    for window in origins:
        signal = rng.uniform(size=50)
        frames.append(
            pd.DataFrame(
                {
                    "customer_id": range(50),
                    "as_of": window.as_of,
                    "signal": signal,
                    "churned": (signal > 0.5).astype(int),
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)

    per_origin, predictions = rolling_origin_backtest(
        panel, origins, feature_cols=["signal"], min_training_origins=1, seed=0
    )

    # Origin B (index 1) is skipped -- only origin C (index 2) is tested.
    assert list(per_origin["as_of"]) == [origins[2].as_of]
    assert set(predictions["as_of"].unique()) == {origins[2].as_of}


def test_fit_at_origin_matches_rolling_origin_backtests_own_purge_and_predictions():
    # Same fixture as the maturity-purge test above: origin C (index 2) is
    # the only testable one, trained on both A and B. fit_at_origin(target
    # index=2) must select the same two training origins and, refit with
    # the same seed on the same rows, reproduce rolling_origin_backtest's
    # own predictions for C exactly -- confirming the duplicated purge
    # logic hasn't drifted from the function it mirrors.
    origins = [
        Window(as_of=pd.Timestamp("2011-01-01"), gap_days=0, horizon_days=100),
        Window(as_of=pd.Timestamp("2011-01-31"), gap_days=0, horizon_days=100),
        Window(as_of=pd.Timestamp("2011-05-31"), gap_days=0, horizon_days=100),
    ]
    rng = np.random.default_rng(0)
    frames = []
    for window in origins:
        signal = rng.uniform(size=50)
        frames.append(
            pd.DataFrame(
                {
                    "customer_id": range(50),
                    "as_of": window.as_of,
                    "signal": signal,
                    "churned": (signal > 0.5).astype(int),
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)

    _, predictions = rolling_origin_backtest(
        panel, origins, feature_cols=["signal"], min_training_origins=1, seed=0
    )
    model, mature_train_windows = fit_at_origin(
        panel, origins, target_index=2, feature_cols=["signal"], seed=0
    )

    assert [w.as_of for w in mature_train_windows] == [origins[0].as_of, origins[1].as_of]
    target_rows = panel[panel["as_of"] == origins[2].as_of].sort_values("customer_id")
    reproduced = model.predict_proba(target_rows[["signal"]])[:, 1]
    assert reproduced == pytest.approx(predictions.sort_values("customer_id")["y_prob"].to_numpy())


def test_fit_at_origin_raises_when_no_training_origin_is_mature():
    # Origin B (index 1) has no mature training origin -- same reason
    # rolling_origin_backtest skips it in the test above.
    origins = [
        Window(as_of=pd.Timestamp("2011-01-01"), gap_days=0, horizon_days=100),
        Window(as_of=pd.Timestamp("2011-01-31"), gap_days=0, horizon_days=100),
    ]
    rng = np.random.default_rng(0)
    signal = rng.uniform(size=20)
    panel = pd.DataFrame(
        {
            "customer_id": range(20),
            "as_of": origins[0].as_of,
            "signal": signal,
            "churned": (signal > 0.5).astype(int),
        }
    )
    with pytest.raises(ValueError, match="no mature training origin"):
        fit_at_origin(panel, origins, target_index=1, feature_cols=["signal"], seed=0)


def test_rolling_origin_backtest_max_lookback_days_restricts_training_origins():
    # Short horizon (10 days) so A/B/C are all mature well before D
    # (as_of 300 days after A) regardless of lookback -- isolates the
    # lookback filter from the maturity filter.
    origins = [
        Window(as_of=pd.Timestamp("2011-01-01"), gap_days=0, horizon_days=10),  # A, day 0
        Window(as_of=pd.Timestamp("2011-01-31"), gap_days=0, horizon_days=10),  # B, day 30
        Window(as_of=pd.Timestamp("2011-03-02"), gap_days=0, horizon_days=10),  # C, day 60
        Window(as_of=pd.Timestamp("2011-10-28"), gap_days=0, horizon_days=10),  # D, day 300
    ]
    rng = np.random.default_rng(0)
    frames = []
    for i, window in enumerate(origins):
        signal = rng.uniform(size=50)
        frames.append(
            pd.DataFrame(
                {
                    "customer_id": [f"{i}-{k}" for k in range(50)],
                    "as_of": window.as_of,
                    "signal": signal,
                    "churned": (signal > 0.5).astype(int),
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)

    # Expanding (default): D trains on A+B+C -- confirmed testable.
    per_origin_expanding, _ = rolling_origin_backtest(
        panel, origins, feature_cols=["signal"], min_training_origins=3, seed=0
    )
    assert list(per_origin_expanding["as_of"]) == [origins[3].as_of]

    # 100-day lookback from D (day 300) -> cutoff day 200: none of A/B/C
    # (days 0/30/60) qualify -- D has no usable training data and is
    # skipped entirely.
    per_origin_100, predictions_100 = rolling_origin_backtest(
        panel,
        origins,
        feature_cols=["signal"],
        min_training_origins=3,
        seed=0,
        max_lookback_days=100,
    )
    assert len(per_origin_100) == 0
    assert len(predictions_100) == 0

    # 250-day lookback from D -> cutoff day 50: only C (day 60) qualifies.
    per_origin_250, _predictions_250 = rolling_origin_backtest(
        panel,
        origins,
        feature_cols=["signal"],
        min_training_origins=3,
        seed=0,
        max_lookback_days=250,
    )
    assert list(per_origin_250["as_of"]) == [origins[3].as_of]
    assert per_origin_250["n"].iloc[0] == 50


def test_frozen_model_backtest_trains_once_and_scores_every_later_origin():
    # Same spacing/horizon as the "tests only origins past the minimum"
    # rolling-origin case: origin 0 is mature relative to origin 1, so the
    # freeze point is origin 1, trained on origin 0's rows alone.
    origins = [
        Window(as_of=pd.Timestamp("2011-01-01"), gap_days=0, horizon_days=30),
        Window(as_of=pd.Timestamp("2011-02-01"), gap_days=0, horizon_days=30),
        Window(as_of=pd.Timestamp("2011-03-01"), gap_days=0, horizon_days=30),
    ]
    rng = np.random.default_rng(0)
    frames = []
    for window in origins:
        signal = rng.uniform(size=100)
        frames.append(
            pd.DataFrame(
                {
                    "customer_id": range(100),
                    "as_of": window.as_of,
                    "signal": signal,
                    "churned": (signal > 0.5).astype(int),
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)

    per_origin, predictions = frozen_model_backtest(
        panel, origins, feature_cols=["signal"], min_training_origins=1, seed=0
    )

    # Frozen at origin 1 (the earliest with mature training data), scored
    # on origins 1 and 2 -- every origin from the freeze point onward, not
    # just the freeze point itself.
    assert list(per_origin["as_of"]) == [origins[1].as_of, origins[2].as_of]
    assert per_origin["origins_since_training"].tolist() == [0, 1]
    assert per_origin["n"].tolist() == [100, 100]
    assert (per_origin["roc_auc"] > 0.9).all()
    assert len(predictions) == 200


def test_frozen_model_backtest_raises_when_nothing_is_ever_mature():
    # Only two origins, horizon (100d) longer than their spacing (30d) --
    # origin 0 is never mature relative to origin 1, so there is no origin
    # to freeze a model at.
    origins = [
        Window(as_of=pd.Timestamp("2011-01-01"), gap_days=0, horizon_days=100),
        Window(as_of=pd.Timestamp("2011-01-31"), gap_days=0, horizon_days=100),
    ]
    rng = np.random.default_rng(0)
    frames = []
    for window in origins:
        signal = rng.uniform(size=20)
        frames.append(
            pd.DataFrame(
                {
                    "customer_id": range(20),
                    "as_of": window.as_of,
                    "signal": signal,
                    "churned": (signal > 0.5).astype(int),
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)

    with pytest.raises(ValueError, match="no origin"):
        frozen_model_backtest(
            panel, origins, feature_cols=["signal"], min_training_origins=1, seed=0
        )


def test_frozen_model_backtest_detects_a_frozen_models_decay():
    # Training data (origin A): signal > 0.5 -> churned = 1. Origins B and
    # C keep the same relationship -- the frozen model (trained on A)
    # should still rank them almost perfectly. Origin D *inverts* it
    # (signal > 0.5 -> churned = 0) -- a model that learned "high signal
    # means churn" now ranks every row backwards, so its ROC AUC on D
    # should collapse toward 0, not just toward chance.
    origins = [
        Window(as_of=pd.Timestamp("2011-01-01"), gap_days=0, horizon_days=10),  # A: train source
        Window(
            as_of=pd.Timestamp("2011-01-31"), gap_days=0, horizon_days=10
        ),  # B: same relationship
        Window(
            as_of=pd.Timestamp("2011-03-02"), gap_days=0, horizon_days=10
        ),  # C: same relationship
        Window(as_of=pd.Timestamp("2011-04-01"), gap_days=0, horizon_days=10),  # D: inverted
    ]
    rng = np.random.default_rng(0)
    frames = []
    for i, window in enumerate(origins):
        signal = rng.uniform(size=100)
        inverted = i == 3
        churned = (signal <= 0.5) if inverted else (signal > 0.5)
        frames.append(
            pd.DataFrame(
                {
                    "customer_id": range(100),
                    "as_of": window.as_of,
                    "signal": signal,
                    "churned": churned.astype(int),
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)

    per_origin, _ = frozen_model_backtest(
        panel, origins, feature_cols=["signal"], min_training_origins=1, seed=0
    )

    by_origins_since = per_origin.set_index("origins_since_training")["roc_auc"]
    assert by_origins_since[0] > 0.9  # origin B, relationship unchanged
    assert by_origins_since[1] > 0.9  # origin C, relationship unchanged
    assert by_origins_since[2] < 0.1  # origin D, relationship inverted
