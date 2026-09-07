"""Tests for KKBox eligibility, as-of features, and label construction.

Expected values are computed by hand against the fixture below, following
the same convention `test_splits.py` uses for `build_asof_panel` -- not
read back from `build_kkbox_asof_panel`'s own output.

Fixture: as_of = 2016-06-01, gap_days = 7, horizon_days = 30 ->
label window [2016-06-08, 2016-07-08). eligibility_lookback_days = 45 ->
eligibility window [2016-04-17, 2016-06-08).

- cust1: three pre-as_of transactions (2015-01-01, 2016-03-01 [a
  cancellation], 2016-05-06 [most recent, expiring 2016-06-05 -- inside the
  eligibility window]) plus a renewal transaction inside the label window
  (2016-06-15) -> eligible, churned=0.
- cust2: one pre-as_of transaction (2016-01-01, expiring 2016-05-01 --
  already lapsed but inside the eligibility window) plus a
  *cancellation-only* transaction inside the label window (2016-06-20) ->
  eligible, churned=1 (the single most important behavioral test in this
  module: a cancellation inside the label window must not count as a
  renewal).
- cust3: expires 2016-04-01, before the eligibility window opens (2016-04-17)
  -> not eligible.
- cust4: expires 2016-08-01, after the eligibility window closes (2016-06-08)
  -> not eligible.
- cust5: expires 2016-05-25 (eligible) and has *no* transaction at all in
  the label window -> churned=1 (censored, not a cancellation).
"""

from __future__ import annotations

import pandas as pd
import pytest

from churnval.kkbox_features import (
    FEATURE_COLS,
    build_kkbox_asof_panel,
    kkbox_naive_features,
)
from churnval.windows import Window

AS_OF = pd.Timestamp("2016-06-01")
WINDOW = Window(as_of=AS_OF, gap_days=7, horizon_days=30)


def _tx(customer_id, transaction_date, membership_expire_date, is_cancel, **overrides):
    row = {
        "customer_id": customer_id,
        "transaction_date": pd.Timestamp(transaction_date),
        "membership_expire_date": pd.Timestamp(membership_expire_date),
        "actual_amount_paid": 149,
        "is_auto_renew": 1,
        "is_cancel": is_cancel,
    }
    row.update(overrides)
    return row


@pytest.fixture
def transactions() -> pd.DataFrame:
    rows = [
        # cust1: eligible, renews inside the label window -> churned=0.
        _tx("cust1", "2015-01-01", "2015-01-31", 0),
        _tx("cust1", "2016-03-01", "2016-03-31", 1),  # a cancellation, not the latest
        _tx("cust1", "2016-05-06", "2016-06-05", 0, actual_amount_paid=149, is_auto_renew=1),
        _tx("cust1", "2016-06-15", "2016-07-15", 0),  # renewal inside the label window
        # cust2: eligible, only a cancellation inside the label window -> churned=1.
        _tx("cust2", "2016-01-01", "2016-05-01", 0, actual_amount_paid=99, is_auto_renew=0),
        _tx("cust2", "2016-06-20", "2016-06-20", 1),  # cancellation inside the label window
        # cust3: expires before the eligibility window opens -> not eligible.
        _tx("cust3", "2015-06-01", "2016-04-01", 0),
        # cust4: expires after the eligibility window closes -> not eligible.
        _tx("cust4", "2016-01-01", "2016-08-01", 0),
        # cust5: eligible, no activity at all in the label window -> churned=1.
        _tx("cust5", "2016-05-01", "2016-05-25", 0, actual_amount_paid=129),
    ]
    return pd.DataFrame(rows)


def test_eligibility_keeps_expiring_soon_and_recently_expired_customers(transactions):
    panel = build_kkbox_asof_panel(transactions, [WINDOW])
    assert set(panel["customer_id"]) == {"cust1", "cust2", "cust5"}


def test_days_until_expiry_and_tenure_hand_computed(transactions):
    panel = build_kkbox_asof_panel(transactions, [WINDOW]).set_index("customer_id")

    # cust1's most recent pre-as_of transaction expires 2016-06-05, 4 days
    # after as_of (2016-06-01).
    assert panel.loc["cust1", "days_until_expiry"] == 4
    # cust1's first-ever transaction is 2015-01-01.
    expected_tenure = (AS_OF - pd.Timestamp("2015-01-01")).days
    assert panel.loc["cust1", "tenure_days"] == expected_tenure

    # cust2's membership already lapsed 31 days before as_of.
    assert panel.loc["cust2", "days_until_expiry"] == -31
    assert panel.loc["cust2", "current_plan_price"] == 99
    assert panel.loc["cust2", "is_auto_renew"] == 0


def test_lookback_aggregates_count_transactions_and_cancellations_separately(transactions):
    panel = build_kkbox_asof_panel(transactions, [WINDOW]).set_index("customer_id")

    # cust1's lookback window (180 days before as_of) is
    # [2015-12-04, 2016-06-01) -- the 2015-01-01 transaction is outside it;
    # the 2016-03-01 cancellation and 2016-05-06 renewal are both inside it.
    assert panel.loc["cust1", "n_transactions_lookback"] == 2
    assert panel.loc["cust1", "n_cancellations_lookback"] == 1

    # cust2's single pre-as_of transaction (2016-01-01) is inside the
    # lookback window and is not a cancellation.
    assert panel.loc["cust2", "n_transactions_lookback"] == 1
    assert panel.loc["cust2", "n_cancellations_lookback"] == 0


def test_label_treats_cancellation_only_as_churned_not_a_renewal(transactions):
    panel = build_kkbox_asof_panel(transactions, [WINDOW]).set_index("customer_id")

    assert panel.loc["cust1", "churned"] == 0  # renewed inside the label window
    assert panel.loc["cust2", "churned"] == 1  # only a cancellation inside the label window
    assert panel.loc["cust5", "churned"] == 1  # no activity at all inside the label window


def test_panel_has_exactly_the_documented_columns(transactions):
    panel = build_kkbox_asof_panel(transactions, [WINDOW])
    assert list(panel.columns) == ["customer_id", "as_of", *FEATURE_COLS, "churned"]


def test_features_recompute_fresh_per_origin():
    # cust1 has one plan through mid-2016, then upgrades to a pricier plan
    # in between two origins -- current_plan_price must reflect that origin's
    # own most-recent-before-as_of transaction, not be frozen at the first
    # one the way a naive full-history feature would be.
    early = Window(as_of=pd.Timestamp("2016-03-01"), gap_days=7, horizon_days=30)
    late = Window(as_of=pd.Timestamp("2016-06-01"), gap_days=7, horizon_days=30)
    transactions = pd.DataFrame(
        [
            _tx("cust1", "2016-01-01", "2016-02-15", 0, actual_amount_paid=99),
            _tx("cust1", "2016-04-01", "2016-05-15", 0, actual_amount_paid=199),
            _tx("cust1", "2016-05-20", "2016-06-05", 0, actual_amount_paid=199),
        ]
    )
    panel = build_kkbox_asof_panel(transactions, [early, late]).set_index(["as_of", "customer_id"])

    assert panel.loc[(early.as_of, "cust1"), "current_plan_price"] == 99
    assert panel.loc[(late.as_of, "cust1"), "current_plan_price"] == 199


def test_kkbox_naive_features_reuses_identical_values_across_origins():
    # naive_features is not as-of-safe by construction -- it should return
    # exactly one row per customer, anchored to the dataset's own last
    # transaction date regardless of any origin.
    transactions = pd.DataFrame(
        [
            _tx("cust1", "2016-01-01", "2016-02-15", 0, actual_amount_paid=99),
            _tx("cust1", "2016-04-01", "2016-05-15", 0, actual_amount_paid=199),
        ]
    )
    features = kkbox_naive_features(transactions)
    assert len(features.loc[["cust1"]]) == 1
    reference_date = transactions["transaction_date"].max()
    assert (
        features.loc["cust1", "tenure_days"] == (reference_date - pd.Timestamp("2016-01-01")).days
    )
    assert features.loc["cust1", "current_plan_price"] == 199  # the later, most recent row


def test_eligibility_gap_days_overrides_the_windows_own_gap_days_for_eligibility_only(
    transactions,
):
    # cust1's membership expires 2016-06-05, 4 days after as_of (2016-06-01)
    # -- eligible under the default (eligibility upper bound = as_of +
    # gap_days = as_of + 7), but NOT eligible once eligibility_gap_days is
    # pinned to 0 (upper bound = as_of + 0 = as_of), even though WINDOW's
    # own gap_days (7) is unchanged and still governs the label window.
    # This is the regression test for the bug this parameter fixes: an
    # earlier version silently changed the eligible population whenever a
    # gap ablation changed gap_days, confounding "what does the gap cost"
    # with "what does a smaller, more-lapsed population cost."
    panel_default = build_kkbox_asof_panel(transactions, [WINDOW])
    assert "cust1" in set(panel_default["customer_id"])

    panel_pinned = build_kkbox_asof_panel(transactions, [WINDOW], eligibility_gap_days=0)
    assert "cust1" not in set(panel_pinned["customer_id"])


def test_current_membership_state_is_deterministic_and_prefers_the_later_expiry_on_a_tied_date():
    # Two transactions for cust1 on the exact same transaction_date -- DuckDB's
    # ASOF JOIN gives no tie-break guarantee for this case (found under
    # adversarial review: repeated calls on identical real data picked
    # different eligible populations and labels, not just different feature
    # values). _dedupe_for_asof_join must resolve the tie before the join
    # ever sees it, deterministically, so both transactions being inside the
    # eligibility window doesn't matter -- only the larger
    # membership_expire_date should win.
    transactions = pd.DataFrame(
        [
            _tx("cust1", "2016-05-06", "2016-06-01", 0, actual_amount_paid=99),
            _tx("cust1", "2016-05-06", "2016-06-05", 0, actual_amount_paid=199),
        ]
    )
    panel = build_kkbox_asof_panel(transactions, [WINDOW]).set_index("customer_id")
    assert panel.loc["cust1", "days_until_expiry"] == 4  # 2016-06-05 - 2016-06-01
    assert panel.loc["cust1", "current_plan_price"] == 199

    repeat = build_kkbox_asof_panel(transactions, [WINDOW]).set_index("customer_id")
    pd.testing.assert_frame_equal(panel, repeat)


def test_lookback_features_fill_zero_not_nan_when_no_transactions_in_the_window():
    # _lookback_aggregates is an inner join over a bounded window -- a
    # customer whose only transaction predates the 180-day lookback window
    # entirely (a long-duration-plan customer, in the real data) doesn't
    # appear in it at all, leaving NaN after build_kkbox_asof_panel's left
    # merge unless filled explicitly (found under adversarial review).
    old_only_transactions = pd.DataFrame([_tx("cust1", "2015-01-01", "2016-05-25", 0)])
    panel = build_kkbox_asof_panel(old_only_transactions, [WINDOW]).set_index("customer_id")
    assert panel.loc["cust1", "n_transactions_lookback"] == 0
    assert panel.loc["cust1", "n_cancellations_lookback"] == 0
