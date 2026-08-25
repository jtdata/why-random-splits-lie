"""Tests for the naive (deliberately-wrong) panel, feature, and label logic.

Expected values are hand-computed against the fixtures below, not read back
from the implementation -- see each fixture's docstring for the arithmetic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churnval.naive_baseline import (
    build_naive_panel,
    eligible_customers,
    naive_features,
    tautological_label,
)
from churnval.windows import Window

# Fixture: four customers. Customer 104's single purchase on 2011-12-31 is
# the latest transaction in the dataset -- it exists purely to anchor
# `last_date`, since `naive_features`/`tautological_label` use the
# dataset-wide max, not any individual customer's own last purchase.
# Horizon = 30 days -> cutoff = 2011-12-31 - 30 days = 2011-12-01.
#
# Customer 101: invoices on 2011-01-01 (qty 2 @ 5.00 = 10.00) and
#   2011-12-15 (qty 1 @ 20.00 = 20.00). Last purchase 2011-12-15 >= cutoff ->
#   active (churned=0). frequency=2, monetary=30.00.
#   recency_days = (2011-12-31 - 2011-12-15).days = 16.
#
# Customer 102: one invoice on 2011-01-01 (qty 1 @ 50.00 = 50.00). Last
#   purchase 2011-01-01 < cutoff -> churned=1. frequency=1, monetary=50.00.
#   recency_days = (2011-12-31 - 2011-01-01).days = 364 (2011 is not a leap
#   year, and Jan 1 -> Dec 31 spans 364 days).
#
# Customer 103: two invoices on 2011-06-01 (qty 3 @ 2.00 = 6.00 and qty 1 @
#   4.00 = 4.00). Last purchase 2011-06-01 < cutoff -> churned=1.
#   frequency=2, monetary=10.00.
#   recency_days = (2011-12-31 - 2011-06-01).days = 213 (day-of-year 152 for
#   Jun 1 vs. 365 for Dec 31 in a non-leap year).
#
# Customer 104: one invoice on 2011-12-31 (qty 1 @ 100.00 = 100.00). Last
#   purchase == last_date -> churned=0, recency_days=0, frequency=1,
#   monetary=100.00.


@pytest.fixture
def transactions() -> pd.DataFrame:
    rows = [
        ("A", 101, "2011-01-01", 2, 5.0),
        ("B", 101, "2011-12-15", 1, 20.0),
        ("C", 102, "2011-01-01", 1, 50.0),
        ("D", 103, "2011-06-01", 3, 2.0),
        ("E", 103, "2011-06-01", 1, 4.0),
        ("F", 104, "2011-12-31", 1, 100.0),
    ]
    df = pd.DataFrame(rows, columns=["invoice", "customer_id", "invoice_date", "quantity", "price"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["revenue"] = df["quantity"] * df["price"]
    return df


def test_tautological_label_uses_last_dataset_date_as_reference(transactions):
    label = tautological_label(transactions, horizon_days=30)
    assert label.to_dict() == {101: 0, 102: 1, 103: 1, 104: 0}


def test_naive_features_recency_frequency_monetary(transactions):
    features = naive_features(transactions)
    assert features.loc[101].to_dict() == {"recency_days": 16, "frequency": 2, "monetary": 30.0}
    assert features.loc[102].to_dict() == {"recency_days": 364, "frequency": 1, "monetary": 50.0}
    assert features.loc[103].to_dict() == {"recency_days": 213, "frequency": 2, "monetary": 10.0}
    assert features.loc[104].to_dict() == {"recency_days": 0, "frequency": 1, "monetary": 100.0}


# Fixture for eligibility and panel construction. as_of = 2011-03-01.
#
#   default lookback (365d) window: [2010-03-01, 2011-03-01)
#   custom  lookback ( 60d) window: [2010-12-31, 2011-03-01)
#   label window (gap=0, horizon=30d): [2011-03-01, 2011-03-31)
#
# Customer 201: purchase 2011-01-15 (inside both lookback windows) and
#   2011-03-10 (inside the label window) -> eligible under either lookback;
#   active (churned=0).
# Customer 202: purchase 2011-01-20 (inside both lookback windows), nothing
#   in the label window -> eligible under either lookback; churned=1.
# Customer 203: purchase 2010-11-01 -- inside the 365-day window
#   (2010-11-01 >= 2010-03-01) but *outside* the 60-day window
#   (2010-11-01 < 2010-12-31). Nothing in the label window -> eligible only
#   under the default lookback; churned=1 when included.
# Customer 204: purchase 2009-06-01 only -- outside even the 365-day window
#   (2009-06-01 < 2010-03-01) -- never eligible, despite also having a
#   purchase inside the label window, which must not rescue it.


@pytest.fixture
def panel_transactions() -> pd.DataFrame:
    rows = [
        ("A", 201, "2011-01-15", 1, 10.0),
        ("B", 201, "2011-03-10", 1, 10.0),
        ("C", 202, "2011-01-20", 1, 20.0),
        ("D", 203, "2010-11-01", 1, 5.0),
        ("E", 204, "2009-06-01", 1, 1.0),
        ("F", 204, "2011-03-05", 1, 1.0),
    ]
    df = pd.DataFrame(rows, columns=["invoice", "customer_id", "invoice_date", "quantity", "price"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["revenue"] = df["quantity"] * df["price"]
    return df


def test_eligible_customers_uses_the_given_lookback(panel_transactions):
    as_of = pd.Timestamp("2011-03-01")
    assert set(eligible_customers(panel_transactions, as_of, lookback_days=60)) == {201, 202}
    assert set(eligible_customers(panel_transactions, as_of, lookback_days=365)) == {201, 202, 203}


def test_build_naive_panel_excludes_ineligible_customers_and_labels_the_rest(panel_transactions):
    window = Window(as_of=pd.Timestamp("2011-03-01"), gap_days=0, horizon_days=30)
    panel = build_naive_panel(panel_transactions, [window])

    # customer 204 never appears: its only purchase inside the eligibility
    # lookback is more than 365 days before as_of, and its purchase inside
    # the label window does not make it eligible.
    assert set(panel["customer_id"]) == {201, 202, 203}
    assert panel.set_index("customer_id")["churned"].to_dict() == {201: 0, 202: 1, 203: 1}


def test_build_naive_panel_reuses_full_history_features_across_origins(panel_transactions):
    # Customer 201 appears at two origins with the same recency/frequency/
    # monetary features both times -- mistake #1: the features do not
    # respect either origin's as_of.
    early = Window(as_of=pd.Timestamp("2011-01-20"), gap_days=0, horizon_days=30)
    late = Window(as_of=pd.Timestamp("2011-03-01"), gap_days=0, horizon_days=30)
    panel = build_naive_panel(panel_transactions, [early, late])

    rows_201 = panel.loc[panel["customer_id"] == 201]
    assert len(rows_201) == 2
    for column in ("recency_days", "frequency", "monetary"):
        assert (rows_201[column] == rows_201[column].iloc[0]).all()
