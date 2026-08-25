"""Tests for the as-of-safe RFM preview used in notebook 01's comparison chart.

Expected values are hand-computed against the fixture below.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churnval.asof_preview import asof_recency_frequency_monetary

# as_of = 2011-03-01.
#
# Customer 301: purchases 2011-01-01 (qty 1 @ 10.00 = 10.00) and 2011-02-15
#   (qty 1 @ 5.00 = 5.00), both strictly before as_of.
#   recency_days = (2011-03-01 - 2011-02-15).days = 14. frequency=2,
#   monetary=15.00.
#
# Customer 302: purchase 2011-01-01 (qty 2 @ 20.00 = 40.00, before as_of) and
#   2011-03-15 (qty 1 @ 1.00 = 1.00, AFTER as_of -- must be excluded).
#   recency_days = (2011-03-01 - 2011-01-01).days = 59. frequency=1 (only
#   the pre-as_of invoice), monetary=40.00 (only the pre-as_of invoice).


@pytest.fixture
def transactions() -> pd.DataFrame:
    rows = [
        ("A", 301, "2011-01-01", 1, 10.0),
        ("B", 301, "2011-02-15", 1, 5.0),
        ("C", 302, "2011-01-01", 2, 20.0),
        ("D", 302, "2011-03-15", 1, 1.0),
    ]
    df = pd.DataFrame(rows, columns=["invoice", "customer_id", "invoice_date", "quantity", "price"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["revenue"] = df["quantity"] * df["price"]
    return df


def test_asof_features_exclude_transactions_on_or_after_as_of(transactions):
    features = asof_recency_frequency_monetary(transactions, pd.Timestamp("2011-03-01"))
    assert features.loc[301].to_dict() == {"recency_days": 14, "frequency": 2, "monetary": 15.0}
    assert features.loc[302].to_dict() == {"recency_days": 59, "frequency": 1, "monetary": 40.0}
