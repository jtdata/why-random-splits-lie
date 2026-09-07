"""Tests for the as-of-safe RFM features.

Expected values are hand-computed against the fixture below.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churnval.features import asof_features

# as_of = 2011-04-01.
#
# Customer 401: purchases 2011-01-10 (qty 1 @ 15.00 = 15.00) and 2011-03-20
#   (qty 2 @ 5.00 = 10.00), both strictly before as_of.
#   recency_days = (2011-04-01 - 2011-03-20).days = 12. frequency=2,
#   monetary=25.00.
#
# Customer 402: purchase 2011-02-01 (qty 1 @ 100.00 = 100.00, before as_of)
#   and 2011-04-15 (qty 1 @ 1.00 = 1.00, AFTER as_of -- must be excluded).
#   recency_days = (2011-04-01 - 2011-02-01).days = 59. frequency=1
#   (only the pre-as_of invoice), monetary=100.00 (only the pre-as_of
#   invoice).


@pytest.fixture
def transactions() -> pd.DataFrame:
    rows = [
        ("A", 401, "2011-01-10", 1, 15.0),
        ("B", 401, "2011-03-20", 2, 5.0),
        ("C", 402, "2011-02-01", 1, 100.0),
        ("D", 402, "2011-04-15", 1, 1.0),
    ]
    df = pd.DataFrame(rows, columns=["invoice", "customer_id", "invoice_date", "quantity", "price"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["revenue"] = df["quantity"] * df["price"]
    return df


def test_asof_features_exclude_transactions_on_or_after_as_of(transactions):
    features = asof_features(transactions, pd.Timestamp("2011-04-01"))
    assert features.loc[401].to_dict() == {"recency_days": 12, "frequency": 2, "monetary": 25.0}
    assert features.loc[402].to_dict() == {"recency_days": 59, "frequency": 1, "monetary": 100.0}
