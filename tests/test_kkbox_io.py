"""Tests for KKBox transaction cleaning.

Each cleaning rule in `churnval.kkbox_io._clean_query` gets its own row in a
small synthetic fixture, hand-verified rather than read back from the
implementation -- see `_clean_query`'s own docstring for the reasoning
behind each rule, backed by counts from the real file.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churnval.kkbox_io import _clean

RAW_COLUMNS = [
    "msno",
    "payment_method_id",
    "payment_plan_days",
    "plan_list_price",
    "actual_amount_paid",
    "is_auto_renew",
    "transaction_date",
    "membership_expire_date",
    "is_cancel",
]


def _row(msno, transaction_date, membership_expire_date, is_cancel, **overrides):
    row = {
        "msno": msno,
        "payment_method_id": 41,
        "payment_plan_days": 30,
        "plan_list_price": 149,
        "actual_amount_paid": 149,
        "is_auto_renew": 1,
        "transaction_date": transaction_date,
        "membership_expire_date": membership_expire_date,
        "is_cancel": is_cancel,
    }
    row.update(overrides)
    return row


@pytest.fixture
def raw() -> pd.DataFrame:
    normal = _row("A", 20160101, 20160131, 0)
    duplicate = dict(normal)  # exact duplicate of "A"'s row
    corrupt_sentinel = _row("B", 20160101, 19700101, 0)
    inconsistent_active = _row(
        "C", 20160101, 20151231, 0
    )  # expires before it happened, not a cancel
    legitimate_cancel = _row("D", 20160101, 20151231, 1)  # same ordering, but is_cancel=1 -- keep
    zero_plan_days = _row("E", 20160101, 20160131, 0, payment_plan_days=0)  # keep, known quirk

    return pd.DataFrame(
        [
            normal,
            duplicate,
            corrupt_sentinel,
            inconsistent_active,
            legitimate_cancel,
            zero_plan_days,
        ],
        columns=RAW_COLUMNS,
    )


def test_clean_renames_msno_to_customer_id_and_parses_dates(raw):
    cleaned = _clean(raw)
    row_a = cleaned.loc[cleaned["customer_id"] == "A"].iloc[0]
    assert row_a["transaction_date"] == pd.Timestamp("2016-01-01")
    assert row_a["membership_expire_date"] == pd.Timestamp("2016-01-31")


def test_clean_drops_exact_duplicates(raw):
    cleaned = _clean(raw)
    assert (cleaned["customer_id"] == "A").sum() == 1


def test_clean_drops_corrupt_sentinel_expiry_dates(raw):
    cleaned = _clean(raw)
    assert "B" not in set(cleaned["customer_id"])


def test_clean_drops_inconsistent_active_subscription_rows(raw):
    # expire < transaction and is_cancel=0 -- internally inconsistent, dropped.
    cleaned = _clean(raw)
    assert "C" not in set(cleaned["customer_id"])


def test_clean_keeps_legitimate_cancellations_with_same_date_ordering(raw):
    # expire <= transaction but is_cancel=1 -- a real cancellation, must survive.
    cleaned = _clean(raw)
    assert "D" in set(cleaned["customer_id"])
    row_d = cleaned.loc[cleaned["customer_id"] == "D"].iloc[0]
    assert row_d["is_cancel"] == 1


def test_clean_keeps_zero_payment_plan_days_rows(raw):
    cleaned = _clean(raw)
    assert "E" in set(cleaned["customer_id"])
    row_e = cleaned.loc[cleaned["customer_id"] == "E"].iloc[0]
    assert row_e["payment_plan_days"] == 0
