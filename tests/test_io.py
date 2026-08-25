"""Tests for the Online Retail II cleaning rules.

Exercises `_clean` directly against a synthetic frame shaped like the source
workbook, so these run without the real dataset. Every dropped row is dropped
for a stated reason -- a rule that silently keeps a cancellation or a
returns row would inflate `frequency` and `monetary` downstream.
"""

from __future__ import annotations

import pandas as pd

from churnval.io import _clean

RAW_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]


def _row(invoice, quantity, price, customer_id, stock_code="10001"):
    return [
        invoice,
        stock_code,
        "WIDGET",
        quantity,
        pd.Timestamp("2011-01-01"),
        price,
        customer_id,
        "United Kingdom",
    ]


def test_clean_keeps_only_completed_purchases_with_a_customer_id():
    raw = pd.DataFrame(
        [
            _row("536365", 2, 5.0, 17850.0),  # kept: ordinary purchase
            _row("C536366", 1, 5.0, 17850.0),  # dropped: cancellation (C prefix)
            _row("536367", -1, 5.0, 17850.0),  # dropped: non-positive quantity
            _row("536368", 1, 0.0, 17850.0),  # dropped: non-positive price
            _row("536369", 1, 5.0, None),  # dropped: no customer id
        ],
        columns=RAW_COLUMNS,
    )
    cleaned = _clean(raw)
    assert cleaned["invoice"].tolist() == ["536365"]
    assert cleaned["customer_id"].tolist() == [17850]


def test_clean_computes_revenue_as_quantity_times_price():
    raw = pd.DataFrame([_row("536365", 3, 2.5, 17850.0)], columns=RAW_COLUMNS)
    cleaned = _clean(raw)
    assert cleaned["revenue"].tolist() == [7.5]


def test_clean_parses_invoice_date_to_timestamp():
    raw = pd.DataFrame([_row("536365", 1, 5.0, 17850.0)], columns=RAW_COLUMNS)
    cleaned = _clean(raw)
    assert cleaned["invoice_date"].iloc[0] == pd.Timestamp("2011-01-01")


def test_clean_stock_code_is_string_even_when_mixed_with_numeric_codes():
    # The two source sheets infer StockCode differently -- one sheet reads
    # numeric-looking codes as int, the other as str for alphanumeric codes
    # like "79323P". A mixed int/str object column fails to write to
    # Parquet; this regression-tests that `_clean` normalises it first.
    raw = pd.DataFrame(
        [
            ["536365", 85123, "MUG", 1, pd.Timestamp("2011-01-01"), 5.0, 17850.0, "UK"],
            ["536366", "79323P", "PINK VASE", 1, pd.Timestamp("2011-01-01"), 5.0, 17850.0, "UK"],
        ],
        columns=RAW_COLUMNS,
    )
    cleaned = _clean(raw)
    assert cleaned["stock_code"].tolist() == ["85123", "79323P"]
    assert cleaned["stock_code"].apply(type).eq(str).all()


def test_clean_drops_administrative_line_items():
    # Postage, carriage, manual adjustments, bank charges, discounts, and
    # test rows are not product purchases -- keeping them would build a
    # customer's RFM features out of fees rather than behaviour. See
    # docs/data/online_retail_ii.md for how this list was verified against
    # the real data.
    raw = pd.DataFrame(
        [
            _row("536365", 1, 5.0, 17850.0),  # kept: ordinary purchase
            _row("536366", 1, 4.0, 17850.0, stock_code="POST"),  # dropped: postage
            _row("536367", 1, 4.0, 17850.0, stock_code="M"),  # dropped: manual
            _row("536368", 1, 4.0, 17850.0, stock_code="BANK CHARGES"),  # dropped
            _row("536369", 1, 4.0, 17850.0, stock_code="ADJUST"),  # dropped
            _row("536370", 1, 4.0, 17850.0, stock_code="TEST001"),  # dropped
            _row("536371", 1, 4.0, 17850.0, stock_code="D"),  # dropped: discount
        ],
        columns=RAW_COLUMNS,
    )
    cleaned = _clean(raw)
    assert cleaned["invoice"].tolist() == ["536365"]
