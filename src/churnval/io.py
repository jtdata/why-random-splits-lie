"""Loading and cleaning Online Retail II.

The source is a two-sheet Excel workbook, slow to parse (~1M rows). Cleaning
is split from the read so the cleaning rules can be tested against a small
synthetic frame without touching the real file, and so the parsed result can
be cached to Parquet once.
"""

from __future__ import annotations

import logging

import pandas as pd

from churnval.config import PATHS

logger = logging.getLogger(__name__)

RAW_XLSX = PATHS.raw / "online_retail_ii" / "online_retail_II.xlsx"
CACHE_PARQUET = PATHS.interim / "online_retail_ii.parquet"
SHEETS = ("Year 2009-2010", "Year 2010-2011")

RAW_COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}

#: StockCode values that are administrative line items, not products --
#: postage, carriage, manual adjustments, bank charges, discounts, bad-debt
#: write-offs, and the two "test product" rows. Verified by inspecting the
#: description text behind every non-numeric stock code in the cleaned data
#: (see docs/data/online_retail_ii.md). A customer whose entire history is
#: made of these would get RFM features built from fees, not purchases --
#: `B` ("Adjust bad debt") is included defensively even though its rows
#: currently also lack a customer ID and would be dropped anyway.
EXCLUDED_STOCK_CODES = frozenset(
    {"POST", "DOT", "M", "C2", "BANK CHARGES", "ADJUST", "ADJUST2", "D", "TEST001", "TEST002", "B"}
)


def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules shared by both sheets.

    Drops rows with no customer ID (an anonymous transaction cannot be
    attributed to a churn label), cancellations (an invoice number prefixed
    "C"), non-positive quantity or price (returns and adjustment rows), and
    administrative line items (see `EXCLUDED_STOCK_CODES`) -- none of these
    represent a completed product purchase. Adds `revenue`.

    Args:
        raw: a frame using the source workbook's column names.

    Returns:
        One row per transaction line with columns invoice, stock_code,
        description, quantity, invoice_date, price, customer_id, country,
        revenue.
    """
    df = raw.rename(columns=RAW_COLUMNS)
    df = df.dropna(subset=["customer_id"]).copy()
    df["customer_id"] = df["customer_id"].astype("int64")
    df["invoice"] = df["invoice"].astype(str)
    # StockCode is numeric-looking in most rows but alphanumeric in others
    # (e.g. "79323P"); pandas infers the two source sheets differently, and a
    # mixed int/str object column fails to convert to a single Arrow type.
    df["stock_code"] = df["stock_code"].astype(str)

    is_cancellation = df["invoice"].str.startswith("C")
    is_administrative = df["stock_code"].isin(EXCLUDED_STOCK_CODES)
    df = df.loc[
        ~is_cancellation & ~is_administrative & (df["quantity"] > 0) & (df["price"] > 0)
    ].copy()

    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["revenue"] = df["quantity"] * df["price"]
    return df.reset_index(drop=True)


def load_online_retail_ii(*, force_reload: bool = False) -> pd.DataFrame:
    """Load cleaned Online Retail II transaction lines, cached to Parquet.

    Args:
        force_reload: re-parse the source Excel even if a cached Parquet file
            already exists.

    Returns:
        See `_clean`.

    Raises:
        FileNotFoundError: the source workbook has not been fetched yet.
    """
    if CACHE_PARQUET.exists() and not force_reload:
        return pd.read_parquet(CACHE_PARQUET)

    if not RAW_XLSX.exists():
        raise FileNotFoundError(f"{RAW_XLSX} not found. Run `uv run churnval fetch retail` first.")

    logger.info("reading %s (both sheets) -- this is the slow part", RAW_XLSX.name)
    frames = [pd.read_excel(RAW_XLSX, sheet_name=sheet) for sheet in SHEETS]
    df = _clean(pd.concat(frames, ignore_index=True))

    PATHS.ensure()
    df.to_parquet(CACHE_PARQUET, index=False)
    return df
