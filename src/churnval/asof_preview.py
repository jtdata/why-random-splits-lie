"""A minimal as-of-safe RFM computation, for one comparison chart only.

`01_naive_baseline` uses this to show, concretely, how much `naive_features`'
full-history aggregation differs from computing the same features strictly
before each row's own `as_of`. It is NOT the corrected feature pipeline for
this project -- `03_temporal_protocol` builds and tests that from scratch,
and may reasonably differ from this (eligibility handling, additional
features, performance). Keeping this out of `naive_baseline.py` is
deliberate: that module's docstring promises nothing in it is reused past
notebook 01, and this function's whole purpose is a preview of the opposite
of naive -- conflating the two would be confusing.
"""

from __future__ import annotations

import pandas as pd


def asof_recency_frequency_monetary(
    transactions: pd.DataFrame, as_of: pd.Timestamp
) -> pd.DataFrame:
    """Recency/Frequency/Monetary computed from transactions strictly before `as_of`.

    Args:
        transactions: cleaned transaction lines (see
            `churnval.io.load_online_retail_ii`).
        as_of: the scoring occasion. Only transactions with
            `invoice_date < as_of` are used -- matching the feature-window
            convention in `churnval.windows`.

    Returns:
        DataFrame indexed by `customer_id`: recency_days (days from `as_of`
        to that customer's most recent prior purchase), frequency (distinct
        invoices), monetary (summed revenue) -- all computed only from
        history available strictly before `as_of`.
    """
    before = transactions.loc[transactions["invoice_date"] < as_of]
    grouped = before.groupby("customer_id")
    return pd.DataFrame(
        {
            "recency_days": (as_of - grouped["invoice_date"].max()).dt.days,
            "frequency": grouped["invoice"].nunique(),
            "monetary": grouped["revenue"].sum(),
        }
    )
