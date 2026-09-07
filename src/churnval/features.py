"""As-of-safe RFM features -- the tested, canonical replacement for
`naive_baseline.naive_features`.

Deliberately not shared code with `naive_baseline.py` (whose docstring says
nothing in it is reused past notebook 02) or `asof_preview.py` (a one-off
comparison helper built for a single chart in `01_naive_baseline`, whose own
docstring says `03_temporal_protocol` builds and tests its own version and
may reasonably differ from it). This module is that version: the one later
notebooks are meant to import.
"""

from __future__ import annotations

import pandas as pd


def asof_features(transactions: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Recency/Frequency/Monetary computed strictly before `as_of`.

    Every aggregation uses only transactions with `invoice_date < as_of` --
    the feature-window convention in `churnval.windows`. A customer's value
    changes from one scoring occasion to the next, unlike the naive version.

    Args:
        transactions: cleaned transaction lines (see
            `churnval.io.load_online_retail_ii`).
        as_of: the scoring occasion.

    Returns:
        DataFrame indexed by `customer_id`: recency_days (days from `as_of`
        to that customer's most recent prior purchase), frequency (distinct
        invoices), monetary (summed revenue) -- all computed only from
        history strictly before `as_of`.
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
