"""The deliberately-wrong panel, feature, and split logic for notebook 01.

Reproduces three validation mistakes at once, each mapped to a claim this
project exists to test:

1. **Features ignore as_of.** `naive_features` summarizes a customer's
   *entire* transaction history once, and that same summary is reused at
   every scoring occasion (`as_of`) the customer appears in -- a customer
   scored in month 3 and the same customer scored in month 9 get identical
   recency, frequency and monetary values.
2. **The split ignores entity identity.** The panel built by
   `build_naive_panel` has one row per (customer, as_of); a random row split
   lets the same customer's rows land in both train and test.
3. **No gap.** `GAP_DAYS = 0` -- see `01_naive_baseline` for why this
   matters and what the project default (`churnval.config.DEFAULT_GAP_DAYS`)
   is instead.

The label itself is forward-looking (built from `churnval.windows`, the same
machinery the correct protocol in `03_temporal_protocol` uses) -- the mistake
this notebook demonstrates is not in what "churn" means, it is in how the
features and the split around that label are built. See ADR-0007 for why
`HORIZON_DAYS = 90` and the eligibility rule below were chosen for this
dataset specifically.

`tautological_label` is a separate, deliberately more broken construction
kept only for the closing "why a perfect score is a bug report" section of
the notebook -- it is not part of the three-mistake panel above.

Nothing in this module is imported by notebook 03 onward. See the warning
banner in `01_naive_baseline.py`.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from churnval.windows import Window, mask_label_events

#: Length of the label window. Not the project default of 30 days -- see
#: ADR-0007: a 30-day window on an irregular, largely-wholesale purchase
#: cadence would call most active repeat customers "churned" for not having
#: reordered within a calendar month.
HORIZON_DAYS = 90

#: Deliberately zero. Mistake #3 -- see the module and notebook docstrings.
GAP_DAYS = 0

#: A customer needs at least one purchase in this many days before an as_of
#: to be scored at that origin at all -- the maturity check. See ADR-0007
#: for why >=1 purchase rather than >=2.
ELIGIBILITY_LOOKBACK_DAYS = 365


def naive_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Recency/Frequency/Monetary features over each customer's full history.

    Recency is measured to the last date in the whole dataset, not to any
    particular as_of -- this is mistake #1, made concrete. The same row is
    reused for every origin a customer appears in by `build_naive_panel`.

    Args:
        transactions: cleaned transaction lines (see
            `churnval.io.load_online_retail_ii`).

    Returns:
        DataFrame indexed by `customer_id`: recency_days (int), frequency
        (distinct invoices), monetary (summed revenue).
    """
    last_date = transactions["invoice_date"].max()
    grouped = transactions.groupby("customer_id")
    return pd.DataFrame(
        {
            "recency_days": (last_date - grouped["invoice_date"].max()).dt.days,
            "frequency": grouped["invoice"].nunique(),
            "monetary": grouped["revenue"].sum(),
        }
    )


def eligible_customers(
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int = ELIGIBILITY_LOOKBACK_DAYS,
) -> pd.Index:
    """Customers with at least one purchase in `lookback_days` before as_of.

    The maturity check for this panel: a customer with nothing in the
    lookback window has no recent activity to be scored on at this origin,
    matching what a real deployment would exclude rather than silently
    treating as a non-churner.

    Args:
        transactions: cleaned transaction lines.
        as_of: the scoring occasion. The lookback window is
            `[as_of - lookback_days, as_of)`, consistent with the
            feature-window convention in `churnval.windows`.
        lookback_days: width of the lookback window.

    Returns:
        Distinct customer IDs eligible at this origin.
    """
    window_start = as_of - timedelta(days=lookback_days)
    recent = transactions[
        (transactions["invoice_date"] >= window_start) & (transactions["invoice_date"] < as_of)
    ]
    # Named to match `naive_features`' index -- an unnamed Index passed to
    # `.loc[]` silently overwrites the target frame's index name with None,
    # which breaks `reset_index()` downstream in `build_naive_panel`.
    return pd.Index(recent["customer_id"].unique(), name="customer_id")


def build_naive_panel(transactions: pd.DataFrame, origins: list[Window]) -> pd.DataFrame:
    """One row per (customer, as_of): the naive, three-mistakes-at-once panel.

    For each origin: restrict to eligible customers, join each one's
    *entire-history* features (mistake #1 -- see `naive_features`), and set
    the label from `churnval.windows.mask_label_events` on that origin's
    label window -- a customer churns if they made no purchase during it.

    Args:
        transactions: cleaned transaction lines.
        origins: scoring occasions, typically from
            `churnval.windows.rolling_origins`.

    Returns:
        One row per (customer_id, as_of): customer_id, as_of, recency_days,
        frequency, monetary, churned.
    """
    features = naive_features(transactions)
    rows = []
    for window in origins:
        eligible = eligible_customers(transactions, window.as_of)
        in_window = mask_label_events(transactions, window, "invoice_date")
        purchasers = set(transactions.loc[in_window, "customer_id"])

        panel = features.loc[eligible].copy()
        panel["as_of"] = window.as_of
        panel["churned"] = (~panel.index.isin(purchasers)).astype(int)
        rows.append(panel.reset_index())
    return pd.concat(rows, ignore_index=True)


def tautological_label(transactions: pd.DataFrame, horizon_days: int = HORIZON_DAYS) -> pd.Series:
    """Churn = no purchase in the final `horizon_days` of the dataset.

    Anchored to the *same* reference date -- `transactions["invoice_date"]
    .max()` -- that `naive_features`' `recency_days` uses. This is not
    leakage in the usual probabilistic sense; it is an identity:
    `churned == (recency_days > horizon_days)`, by construction. Kept only
    for the closing section of `01_naive_baseline` that demonstrates a
    perfect offline score is a bug report, not a result -- not part of
    `build_naive_panel`.

    Args:
        transactions: cleaned transaction lines.
        horizon_days: width of the trailing "active" window.

    Returns:
        Series indexed by `customer_id`, named `churned`.
    """
    last_date = transactions["invoice_date"].max()
    cutoff = last_date - timedelta(days=horizon_days)
    last_purchase = transactions.groupby("customer_id")["invoice_date"].max()
    return (last_purchase < cutoff).astype(int).rename("churned")
