"""Eligibility, as-of features, and label construction for KKBox.

The contractual analogue of `churnval.splits` + `churnval.features` combined
into one module -- there is no KKBox-specific backtest logic to put in a
`kkbox_splits.py`: `rolling_origin_backtest`/`fit_at_origin`/
`frozen_model_backtest` (`churnval.splits`) are reused unchanged, since they
only ever touch `panel["as_of"]`/`panel["customer_id"]`/`panel["churned"]`/
caller-supplied `feature_cols`, never a retail-specific column name. Only
panel *construction* is dataset-specific, so it all lives here. See
ADR-0014 for the full reasoning behind what's reused vs. rebuilt.

`build_kkbox_asof_panel` computes eligibility, features, and the label for
every origin in one set of vectorized DuckDB SQL passes, not the
per-origin Python loop `churnval.splits.build_asof_panel` uses -- at
21.5M transaction rows and ~18 origins, repeating a full-table pandas
filter+groupby per origin does not scale the way it does for Online Retail
II's ~800K rows and ~10 origins. See ADR-0015.

Eligibility is expiry-based, not activity-based: a customer counts as
"eligible" at an origin if their *current* membership's expiry falls in
`[as_of - ELIGIBILITY_LOOKBACK_DAYS, as_of + gap_days)` -- due for a renewal
decision around now, not merely "has bought something recently" the way
Online Retail II's `eligible_customers` works. A retail-style 365-day
activity lookback scores 1.05M-1.74M customers per origin against the real
data (measured directly, not estimated), most of whom aren't due for a
renewal decision for months; the expiry-based rule scores a smaller, bounded
population instead -- 200K-424K per origin across the real 20-origin
backtest, fluctuating with signup seasonality rather than growing
monotonically (see ADR-0013). `ELIGIBILITY_LOOKBACK_DAYS` is wider than
KKBox's own 30-day grace convention specifically to catch the
non-30/31-day `payment_plan_days` tail present in the real data (90/100/180/
195/410-day plans exist alongside the dominant monthly one).
"""

from __future__ import annotations

import duckdb
import pandas as pd

from churnval.windows import Window

#: A customer's current membership may already have lapsed by this many days
#: and still count as an active churn-prevention target, or may not lapse
#: for this many days yet. See module docstring and ADR-0013.
ELIGIBILITY_LOOKBACK_DAYS = 45

#: Trailing window for count features (`n_transactions_lookback`,
#: `n_cancellations_lookback`) -- about six monthly cycles, wide enough to
#: see a cancel-then-resubscribe pattern without reaching into a
#: long-tenured customer's whole history. See ADR-0013.
FEATURE_LOOKBACK_DAYS = 180

#: The six as-of-safe features `build_kkbox_asof_panel` produces -- the
#: contractual analogue of Online Retail II's RFM three. Each is justified
#: individually in ADR-0013, not ported wholesale from the retail feature
#: set.
FEATURE_COLS = [
    "days_until_expiry",
    "tenure_days",
    "current_plan_price",
    "is_auto_renew",
    "n_transactions_lookback",
    "n_cancellations_lookback",
]


def _origins_frame(origins: list[Window], eligibility_gap_days: int | None) -> pd.DataFrame:
    """`origins` as a small DataFrame DuckDB can join against directly.

    `eligibility_gap_days` is a separate column from each origin's own
    `gap_days`, deliberately -- see `build_kkbox_asof_panel`'s docstring for
    why eligibility's upper bound must not be silently tied to whatever
    `gap_days` an ablation happens to be varying.
    """
    return pd.DataFrame(
        {
            "as_of": [w.as_of for w in origins],
            "gap_days": [w.gap_days for w in origins],
            "eligibility_gap_days": [
                w.gap_days if eligibility_gap_days is None else eligibility_gap_days
                for w in origins
            ],
            "label_start": [w.label_start for w in origins],
            "label_end": [w.label_end for w in origins],
        }
    )


def _dedupe_for_asof_join(transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per `(customer_id, transaction_date)`, for the ASOF join only.

    DuckDB's `ASOF JOIN` gives no tie-break guarantee when multiple rows
    share the same join key. In the real data, 255,595
    `(customer_id, transaction_date)` groups have more than one row, and
    94.6% of those groups have a *different* `membership_expire_date`
    across the tied rows -- which made `_current_membership_state`
    genuinely nondeterministic: two calls on identical input picked
    different eligible populations and different labels, not merely
    different feature values (found under adversarial review of this
    module -- reproduced directly by calling `build_kkbox_asof_panel` twice
    on the same cached transactions and diffing the resulting
    `(as_of, customer_id)` sets, which differed by ~0.8% of the panel).

    Deduping once, here, with an explicit deterministic tie-break -- the
    largest `membership_expire_date` wins, the same choice the superseded
    range-join version made via `ORDER BY transaction_date DESC,
    membership_expire_date DESC` -- removes the ambiguity before it reaches
    the join, using a stable sort so any *remaining* tie (same date AND
    same expiry) resolves the same way every run rather than by whichever
    order DuckDB happens to scan rows in. `_lookback_aggregates`/`_label`
    still see every real transaction, duplicates included -- only the
    "current membership state" lookup needs a single row per date.
    """
    return transactions.sort_values(
        ["customer_id", "transaction_date", "membership_expire_date"],
        ascending=[True, True, False],
        kind="stable",
    ).drop_duplicates(subset=["customer_id", "transaction_date"], keep="first")


def _current_membership_state(
    con: duckdb.DuckDBPyConnection, eligibility_lookback_days: int
) -> pd.DataFrame:
    """Each eligible customer's most recent pre-`as_of` membership state, per origin.

    Requires `transactions`, `transactions_asof` (see `_dedupe_for_asof_join`)
    and `origins` (see `_origins_frame`) already registered on `con`. "Most
    recent transaction strictly before `as_of`, per customer, per origin" is
    exactly what an `ASOF JOIN` is built for -- DuckDB implements it with a
    merge-based algorithm designed for nearest-prior-row lookups, not a full
    inequality join. An earlier version of this function used `origins JOIN
    transactions ON transaction_date < as_of` (even bounded to a 500-day
    window) instead -- that join's cost is closer to `len(origins) x
    len(transactions)` before DuckDB can narrow it, and it did not finish in
    several minutes against the real 21.5M-row file. The `ASOF JOIN` below,
    measured directly against the same file, takes about 37s for all ~20
    origins. It joins against `transactions_asof`, not `transactions`
    directly -- see `_dedupe_for_asof_join` for why. The customer's *global*
    first-ever transaction (needed for `tenure_days`, which is not bounded to
    one origin at all) is computed once, independently of origins, from the
    undeduped `transactions` table, and joined back in by `customer_id`
    alone.

    The eligibility upper bound uses `eligibility_gap_days`, a column
    `origins` carries *separately* from each `Window`'s own `gap_days` --
    see `build_kkbox_asof_panel`'s docstring for why conflating the two
    would confound a gap ablation with a change in who's eligible at all.

    Returns:
        One row per (as_of, customer_id) among the eligible population:
        `as_of`, `customer_id`, `days_until_expiry`, `tenure_days`,
        `current_plan_price`, `is_auto_renew`.
    """
    return con.execute(f"""
        WITH first_transaction AS (
            SELECT customer_id, MIN(transaction_date) AS first_transaction_date
            FROM transactions
            GROUP BY customer_id
        ),
        origin_customers AS (
            SELECT o.as_of, o.eligibility_gap_days, c.customer_id
            FROM origins o
            CROSS JOIN (SELECT DISTINCT customer_id FROM transactions) c
        ),
        latest AS (
            SELECT
                oc.as_of,
                oc.eligibility_gap_days,
                t.customer_id,
                t.membership_expire_date,
                t.actual_amount_paid,
                t.is_auto_renew
            FROM origin_customers oc
            ASOF JOIN transactions_asof t
              ON t.customer_id = oc.customer_id AND t.transaction_date < oc.as_of
        )
        SELECT
            l.as_of,
            l.customer_id,
            date_diff('day', l.as_of, l.membership_expire_date) AS days_until_expiry,
            date_diff('day', ft.first_transaction_date, l.as_of) AS tenure_days,
            l.actual_amount_paid AS current_plan_price,
            l.is_auto_renew
        FROM latest l
        JOIN first_transaction ft ON ft.customer_id = l.customer_id
        WHERE l.membership_expire_date >= l.as_of - INTERVAL '{eligibility_lookback_days} days'
          AND l.membership_expire_date < l.as_of + INTERVAL (l.eligibility_gap_days) DAY
    """).df()


def _lookback_aggregates(
    con: duckdb.DuckDBPyConnection, feature_lookback_days: int
) -> pd.DataFrame:
    """Transaction and cancellation counts in the `feature_lookback_days` before `as_of`.

    Requires `eligible` (as_of, customer_id) and `transactions` registered
    on `con`. Joined by `customer_id` (an equi-join DuckDB can hash) with
    the date range as a secondary filter, not the other way around --
    `eligible` is already the small, per-origin-bounded population, so this
    join is far cheaper than `_current_membership_state`'s origins x
    transactions join.

    Returns:
        One row per (as_of, customer_id) in `eligible`: `as_of`,
        `customer_id`, `n_transactions_lookback`, `n_cancellations_lookback`.
    """
    return con.execute(f"""
        SELECT
            e.as_of,
            e.customer_id,
            COUNT(*) AS n_transactions_lookback,
            SUM(CASE WHEN t.is_cancel = 1 THEN 1 ELSE 0 END) AS n_cancellations_lookback
        FROM eligible e
        JOIN transactions t
          ON t.customer_id = e.customer_id
         AND t.transaction_date >= e.as_of - INTERVAL '{feature_lookback_days} days'
         AND t.transaction_date < e.as_of
        GROUP BY e.as_of, e.customer_id
    """).df()


def _label(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Churn label per (as_of, customer_id) in `eligible`.

    `churned = 0` iff a non-cancellation transaction (`is_cancel = 0`) lands
    in `[label_start, label_end)` -- a cancellation inside the label window
    is *stronger* evidence of churn, not a renewal, so it must not save a
    customer from `churned = 1` the way any transaction would under a naive
    "did they transact again" rule.

    Requires `eligible` and `origins` (for `label_start`/`label_end`) and
    `transactions` registered on `con`.

    Returns:
        One row per (as_of, customer_id) in `eligible`: `as_of`,
        `customer_id`, `churned`.
    """
    return con.execute("""
        WITH renewals AS (
            SELECT DISTINCT e.as_of, e.customer_id
            FROM eligible e
            JOIN origins o ON o.as_of = e.as_of
            JOIN transactions t
              ON t.customer_id = e.customer_id
             AND t.is_cancel = 0
             AND t.transaction_date >= o.label_start
             AND t.transaction_date < o.label_end
        )
        SELECT
            e.as_of,
            e.customer_id,
            CASE WHEN r.customer_id IS NULL THEN 1 ELSE 0 END AS churned
        FROM eligible e
        LEFT JOIN renewals r ON r.as_of = e.as_of AND r.customer_id = e.customer_id
    """).df()


def build_kkbox_asof_panel(
    transactions: pd.DataFrame,
    origins: list[Window],
    *,
    eligibility_lookback_days: int = ELIGIBILITY_LOOKBACK_DAYS,
    feature_lookback_days: int = FEATURE_LOOKBACK_DAYS,
    eligibility_gap_days: int | None = None,
) -> pd.DataFrame:
    """One row per (customer_id, as_of): the corrected KKBox panel.

    Computed as vectorized DuckDB SQL across every origin at once -- see the
    module docstring for why a per-origin pandas loop
    (`churnval.splits.build_asof_panel`'s shape) doesn't scale here.

    Args:
        transactions: cleaned transaction lines, e.g. from
            `churnval.kkbox_io.load_kkbox_transactions`.
        origins: scoring occasions, typically from
            `churnval.windows.rolling_origins`.
        eligibility_lookback_days: see `ELIGIBILITY_LOOKBACK_DAYS`.
        feature_lookback_days: see `FEATURE_LOOKBACK_DAYS`.
        eligibility_gap_days: overrides each origin's own `gap_days` for the
            eligibility upper bound only (the label still uses each
            origin's real `gap_days`). Defaults to `None`, which uses each
            origin's own `gap_days` -- the normal case. Pass an explicit
            value when comparing panels built with *different* `gap_days`
            for their labels (a gap ablation, e.g. `07_generalisation`'s
            no-gap comparison): without this, changing `gap_days` moves the
            eligibility window's upper bound too, which silently changes
            *who is eligible at all* (a customer whose membership expires
              5 days after `as_of` is eligible when `gap_days=7` but not
            when `gap_days=0`) on top of changing the label -- confounding
            "what does the gap alone cost" with "what does a smaller
            eligible population, skewed toward already-lapsed customers,
            cost." Pin this to the *honest* protocol's own `gap_days` when
            building an ablation panel, so only the label changes.

    Returns:
        One row per (customer_id, as_of): `customer_id`, `as_of`, plus
        `FEATURE_COLS`, plus `churned`.
    """
    con = duckdb.connect()
    try:
        con.register("transactions", transactions)
        con.register("transactions_asof", _dedupe_for_asof_join(transactions))
        con.register("origins", _origins_frame(origins, eligibility_gap_days))

        membership_state = _current_membership_state(con, eligibility_lookback_days)
        con.register("eligible", membership_state[["as_of", "customer_id"]])

        lookback = _lookback_aggregates(con, feature_lookback_days)
        label = _label(con)
    finally:
        con.close()

    panel = membership_state.merge(lookback, on=["as_of", "customer_id"], how="left")
    panel = panel.merge(label, on=["as_of", "customer_id"], how="left")
    # `_lookback_aggregates` is an inner join over a bounded window -- a
    # customer with zero transactions in the last `feature_lookback_days`
    # (a long-duration-plan customer whose only recent transaction is older
    # than the window, e.g. one of the real data's 410-day plans) simply
    # doesn't appear in it, leaving `NaN` here after the left merge. That's
    # a real "no recent activity" signal, not missing data -- zero is the
    # correct value, not an unknown one, so it's filled rather than left for
    # LightGBM's native (and silent) missing-value handling to absorb.
    panel[["n_transactions_lookback", "n_cancellations_lookback"]] = panel[
        ["n_transactions_lookback", "n_cancellations_lookback"]
    ].fillna(0)
    return panel[["customer_id", "as_of", *FEATURE_COLS, "churned"]]


def kkbox_naive_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Full-history subscription-state features -- ignores `as_of` on purpose.

    The KKBox analogue of `naive_baseline.naive_features`: mistake #1 (a
    naive baseline's features ignore `as_of`) made concrete for contractual
    data. The same row is reused for every origin a customer appears in --
    that reuse happens in the notebook, not here; this function only
    computes the (not as-of-safe) values, from each customer's single most
    recent transaction in their *entire* history and their full-history
    transaction/cancellation counts.

    One pandas groupby over the whole table -- not DuckDB, a single pass,
    well within pandas' comfort zone even at this row count
    (`docs/standards/coding.md`).

    Args:
        transactions: cleaned transaction lines.

    Returns:
        DataFrame indexed by `customer_id`: `days_until_expiry` (relative to
        the dataset's own last transaction date, not any row's `as_of` --
        the naive mistake, made concrete), `tenure_days` (relative to the
        same fixed reference), `current_plan_price`, `is_auto_renew`,
        `n_transactions_lookback`, `n_cancellations_lookback` (both over the
        customer's *entire* history, not a lookback window).
    """
    reference_date = transactions["transaction_date"].max()
    latest = transactions.sort_values("transaction_date").groupby("customer_id").tail(1)
    latest = latest.set_index("customer_id")
    grouped = transactions.groupby("customer_id")

    return pd.DataFrame(
        {
            "days_until_expiry": (latest["membership_expire_date"] - reference_date).dt.days,
            "tenure_days": (reference_date - grouped["transaction_date"].min()).dt.days,
            "current_plan_price": latest["actual_amount_paid"],
            "is_auto_renew": latest["is_auto_renew"],
            "n_transactions_lookback": grouped.size(),
            "n_cancellations_lookback": grouped["is_cancel"].sum(),
        }
    )
