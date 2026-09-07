"""Loading and cleaning KKBox subscription transactions.

At 21.5M rows / 1.73GB, `transactions.csv` is the first source file in this
project past pandas' comfort threshold (`docs/standards/coding.md`): the
read-and-clean step runs as DuckDB SQL directly against the raw CSV (a full
scan takes seconds, measured directly against the real download) rather than
a pandas `read_csv`, which would be slower and far heavier on memory while
parsing. `_clean`'s SQL is the single source of truth for the cleaning rules,
shared between the real large-file path (`load_kkbox_transactions`, which
never materializes the full raw file in pandas -- it streams straight from
CSV to a cleaned Parquet cache) and this module's own tests, which run the
identical query against a tiny in-memory DataFrame. Every function downstream
of the cached Parquet still receives a plain pandas `DataFrame`, so nothing
else in this project needs to know DuckDB was involved.

`is_cancel = 1` rows are kept, not dropped, unlike `churnval.io`'s Online
Retail II cleaning (which drops cancelled invoices outright): a cancellation
is a real, informative subscription event here -- it feeds
`churnval.kkbox_features`'s cancellation-count feature -- and is excluded
only at label-construction time, where a cancellation inside the label
window is treated as *stronger* evidence of churn, not weaker.

`members_v3.csv` is deliberately not loaded anywhere in this project. Its
`bd` (age) column is outside `[1, 100]` for 67% of rows and `gender` is null
in 65% -- unusable as a feature source, and the corrected protocol (`03`
already proved this for Online Retail II) doesn't need demographics on top
of transaction-derived features anyway. See `notebooks/07_generalisation.py`
for the counts and `docs/data/kkbox.md` for the dataset card.
"""

from __future__ import annotations

import logging

import duckdb
import pandas as pd

from churnval.config import PATHS

logger = logging.getLogger(__name__)

RAW_TRANSACTIONS_CSV = PATHS.raw / "kkbox" / "transactions.csv"
CACHE_PARQUET = PATHS.interim / "kkbox_transactions.parquet"


def _clean_query(source: str) -> str:
    """The cleaning rules, as SQL against any relation named `source` with the raw schema.

    Kept as a query-builder (not inlined twice) so `_clean` (tested against a
    tiny in-memory frame) and `load_kkbox_transactions` (run against the full
    CSV via `read_csv_auto`, never touching pandas until the result is
    written) share the identical rules rather than two copies that could
    drift.

    Drops (counts verified directly against the real file with DuckDB
    `SELECT COUNT(*)` queries, not estimated): exact full-row duplicates
    (3,339); rows with a corrupt sentinel `membership_expire_date` before
    2000-01-01 (1,778 rows -- these would otherwise produce absurd
    `days_until_expiry` outliers); rows where
    `membership_expire_date < transaction_date` *and* `is_cancel = 0`
    (6,460 -- internally inconsistent, an active-subscription transaction
    claiming to expire before it happened, not explainable as a legitimate
    cancellation the way an `is_cancel = 1` row with the same date ordering
    is). These three sets overlap (1,530 rows are both a corrupt sentinel
    and date-inconsistent), so the net rows dropped from the real file is
    10,045, not the sum of the three counts.

    Keeps, deliberately, as known quirks rather than corrected: `is_cancel = 1`
    rows regardless of amount (see module docstring); zero-`actual_amount_paid`
    rows (9.5% of the real file, mostly not cancellations -- plausibly free
    trials or full discounts); `payment_plan_days = 0` rows (4.0% of the real
    file, 851,535 of which have `is_cancel = 0` and a normal future expiry --
    looks like a legitimate promotional pattern, not garbage).
    """
    return f"""
        WITH parsed AS (
            SELECT DISTINCT
                msno AS customer_id,
                payment_method_id,
                payment_plan_days,
                plan_list_price,
                actual_amount_paid,
                is_auto_renew,
                strptime(CAST(transaction_date AS VARCHAR), '%Y%m%d') AS transaction_date,
                strptime(CAST(membership_expire_date AS VARCHAR), '%Y%m%d')
                    AS membership_expire_date,
                is_cancel
            FROM {source}
        )
        SELECT *
        FROM parsed
        WHERE membership_expire_date >= TIMESTAMP '2000-01-01'
          AND NOT (membership_expire_date < transaction_date AND is_cancel = 0)
    """


def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply `_clean_query` to an in-memory frame using the raw CSV's column names.

    Args:
        raw: columns `msno, payment_method_id, payment_plan_days,
            plan_list_price, actual_amount_paid, is_auto_renew,
            transaction_date, membership_expire_date, is_cancel` --
            `transaction_date`/`membership_expire_date` as integer `YYYYMMDD`,
            matching the raw CSV exactly.

    Returns:
        Same columns, `customer_id` renamed from `msno`, dates parsed to
        `datetime64`, cleaned per `_clean_query`'s docstring.
    """
    con = duckdb.connect()
    try:
        con.register("raw", raw)
        return con.execute(_clean_query("raw")).df()
    finally:
        con.close()


def load_kkbox_transactions(*, force_reload: bool = False) -> pd.DataFrame:
    """Load cleaned KKBox transaction lines, cached to Parquet.

    Unlike `churnval.io.load_online_retail_ii`, the uncached path never reads
    the full raw CSV into pandas: DuckDB streams `read_csv_auto` straight
    into `COPY ... TO parquet`, and only the cleaned, cached result is ever
    handed to pandas.

    Args:
        force_reload: re-run the DuckDB clean even if a cached Parquet file
            already exists.

    Returns:
        See `_clean_query`.

    Raises:
        FileNotFoundError: the source CSV has not been fetched yet.
    """
    if CACHE_PARQUET.exists() and not force_reload:
        return pd.read_parquet(CACHE_PARQUET)

    if not RAW_TRANSACTIONS_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_TRANSACTIONS_CSV} not found. Run `uv run churnval fetch kkbox` first."
        )

    logger.info("cleaning %s via DuckDB -- this is the slow part", RAW_TRANSACTIONS_CSV.name)
    PATHS.ensure()
    con = duckdb.connect()
    try:
        source = f"read_csv_auto('{RAW_TRANSACTIONS_CSV.as_posix()}')"
        con.execute(
            f"COPY ({_clean_query(source)}) TO '{CACHE_PARQUET.as_posix()}' (FORMAT PARQUET)"
        )
    finally:
        con.close()
    return pd.read_parquet(CACHE_PARQUET)
