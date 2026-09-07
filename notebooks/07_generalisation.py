# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 07 — Generalisation
#
# **Question this notebook answers:** does the same failure, random-split
# validation inflating offline churn metrics, appear in *contractual*
# churn as well as Online Retail II's non-contractual case? And does the
# protocol built to fix it (`03`-`06`) transfer to a dataset roughly 25x
# larger?
#
# **Consumes:** `data/raw/kkbox/transactions.csv` (21.5M rows, 1.73GB,
# 2015-01 to 2017-02) via the `churnval.kkbox_io` and
# `churnval.kkbox_features` modules. The schema is genuinely different from
# Online Retail II's, so eligibility, features and the label are rebuilt for
# this dataset (see
# [ADR-0013](../docs/adr/0013-kkbox-eligibility-and-label-definition.md)).
# Everything downstream of the panel, `rolling_origin_backtest`,
# `fit_at_origin`, all of `churnval.evaluation` and `churnval.calibration`,
# and `churnval.hazard` with one added parameter, is reused **unchanged**
# from `03`-`06`; see
# [ADR-0014](../docs/adr/0014-kkbox-module-reuse-strategy.md). Also
# `reports/results_naive.json` and `results_temporal.json` from `01`/`03`
# for the final side-by-side chart.
#
# **Produces:** a KKBox naive-vs-corrected replication of `04`'s headline
# finding; a KKBox calibration check (`05`'s machinery, reused); a KKBox
# hazard comparison (`06`'s machinery, reused); a leakage audit; and the
# portable checklist written to `docs/checklist.md`.
#
# **Runtime:** the two genuinely new KKBox-scale steps were measured rather
# than estimated. Loading and cleaning `transactions.csv` via DuckDB takes
# about 20s; building the full 20-origin as-of panel takes about 84s (see
# [ADR-0015](../docs/adr/0015-duckdb-for-kkbox-scale.md) for why a
# per-origin pandas loop, `03`'s own shape, does not scale to this file).
# The backtest and the hazard person-period build, the latter still `06`'s
# unvectorized Python loop, are timed live in their own cells below.
#
# ---
#
# `01` through `06` built and validated this project's protocol against one
# dataset: Online Retail II, non-contractual churn, where the event has to
# be inferred from a purchase gap. ADR-0003 chose that dataset alongside
# KKBox specifically so this question could be asked later. Does any of it
# depend on Online Retail II's particular shape, or does it hold where churn
# is an explicit, observed event (a subscription's own expiry), at a scale
# where pandas stops being an option?


# %% [markdown]
# ## Loading and cleaning transactions.csv
#
# `churnval.kkbox_io.load_kkbox_transactions` never reads the full 21.5M-row
# raw CSV into pandas. DuckDB streams the clean straight into a cached
# Parquet file, and only that cached result is handed to pandas. See the
# module docstring and ADR-0015 for the cleaning rules.

# %% jupyter={"source_hidden": true}
import json
import time
from datetime import timedelta

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from churnval.calibration import brier_decomposition, fit_isotonic, fit_platt, reliability_table
from churnval.config import PATHS, SEED
from churnval.evaluation import score
from churnval.hazard import (
    build_person_period_panel,
    fit_hazard_at_origin,
    hazard_curve,
    hazard_rolling_origin_backtest,
)
from churnval.kkbox_features import FEATURE_COLS, build_kkbox_asof_panel, kkbox_naive_features
from churnval.kkbox_io import RAW_TRANSACTIONS_CSV, load_kkbox_transactions
from churnval.plotting import PALETTE, set_style
from churnval.splits import fit_at_origin, rolling_origin_backtest
from churnval.windows import rolling_origins

PATHS.ensure()
set_style()

transactions = load_kkbox_transactions()
print(
    f"{len(transactions):,} transaction lines, "
    f"{transactions['customer_id'].nunique():,} customers, "
    f"{transactions['transaction_date'].min():%Y-%m-%d} to "
    f"{transactions['transaction_date'].max():%Y-%m-%d}"
)

# %% [markdown]
# ## What cleaning removed, and what it deliberately kept
#
# The same discipline `01`'s dataset-card work established: every cleaning
# rule gets a live count against the real data. This doubles as
# leakage-audit evidence later.

# %% jupyter={"source_hidden": true}
con = duckdb.connect()
src = f"read_csv_auto('{RAW_TRANSACTIONS_CSV.as_posix()}')"
raw_count = con.execute(f"SELECT COUNT(*) FROM {src}").fetchone()[0]
duplicate_count = (
    raw_count - con.execute(f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM {src})").fetchone()[0]
)
sentinel_count = con.execute(f"""
    SELECT COUNT(*) FROM {src}
    WHERE strptime(CAST(membership_expire_date AS VARCHAR), '%Y%m%d') < TIMESTAMP '2000-01-01'
""").fetchone()[0]
inconsistent_count = con.execute(f"""
    SELECT COUNT(*) FROM {src}
    WHERE strptime(CAST(membership_expire_date AS VARCHAR), '%Y%m%d')
        < strptime(CAST(transaction_date AS VARCHAR), '%Y%m%d')
      AND is_cancel = 0
""").fetchone()[0]
con.close()

pd.Series(
    {
        "raw rows": raw_count,
        "exact duplicates (dropped)": duplicate_count,
        "corrupt sentinel expiry dates (dropped)": sentinel_count,
        "expiry-before-transaction, not a cancellation (dropped)": inconsistent_count,
        "net rows after cleaning": len(transactions),
    }
)

# %% [markdown]
# The net drop is far smaller than the sum of the three rules, because they
# overlap: a corrupt sentinel date is often also date-inconsistent.
# `is_cancel = 1` rows are kept regardless of date ordering, since a
# cancellation is a real event here and feeds a feature below. See
# `docs/data/kkbox.md` for the quirks kept deliberately (zero-amount
# transactions, `payment_plan_days = 0`) rather than corrected.
#
# Because the rules overlap, a chart of those raw counts would double-count
# and not sum to the net drop. Making the categories disjoint means picking
# a precedence order and assigning each dropped row to the *first* rule it
# matches: exact duplicate first, then corrupt sentinel expiry, then
# expiry-before-transaction. Every dropped row then falls in exactly one
# bucket, so the counts sum to the net drop. Checked below.

# %% jupyter={"source_hidden": true}
con = duckdb.connect()
attribution = con.execute(f"""
    WITH ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY msno, payment_method_id, payment_plan_days, plan_list_price,
                             actual_amount_paid, is_auto_renew, transaction_date,
                             membership_expire_date, is_cancel
                ORDER BY (SELECT NULL)
            ) AS dup_rank
        FROM {src}
    )
    SELECT
        CASE
            WHEN dup_rank > 1 THEN 'exact duplicate'
            WHEN strptime(CAST(membership_expire_date AS VARCHAR), '%Y%m%d') < TIMESTAMP '2000-01-01'
                THEN 'corrupt sentinel expiry'
            WHEN strptime(CAST(membership_expire_date AS VARCHAR), '%Y%m%d')
                 < strptime(CAST(transaction_date AS VARCHAR), '%Y%m%d') AND is_cancel = 0
                THEN 'expiry before transaction'
            ELSE 'kept'
        END AS reason,
        COUNT(*) AS n
    FROM ranked
    GROUP BY reason
""").df()
con.close()

dropped_attribution = (
    attribution.loc[attribution["reason"] != "kept"]
    .set_index("reason")["n"]
    .reindex(["exact duplicate", "corrupt sentinel expiry", "expiry before transaction"])
)
assert dropped_attribution.sum() == raw_count - len(transactions), (
    "disjoint attribution must sum to exactly the net drop"
)
dropped_attribution

# %% jupyter={"source_hidden": true}
pct_of_dropped = dropped_attribution / dropped_attribution.sum()
pct_of_raw = dropped_attribution / raw_count

fig, ax = plt.subplots(figsize=(8.5, 3.5))
bars = ax.barh(dropped_attribution.index, dropped_attribution.to_numpy(), color=PALETTE["neutral"])
for bar, n, p_dropped, p_raw in zip(
    bars,
    dropped_attribution.to_numpy(),
    pct_of_dropped.to_numpy(),
    pct_of_raw.to_numpy(),
    strict=True,
):
    ax.text(
        bar.get_width(),
        bar.get_y() + bar.get_height() / 2,
        f"  {n:,} rows -- {p_dropped:.1%} of dropped, {p_raw:.3%} of raw",
        va="center",
        fontsize=9,
    )
ax.set_xlim(0, dropped_attribution.max() * 2.3)
ax.set_xlabel(
    "rows dropped (disjoint attribution, duplicates -> sentinel -> inconsistent precedence)"
)
ax.set_title(
    "Expiry-before-transaction rows, not duplicates, are the largest\nshare of what cleaning drops",
    loc="left",
    fontsize=10,
)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(PATHS.figures / "kkbox_cleaning_attribution.png", dpi=200)
# fig

# %% [markdown]
# `members_v3.csv` is inspected once, here, and then never loaded anywhere
# else in this project. The two columns below are why.

# %% jupyter={"source_hidden": true}
members_sample = pd.read_csv(PATHS.raw / "kkbox" / "members_v3.csv", usecols=["bd", "gender"])
bd_out_of_range = ((members_sample["bd"] < 1) | (members_sample["bd"] > 100)).mean()
gender_null = members_sample["gender"].isna().mean()
print(
    f"bd (age) outside [1, 100]: {bd_out_of_range:.0%} of rows -- "
    f"min={members_sample['bd'].min()}, max={members_sample['bd'].max()}\n"
    f"gender null: {gender_null:.0%} of rows"
)

# %% [markdown]
# ## Window constants: no override needed
#
# Online Retail II needed a dataset-specific `HORIZON_DAYS = 90` override
# (ADR-0007) because its non-contractual purchase cadence did not fit the
# project default. `config.py`'s `DEFAULT_HORIZON_DAYS = 30` was written
# with KKBox's monthly billing cadence in mind, which the real
# `payment_plan_days` distribution below confirms rather than assumes.

# %% jupyter={"source_hidden": true}
plan_days_distribution = (
    transactions["payment_plan_days"]
    .value_counts(normalize=True)
    .sort_values(ascending=False)
    .head(8)
)
plan_days_distribution

# %% [markdown]
# The same eight values ordered by plan length rather than frequency, so the
# gap between the dominant monthly cadence and the tail is legible. The 30-
# and 31-day bars are highlighted.

# %% jupyter={"source_hidden": true}
plan_days_ordered = plan_days_distribution.sort_index()
bar_colors = [
    PALETTE["emphasis"] if days in (30, 31) else PALETTE["neutral"]
    for days in plan_days_ordered.index
]
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(plan_days_ordered.index.astype(str), plan_days_ordered.to_numpy(), color=bar_colors)
for bar, pct in zip(bars, plan_days_ordered.to_numpy(), strict=True):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{pct:.1%}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
ax.set_xlabel("payment_plan_days")
ax.set_ylabel("share of transactions")
ax.set_title(
    f"{plan_days_ordered.loc[30]:.0%} of transactions are on a 30-day plan -- "
    "the default horizon needs no override",
    loc="left",
    fontsize=10,
)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(PATHS.figures / "kkbox_payment_plan_days.png", dpi=200)
# fig

# %% [markdown]
# 88% of transactions are on a 30-day plan and another 3.6% on 31-day, so
# `DEFAULT_HORIZON_DAYS = 30` and `DEFAULT_GAP_DAYS = 7` are used as-is,
# a deliberate contrast with Online Retail II. The tail beyond the monthly
# cadence (7-day trials; 90, 100, 180, 195 and 410-day plans) is exactly why
# eligibility uses a wider lookback than KKBox's own 30-day grace period.

# %% [markdown]
# ## Eligibility: expiry-based, not activity-based
#
# Online Retail II's eligibility rule is activity-based: at least one
# purchase in the 365 days before `as_of`. Applied to KKBox directly, that
# rule scores over a million ever-present customers per origin, most of whom
# are not due for a renewal decision for months. Measured against the real
# data rather than assumed:

# %% jupyter={"source_hidden": true}
activity_lookback_counts = {}
for as_of_str in ("2015-08-01", "2016-02-01", "2016-08-01", "2017-01-01"):
    as_of = pd.Timestamp(as_of_str)
    window_start = as_of - timedelta(days=365)
    mask = (transactions["transaction_date"] >= window_start) & (
        transactions["transaction_date"] < as_of
    )
    activity_lookback_counts[as_of_str] = transactions.loc[mask, "customer_id"].nunique()
pd.Series(
    activity_lookback_counts, name="365-day activity-lookback eligible customers"
).sort_values(ascending=False)

# %% [markdown]
# `build_kkbox_asof_panel` uses an expiry-based rule instead: a customer is
# eligible if their *current* membership expires in
# `[as_of - 45 days, as_of + gap_days)`, meaning they are due for a renewal
# decision around now, which is the contractual question this protocol
# actually asks. `ELIGIBILITY_LOOKBACK_DAYS = 45` is wider than KKBox's own
# 30-day grace convention specifically to catch the non-30/31-day tail shown
# above. See [ADR-0013](../docs/adr/0013-kkbox-eligibility-and-label-definition.md)
# for the full reasoning, including why this project's label does not
# reproduce KKBox's own churn definition byte for byte.

# %% [markdown]
# ## Building the origins and the corrected panel
#
# The same `rolling_origins` every prior notebook uses. The panel build is
# the one step that needed a different implementation strategy at this
# scale; see ADR-0015.

# %% jupyter={"source_hidden": true}
first_as_of = transactions["transaction_date"].min().normalize() + timedelta(days=181)
last_event = transactions["transaction_date"].max()
origins = rolling_origins(first_as_of, last_event, gap_days=7, horizon_days=30, step_days=30)
print(
    f"{len(origins)} origins, {origins[0].as_of:%Y-%m-%d} to {origins[-1].as_of:%Y-%m-%d} "
    f"(Online Retail II had {10})"
)

# %% jupyter={"source_hidden": true}
panel = build_kkbox_asof_panel(transactions, origins)
print(
    f"{len(panel):,} panel rows, {panel['churned'].mean():.1%} churned overall, "
    f"{len(panel) / len(origins):,.0f} eligible customers per origin on average"
)
panel_by_origin = panel.groupby("as_of").agg(
    n=("customer_id", "size"), prevalence=("churned", "mean")
)
panel_by_origin

# %% [markdown]
# Eligible population and churn prevalence by origin, oldest to newest, as
# two panels sharing one x axis rather than one chart with two y axes. A
# headcount and a rate do not belong on the same scale, and
# `docs/standards/notebooks.md` rules out dual axes for that reason.

# %% jupyter={"source_hidden": true}
fig, axes = plt.subplots(2, 1, figsize=(9.5, 6), sharex=True)
axes[0].plot(panel_by_origin.index, panel_by_origin["n"], marker="o", color=PALETTE["train"])
axes[0].set_ylabel("eligible customers (n)")
axes[0].tick_params(labelbottom=False)
axes[0].set_title(
    f"The eligible population ranges {panel_by_origin['n'].min():,.0f}-"
    f"{panel_by_origin['n'].max():,.0f} per origin, no clean trend",
    loc="left",
    fontsize=10,
)

axes[1].plot(
    panel_by_origin.index, panel_by_origin["prevalence"], marker="o", color=PALETTE["churned"]
)
peak_origin = panel_by_origin["prevalence"].idxmax()
axes[1].set_ylabel("churn prevalence")
axes[1].set_ylim(0, 1)
axes[1].set_xlabel("as_of")
axes[1].set_title(
    f"Prevalence swings with it -- {peak_origin:%Y-%m-%d} peaks at "
    f"{panel_by_origin['prevalence'].max():.0%}",
    loc="left",
    fontsize=10,
)
for ax in axes:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(PATHS.figures / "kkbox_origins_population_prevalence.png", dpi=200)
# fig

# %% [markdown]
# ### One consequence of the gap that Online Retail II never exposed
#
# The 7-day gap does something on contractual data it does not do on retail
# data, and it is worth measuring before any result rests on it. A customer
# eligible at an origin is by definition near a renewal decision. If they
# renew *during the gap*, on a 30-day plan, their next decision lands
# roughly 30 days later, at or just past the far edge of the label window
# `[as_of + 7, as_of + 37)`. Whether they are labelled churned then depends
# on which side of that edge their next renewal falls, which is close to a
# coin flip driven by plan length rather than by any intention to leave.
#
# On Online Retail II this was harmless. A purchase inside the gap simply
# did not count toward the label, and the next purchase was not scheduled by
# a contract. Here the contract schedules it. The cell below measures how
# much of the panel this touches.

# %% jupyter={"source_hidden": true}
start = time.perf_counter()
con = duckdb.connect()
try:
    con.register("panel", panel[["customer_id", "as_of", "churned"]])
    con.register("renewals", transactions.loc[transactions["is_cancel"] == 0])
    gap_effect = con.execute("""
        WITH gap_renewals AS (
            SELECT DISTINCT p.as_of, p.customer_id
            FROM panel p
            JOIN renewals t
              ON t.customer_id = p.customer_id
             AND t.transaction_date >= p.as_of
             AND t.transaction_date < p.as_of + INTERVAL 7 DAY
        )
        SELECT
            CASE WHEN g.customer_id IS NULL
                 THEN 'no renewal during the gap'
                 ELSE 'renewed during the gap' END AS bucket,
            COUNT(*) AS n,
            AVG(p.churned) AS churn_rate
        FROM panel p
        LEFT JOIN gap_renewals g ON g.as_of = p.as_of AND g.customer_id = p.customer_id
        GROUP BY 1
        ORDER BY 1
    """).df()
finally:
    con.close()
gap_effect["share of panel"] = gap_effect["n"] / gap_effect["n"].sum()
print(f"computed in {time.perf_counter() - start:.0f}s")
gap_effect

# %% [markdown]
# A large share of the eligible population renews during the gap, and a
# substantial minority of *those* customers are still labelled churned,
# because their next renewal after that falls outside the label window. This
# is not a leak, and it is not obviously the wrong choice: the definition
# says the outcome window opens when action becomes possible, and a customer
# who renews and then lapses genuinely did lapse. But it means the KKBox
# label is measuring something subtly different from Online Retail II's, and
# in particular it means the no-gap ablation below is **not** the same
# experiment `04` ran. Read the next two sections with that in mind.

# %% [markdown]
# ## The corrected backtest
#
# `rolling_origin_backtest`, unchanged, the same function `03` built and
# ADR-0009 documents. This cell contains no KKBox-specific logic at all.

# %% jupyter={"source_hidden": true}
per_origin_corrected, predictions_corrected = rolling_origin_backtest(
    panel, origins, feature_cols=FEATURE_COLS, min_training_origins=3, seed=SEED
)
per_origin_corrected[["as_of", "n", "prevalence", "roc_auc", "pr_auc", "brier"]]

# %% [markdown]
# Four panels sharing the `as_of` axis: one per metric, plus prevalence in
# its own panel rather than a second y axis on any of the three, so the
# highest-prevalence origin is readable without distorting the metric
# scales.

# %% jupyter={"source_hidden": true}
fig, axes = plt.subplots(4, 1, figsize=(9.5, 11), sharex=True)
backtest_metric_specs = [
    ("roc_auc", "ROC AUC"),
    ("pr_auc", "PR AUC"),
    ("brier", "Brier (lower is better)"),
]
for ax, (col, label) in zip(axes[:3], backtest_metric_specs, strict=True):
    ax.plot(
        per_origin_corrected["as_of"], per_origin_corrected[col], marker="o", color=PALETTE["train"]
    )
    ax.set_ylabel(label)
    ax.tick_params(labelbottom=False)

axes[3].plot(
    per_origin_corrected["as_of"],
    per_origin_corrected["prevalence"],
    marker="o",
    color=PALETTE["neutral"],
    alpha=0.7,
)
hardest = per_origin_corrected.loc[per_origin_corrected["prevalence"].idxmax()]
axes[3].scatter([hardest["as_of"]], [hardest["prevalence"]], color=PALETTE["emphasis"], zorder=3)
axes[3].annotate(
    f"{hardest['as_of']:%Y-%m-%d}: {hardest['prevalence']:.0%} prevalence",
    xy=(hardest["as_of"], hardest["prevalence"]),
    xytext=(0, 8),
    textcoords="offset points",
    ha="center",
    fontsize=9,
    color=PALETTE["emphasis"],
)
axes[3].set_ylabel("prevalence")
axes[3].set_xlabel("as_of")

axes[0].set_title(
    "Ranking and calibration both dip at the highest-prevalence origins", loc="left", fontsize=10
)
for ax in axes:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(PATHS.figures / "kkbox_corrected_backtest_by_origin.png", dpi=200)
# fig

# %% [markdown]
# ROC AUC is consistently well above Online Retail II's 0.74-0.79 range,
# often above 0.9. That is expected rather than a sign of a stronger model.
# KKBox's `days_until_expiry` is close to a direct read of the label
# mechanism (a subscription already lapsed is likely to stay lapsed), where
# Online Retail II's RFM features are only ever a proxy for unobserved
# intent. The regime swings are real too: 2016-07-25 is the hardest origin,
# and it is also the highest-prevalence one, the same prevalence-driven
# variation `03` showed for Online Retail II.

# %% jupyter={"source_hidden": true}
corrected_pooled = score(
    predictions_corrected["y_true"], predictions_corrected["y_prob"], label="KKBox corrected"
)
corrected_pooled.as_row()

# %% [markdown]
# ## The naive KKBox replication
#
# Reusing the honest panel's own population as the common reference (the
# same argument ADR-0010 made for Online Retail II), each of `01`'s three
# mistakes gets the smallest change that still isolates it. Full
# notebook-for-notebook parity with `01` and `02` would be its own project;
# this is a scope choice, not an ADR-worthy one.
#
# Mistakes #1 and #2 come from a single ordinary `train_test_split` below,
# not two separate changes. Swapping in `kkbox_naive_features` (mistake #1,
# features frozen to the dataset's last transaction date instead of each
# row's `as_of`) is explicit. Mistake #2 falls out for free from the same
# cell: `kkbox_naive_features` returns one static row per customer, reused
# across every origin that customer appears in, so a row-level split with no
# grouping lets the same customer land on both sides without any extra code.

# %% jupyter={"source_hidden": true}
naive_features_table = kkbox_naive_features(transactions)
naive_panel = panel[["customer_id", "as_of", "churned"]].merge(
    naive_features_table.reset_index(), on="customer_id", how="left"
)
print(f"{len(naive_panel):,} rows -- same population as the corrected panel, naive features")

X_naive = naive_panel[FEATURE_COLS]
y_naive = naive_panel["churned"]
X_train, X_test, y_train, y_test = train_test_split(
    X_naive, y_naive, test_size=0.2, random_state=SEED, stratify=y_naive
)

naive_model = LGBMClassifier(random_state=SEED, verbosity=-1)
naive_model.fit(X_train, y_train)
naive_prob = naive_model.predict_proba(X_test)[:, 1]
naive_result = score(y_test, naive_prob, label="KKBox naive (mistakes #1+#2)")
naive_result.as_row()

# %% [markdown]
# Naive vs. corrected, one panel per metric, with value labels and the gap
# stated rather than left to the eye. Brier's "better" direction is the
# reverse of the other two, stated in the y-axis label.

# %% jupyter={"source_hidden": true}
naive_vs_corrected_metrics = [
    ("roc_auc", "ROC AUC (higher is better)"),
    ("pr_auc", "PR AUC (higher is better)"),
    ("brier", "Brier (lower is better)"),
]
fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
for ax, (col, label) in zip(axes, naive_vs_corrected_metrics, strict=True):
    naive_v = getattr(naive_result, col)
    corrected_v = getattr(corrected_pooled, col)
    bars = ax.bar(
        ["naive", "corrected"], [naive_v, corrected_v], color=[PALETTE["both"], PALETTE["test"]]
    )
    for bar, v in zip(bars, [naive_v, corrected_v], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9
        )
    ax.set_ylabel(label)
    ax.set_title(f"gap = {naive_v - corrected_v:+.3f}", loc="left", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.suptitle("Naive KKBox: better-looking ranking, and a Brier score that misleads", fontsize=10.5)
fig.tight_layout()
fig.savefig(PATHS.figures / "kkbox_naive_vs_corrected_metrics.png", dpi=200)
# fig

# %% [markdown]
# Brier's gap runs the "wrong" way: naive's sits *below* corrected's, which
# would read as better calibration if the two numbers came from comparable
# evaluations. They do not. Naive is scored on one random, stratified 20%
# slice of one static population, while corrected pools predictions across
# 17 test origins whose prevalence swings sharply. Brier is sensitive to
# exactly that kind of regime mixing, and a single-snapshot evaluation never
# encounters it. The naive bar's lower Brier is an artefact of an easier
# evaluation, not a better-calibrated model. The ROC AUC and PR AUC panels,
# which do not reverse, are the honest read on naive's inflated ranking.

# %% [markdown]
# ### Mistake #3, and why it is not the same experiment `04` ran
#
# The no-gap arm reuses `build_kkbox_asof_panel` and `rolling_origins`
# unchanged with `gap_days=0`, plus one addition Online Retail II did not
# need. `eligibility_gap_days=7` pins the eligibility window's upper bound
# to the honest protocol's gap regardless of what `gap_days` does to the
# label. Without it, a customer whose membership expires five days after
# `as_of` is eligible under the real protocol but not under the ablation,
# silently shrinking the population toward already-lapsed customers and
# confounding "what does the gap cost" with "what does a harder population
# cost". Found while building this notebook (the first run showed 79% churn
# for this arm against about 48% for the others) and fixed in
# `churnval.kkbox_features` rather than left as a caveat.
#
# Pinning eligibility fixes the population but cannot fix the label. With
# `gap_days=0` the label window becomes `[as_of, as_of + 30)`, so every
# customer who renews during what was the gap is now counted as a renewal
# and labelled `churned=0`. The cell above measured how many customers that
# is. So this ablation changes *which customers are churners*, not just the
# window's boundary, and it changes it in the direction of an easier
# problem: a renewal in the first seven days is close to deterministic given
# `days_until_expiry`. The resulting number is a legitimate measurement of
# something, but it is not comparable to `04`'s no-gap arm, where the gap
# moved a boundary and nothing else. The prevalence difference between the
# two arms is the tell.

# %% jupyter={"source_hidden": true}
origins_nogap = rolling_origins(first_as_of, last_event, gap_days=0, horizon_days=30, step_days=30)
panel_nogap = build_kkbox_asof_panel(transactions, origins_nogap, eligibility_gap_days=7)
per_origin_nogap, predictions_nogap = rolling_origin_backtest(
    panel_nogap, origins_nogap, feature_cols=FEATURE_COLS, min_training_origins=3, seed=SEED
)
nogap_result = score(predictions_nogap["y_true"], predictions_nogap["y_prob"], label="KKBox no-gap")
nogap_result.as_row()

# %% [markdown]
# The two arms are built on the same eligible population but do not agree on
# who churned, which is the point made above, quantified:

# %% jupyter={"source_hidden": true}
label_shift = pd.DataFrame(
    [
        {
            "arm": "corrected (gap=7)",
            "n": len(panel),
            "prevalence": panel["churned"].mean(),
        },
        {
            "arm": "no-gap (gap=0)",
            "n": len(panel_nogap),
            "prevalence": panel_nogap["churned"].mean(),
        },
    ]
).set_index("arm")
label_shift["prevalence delta vs. corrected"] = (
    label_shift["prevalence"] - label_shift.loc["corrected (gap=7)", "prevalence"]
)
label_shift

# %% [markdown]
# ## The headline comparison
#
# KKBox's naive, no-gap and corrected numbers alongside Online Retail II's
# `01`/`03` results loaded from `reports/`, the same convention `04` used.

# %% jupyter={"source_hidden": true}
retail_naive = json.loads((PATHS.reports / "results_naive.json").read_text())
retail_temporal = json.loads((PATHS.reports / "results_temporal.json").read_text())

results_path = PATHS.reports / "results_kkbox_naive.json"
results_path.write_text(json.dumps(naive_result.as_row(), indent=2))
results_path = PATHS.reports / "results_kkbox_nogap.json"
results_path.write_text(json.dumps(nogap_result.as_row(), indent=2))
results_path = PATHS.reports / "results_kkbox_temporal.json"
results_path.write_text(json.dumps(corrected_pooled.as_row(), indent=2))

headline_table = pd.DataFrame(
    [
        {"dataset": "Online Retail II", **retail_naive},
        {"dataset": "Online Retail II", **retail_temporal},
        {"dataset": "KKBox", **naive_result.as_row()},
        {"dataset": "KKBox", **nogap_result.as_row()},
        {"dataset": "KKBox", **corrected_pooled.as_row()},
    ]
).set_index(["dataset", "label"])
headline_table[["n", "prevalence", "roc_auc", "pr_auc", "brier"]]

# %% jupyter={"source_hidden": true}
kkbox_gap = naive_result.roc_auc - corrected_pooled.roc_auc
retail_gap = retail_naive["roc_auc"] - retail_temporal["roc_auc"]
pd.Series(
    {"KKBox naive-vs-corrected ROC AUC gap": kkbox_gap, "Online Retail II ROC AUC gap": retail_gap}
)

# %% [markdown]
# One panel per metric, grouped bars per dataset, so Online Retail II and
# KKBox stay structurally comparable (naive vs. corrected, two bars each)
# rather than KKBox getting a third bar for its no-gap ablation. That result
# is marked as a tick on KKBox's corrected bar instead, at the height it
# actually scored, because as the section above explains it is not a third
# condition in the same series. Each dataset's prevalence is on its x-tick,
# because Brier is not comparable across different base rates.

# %% jupyter={"source_hidden": true}
headline_metrics = [
    ("roc_auc", "ROC AUC (higher is better)"),
    ("pr_auc", "PR AUC (higher is better)"),
    ("brier", "Brier (lower is better)"),
]
bar_width = 0.35
group_x = [0, 1]

fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
for ax, (col, label) in zip(axes, headline_metrics, strict=True):
    naive_vals = [retail_naive[col], getattr(naive_result, col)]
    corrected_vals = [retail_temporal[col], getattr(corrected_pooled, col)]
    naive_bars = ax.bar(
        [x - bar_width / 2 for x in group_x],
        naive_vals,
        width=bar_width,
        color=PALETTE["both"],
        label="naive",
    )
    corrected_bars = ax.bar(
        [x + bar_width / 2 for x in group_x],
        corrected_vals,
        width=bar_width,
        color=PALETTE["test"],
        label="corrected",
    )
    for bar, v in zip([*naive_bars, *corrected_bars], [*naive_vals, *corrected_vals], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8
        )

    # KKBox no-gap: a tick on KKBox's corrected bar, not a third bar.
    kkbox_corrected_bar = corrected_bars[1]
    nogap_v = getattr(nogap_result, col)
    ax.plot(
        [
            kkbox_corrected_bar.get_x(),
            kkbox_corrected_bar.get_x() + kkbox_corrected_bar.get_width(),
        ],
        [nogap_v, nogap_v],
        color=PALETTE["emphasis"],
        lw=2.5,
        solid_capstyle="butt",
        zorder=3,
    )

    ax.set_xticks(group_x)
    ax.set_xticklabels(
        [
            f"Online Retail II\n(naive {retail_naive['prevalence']:.0%}, "
            f"corrected {retail_temporal['prevalence']:.0%})",
            f"KKBox\n(naive {naive_result.prevalence:.0%}, corrected {corrected_pooled.prevalence:.0%})",
        ],
        fontsize=8,
    )
    ax.set_ylabel(label)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].legend(frameon=False, fontsize=9, loc="lower left")
fig.suptitle(
    "The random-split lie replicates in contractual churn, at a fraction of the size", fontsize=11
)
fig.tight_layout(rect=[0, 0.07, 1, 0.95])
fig.text(
    0.5,
    0.015,
    "Prevalence differs by dataset and split, so Brier (panel 3) is not directly comparable across groups. "
    "The blue tick on KKBox's corrected bar marks the no-gap ablation, which scores a differently-labelled panel.",
    ha="center",
    fontsize=8.5,
    color=PALETTE["reference_line"],
)
fig.savefig(PATHS.figures / "kkbox_vs_retail_gap.png", dpi=200)
# fig

# %% [markdown]
# The failure replicates, but at a fraction of Online Retail II's size:
# KKBox's naive-vs-corrected gap is 0.027 ROC AUC against Online Retail II's
# 0.138. Both are computed above. The most direct reading is a ceiling
# effect. KKBox's corrected model already scores 0.907, close to what a
# metric bounded at 1.0 allows, mostly because `days_until_expiry` is nearly
# a structural readout of the label. There is not much headroom left for a
# leaky version of the same features to inflate into. Online Retail II's
# honest features are a much weaker proxy for intent at 0.765, leaving far
# more room for the naive mistakes to look like skill.
#
# The lesson is not "the gap is smaller for contractual churn in general".
# It is that the size of the lie scales with how much genuine signal the
# honest model already has, which is a property of the features and the
# problem rather than of contractual versus non-contractual as a category.
#
# The no-gap arm scores higher than either, but as the label-shift table
# showed it is scoring a panel with different labels and a different
# prevalence, so it does not slot into that comparison as "the gap alone".
# Read on its own, it says that folding the first seven days back into the
# label makes the problem easier, which is what should be expected when the
# added days contain near-deterministic renewals. It does not measure what
# `04`'s no-gap arm measured. Stating that plainly costs this notebook a
# tidy cross-dataset claim about gaps, and the claim was not supportable.

# %% [markdown]
# ## An informal cross-check against KKBox's own labels
#
# `train.csv` carries the Kaggle competition's `is_churn` label for a
# specific scoring cutoff that does not line up with these rolling origins.
# This is a plausibility check, not a validation: different label definition
# (ADR-0013), different population, one cutoff against 20 origins.

# %% jupyter={"source_hidden": true}
official_labels = pd.read_csv(PATHS.raw / "kkbox" / "train.csv").rename(
    columns={"msno": "customer_id"}
)
last_origin_customers = panel.loc[panel["as_of"] == origins[-1].as_of, ["customer_id", "churned"]]
joined = last_origin_customers.merge(official_labels, on="customer_id", how="inner")
agreement = (joined["churned"] == joined["is_churn"]).mean()
print(
    f"{len(joined):,} customers in both this notebook's last origin and train.csv; "
    f"{agreement:.1%} agreement between this project's label and KKBox's official one"
)

# %% [markdown]
# ## Calibration
#
# `churnval.calibration` reused unchanged, with the same three-way temporal
# split `05` established (ADR-0011): the classifier trains through a mature
# origin, a later origin fits the calibrator, a later-still origin evaluates
# it, reusing `fit_at_origin` rather than new logic.

# %% jupyter={"source_hidden": true}
calibration_as_of = per_origin_corrected["as_of"].iloc[-2]
evaluation_as_of = per_origin_corrected["as_of"].iloc[-1]
calibration_index = [w.as_of for w in origins].index(calibration_as_of)

calibration_model, _ = fit_at_origin(
    panel, origins, calibration_index, feature_cols=FEATURE_COLS, seed=SEED
)
calibration_rows = panel.loc[panel["as_of"] == calibration_as_of]
evaluation_rows = panel.loc[panel["as_of"] == evaluation_as_of]

calibration_raw_prob = calibration_model.predict_proba(calibration_rows[FEATURE_COLS])[:, 1]
evaluation_raw_prob = calibration_model.predict_proba(evaluation_rows[FEATURE_COLS])[:, 1]

platt_model = fit_platt(calibration_raw_prob, calibration_rows["churned"].to_numpy())
isotonic_model = fit_isotonic(calibration_raw_prob, calibration_rows["churned"].to_numpy())
evaluation_platt_prob = platt_model.predict_proba(evaluation_raw_prob.reshape(-1, 1))[:, 1]
evaluation_isotonic_prob = isotonic_model.predict(evaluation_raw_prob)

# %% [markdown]
# `05`'s reliability curve, reused unchanged. Each point is one bin of the
# evaluation slice: mean predicted probability against observed churn rate,
# with a perfectly calibrated model on the diagonal.

# %% jupyter={"source_hidden": true}
kkbox_variants = {
    "uncalibrated": evaluation_raw_prob,
    "Platt": evaluation_platt_prob,
    "isotonic": evaluation_isotonic_prob,
}
kkbox_variant_colours = {
    "uncalibrated": PALETTE["neutral"],
    "Platt": PALETTE["train"],
    "isotonic": PALETTE["test"],
}
fig, ax = plt.subplots(figsize=(7, 6.5))
ax.plot([0, 1], [0, 1], color=PALETTE["reference_line"], lw=1, ls="--", label="perfect calibration")
for name, prob in kkbox_variants.items():
    table = reliability_table(evaluation_rows["churned"].to_numpy(), prob, n_bins=10)
    ax.plot(
        table["mean_predicted"],
        table["observed_rate"],
        marker="o",
        color=kkbox_variant_colours[name],
        label=name,
    )
ax.set_xlabel("mean predicted probability (evaluation slice, per bin)")
ax.set_ylabel("observed churn rate (evaluation slice, per bin)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title(
    "KKBox: the uncalibrated model already sits close to the diagonal",
    loc="left",
    fontsize=10,
)
ax.legend(frameon=False, fontsize=9, loc="upper left")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(PATHS.figures / "kkbox_reliability_curves.png", dpi=200)
# fig

# %% jupyter={"source_hidden": true}
kkbox_decomposition = pd.DataFrame(
    [
        {
            "variant": name,
            **brier_decomposition(evaluation_rows["churned"].to_numpy(), prob),
        }
        for name, prob in [
            ("uncalibrated", evaluation_raw_prob),
            ("Platt", evaluation_platt_prob),
            ("isotonic", evaluation_isotonic_prob),
        ]
    ]
).set_index("variant")
kkbox_decomposition

# %% jupyter={"source_hidden": true}
kkbox_best_calibration = kkbox_decomposition["reliability"].idxmin()
print(
    f"lowest reliability term: {kkbox_best_calibration} "
    f"({kkbox_decomposition.loc[kkbox_best_calibration, 'reliability']:.5f} vs. "
    f"{kkbox_decomposition.drop(kkbox_best_calibration)['reliability'].min():.5f} "
    "for the next-best variant)"
)

# %% [markdown]
# The winner here is the *uncalibrated* model, not either calibrator, the
# opposite of `05`'s Online Retail II result where calibrating was a
# several-fold improvement. That is consistent with the ceiling-effect
# reading above: a model built mostly around `days_until_expiry` produces
# raw probabilities already close to calibrated at this origin, so Platt and
# isotonic have little to correct and each adds a little noise instead. Both
# calibrators are fitted on a separate, later slice than the classifier
# (ADR-0011's discipline, reused unchanged), so this is not recovered
# training-set overfitting. Read from the table, not asserted in advance,
# the same way `05` reported the opposite result.

# %% [markdown]
# ## The hazard model
#
# `churnval.hazard` reused with one difference from `06`: `time_col` is
# passed explicitly (ADR-0014). KKBox also needs one thing Online Retail
# II's cleaning handled implicitly. `01`'s cleaning dropped cancelled
# invoices outright, so any remaining transaction was safely "the event".
# KKBox's cleaning keeps `is_cancel = 1` rows on purpose, since they are a
# real feature. Passing the full transaction log to
# `build_person_period_panel` unfiltered would let a cancellation-only
# transaction count as the event, exactly backwards from
# `build_kkbox_asof_panel`'s own label rule, which treats a cancellation
# inside the label window as churn rather than a renewal. Filtering to
# renewals first is what keeps the hazard model's notion of "event"
# consistent with the binary label it is compared against.
#
# One more parameter needs a KKBox-specific value. `PERIOD_DAYS` defaults to
# 30, chosen in ADR-0012 for Online Retail II's 90-day horizon. KKBox's
# horizon is 30 days, so the same default would give exactly *one* period,
# collapsing the hazard model back into the binary classifier. ADR-0012
# anticipated this: `period_days=10` gives three periods, the same count
# Online Retail II used, scaled to this dataset's horizon rather than
# reusing an absolute day count.
#
# `build_person_period_panel` expands via a per-origin, per-customer Python
# loop, the same shape ADR-0015 measured and rejected for the panel build at
# this scale. It was never revisited, which an adversarial review caught:
# this cell's runtime went undisclosed while the two comparably sized steps
# around it were both measured. It is timed below rather than rewritten,
# since rewriting means changing a function two notebooks depend on for its
# exact current behaviour.

# %% jupyter={"source_hidden": true}
start = time.perf_counter()
renewal_transactions = transactions.loc[transactions["is_cancel"] == 0]
person_period_panel = build_person_period_panel(
    renewal_transactions,
    origins,
    panel,
    feature_cols=FEATURE_COLS,
    time_col="transaction_date",
    period_days=10,
)
elapsed = time.perf_counter() - start
print(
    f"{len(person_period_panel):,} person-period rows "
    f"({len(person_period_panel) / len(panel):.2f}x the original panel), "
    f"{person_period_panel['event'].mean():.1%} of rows are an event "
    f"-- built in {elapsed:.0f}s"
)

# %% jupyter={"source_hidden": true}
per_origin_hazard, predictions_hazard = hazard_rolling_origin_backtest(
    person_period_panel,
    panel,
    origins,
    feature_cols=FEATURE_COLS,
    min_training_origins=3,
    seed=SEED,
    period_days=10,
)
assert list(per_origin_hazard["as_of"]) == list(per_origin_corrected["as_of"]), (
    "hazard and binary backtests tested different origins"
)
hazard_pooled = score(
    predictions_hazard["y_true"], predictions_hazard["y_prob"], label="KKBox hazard"
)
pd.DataFrame([corrected_pooled.as_row(), hazard_pooled.as_row()]).set_index("label")[
    ["n", "roc_auc", "pr_auc", "brier"]
]

# %% [markdown]
# Same convention as the naive-vs-corrected chart: value labels, the gap
# between the two bars, and Brier's reversed direction in the y-axis label.

# %% jupyter={"source_hidden": true}
hazard_vs_binary_metrics = [
    ("roc_auc", "ROC AUC (higher is better)"),
    ("pr_auc", "PR AUC (higher is better)"),
    ("brier", "Brier (lower is better)"),
]
fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
for ax, (col, label) in zip(axes, hazard_vs_binary_metrics, strict=True):
    binary_v = getattr(corrected_pooled, col)
    hazard_v = getattr(hazard_pooled, col)
    bars = ax.bar(
        ["binary", "hazard"], [binary_v, hazard_v], color=[PALETTE["test"], PALETTE["train"]]
    )
    for bar, v in zip(bars, [binary_v, hazard_v], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9
        )
    ax.set_ylabel(label)
    ax.set_title(f"gap = {hazard_v - binary_v:+.4f}", loc="left", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.suptitle(
    "The hazard model matches the binary classifier almost exactly on aggregate metrics",
    fontsize=10.5,
)
fig.tight_layout()
fig.savefig(PATHS.figures / "kkbox_hazard_vs_binary_metrics.png", dpi=200)
# fig

# %% [markdown]
# ### Where the risk is, for a contractual product
#
# The same period-by-period view `06` built, reused via
# `fit_hazard_at_origin` and `hazard_curve` unchanged, at the same
# evaluation origin the calibration section used.

# %% jupyter={"source_hidden": true}
evaluation_index = [w.as_of for w in origins].index(evaluation_as_of)
hazard_model, _ = fit_hazard_at_origin(
    person_period_panel,
    origins,
    evaluation_index,
    feature_cols=FEATURE_COLS,
    seed=SEED,
)
n_periods_kkbox = 3
curve = hazard_curve(
    hazard_model, evaluation_rows.reset_index(drop=True), FEATURE_COLS, n_periods_kkbox
)
curve = curve.sort_values(["customer_id", "period"])
curve["survival"] = curve.groupby("customer_id")["hazard"].transform(lambda h: (1 - h).cumprod())
kkbox_period_summary = curve.groupby("period").agg(
    mean_hazard=("hazard", "mean"), mean_survival=("survival", "mean")
)
kkbox_period_summary

# %% [markdown]
# `mean_survival` above is averaged the only correct way: each customer's
# own survival curve is computed first, period by period, and *then*
# averaged across customers. Averaging hazards first and accumulating
# afterwards answers a different question, because survival is a
# multiplicative function of a customer's own hazard sequence and that
# nonlinearity does not commute with averaging across customers.
# `cumulative_hazard` is derived from this same correctly averaged
# `mean_survival` via `-log(S)`, so the two panels below agree with each
# other exactly rather than being independently approximated curves that
# could drift apart.

# %% jupyter={"source_hidden": true}
kkbox_period_summary["cumulative_hazard"] = -np.log(kkbox_period_summary["mean_survival"])
kkbox_period_summary

# %% jupyter={"source_hidden": true}
period_days_kkbox = 10
day_boundaries = [7 + p * period_days_kkbox for p in range(n_periods_kkbox + 1)]  # [7, 17, 27, 37]
hazard_step = [*kkbox_period_summary["mean_hazard"], kkbox_period_summary["mean_hazard"].iloc[-1]]
cumhaz_step = [0.0, *kkbox_period_summary["cumulative_hazard"]]
survival_step = [1.0, *kkbox_period_summary["mean_survival"]]

fig, axes = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True)
axes[0].step(day_boundaries, hazard_step, where="post", marker="o", color=PALETTE["train"])
axes[0].set_ylabel("hazard\n(per 10-day period)")
axes[0].set_title(
    "KKBox's risk arrives at the end of the window, not the start --\n"
    "the opposite of Online Retail II's near-flat hazard",
    loc="left",
    fontsize=10,
)

axes[1].step(day_boundaries, cumhaz_step, where="post", marker="o", color=PALETTE["train"])
axes[1].set_ylabel("cumulative hazard\n(-log(survival))")

axes[2].step(day_boundaries, survival_step, where="post", marker="o", color=PALETTE["train"])
axes[2].set_ylim(0, 1.02)
axes[2].set_ylabel("survival\nprobability")
axes[2].set_xlabel("days after as_of (10-day period resolution -- flat within each period)")
axes[2].set_xticks(day_boundaries)

for ax in axes:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(PATHS.figures / "kkbox_hazard_curve.png", dpi=200)
# fig

# %% [markdown]
# `kkbox_period_summary` puts numbers on the shape: mean predicted hazard is
# low in days 7-17 and 17-27 after `as_of`, then jumps sharply in the final
# period, days 27-37. Almost the entire window's risk lands in its last ten
# days. Unlike Online Retail II's near-flat hazard, this is a real
# concentration rather than a risk-set artefact: the hazard itself, which is
# already conditional on still being at risk, is what jumps.
#
# The mechanism is the contract. Online Retail II's window opens
# `GAP_DAYS` after the customer's most recent purchase, and nothing
# schedules the next one. KKBox's `days_until_expiry` ties each customer to
# a specific renewal deadline, and for most eligible customers, given the
# dominant 30-day plan and an eligibility window centred on `as_of`, that
# deadline falls late in the window rather than early. Reused machinery,
# genuinely different curve. The shape is a property of the two datasets'
# label mechanics, not of the hazard code.

# %% [markdown]
# ---
# ## Leakage audit
#
# Covers this notebook's genuinely new surface: KKBox eligibility, feature
# and label construction. Everything downstream of the panel is reused
# unchanged and was audited in `03`-`06`.
#
# | # | Component | Leak class | Evidence | Severity | Fix |
# |---|---|---|---|---|---|
# | 1 | `rolling_origin_backtest`, `evaluation`, `calibration`, `hazard` (reused) | Inherited; audited in `03`-`06` | Called unchanged (`hazard` with one added `time_col` parameter, ADR-0014); no KKBox-specific logic lives in any of them | n/a | n/a |
# | 2 | `_current_membership_state`'s ASOF JOIN uses strictly-before-`as_of` transactions | Clean, verified by reading the SQL | `ASOF JOIN transactions t ON t.customer_id = oc.customer_id AND t.transaction_date < oc.as_of`, a strict `<` matching every other as-of boundary in this project | n/a | n/a |
# | 3 | `_lookback_aggregates` window is strictly before `as_of` | Clean | `t.transaction_date >= e.as_of - INTERVAL ... AND t.transaction_date < e.as_of` | n/a | n/a |
# | 4 | Label treats a cancellation inside the label window as churn, not a renewal | Clean, regression tested | `tests/test_kkbox_features.py::test_label_treats_cancellation_only_as_churned_not_a_renewal` | n/a | n/a |
# | 5 | Hazard model's "event" definition vs. the binary label's | Found while building this notebook, fixed | An unfiltered transaction table passed to `build_person_period_panel` would let a cancellation-only transaction count as a return, exactly backwards from the binary label's rule. `03`-`06`'s cleaning already dropped cancellations, so this mismatch had no way to surface there | High; would have made the hazard-vs-binary comparison internally inconsistent | Fixed via `renewal_transactions`; regression-tested in `tests/test_kkbox_hazard_consistency.py` |
# | 6 | `train.csv` cross-check isolation | Clean | Computed after every model and backtest is built; nothing from the official labels feeds back into the panel, any backtest, or any fitted model | n/a | n/a |
# | 7 | ASOF JOIN tie-break on same-date rows | Found under adversarial review, fixed | Originally rated Low ("not expected to matter much"). Independently reproduced: two in-process calls to `build_kkbox_asof_panel` on identical cached data picked different `(as_of, customer_id)` populations for about 0.8% of the panel, because 255,595 same-date transaction groups exist and 94.6% of them disagree on `membership_expire_date`. That changes *eligibility and labels*, not merely a feature value, so the original rating was wrong on both counts | High (reclassified from Low); the headline numbers were not reproducible run to run before the fix | `_dedupe_for_asof_join`, a deterministic largest-expiry-wins tie-break applied before the join; regression-tested; recorded in ADR-0015's addendum |
# | 8 | Eligibility window's upper bound tied to whatever `gap_days` an ablation varied | Found while building this notebook, fixed | The no-gap ablation's first run showed 79% churn prevalence against about 48% for every other arm, because `gap_days=0` also narrowed the eligibility window's upper bound, excluding customers whose membership had not lapsed yet and skewing the population toward already-lapsed customers. Not a label leak, a population-comparability confound | High; would have made the no-gap number measure "gap plus population shift" | `eligibility_gap_days` parameter, pinned to 7 here; regression-tested |
# | 9 | `n_transactions_lookback` and `n_cancellations_lookback` can be `NaN` at scoring time | Found under adversarial review, fixed | `_lookback_aggregates` inner-joins over a bounded window, so a customer with zero transactions in the last `FEATURE_LOOKBACK_DAYS` does not appear and is left `NaN` after the left merge. Affected about 4.0% of panel rows, undisclosed and untested before that review; LightGBM's native missing-value handling absorbed it silently | Medium; a real, previously undisclosed maturity gap over a non-trivial share of the population | Explicit `fillna(0)` with rationale (zero transactions observed is a real signal, not missing data); regression-tested |
# | 10 | The gap changes label membership on contractual data | Found in this revision, disclosed rather than fixed | Because eligible customers are near a renewal decision by construction, a renewal inside the 7-day gap pushes their next decision to the far edge of the label window. The `gap_effect` cell above measures how much of the panel this touches. Not a leak: the label is temporally sound either way. But it means the `gap_days=0` ablation changes *who counts as a churner*, so its score is not comparable to `04`'s no-gap arm, where the gap moved only a boundary | Medium; affects interpretation of the no-gap arm, not the correctness of the naive-vs-corrected headline | Disclosed above and in the headline discussion; the earlier draft's cross-dataset claim about gap contribution is withdrawn |
# | 11 | KKBox results still not bit-reproducible across runs | Found in this revision, open | Row 7 fixed the nondeterministic ASOF tie-break, but two full runs of this notebook on identical cached inputs still disagree in the 3rd-4th decimal: corrected ROC AUC 0.907547 then 0.906982, no-gap 0.934496 then 0.933421, corrected Brier 0.121644 then 0.122609. Online Retail II's numbers over the same two runs reproduce to 1e-9 (`04` asserts this), so the cause is scale-dependent, not a logic error. The leading suspect is LightGBM's runtime auto-choice between row-wise and column-wise histogram construction, which is decided by a timing benchmark at fit time and therefore varies with machine load; it would show up only on data large enough to trigger the choice, which is exactly the pattern here. Not confirmed — the diagnostic is to pin `force_row_wise=True` (or `deterministic=True`) and re-run | Medium; no headline claim in this notebook depends on the 3rd decimal, and none of the qualitative findings move, but "the same code on the same data gives the same answer" does not currently hold for this dataset | Not fixed here: pinning LightGBM's histogram mode changes the model configuration ADR-0008 fixes across every notebook, so it needs its own ADR and a full re-run of `01`-`07`, not a quiet parameter edit inside the last notebook |
#
# Six real defects found in the course of this notebook's work (rows 5, 7,
# 8, 9, 10, 11). Two surfaced while building it, by writing the hazard
# section's prose and by running the naive/no-gap comparison against real
# data. Two more surfaced under a separate adversarial review after the
# notebook first ran clean end to end, which is why that review is its own
# pass: a notebook running without errors and producing plausible-looking
# numbers is not the same claim as its numbers being correct or
# reproducible. The last two surfaced on a later reading pass over the prose
# itself, one of them only because that pass re-ran the notebook and
# compared the output against the numbers the previous run had written.

# %% [markdown]
# ---
# ## Closing
#
# **Finding:** the random-split lie replicates on contractual churn, at 25x
# the row count, using a genuinely different eligibility rule, feature set
# and label: 0.934 ROC AUC naive against 0.907 corrected, a gap of 0.027,
# next to Online Retail II's 0.138. The gap is real but far smaller, and the
# reason is measured rather than assumed. KKBox's corrected model already
# sits close to ROC AUC's ceiling, mostly on the strength of
# `days_until_expiry`, leaving little room for a leaky version of the same
# features to inflate into. The size of the lie scales with how little
# genuine signal the honest model has, not with the kind of churn.
#
# The gap ablation does not transfer. On this dataset `gap_days=0` changes
# which customers count as churners, not just where the label window's
# boundary sits, because eligible customers are near a renewal decision by
# construction and many of them renew inside the gap. Its score is a
# legitimate number for a different labelling of the problem, and the
# earlier draft's claim that it showed the gap contributing more here than
# on Online Retail II is withdrawn. Nothing in this project measures gap
# contribution on both datasets in a comparable way.
#
# Two results did not have to point the same way as `05` and `06`, and one
# did not. Calibration here is won by the *uncalibrated* model: Platt and
# isotonic both make reliability slightly worse, the opposite of `05`'s
# several-fold improvement, again consistent with the ceiling effect. The
# hazard model matches the binary classifier's aggregate ranking almost
# exactly, as in `06`, but where it concentrates risk is close to a mirror
# image. KKBox's risk lands overwhelmingly in the window's final ten days,
# and that is a genuine concentration in the hazard itself, where Online
# Retail II's hazard was near-flat and only its survival curve looked
# front-loaded. Reused code, opposite shape, driven by what each label is
# built from.
#
# These numbers are from a run that includes the two fixes an adversarial
# review found: a nondeterministic tie-break that changed the eligible
# population and labels across runs, and a silent `NaN` in two lookback
# features for about 4% of the panel. The headline shape survived both, but
# several exact figures moved by a few thousandths to a percentage point.
#
# One caveat on those figures, found by re-running this notebook and
# diffing the output against what the previous run had written: KKBox's
# numbers are still not bit-reproducible, moving by up to about 0.001
# between runs on identical inputs, where Online Retail II's reproduce
# exactly. Audit row 11 has the evidence and the leading suspect. Nothing
# qualitative in this notebook turns on the third decimal, but quote these
# figures to three places rather than six, and treat the KKBox-vs-retail
# comparison as a comparison of sizes rather than of exact values.
#
# **Next:** this closes the notebook sequence. `docs/checklist.md` distills
# what `00`-`07` found into a portable, dataset-agnostic reference.
