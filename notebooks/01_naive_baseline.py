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
# # 01 — Naive baseline
#
# > ## ⚠️ THIS NOTEBOOK IS WRONG ON PURPOSE
# >
# > Everything through "The number" below reproduces three validation
# > mistakes real churn tutorials make, at once. The resulting metric is not
# > a legitimate estimate of anything. It exists so that notebook
# > `02_leakage_diagnosis` has a real number to take apart, and so that
# > notebook `04_the_gap` has something to subtract the honest number from.
# > A fourth, more extreme version closes the notebook separately — see
# > "The version that scores 1.000" near the end.
# >
# > **Do not reuse anything in `src/churnval/naive_baseline.py` outside this
# > notebook.** The correct feature and split logic is built from scratch in
# > `03_temporal_protocol`.
#
# **Question this notebook answers:** what does the conventional approach
# report, if it is built faithfully the way most tutorials build it?
#
# **Consumes:** `data/raw/online_retail_ii/online_retail_ii.xlsx`, fetched by
# `uv run churnval fetch retail`.
#
# **Produces:** the "Random split (the wrong way)" row of the README results
# table, written to `reports/results_naive.json`; a dataset card for Online
# Retail II at `docs/data/online_retail_ii.md`; supporting diagnostic charts
# in `reports/figures/`.
#
# **Runtime:** a few minutes — the dataset is ~1M rows, read once from Excel
# and cached to Parquet.
#
# ---
#
# The label itself is forward-looking and built honestly: at each of several
# scoring occasions (`as_of` dates, from `churnval.windows.rolling_origins`),
# a customer "churns" if they make no purchase during the label window that
# opens after that `as_of`. Getting the label right is not the point of this
# notebook — the three mistakes below are:
#
# 1. **Features ignore `as_of`.** Recency, Frequency and Monetary value are
#    computed once over a customer's *entire* history and reused unchanged
#    at every scoring occasion that customer appears in. A customer scored
#    in December and the same customer scored the following August get
#    identical features.
# 2. **The split ignores entity identity.** Each (customer, as_of) pair is
#    one row. A random 80/20 row split lets the same customer's rows land in
#    both train and test — the model can see how a customer's history plays
#    out at one point in time while being tested on that same customer at
#    another.
# 3. **No gap.** `GAP_DAYS = 0` — features and the label window touch
#    directly, with none of the operational lead time
#    `churnval.config.DEFAULT_GAP_DAYS` represents elsewhere in this repo.
#
# None of this is hidden. It is built the same way a tutorial would build it,
# so that what it costs can be measured honestly in the notebooks that
# follow. See [ADR-0007](../docs/adr/0007-online-retail-ii-naive-panel-definition.md)
# for why this dataset uses a 90-day horizon and a 365-day eligibility
# lookback rather than the project-wide defaults.

# %% [markdown]
# ## Load the data
#
# Online Retail II ships as a two-sheet Excel workbook. `churnval.io` parses
# it once, drops rows that are not completed purchases (cancellations,
# non-positive quantity or price, missing customer ID), and caches the result
# to Parquet — every call after the first is a Parquet read. See
# `docs/data/online_retail_ii.md` for the dataset card.

# %%
import json
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from churnval.asof_preview import asof_recency_frequency_monetary
from churnval.config import DEFAULT_STEP_DAYS, PATHS, SEED
from churnval.evaluation import score
from churnval.io import load_online_retail_ii
from churnval.leakage import assert_no_dominant_single_feature, single_feature_auc
from churnval.naive_baseline import (
    GAP_DAYS,
    HORIZON_DAYS,
    build_naive_panel,
    naive_features,
    tautological_label,
)
from churnval.plotting import PALETTE, set_style
from churnval.windows import rolling_origins

PATHS.ensure()
set_style()
transactions = load_online_retail_ii()
print(
    f"{len(transactions):,} transaction lines, "
    f"{transactions['customer_id'].nunique():,} customers, "
    f"{transactions['invoice_date'].min():%Y-%m-%d} to "
    f"{transactions['invoice_date'].max():%Y-%m-%d}"
)

# %% [markdown]
# ## Build the scoring occasions
#
# `rolling_origins` (from `churnval.windows` — the same module the correct
# protocol in `03_temporal_protocol` uses) generates monthly `as_of` dates.
# The first occasion is pushed a full `ELIGIBILITY_LOOKBACK_DAYS` (365) past
# the start of the data, so every origin has a full lookback window behind
# it; the last is whatever origin's label window still fits inside the data.
# `gap_days=0` is mistake #3 from the banner above, passed explicitly rather
# than left implicit.

# %%
first_as_of = transactions["invoice_date"].min().normalize() + timedelta(days=365)
last_event = transactions["invoice_date"].max()
origins = rolling_origins(
    first_as_of,
    last_event,
    gap_days=GAP_DAYS,
    horizon_days=HORIZON_DAYS,
    step_days=DEFAULT_STEP_DAYS,
)
print(f"{len(origins)} origins, {origins[0].as_of:%Y-%m-%d} to {origins[-1].as_of:%Y-%m-%d}")

# %% [markdown]
# ## Build the naive panel — mistake #1
#
# `build_naive_panel` joins each customer's *entire-history* RFM features
# (`naive_features`, computed once) onto every origin they are eligible for.
# Nothing here recomputes a feature from data strictly before its row's
# `as_of` — that correct behaviour is what `03_temporal_protocol` adds.

# %%
panel = build_naive_panel(transactions, origins)
print(
    f"{len(panel):,} (customer, as_of) rows, {panel['customer_id'].nunique():,} distinct "
    f"customers, {panel['churned'].mean():.1%} churned overall"
)

# %% [raw]
# Churn rate by origin — if the panel were free of the leak, this line would
# still vary origin to origin (real seasonality), but the *features* joined
# onto each origin would not: a customer's recency, frequency and monetary
# value are frozen at the dataset's own end regardless of which origin's row
# they are attached to.

# %%
by_origin = panel.groupby("as_of")["churned"].agg(n="size", churn_rate="mean")
by_origin

# %% [markdown]
# ## Naive features vs. computing them strictly before `as_of`
#
# `naive_features` ignores `as_of` entirely — that's mistake #1 in words.
# What does it cost in the actual numbers? `asof_recency_frequency_monetary`
# (`churnval.asof_preview`) recomputes recency for the same (customer,
# as_of) rows using only transactions strictly before each row's own
# `as_of` — a lightweight preview of "as-of-safe," not the corrected
# pipeline itself (`03_temporal_protocol` builds and tests that from
# scratch, and may reasonably differ from this).

# %%
asof_frames = []
for window in origins:
    asof_feats = asof_recency_frequency_monetary(transactions, window.as_of)
    row_customers = panel.loc[panel["as_of"] == window.as_of, "customer_id"]
    sub = asof_feats.reindex(row_customers).reset_index().rename(columns={"index": "customer_id"})
    sub["as_of"] = window.as_of
    asof_frames.append(sub)
asof_panel = pd.concat(asof_frames, ignore_index=True)

comparison = panel.merge(asof_panel, on=["customer_id", "as_of"], suffixes=("_naive", "_asof"))
comparison["recency_delta"] = comparison["recency_days_naive"] - comparison["recency_days_asof"]
comparison.groupby("churned")["recency_delta"].median().rename(
    "median naive-minus-asof recency (days)"
)

# %% [markdown]
# For customers who go on to purchase again (`churned=0`), the naive feature
# understates staleness slightly — it's measured to the dataset's own end,
# which is usually later than the row's `as_of`. For customers who don't
# (`churned=1`), it *overstates* staleness by months: the naive value
# reflects how long it's been since their last purchase as of December 2011,
# not as of this row's own scoring date, which can make an otherwise-recent
# lapse look far more stale than it was at the time.

# %%
sample = comparison.sample(n=min(3000, len(comparison)), random_state=SEED)
fig, ax = plt.subplots(figsize=(8, 6.5))
for label, colour in ((0, PALETTE["active"]), (1, PALETTE["churned"])):
    sub = sample[sample["churned"] == label]
    ax.scatter(
        sub["recency_days_asof"],
        sub["recency_days_naive"],
        s=8,
        alpha=0.35,
        color=colour,
        label=f"churned={label}",
    )
lim = max(comparison["recency_days_asof"].max(), comparison["recency_days_naive"].max())
ax.plot(
    [0, lim],
    [0, lim],
    color=PALETTE["reference_line"],
    lw=1,
    ls="--",
    label="naive = as-of-correct",
)
ax.set_xlabel("recency_days, computed strictly before as_of")
ax.set_ylabel("recency_days, naive_features (full history, ignores as_of)")
ax.set_title(
    "Naive recency overstates staleness for churned customers,\nunderstates it for active ones",
    loc="left",
)
ax.legend(frameon=False, markerscale=3, fontsize=8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(PATHS.figures / "naive_vs_asof_recency.png", dpi=200)
# fig

# %% [markdown]
# ## Random split across panel rows — mistake #2
#
# The split is row-level and ignores `customer_id` entirely — stratified
# only on the label, to keep class balance stable. The count below is the
# leak made visible: most customers who land in the test set were *already
# seen*, at a different `as_of`, during training.

# %%
X = panel[["recency_days", "frequency", "monetary"]]
y = panel["churned"]

X_train, X_test, y_train, y_test, cust_train, cust_test = train_test_split(
    X, y, panel["customer_id"], test_size=0.2, random_state=SEED, stratify=y
)
overlap = set(cust_train) & set(cust_test)
print(
    f"{cust_test.nunique():,} distinct customers in the test set; "
    f"{len(overlap):,} of them ({len(overlap) / cust_test.nunique():.1%}) also appear in train"
)

# %%
train_only = len(set(cust_train) - set(cust_test))
test_only = len(set(cust_test) - set(cust_train))

fig, ax = plt.subplots(figsize=(8, 3.2))
categories = ["train only", "test only", "seen in both"]
counts = [train_only, test_only, len(overlap)]
colors = [PALETTE["train"], PALETTE["test"], PALETTE["both"]]
ax.barh(categories, counts, color=colors)
for i, c in enumerate(counts):
    ax.text(c, i, f" {c:,}", va="center", fontsize=9)
ax.set_xlabel("distinct customers")
ax.set_title(
    f"{len(overlap) / cust_test.nunique():.0%} of test customers were already seen\n"
    "at another as_of, during training",
    loc="left",
)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(PATHS.figures / "naive_customer_overlap.png", dpi=200)
# fig

# %% [markdown]
# ### Even the *rows* that land on different sides are the same people
#
# The chart above counts customers. This one plots individual panel rows —
# one point per (customer, as_of) — against time, for a sample of customers
# grouped by whether their rows ended up in train only, test only, or both.
# For "seen in both" customers, train- and test-coloured points sit right
# next to each other on the same customer's timeline: the split intermingles
# them, it doesn't separate them.

# %%
row_split = pd.Series("test", index=panel.index)
row_split.loc[X_train.index] = "train"

cust_sides = panel.assign(side=row_split).groupby("customer_id")["side"].agg(set)
cust_category = cust_sides.map(
    lambda sides: "seen in both" if len(sides) == 2 else next(iter(sides)) + " only"
)

per_category = 40
sampled_customers = pd.concat(
    [
        cust_category[cust_category == cat].sample(
            n=min(per_category, (cust_category == cat).sum()), random_state=SEED
        )
        for cat in ("seen in both", "train only", "test only")
    ]
)
y_position = pd.Series(range(len(sampled_customers)), index=sampled_customers.index)

plot_rows = panel[panel["customer_id"].isin(sampled_customers.index)].copy()
plot_rows["side"] = row_split.loc[plot_rows.index].to_numpy()
plot_rows["y"] = plot_rows["customer_id"].map(y_position)

# Every row at one origin shares the exact same as_of date, so an unjittered
# scatter draws a solid vertical line per origin -- jitter the x-position by
# a few days (well inside the 30-day origin spacing) so overlapping points
# separate visually.
rng = np.random.default_rng(SEED)
plot_rows["as_of_jittered"] = plot_rows["as_of"] + pd.to_timedelta(
    rng.uniform(-6, 6, size=len(plot_rows)), unit="D"
)

fig, ax = plt.subplots(figsize=(8, 6))
for side, colour in (("train", PALETTE["train"]), ("test", PALETTE["test"])):
    sub = plot_rows[plot_rows["side"] == side]
    ax.scatter(sub["as_of_jittered"], sub["y"], s=16, color=colour, label=side, alpha=0.85)

boundary = 0
for cat in ("seen in both", "train only", "test only"):
    n_cat = int((sampled_customers == cat).sum())
    if n_cat == 0:
        continue
    ax.axhline(boundary - 0.5, color=PALETTE["reference_line"], lw=0.6)
    ax.text(
        plot_rows["as_of"].min(), boundary + n_cat / 2 - 0.5, f"  {cat}", va="center", fontsize=8
    )
    boundary += n_cat

ax.set_yticks([])
ax.set_xlabel("as_of")
ax.set_ylabel(f"{len(sampled_customers)} sampled customers, grouped by split category")
ax.set_title(
    '"Seen in both" customers have train- and test-coloured rows mixed across time',
    loc="left",
)
ax.legend(frameon=False, fontsize=8)
for side_spine in ("top", "right"):
    ax.spines[side_spine].set_visible(False)
fig.tight_layout()
fig.savefig(PATHS.figures / "naive_split_over_time.png", dpi=200)
# fig

# %% [markdown]
# ## Fit and score
#
# LightGBM, the same model family this repo reuses in `03_temporal_protocol`
# and `04_the_gap`, so that the difference measured later is attributable to
# the validation design and not to a change of model.

# %%
model = LGBMClassifier(random_state=SEED, verbosity=-1)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]

result = score(y_test, y_prob, label="Random split (the wrong way)")
result

# %% [markdown]
# ## What that AUC actually looks like
#
# A single ranking number hides shape. The panels below show the same
# `y_test`/`y_prob` as a confusion matrix (threshold 0.5), a ROC curve, and
# a precision-recall curve — the pairing this project's validation standard
# requires: a ranking view and a decision view together, not a ranking
# number alone.

# %%
y_pred = (y_prob >= 0.5).astype(int)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["active", "churned"],
    colorbar=False,
    cmap="Purples",
    ax=axes[0],
)
axes[0].set_title("Confusion matrix (threshold 0.5)", loc="left", fontsize=10)

RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    ax=axes[1],
    curve_kwargs={"color": PALETTE["active"]},
    plot_chance_level=True,
    despine=True,
)
axes[1].set_title(f"ROC curve (AUC {result.roc_auc:.3f})", loc="left", fontsize=10)

PrecisionRecallDisplay.from_predictions(
    y_test,
    y_prob,
    ax=axes[2],
    curve_kwargs={"color": PALETTE["churned"]},
    plot_chance_level=True,
    despine=True,
)
axes[2].set_title(f"Precision-recall curve (AUC {result.pr_auc:.3f})", loc="left", fontsize=10)

for ax in axes:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.suptitle(
    "Random split (the wrong way): every view agrees the model looks strong — that's the inflation",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(PATHS.figures / "naive_model_diagnostics.png", dpi=200)
# fig

# %% [markdown]
# ## Sanity check: no single feature is doing this alone
#
# `01`'s earlier draft anchored the label to the same reference date as
# `recency_days`, making the two identical — a bug, not a demonstration (see
# the closing section below). `assert_no_dominant_single_feature` is a fast
# guard against that happening silently: it fails loudly if any one feature
# alone nearly determines the label. Here it should pass — the inflation in
# this notebook comes from the panel and the split, not from a tautological
# feature.

# %%
assert_no_dominant_single_feature(X, y, threshold=0.99)

# %% [markdown]
# ## How much is each feature carrying?
#
# A drop-one ablation on the same train/test split: refit with each feature
# removed in turn and see how much ROC AUC is lost. If the inflation were
# concentrated in one feature, removing it would collapse the score — as it
# does in the tautological section below, where dropping `recency_days`
# alone would be the whole story.

# %%
ablation = {"all three (baseline)": result.roc_auc}
for dropped in X.columns:
    cols = [c for c in X.columns if c != dropped]
    m = LGBMClassifier(random_state=SEED, verbosity=-1)
    m.fit(X_train[cols], y_train)
    p = m.predict_proba(X_test[cols])[:, 1]
    ablation[f"drop {dropped}"] = roc_auc_score(y_test, p)
pd.Series(ablation).sort_values(ascending=False)

# %% [markdown]
# ## Feature importance
#
# LightGBM's split-count importance for the fitted model. There are only
# three features in this panel, so "top 10" is just "all of them" — but the
# same cell works unchanged if a later notebook adds more.

# %%
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
top_importances = importances.tail(10)

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.barh(top_importances.index, top_importances.to_numpy(), color=PALETTE["active"])
ax.set_xlabel("LightGBM split-count importance")
ax.set_title("No lone dominant feature — all three carry real weight", loc="left")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(PATHS.figures / "naive_feature_importance.png", dpi=200)
# fig

# %% [markdown]
# ## The number
#
# This is the "Random split (the wrong way)" row of the README results
# table. It is written to `reports/results_naive.json` so that
# `04_the_gap` reads it rather than retyping it.

# %%
result_path = PATHS.reports / "results_naive.json"
result_path.write_text(json.dumps(result.as_row(), indent=2))
result.as_row()

# %% [markdown]
# ## A closer look: scoring a month we never trained on
#
# Everything above evaluates on a random slice of the same panel the model
# trained on. `origins[-1]` is the panel's last scoring occasion — the most
# recent 90-day outcome the data can actually confirm. Holding that whole
# month out, training only on the origins before it, and scoring on it is a
# genuine look-forward evaluation — even though the features are still the
# naive, `as_of`-blind ones from mistake #1.

# %%
holdout_as_of = origins[-1].as_of
train_mask = panel["as_of"] < holdout_as_of
holdout_mask = panel["as_of"] == holdout_as_of

holdout_model = LGBMClassifier(random_state=SEED, verbosity=-1)
holdout_model.fit(X[train_mask], y[train_mask])
holdout_prob = holdout_model.predict_proba(X[holdout_mask])[:, 1]
holdout_result = score(
    y[holdout_mask], holdout_prob, label="Look-forward holdout (same naive features)"
)

holdout_customers = panel.loc[holdout_mask, "customer_id"]
already_seen = set(panel.loc[train_mask, "customer_id"]) & set(holdout_customers)
print(
    f"holdout month {holdout_as_of:%Y-%m-%d}: {holdout_mask.sum():,} rows, "
    f"{len(already_seen) / holdout_customers.nunique():.1%} of its customers were already "
    "scored at an earlier origin"
)
holdout_result

# %% [markdown]
# Even a genuine forward holdout degrades relative to the random split
# above — degradation the random split hides completely. The gap stays only
# partial, because the features themselves are still unfixed:
# `recency_days`/`frequency`/`monetary` are still each customer's whole
# history, and most of the holdout month's customers were already scored,
# with those same features, at an earlier origin.

# %%
metric_labels = {
    "roc_auc": "ROC AUC (higher better)",
    "pr_auc": "PR AUC (higher better)",
    "brier": "Brier (lower better)",
}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, metric in zip(axes, metric_labels, strict=True):
    values = [getattr(result, metric), getattr(holdout_result, metric)]
    ax.bar(
        ["random split", "look-forward\nholdout"],
        values,
        color=[PALETTE["train"], PALETTE["test"]],
    )
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(metric_labels[metric], fontsize=10, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.suptitle(
    "Even with the same leaky features, a real forward holdout scores worse than a random split",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(PATHS.figures / "naive_holdout_degradation.png", dpi=200)
# fig

# %% [markdown]
# ---
# ## The version that scores 1.000, and why a perfect score is a bug report
#
# An earlier draft of this notebook defined the label directly from the same
# reference date `recency_days` is measured to —
# `churnval.naive_baseline.tautological_label` still implements that version,
# kept only for this closing demonstration. It is not part of the panel
# above and nothing here feeds `reports/results_naive.json`.

# %%
tautological_frame = naive_features(transactions).join(
    tautological_label(transactions, horizon_days=HORIZON_DAYS), how="inner"
)
X_taut = tautological_frame[["recency_days", "frequency", "monetary"]]
y_taut = tautological_frame["churned"]

X_taut_train, X_taut_test, y_taut_train, y_taut_test = train_test_split(
    X_taut, y_taut, test_size=0.2, random_state=SEED, stratify=y_taut
)
taut_model = LGBMClassifier(random_state=SEED, verbosity=-1)
taut_model.fit(X_taut_train, y_taut_train)
taut_prob = taut_model.predict_proba(X_taut_test)[:, 1]
score(y_taut_test, taut_prob, label="tautology, random split")

# %% [markdown]
# `recency_days > HORIZON_DAYS` and `churned` are the same boolean by
# construction — not correlated, *identical*. `single_feature_auc` shows it
# directly: `recency_days` alone, with no model at all, already scores
# essentially the maximum.

# %%
single_feature_auc(X_taut, y_taut)

# %% [markdown]
# A tempting but wrong explanation is "the random split let the model
# cheat." It didn't — the identity holds for every row regardless of which
# split a row lands in. To prove it, split customers by *acquisition
# cohort* instead: earliest 80% of customers by first-purchase date train
# the model, the most recently acquired 20% test it. This is about as
# strict a temporal split as a single-snapshot panel can have, and the
# result is unchanged.

# %%
first_purchase = transactions.groupby("customer_id")["invoice_date"].min()
cutoff = first_purchase.quantile(0.8)
cohort_frame = tautological_frame.join(first_purchase.rename("first_purchase"))
is_train = cohort_frame["first_purchase"] < cutoff

cohort_model = LGBMClassifier(random_state=SEED, verbosity=-1)
cohort_model.fit(
    cohort_frame.loc[is_train, ["recency_days", "frequency", "monetary"]],
    cohort_frame.loc[is_train, "churned"],
)
cohort_prob = cohort_model.predict_proba(
    cohort_frame.loc[~is_train, ["recency_days", "frequency", "monetary"]]
)[:, 1]
score(cohort_frame.loc[~is_train, "churned"], cohort_prob, label="tautology, cohort split")

# %% [markdown]
# Same identity, same near-1.000 score, under a split with no row-level
# leakage at all. `assert_no_dominant_single_feature` is what should have
# caught this before any model was trained — this is what it reports when
# pointed at the tautological frame:

# %%
try:
    assert_no_dominant_single_feature(X_taut, y_taut, threshold=0.99)
except ValueError as exc:
    print(exc)

# %% [markdown]
# ---
# ## Leakage audit
#
# Run against the main panel (features, split, and label construction) and
# the tautological closing section, following the project's `leakage-audit`
# skill, then independently re-checked by the project's `validation-reviewer`
# agent. One of the reviewer's findings changed what's below: an earlier
# draft of this section asserted several numbers (an adversarial-validation
# AUC, a label-shuffle AUC, a temporal-split-sensitivity AUC) that no code
# cell in this notebook actually produced — a violation of this project's own
# "no results hardcoded into prose" rule. Rather than add three more model
# fits to backfill numbers that belong to `02_leakage_diagnosis` anyway (its
# stated job, per `docs/timeline.md`, is exactly "adversarial-validation AUC;
# ablation results"), those checks are named below without specific figures,
# and left for that notebook to do properly. Everything with a number below
# is computed in this notebook.
#
# The reviewer also caught a real second issue: the causal story originally
# told for the "overlapping label windows" finding was wrong. The cell below
# corrects it.

# %%
pivot = panel.pivot_table(index="customer_id", columns="as_of", values="churned")
origins_sorted = sorted(pivot.columns)
p_churn = y.mean()
independence_baseline = p_churn**2 + (1 - p_churn) ** 2
print(
    f"agreement expected under independence: {independence_baseline:.1%} (base churn rate {p_churn:.1%})"
)

overlap_rows = []
for lag_steps in (1, 2, 3):
    agreements = []
    for i in range(len(origins_sorted) - lag_steps):
        pair = pivot[[origins_sorted[i], origins_sorted[i + lag_steps]]].dropna()
        agreements.append((pair.iloc[:, 0] == pair.iloc[:, 1]).mean())
    overlap_rows.append(
        {
            "origins apart (days)": lag_steps * DEFAULT_STEP_DAYS,
            "label-window overlap (days)": max(HORIZON_DAYS - lag_steps * DEFAULT_STEP_DAYS, 0),
            "same-customer label agreement": pd.Series(agreements).mean(),
        }
    )
pd.DataFrame(overlap_rows)

# %% [markdown]
# Agreement falls as the window overlap shrinks, but even at zero overlap
# (origins a full `HORIZON_DAYS` apart) it stays well above the independence
# baseline. Most of the excess is not the overlapping-window artifact — it's
# that some customers are durably loyal and others durably lapsed, so the
# same customer's label tends to agree with itself across origins regardless
# of window overlap. The overlap adds a smaller, real effect on top of that.
#
# | # | Component | Leak class | Evidence | Severity | Fix |
# |---|---|---|---|---|---|
# | 1 | `naive_features` (recency/frequency/monetary) | Feature-time leakage — full-history aggregation ignores `as_of`, and includes the very transaction that determines the label | The purchase that makes a row `churned=0` is itself counted into that customer's `frequency`/`monetary` and lowers `recency_days`, at *every* origin that customer appears in, not only the one the purchase falls inside. Single-feature AUCs 0.82-0.86 (computed above); the drop-one ablation above shows no single feature is "the whole leak" — even removing the strongest one (`recency_days`) only brings AUC down to ~0.87, well above chance, not down to it | High (intentional — mistake #1) | Compute features from transactions strictly before each row's own `as_of` (`03_temporal_protocol`) |
# | 2 | Row-level random split | Split leakage — entity overlap across train/test at different `as_of` | Customer-overlap chart above | High (intentional — mistake #2) | Group the split by `customer_id` (e.g. `GroupShuffleSplit`), or evaluate per-origin |
# | 3 | `GAP_DAYS = 0` | Split leakage — no operational lead time; `feature_end == label_start` | Passed explicitly to `rolling_origins`, vs. `config.DEFAULT_GAP_DAYS = 7` used elsewhere | Medium (intentional — mistake #3; secondary here since mistake #1 already reads future data regardless of gap) | Use `DEFAULT_GAP_DAYS` in `03_temporal_protocol` |
# | 4 | Same-customer label agreement across origins | **Not previously flagged**, and corrected once during review — see the cell above | Agreement is highest at maximum window overlap and falls as overlap shrinks, but stays well above the independence baseline even at zero overlap — mostly ordinary churn-propensity persistence, not the window-overlap artifact alone | Medium — compounds mistake #2: rows for the same customer are not independent draws, for two separate reasons, not one | A correct rolling-origin backtest needs cluster-robust variance (e.g. block bootstrap by `customer_id`) regardless of `step_days` — spacing origins further apart removes only the smaller, overlap-driven component |
# | 5 | Eligibility rule (`>=1` purchase in 365d before `as_of`) | Checked — **clean** | `eligible_customers` filters strictly `< as_of`, matching the feature-window convention in `churnval.windows`; it does not itself look forward | n/a | n/a — see [ADR-0007](../docs/adr/0007-online-retail-ii-naive-panel-definition.md) for the `>=1` vs. `>=2` purchase question, which is a modelling choice, not a leak |
# | 6 | Label maturity / open-window censoring | Checked — **clean** | `rolling_origins` excludes any origin whose label window would extend past the dataset's last event, so no origin's label is truncated and no censored customer is silently coded as a non-churner | n/a | n/a |
# | 7 | Results file vs. tautological section | Checked — **clean** | `reports/results_naive.json` is written before the tautological section begins; nothing after that point writes to any file, so the perfect score below cannot contaminate the reported number | n/a | n/a |
# | 8 | `io.py` administrative stock codes (`POST`, `M`, `BANK CHARGES`, …) | **Found during review, fixed** | 25 customers had their entire purchase history built from postage/manual/bank-charge/test line items, not real purchases, before `_clean` excluded them — see `docs/data/online_retail_ii.md` | Medium (data-quality, not this notebook's intentional leak) | Fixed in `churnval.io.EXCLUDED_STOCK_CODES`; regression-tested in `tests/test_io.py` |
#
# **Checked, not added as numbered rows because they don't yet have a code
# cell here** — adversarial validation (train rows vs. test rows) and a
# label-shuffle-within-`as_of` test were both explored manually while
# auditing this notebook. Both are legitimate diagnostics, but they are
# `02_leakage_diagnosis`'s stated deliverables, not this notebook's — adding
# ad hoc versions here to report a number would mean either duplicating that
# notebook's work or hardcoding a number no cell produced, which is exactly
# the mistake this section's introduction above was rewritten to stop doing.
#
# **Answering the three questions this audit was asked to settle:**
#
# - **Is there a fourth, unintentional leak?** Yes — row 4, though the
#   mechanism is not quite "overlapping windows" as first claimed; see the
#   correction above.
# - **Does the eligibility rule leak?** No — row 5, clean by construction.
# - **Does reusing `SEED` across this notebook's three model fits matter?**
#   No. The seed only controls internal split-shuffling and tree randomness
#   *within* each independently-fit model — no state crosses between fits,
#   consistent with the project standard of one seed constant used everywhere
#   (`config.SEED`). For the tautological section specifically, the direct
#   `single_feature_auc` identity check (computed above) already shows the
#   ≈1.000 score is a deterministic consequence of
#   `churned == (recency_days > HORIZON_DAYS)`, not a favourable draw — it
#   would reproduce under any seed.

# %% [markdown]
# ---
# ## Closing
#
# **Finding:** built faithfully, the conventional approach reports ROC AUC ≈
# 0.90 (see `reports/results_naive.json` for the full, current bundle). That
# number is not a legitimate estimate of anything — it comes from features
# that read future transactions regardless of `as_of`, a split that lets the
# same customer teach and test the model at different points in time, and no
# gap between scoring and the outcome window. None of the three is subtle on
# its own; a fourth, non-obvious effect compounds the second — rows for the
# same customer are not independent draws for two separate reasons, not one,
# as the leakage audit above shows.
#
# The closing section went further and built a version that scores a literal
# 1.000, on purpose, to make a narrower point survive on its own: a perfect
# offline score is a bug report, not a result, and a temporal split does not
# by itself rule that out — the feature and the label have to be checked for
# being the same fact wearing two names.
#
# **Next:** `02_leakage_diagnosis` takes the ≈0.90 number apart properly —
# the adversarial-validation and ablation diagnostics this notebook
# deliberately left out, built as reusable tools rather than one-off checks,
# plus the leakage table its own opening question asks for. `03_temporal_protocol`
# then rebuilds the panel the honest way: features computed strictly before
# each row's own `as_of`, a split grouped by `customer_id`, and
# `config.DEFAULT_GAP_DAYS` instead of zero — so that `04_the_gap` can report,
# for the first time in this repo, an actual measurement of how much the
# number on this page was lying.
