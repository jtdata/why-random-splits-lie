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
# > mistakes that real churn tutorials make, all at once. The resulting metric
# > is not a legitimate estimate of anything. It exists so that
# > `02_leakage_diagnosis` has a real number to take apart and `04_the_gap`
# > has something to subtract the honest number from. A fourth, more extreme
# > version closes the notebook separately ("The version that scores 1.000").
# >
# > **Do not reuse anything in `src/churnval/naive_baseline.py` outside this
# > notebook.** The correct feature and split logic is built from scratch in
# > `03_temporal_protocol`.
#
# **Question this notebook answers:** What does the conventional approach
# report, if it is built faithfully the way most tutorials build it?
#
# **Consumes:** `data/raw/online_retail_ii/online_retail_ii.xlsx`, fetched by
# `uv run churnval fetch retail`.
#
# **Produces:** The "Random split (the wrong way)" row of the README results
# table, written to `reports/results_naive.json`; a dataset card for Online
# Retail II at `docs/data/online_retail_ii.md`; supporting diagnostic charts
# in `reports/figures/`.
#
# **Runtime:** A few minutes. The dataset is about 1M rows, read once from
# Excel and cached to Parquet.
#
# ---
#
# The label itself is built honestly and looks forward: at each of several
# scoring occasions (`as_of` dates from `churnval.windows.rolling_origins`), a
# customer "churns" if they make no purchase during the label window that
# opens after that `as_of`. The three mistakes are in everything around the
# label:
#
# 1. **Features ignore `as_of`.** Recency, frequency and monetary value are
#    computed once over a customer's *entire* history and reused unchanged
#    at every scoring occasion that customer appears in. A customer scored
#    in December and the same customer scored the following August get
#    identical features.
# 2. **The split ignores entity identity.** Each (customer, `as_of`) pair is
#    one row. A random 80/20 row split lets the same customer's rows land in
#    both train and test, so the model can see how a customer's history plays
#    out at one point in time while being tested on that same customer at
#    another.
# 3. **No gap.** `GAP_DAYS = 0`. Features and the label window touch, with
#    none of the operational lead time that `churnval.config.DEFAULT_GAP_DAYS`
#    represents elsewhere in this repo.
#
# None of this is hidden. It is built the way a tutorial would build it, so
# that the cost can be measured in the notebooks that follow. See
# [ADR-0007](../docs/adr/0007-online-retail-ii-naive-panel-definition.md)
# for why this dataset uses a 90-day horizon and a 365-day eligibility
# lookback rather than the project-wide defaults.

# %% [markdown]
# ## Load the data
#
# Online Retail II ships as a two-sheet Excel workbook. `churnval.io` parses
# it once, drops rows that are not completed purchases (cancellations,
# non-positive quantity or price, missing customer ID), and caches the result
# to Parquet. Every call after the first is a Parquet read. See
# `docs/data/online_retail_ii.md` for the dataset card.

# %% jupyter={"source_hidden": true}
import json
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
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
from churnval.plotting import ORDINAL_BLUE_10, PALETTE, set_style
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
# `rolling_origins` (from `churnval.windows`, the same module the correct
# protocol in `03_temporal_protocol` uses) generates monthly `as_of` dates.
# The first occasion is pushed a full `ELIGIBILITY_LOOKBACK_DAYS` (365) past
# the start of the data, so every origin has a full lookback window behind
# it. The last is whichever origin's label window still fits inside the
# data. `gap_days=0` is mistake #3, passed explicitly rather than left
# implicit.

# %% jupyter={"source_hidden": true}
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
# ## Build the naive panel: mistake #1
#
# `build_naive_panel` joins each customer's *entire-history* RFM features
# (`naive_features`, computed once) onto every origin they are eligible for.
# Nothing here recomputes a feature from data strictly before its row's
# `as_of`. That is what `03_temporal_protocol` adds.

# %% jupyter={"source_hidden": true}
panel = build_naive_panel(transactions, origins)
print(
    f"{len(panel):,} (customer, as_of) rows, {panel['customer_id'].nunique():,} distinct "
    f"customers, {panel['churned'].mean():.1%} churned overall"
)

# %% [markdown]
# Churn rate by origin. The rate varies from origin to origin because of
# real seasonality. The features joined onto each origin do not vary at all:
# a customer's recency, frequency and monetary value are frozen at the
# dataset's own end, whichever origin's row they are attached to.

# %% jupyter={"source_hidden": true}
by_origin = panel.groupby("as_of")["churned"].agg(n="size", churn_rate="mean")
by_origin

# %% [markdown]
# ## Naive features vs. computing them strictly before `as_of`
#
# `naive_features` ignores `as_of`. What does that cost in actual numbers?
# `asof_recency_frequency_monetary` (`churnval.asof_preview`) recomputes
# recency for the same (customer, `as_of`) rows using only transactions
# strictly before each row's own `as_of`. This is a lightweight preview of
# "as-of-safe" for one chart, not the corrected pipeline. `03_temporal_protocol`
# builds and tests that properly.

# %% jupyter={"source_hidden": true}
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
# understates staleness slightly, because it is measured to the dataset's end
# and their later purchases pull it down. For customers who do not
# (`churned=1`), it overstates staleness by months: the naive value reflects
# how long it had been since their last purchase as of December 2011, not as
# of the row's own scoring date. A lapse that was recent at the time looks
# ancient.

# %% jupyter={"source_hidden": true}
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
# ### The same points, coloured by scoring date instead of by label
#
# The chart above pools all ten origins and colours only by `churned`, which
# leaves the diagonal streaks unexplained. Colouring each point by its
# scoring date instead (one shade per origin, earliest lightest) resolves
# every streak into the cohort scored on one date. Marker shape keeps
# `churned` visible without a second colour channel.

# %% jupyter={"source_hidden": true}
as_of_order = sorted(sample["as_of"].unique())
as_of_rank = {origin: i for i, origin in enumerate(as_of_order)}
sample_ranked = sample.assign(as_of_rank=sample["as_of"].map(as_of_rank))

cmap = ListedColormap(ORDINAL_BLUE_10)
norm = BoundaryNorm(range(len(as_of_order) + 1), cmap.N)

fig, ax = plt.subplots(figsize=(8, 6.5))
markers = {0: ("o", "active"), 1: ("X", "churned")}
for label, (marker, _marker_name) in markers.items():
    sub = sample_ranked[sample_ranked["churned"] == label]
    ax.scatter(
        sub["recency_days_asof"],
        sub["recency_days_naive"],
        c=sub["as_of_rank"],
        cmap=cmap,
        norm=norm,
        marker=marker,
        s=16 if marker == "o" else 24,
        alpha=0.6,
        linewidths=0.7 if marker == "X" else 0,
    )
ax.plot([0, lim], [0, lim], color=PALETTE["reference_line"], lw=1, ls="--")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax, ticks=[i + 0.5 for i in range(len(as_of_order))])
cbar.ax.set_yticklabels([origin.strftime("%Y-%m-%d") for origin in as_of_order], fontsize=7)
cbar.set_label("as_of")
cbar.outline.set_visible(False)

shape_legend = [
    Line2D(
        [0],
        [0],
        marker=marker,
        color="none",
        markerfacecolor=PALETTE["reference_line"],
        markeredgecolor=PALETTE["reference_line"],
        markersize=7,
        label=marker_name,
    )
    for marker, marker_name in markers.values()
]
ax.legend(handles=shape_legend, frameon=False, fontsize=8, loc="upper left")
ax.set_xlabel("recency_days, computed strictly before as_of")
ax.set_ylabel("recency_days, naive_features (full history, ignores as_of)")
ax.set_title(
    "Each colour is one scoring date — the streaks above are as_of cohorts, not noise",
    loc="left",
)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(PATHS.figures / "naive_vs_asof_recency_by_origin.png", dpi=200)
# fig

# %% [markdown]
# ## Random split across panel rows: mistake #2
#
# The split is row-level and ignores `customer_id` entirely. It is stratified
# on the label only, to keep class balance stable. The count below makes the
# leak visible: most customers who land in the test set were *already seen*,
# at a different `as_of`, during training.

# %% jupyter={"source_hidden": true}
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

# %% jupyter={"source_hidden": true}
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
# ### The rows on different sides of the split are the same people
#
# The chart above counts customers. This one plots individual panel rows
# (one point per customer per `as_of`) against time, for a sample of
# customers grouped by whether their rows ended up in train only, test only,
# or both. For the "seen in both" group, train- and test-coloured points sit
# next to each other on the same customer's timeline. The split mixes them
# rather than separating them.

# %% jupyter={"source_hidden": true}
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
# LightGBM, the same model family this repo uses in `03_temporal_protocol`
# and `04_the_gap`, so that the difference measured later comes from the
# validation design and not from a change of model.

# %% jupyter={"source_hidden": true}
model = LGBMClassifier(random_state=SEED, verbosity=-1)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]

result = score(y_test, y_prob, label="Random split (the wrong way)")
result

# %% [markdown]
# ## What that AUC looks like
#
# A single ranking number hides shape. The panels below show the same
# `y_test`/`y_prob` as a confusion matrix at a 0.5 threshold, a ROC curve,
# and a precision-recall curve. This repo's validation standard asks for a
# ranking view and a decision view together, never a ranking number alone.

# %% jupyter={"source_hidden": true}
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
# An earlier draft of this notebook anchored the label to the same reference
# date as `recency_days`, which made the two identical. That is a bug, not a
# demonstration, and it is kept as the closing section below.
# `assert_no_dominant_single_feature` is a cheap guard against that class of
# bug: it fails loudly if any one feature alone nearly determines the label.
# Here it should pass. The inflation in this notebook comes from the panel
# and the split, not from a tautological feature.

# %% jupyter={"source_hidden": true}
assert_no_dominant_single_feature(X, y, threshold=0.99)

# %% [markdown]
# ## How much is each feature carrying?
#
# A drop-one ablation on the same train/test split: refit with each feature
# removed in turn and see how much ROC AUC is lost. If the inflation were
# concentrated in one feature, removing it would collapse the score. (It
# does in the tautological section below, where dropping `recency_days`
# alone is the whole story.)

# %% jupyter={"source_hidden": true}
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
# LightGBM's split-count importance for the fitted model. Split count is a
# coarse measure (it counts how often a feature was used, not how much it
# helped), and with three features it says little the ablation above did not
# already say. It is kept because the same cell works unchanged if a later
# notebook adds more features.

# %% jupyter={"source_hidden": true}
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
# table. It is written to `reports/results_naive.json` so that `04_the_gap`
# reads it rather than retyping it.

# %% jupyter={"source_hidden": true}
result_path = PATHS.reports / "results_naive.json"
result_path.write_text(json.dumps(result.as_row(), indent=2))
result.as_row()

# %% [markdown]
# ## A closer look: scoring a month the model never trained on
#
# Everything above evaluates on a random slice of the same panel the model
# trained on. `origins[-1]` is the panel's last scoring occasion, the most
# recent 90-day outcome the data can confirm. Holding that whole month out,
# training only on the origins before it, and scoring on it removes the
# row-split leak (mistake #2) and nothing else. The features are still the
# `as_of`-blind ones from mistake #1, so this is *not* a genuine forward
# evaluation. `naive_features` measures recency to the dataset's last day,
# which for this origin is after the end of its own label window: the
# feature still partially encodes the outcome. `04_the_gap` takes this apart
# properly; the point here is only that the random split hides even the
# partial degradation a time-based holdout exposes.

# %% jupyter={"source_hidden": true}
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
# The holdout score is lower than the random split's, and the random split
# would never have shown that. The drop is only partial because the features
# are still unfixed: `recency_days`, `frequency` and `monetary` are each
# customer's whole history, and most of the holdout month's customers were
# already scored with those same features at an earlier origin.

# %% jupyter={"source_hidden": true}
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
    "Holding out the last month degrades all three metrics, even though the features still "
    "read past it",
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
# reference date that `recency_days` is measured to.
# `churnval.naive_baseline.tautological_label` still implements that version,
# kept only for this closing demonstration. It is not part of the panel
# above and nothing here feeds `reports/results_naive.json`.

# %% jupyter={"source_hidden": true}
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
# construction. `single_feature_auc` shows it directly: `recency_days` alone,
# with no model at all, already scores essentially the maximum.

# %% jupyter={"source_hidden": true}
single_feature_auc(X_taut, y_taut)

# %% [markdown]
# A tempting but wrong explanation is "the random split let the model
# cheat". It did not. The identity holds for every row regardless of which
# side of the split it lands on. To prove it, split customers by
# *acquisition cohort* instead: the earliest 80% of customers by
# first-purchase date train the model, the most recently acquired 20% test
# it. That is about as strict a temporal split as a single-snapshot panel
# allows, and the result is unchanged.

# %% jupyter={"source_hidden": true}
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
# leakage at all. `assert_no_dominant_single_feature` should have caught this
# before any model was trained. This is what it reports when pointed at the
# tautological frame:

# %% jupyter={"source_hidden": true}
try:
    assert_no_dominant_single_feature(X_taut, y_taut, threshold=0.99)
except ValueError as exc:
    print(exc)

# %% [markdown]
# ---
# ## Leakage audit
#
# Run against the main panel (features, split, and label construction) and
# the tautological closing section, following `docs/standards/validation.md`.
# Every number cited below is computed in a visible cell of this notebook.
# The adversarial-validation and label-shuffle checks are deliberately left to
# `02_leakage_diagnosis`, whose job they are.
#
# One item needs its own cell first. Rows for the same customer at different
# origins are not independent, and there are two reasons, not one. Adjacent
# origins have overlapping label windows (a 90-day window, a 30-day step), so
# the same purchase can decide two labels. Separately, some customers are
# durably loyal and others durably lapsed, so a customer's label agrees with
# itself across origins even when the windows do not overlap at all. The cell
# below separates the two by measuring same-customer label agreement as a
# function of how far apart the origins are.

# %% jupyter={"source_hidden": true}
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
# baseline. Most of the excess is persistence in churn propensity, not the
# overlapping-window artefact. The overlap adds a smaller, real effect on
# top.
#
# | # | Component | Leak class | Evidence | Severity | Fix |
# |---|---|---|---|---|---|
# | 1 | `naive_features` (recency/frequency/monetary) | Feature-time leakage: full-history aggregation ignores `as_of` and includes the very transaction that determines the label | The purchase that makes a row `churned=0` is counted into that customer's `frequency` and `monetary` and lowers `recency_days`, at *every* origin the customer appears in, not only the origin the purchase falls inside. Single-feature AUCs of 0.82-0.86 (computed above). The drop-one ablation shows no single feature is the whole leak: removing the strongest (`recency_days`) only brings AUC down to about 0.87 | High (intentional, mistake #1) | Compute features from transactions strictly before each row's own `as_of` (`03_temporal_protocol`) |
# | 2 | Row-level random split | Split leakage: entity overlap across train and test at different `as_of` | Customer-overlap chart above | High (intentional, mistake #2) | Group the split by `customer_id` (e.g. `GroupShuffleSplit`), or evaluate per origin |
# | 3 | `GAP_DAYS = 0` | Split leakage: no operational lead time; `feature_end == label_start` | Passed explicitly to `rolling_origins`; `config.DEFAULT_GAP_DAYS = 7` is used elsewhere | Medium (intentional, mistake #3; secondary here because mistake #1 already reads future data regardless of gap) | Use `DEFAULT_GAP_DAYS` in `03_temporal_protocol` |
# | 4 | Same-customer label agreement across origins | Rows for one customer are not independent draws, for two separate reasons (see the cell above) | Agreement is highest at maximum window overlap and falls as overlap shrinks, but stays well above the independence baseline even at zero overlap | Medium; compounds mistake #2 | A rolling-origin backtest needs cluster-aware uncertainty (e.g. a block bootstrap by `customer_id`) regardless of `step_days`; spacing origins further apart removes only the overlap component |
# | 5 | Eligibility rule (at least one purchase in the 365 days before `as_of`) | Clean | `eligible_customers` filters strictly `< as_of`, matching the feature-window convention in `churnval.windows`; it does not look forward | n/a | See [ADR-0007](../docs/adr/0007-online-retail-ii-naive-panel-definition.md) for the one-purchase vs. two-purchase question, which is a modelling choice rather than a leak |
# | 6 | Label maturity / open-window censoring | Clean | `rolling_origins` excludes any origin whose label window would extend past the dataset's last event, so no label is truncated and no censored customer is silently coded as a non-churner | n/a | n/a |
# | 7 | Results file vs. tautological section | Clean | `reports/results_naive.json` is written before the tautological section begins; nothing after that point writes to any file | n/a | n/a |
# | 8 | `io.py` administrative stock codes (`POST`, `M`, `BANK CHARGES`, and so on) | Found during this notebook's review and fixed | 25 customers had their entire purchase history built from postage, manual-adjustment, bank-charge or test line items rather than real purchases, before `_clean` excluded them (see `docs/data/online_retail_ii.md`) | Medium, a data-quality issue rather than one of this notebook's intentional leaks | Fixed in `churnval.io.EXCLUDED_STOCK_CODES`; regression-tested in `tests/test_io.py` |
#
# On reusing one `SEED` across this notebook's model fits: the seed only
# controls shuffling and tree randomness *within* each independently fitted
# model, so no state crosses between fits. The tautological section's score
# is an identity, not a favourable draw, and would reproduce under any seed.

# %% [markdown]
# ---
# ## Closing
#
# **Finding:** built faithfully, the conventional approach reports ROC AUC of
# about 0.90 (`reports/results_naive.json` has the full bundle). That number
# is not a legitimate estimate of anything. It comes from features that read
# future transactions regardless of `as_of`, a split that lets the same
# customer teach and test the model at different points in time, and no gap
# between scoring and the outcome window. A fourth, less obvious effect
# compounds the second: rows for the same customer are correlated for two
# separate reasons, as the audit above shows.
#
# The closing section built a version that scores a literal 1.000 to make a
# narrower point stand on its own: a perfect offline score is a bug report,
# not a result, and a temporal split does not rule it out by itself. The
# feature and the label have to be checked for being the same fact under two
# names.
#
# **Next:** `02_leakage_diagnosis` takes the 0.90 apart with reusable tools:
# adversarial validation, ablation, and the leakage table its opening
# question asks for. `03_temporal_protocol` then rebuilds the panel the
# honest way, with features computed strictly before each row's own `as_of`,
# a real gap, and a rolling-origin backtest, so that `04_the_gap` can report
# how much the number on this page was lying.
