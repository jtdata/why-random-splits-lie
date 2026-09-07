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
# # 03 Temporal protocol
#
# **Question this notebook answers:** what does the correct design look
# like?
#
# **Consumes:** `01`'s cached Parquet (`churnval.io.load_online_retail_ii`);
# `churnval.windows` (`Window`, `rolling_origins`, `mask_feature_events`,
# `mask_label_events`), which are general-purpose temporal utilities; and
# the same problem definition `01` used for this dataset, a 90-day horizon
# and a 365-day, one-purchase eligibility lookback (ADR-0007). That reasoning
# was about the dataset's purchase cadence, not about which notebook does
# the scoring.
#
# **Not consumed, on purpose:** the panel, feature and split logic in
# `churnval.naive_baseline` and all of `churnval.asof_preview`. The one
# exception is the `HORIZON_DAYS` constant, imported as a dataset-level
# fact, not as a dependency on any of that module's wrong-on-purpose logic.
#
# **Produces:** a rolling-origin backtest with features computed strictly
# before each row's own `as_of`, a real operational gap
# (`config.DEFAULT_GAP_DAYS`) between `as_of` and the label window, and a
# model retrained forward at each origin rather than evaluated on one split.
# Reported per origin, to show variance across regimes, and pooled into a
# single honest `Result` written to `reports/results_temporal.json` for
# `04_the_gap` and the README table.
#
# **Runtime:** a few minutes, one model fit per backtest origin plus the
# window and retraining experiments below.
#
# ---
#
# `02_leakage_diagnosis` explained why `01`'s 0.90 was inflated without
# fixing anything. This notebook is the fix, built the way
# `docs/standards/validation.md` requires: an explicit gap, features
# snapshotted as-of, several origins retrained forward, and a maturity check
# on who is even eligible to be scored. It does not yet report calibration
# (`05`) or compare against a hazard framing (`06`). The question here is
# only whether the ranking holds up once the three mistakes from `01` are
# corrected in code rather than in prose.

# %% [markdown]
# ## Load the data and build the scoring occasions
#
# Same cached Parquet and same origins as `01` and `02`, monthly `as_of`
# dates starting a full eligibility lookback after the data begins, except
# that `gap_days` is now `config.DEFAULT_GAP_DAYS` (7) instead of 0. That one
# argument is mistake #3, corrected.

# %%
import json
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier

from churnval.config import DEFAULT_GAP_DAYS, DEFAULT_STEP_DAYS, PATHS, SEED
from churnval.evaluation import score
from churnval.io import load_online_retail_ii
from churnval.naive_baseline import HORIZON_DAYS
from churnval.plotting import PALETTE, set_style
from churnval.splits import build_asof_panel, frozen_model_backtest, rolling_origin_backtest
from churnval.windows import rolling_origins

PATHS.ensure()
set_style()
transactions = load_online_retail_ii()
first_as_of = transactions["invoice_date"].min().normalize() + timedelta(days=365)
last_event = transactions["invoice_date"].max()
origins = rolling_origins(
    first_as_of,
    last_event,
    gap_days=DEFAULT_GAP_DAYS,
    horizon_days=HORIZON_DAYS,
    step_days=DEFAULT_STEP_DAYS,
)
print(f"{len(origins)} origins, {origins[0].as_of:%Y-%m-%d} to {origins[-1].as_of:%Y-%m-%d}")

# %% [markdown]
# `HORIZON_DAYS` is imported from `naive_baseline` rather than redefined. It
# is a property of this dataset (ADR-0007), not of the naive pipeline, and
# retyping the literal `90` here would let the two notebooks drift apart
# silently if it ever changed.

# %% [markdown]
# ## Build the corrected panel
#
# `build_asof_panel` (`churnval.splits`) computes each customer's
# recency, frequency and monetary value *strictly before that row's own
# `as_of`* (`churnval.features.asof_features`), recomputed at every origin.
# The label is the same forward-looking definition `01` already used
# correctly. What changes is the gap and the features, not what "churn"
# means.

# %%
panel = build_asof_panel(transactions, origins)
print(
    f"{len(panel):,} (customer, as_of) rows, {panel['customer_id'].nunique():,} distinct "
    f"customers, {panel['churned'].mean():.1%} churned overall"
)

# %% [markdown]
# Churn rate by origin. This is real seasonal movement; every origin's
# features are computed fresh from data available at that origin alone.

# %%
by_origin = panel.groupby("as_of")["churned"].agg(n="size", churn_rate="mean")
by_origin

# %% [markdown]
# ## The rolling-origin backtest
#
# `rolling_origin_backtest` (`churnval.splits`) retrains at each origin on
# every *mature* panel row from strictly earlier origins, then scores on
# that origin's own rows. An expanding window, not a single split (ADR-0009
# explains the choice of expanding over a fixed lookback).
#
# "Mature" is doing real work in that sentence. The first version of this
# function pooled every earlier origin for training without checking whether
# that origin's own label window had closed by the test origin's `as_of`.
# With a 90-day horizon and a 7-day gap, a label is not resolved until 97
# days after its own origin, more than three of this dataset's 30-day steps.
# So the three most recent origins before any test origin are still
# unresolved at that point in simulated time. Training on them means
# training on labels that are correct in the offline dataset but would not
# yet be knowable in a live deployment. `rolling_origin_backtest` now
# excludes any training origin whose label window has not closed before the
# test origin's `as_of`, and skips a test origin entirely if that leaves it
# with no mature training data. That is what happens to the first candidate
# origin here, leaving six origins scored rather than seven.

# %%
feature_cols = ["recency_days", "frequency", "monetary"]
per_origin, predictions = rolling_origin_backtest(
    panel, origins, feature_cols=feature_cols, min_training_origins=3, seed=SEED
)
per_origin

# %% [markdown]
# ### What the maturity check cost the number
#
# Reusing the same panel and origins, this rebuilds the pooled result the
# way the earlier, buggy version would have: every strictly earlier origin
# pooled, no maturity check. `rolling_origin_backtest` does not expose that
# path, since there is no legitimate reason to call it, so it is
# reconstructed inline for this one comparison.

# %%
naive_pooled_predictions = []
for i in range(3, len(origins)):
    train_as_of = [w.as_of for w in origins[:i]]
    test_as_of = origins[i].as_of
    train = panel[panel["as_of"].isin(train_as_of)]
    test = panel[panel["as_of"] == test_as_of]
    unfiltered_model = LGBMClassifier(random_state=SEED, verbosity=-1)
    unfiltered_model.fit(train[feature_cols], train["churned"])
    naive_pooled_predictions.append(
        pd.DataFrame(
            {
                "y_true": test["churned"].to_numpy(),
                "y_prob": unfiltered_model.predict_proba(test[feature_cols])[:, 1],
            }
        )
    )
naive_pooled_predictions = pd.concat(naive_pooled_predictions, ignore_index=True)
unfiltered_result = score(
    naive_pooled_predictions["y_true"],
    naive_pooled_predictions["y_prob"],
    label="unfiltered (bug)",
)
maturity_filtered_result = score(
    predictions["y_true"], predictions["y_prob"], label="maturity-filtered (fixed)"
)
pd.DataFrame([unfiltered_result.as_row(), maturity_filtered_result.as_row()]).set_index("label")[
    ["roc_auc", "pr_auc", "brier"]
]

# %% [markdown]
# The unfiltered version also tests one extra origin (the one with zero
# mature training data, which the fixed version skips), so the two rows are
# not scored on identical rows. The point is the direction rather than the
# precise delta: training on unresolved labels inflated ROC AUC, and the
# maturity check was worth adding before trusting this notebook's headline
# number.

# %% [markdown]
# ## Does performance hold up across regimes?
#
# A single split is one draw; this is six. If the honest number depended
# heavily on which slice of time got tested, it would show up here as large
# swings from origin to origin.

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
metrics = [("roc_auc", "ROC AUC"), ("pr_auc", "PR AUC"), ("brier", "Brier")]
for ax, (col, title) in zip(axes, metrics, strict=True):
    ax.plot(per_origin["as_of"], per_origin[col], marker="o", color=PALETTE["test"])
    ax.set_title(title, loc="left", fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.suptitle(
    "ROC AUC holds up across origins; PR AUC and Brier drift with prevalence",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(PATHS.figures / "temporal_backtest_by_origin.png", dpi=200)
# fig

# %% [markdown]
# The later origins look worse on two of the three panels, but the churn
# rate itself is falling across the whole panel (68% at the earliest origin
# in `by_origin` above, 50% at the latest), and PR AUC and Brier both move
# with the base rate even when ranking quality does not. Correlating each
# metric against `prevalence` across the six tested origins:

# %%
per_origin[["prevalence", "roc_auc", "pr_auc", "brier"]].corr()["prevalence"]

# %% [markdown]
# With six points these correlations are descriptive, not a test. Read
# that way: PR AUC tracks prevalence almost exactly (r about 0.98) and Brier
# moves inversely with it (r about -0.87), which is the mechanical
# behaviour of those two metrics, not evidence the model is failing on later
# data. ROC AUC, the prevalence-invariant metric, correlates weakly (r about
# 0.35) and stays in a narrow band (0.742-0.780). By that read the ranking
# holds up, with the first tested origin, which has the least mature
# training history behind it, as the softest point.

# %% [markdown]
# ## Does the training window need to be this wide?
#
# ADR-0009 chose an expanding window over a fixed lookback on the grounds
# that nothing in this dataset suggests purchase patterns go stale. That
# was a stated assumption, not a measurement. This section tests it: rerun
# the same backtest with a *sliding* window instead of an expanding one, at
# several widths, and see whether older data helps, hurts, or does nothing.
#
# Not every width is possible. A training origin still has to be mature
# (its label window closed before the test origin's `as_of`), and the
# earliest that can happen is `HORIZON_DAYS + GAP_DAYS` = 97 days after the
# training origin's own `as_of`, more than three 30-day steps. A sliding
# window narrower than 120 days can never contain a mature origin at all,
# so every test origin would be skipped. Past 270 days a sliding window can
# no longer exclude anything the expanding window includes, so it stops
# being a different experiment. 120, 150, 180 and 210 days are the four
# narrowest widths that are both possible and different. Each is a true
# fixed window: 120 days always trains on exactly the one most recently
# matured origin, and 210 always trains on the four most recent, dropping
# the fifth-oldest as it slides forward.

# %%
lookback_days_list = [120, 150, 180, 210]
sliding_results = {}
for lookback in lookback_days_list:
    po, preds = rolling_origin_backtest(
        panel,
        origins,
        feature_cols=feature_cols,
        min_training_origins=3,
        seed=SEED,
        max_lookback_days=lookback,
    )
    sliding_results[lookback] = (po, preds)
    print(f"{lookback}d window: {len(po)} origins tested, {len(preds):,} predictions")

# %% [markdown]
# The two boundary claims above, confirmed:

# %%
po_90, preds_90 = rolling_origin_backtest(
    panel,
    origins,
    feature_cols=feature_cols,
    min_training_origins=3,
    seed=SEED,
    max_lookback_days=90,
)
print(
    f"90d window: {len(po_90)} origins tested (0 expected, shorter than the 97-day maturity floor)"
)

po_365, _ = rolling_origin_backtest(
    panel,
    origins,
    feature_cols=feature_cols,
    min_training_origins=3,
    seed=SEED,
    max_lookback_days=365,
)
print(f"365d window matches expanding exactly: {po_365.equals(per_origin)}")

# %% [markdown]
# ### Per-origin performance, one line per window width
#
# The same chart as above, with a line per sliding-window width (lightest is
# narrowest, darkest is widest) against the expanding window already
# computed (dashed).

# %%
lookback_colors = {120: "#F5B899", 150: "#EF9B6C", 180: "#EB6834", 210: "#B84A1F"}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, (col, title) in zip(axes, metrics, strict=True):
    for lookback in lookback_days_list:
        po, _ = sliding_results[lookback]
        ax.plot(
            po["as_of"],
            po[col],
            marker="o",
            markersize=4,
            lw=1.5,
            color=lookback_colors[lookback],
            label=f"{lookback}d",
        )
    ax.plot(
        per_origin["as_of"],
        per_origin[col],
        marker="o",
        markersize=4,
        lw=2,
        ls="--",
        color=PALETTE["test"],
        label="expanding",
    )
    ax.set_title(title, loc="left", fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].legend(frameon=False, fontsize=7, loc="lower left")
fig.suptitle("Wider sliding windows converge toward the expanding-window result", fontsize=10)
fig.tight_layout()
fig.savefig(PATHS.figures / "temporal_backtest_sliding_window.png", dpi=200)
# fig

# %% [markdown]
# ### Does older data help or hurt?
#
# The per-origin view is noisy across only six points. Pooling each window
# width's held-out predictions into one number answers the question
# directly, the same way the expanding window's own headline number is built
# below.

# %%
pooled_by_window = []
for lookback in lookback_days_list:
    _, preds = sliding_results[lookback]
    pooled_by_window.append(
        score(preds["y_true"], preds["y_prob"], label=f"{lookback}d sliding").as_row()
    )
pooled_by_window.append(
    score(predictions["y_true"], predictions["y_prob"], label="expanding (default)").as_row()
)
window_comparison = pd.DataFrame(pooled_by_window).set_index("label")[
    ["roc_auc", "pr_auc", "brier"]
]
window_comparison

# %% [markdown]
# Older data helps, up to a point, and never hurts within the range that can
# be tested here. ROC AUC rises from the tightest window (120 days, one
# training origin) through 150 and 180 days, then is flat across 180, 210
# and the expanding window, three settings that already share most of their
# training rows. PR AUC and Brier tell the same story. The opposite claim,
# that recent-only data is enough, does not survive the 120-day column.
# ADR-0009's assumption holds, and now has a number behind it.

# %% [markdown]
# ## How often does the model need retraining?
#
# A different question. The sliding-window comparison asked *how much
# history* a retrained model should train on; this one asks whether
# retraining at every origin is necessary at all. `frozen_model_backtest`
# (`churnval.splits`) fits one model at the earliest origin with mature
# training data, the same origin and the same training rows
# `rolling_origin_backtest` uses for its first tested origin, and then
# scores every later origin with that same model. If a five-month-old model
# scores about as well as a freshly retrained one, monthly retraining is not
# earning its cost. If it decays, the decay is the argument for retraining.

# %%
per_origin_frozen, predictions_frozen = frozen_model_backtest(
    panel, origins, feature_cols=feature_cols, min_training_origins=3, seed=SEED
)
per_origin_frozen

# %% [markdown]
# `origins_since_training = 0` is the freeze point itself. The frozen model
# *is* the retrained model there, so the two approaches cannot differ yet.

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, (col, title) in zip(axes, metrics, strict=True):
    ax.plot(
        per_origin_frozen["as_of"],
        per_origin_frozen[col],
        marker="o",
        lw=2,
        color=PALETTE["neutral"],
        label="frozen (trained once)",
    )
    ax.plot(
        per_origin["as_of"],
        per_origin[col],
        marker="o",
        lw=2,
        ls="--",
        color=PALETTE["test"],
        label="retrained every origin",
    )
    ax.set_title(title, loc="left", fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle(
    "A frozen model falls behind a retrained one, gradually rather than sharply", fontsize=10
)
fig.tight_layout()
fig.savefig(PATHS.figures / "temporal_backtest_frozen_vs_retrained.png", dpi=200)
# fig

# %% [markdown]
# The gap as a function of the frozen model's age is the clearest way to
# answer "how often":

# %%
gap = per_origin_frozen.set_index("as_of")["roc_auc"] - per_origin.set_index("as_of")["roc_auc"]
gap.index = per_origin_frozen["origins_since_training"].to_numpy()
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.axhline(0, color=PALETTE["reference_line"], lw=1, ls="--")
ax.plot(gap.index, gap.to_numpy(), marker="o", color=PALETTE["both"])
ax.set_xlabel("origins since the frozen model was trained (≈ months)")
ax.set_ylabel("ROC AUC, frozen minus retrained")
ax.set_title("The cost of not retraining grows with the model's age", loc="left")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(PATHS.figures / "temporal_backtest_frozen_gap.png", dpi=200)
# fig

# %% [markdown]
# ## Pooling both approaches for one comparison

# %%
pooled_frozen = score(
    predictions_frozen["y_true"], predictions_frozen["y_prob"], label="frozen (trained once)"
)
pooled_retrained = score(
    predictions["y_true"], predictions["y_prob"], label="retrained every origin"
)
pd.DataFrame([pooled_frozen.as_row(), pooled_retrained.as_row()]).set_index("label")[
    ["roc_auc", "pr_auc", "brier"]
]

# %% [markdown]
# Retraining wins, but not by a cliff. Pooled across the backtest the frozen
# model gives up about 0.02 ROC AUC. The degradation is gradual (the gap
# chart widens roughly in step with the model's age, with no sudden
# collapse), which is useful operational information: this panel does not
# need retraining after every origin to stay usable, but going indefinitely
# without retraining has a real and growing cost, largest at five origins
# out, the furthest point tested here. Note the frozen model was trained on
# the smallest training set in the backtest (one mature origin), so part of
# what the retrained line gains is simply more data, not only fresher data.
# The sliding-window comparison above is the cleaner read on that.

# %% [markdown]
# ## The honest number
#
# Pool every backtest origin's held-out predictions and score once. This is
# not the same as averaging the six per-origin AUCs, which would answer a
# different question (the typical origin) from this one (overall ranking
# quality across everything tested). This is the "Temporal split with gap,
# rolling origin" row of the README results table, written to
# `reports/results_temporal.json` for `04_the_gap` to read.

# %%
pooled_result = score(
    predictions["y_true"], predictions["y_prob"], label="Temporal split with gap, rolling origin"
)
result_path = PATHS.reports / "results_temporal.json"
result_path.write_text(json.dumps(pooled_result.as_row(), indent=2))
pooled_result.as_row()

# %% [markdown]
# For comparison, and only as a check that this notebook is pointed the
# right way before closing it. `04_the_gap` does the full analysis.

# %%
naive_result_path = PATHS.reports / "results_naive.json"
naive_result = json.loads(naive_result_path.read_text())
comparison = pd.DataFrame([naive_result, pooled_result.as_row()]).set_index("label")
comparison[["roc_auc", "pr_auc", "brier"]]

# %% [markdown]
# ---
# ## Leakage audit
#
# Run against `features.py` and `splits.py`. This notebook is meant to be
# the correct one, which makes a clean audit the whole point. The maturity
# check above was not clean on the first pass, and that is recorded here
# rather than folded into the code with no trace.
#
# | # | Component | Leak class | Evidence | Severity | Fix |
# |---|---|---|---|---|---|
# | 1 | `asof_features` | Clean | Filters `invoice_date < as_of` before every aggregation. `tests/test_features.py` covers both inclusion (before `as_of`) and exclusion (on or after `as_of`) with hand-computed values | n/a | n/a |
# | 2 | `eligible_customers` | Clean | Filters strictly `< as_of`, matching the feature-window convention. Same rule as `01`'s already-audited version, reimplemented rather than imported | n/a | n/a |
# | 3 | Label construction, `GAP_DAYS = 7` | Clean, and now real | `mask_label_events` uses `window.label_start = as_of + gap_days`, verified against the origins table (e.g. `as_of=2011-03-01` gives `label_start=2011-03-08`). Half-open `[label_start, label_end)` convention covered by `tests/test_windows.py`. One consequence to be aware of: a purchase that lands *inside* the gap does not count toward the label, so a customer who buys on day 3 and then goes quiet for 90 days is labelled churned. That is consistent with the definition (the outcome window opens when action becomes possible) but it is a choice, and `07` shows it matters much more for a contractual dataset | n/a | n/a |
# | 4 | `rolling_origin_backtest`'s training-origin selection | Found and fixed while building this notebook | The first version pooled every earlier origin for training with no check on whether that origin's label window had closed by the test origin's `as_of`. With `HORIZON_DAYS=90` and `GAP_DAYS=7` a label resolves 97 days after its origin, more than three 30-day steps, so the three most recent origins before any test origin were still open every time. The comparison cell above quantifies it: pooled ROC AUC 0.788 unfiltered vs. 0.765 filtered, and the first candidate test origin had zero mature training origins and is dropped | High; an unaudited maturity gap in the "correct" notebook would have undercut the comparison `04_the_gap` is built on | `rolling_origin_backtest` filters training origins by `w.label_end <= test_window.as_of` and skips any test origin left with no mature training data; regression-tested in `tests/test_splits.py` |
# | 5 | Customers repeating across training origins | Not a leak, unlike `01`'s mistake #2 | A customer can appear in an earlier training origin and a later test origin. What made `01`'s split wrong was that a row's features and label were not bound to that row's own `as_of`. Here every row's features and label are bound to its own origin, and a real customer persisting across origins is what deployment looks like | n/a | n/a |
# | 6 | Pooling for the headline number vs. averaging per-origin metrics | Pooling is the right weighting; one limitation named | Per-origin `n` ranges 4,317-4,323, within 0.15% of each other, so pooling gives every origin close to equal weight. What pooling does not address: the pooled AUC's pairwise comparisons include prediction pairs from two different models (each origin is a separate refit) scored against two different base rates (68% down to 50%). That is standard for walk-forward out-of-sample predictions but not the same guarantee as one model scored once. The per-origin chart is the mitigation; it is how a reader would notice if pooling were hiding a regime-specific collapse | n/a | n/a |

# %% [markdown]
# ---
# ## Closing
#
# **Finding:** corrected the honest way, with as-of-safe features, a real
# 7-day gap, and a rolling-origin backtest that only trains on labels that
# would have been knowable at the time, ROC AUC is 0.765 (PR AUC 0.795, Brier
# 0.192) against `01`'s 0.903 (PR AUC 0.944, Brier 0.123). That is the second
# number this notebook produced, not the first. The first version of the
# backtest, before its maturity check, reported 0.788: a smaller version of
# exactly the mistake this project argues against, found inside the notebook
# whose whole job was being correct. Getting the split, the gap and the
# features right does not get everything right. Each new piece of machinery
# needs its own audit, including the machinery built to fix the last one.
#
# ROC AUC stays in a narrow band across the six backtest origins
# (0.742-0.780) and correlates only weakly with the falling churn rate. PR
# AUC and Brier move with prevalence almost exactly, which is expected and
# not itself a sign the model fails on later regimes. Older training data
# helps rather than hurts within the range testable here, and a model frozen
# at the first origin loses about 0.02 ROC AUC over five months.
#
# **Next:** `04_the_gap` puts `01`'s number and this notebook's number in
# the same chart and the same README table, now that both sides of the
# comparison have been through their own audit.
