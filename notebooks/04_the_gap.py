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
# # 04 — The gap
#
# **Question this notebook answers:** how big is the lie?
#
# **Consumes:** `reports/results_naive.json` (`01_naive_baseline`) and
# `reports/results_temporal.json` (`03_temporal_protocol`), plus the two
# pipelines themselves (`churnval.naive_baseline`, `churnval.splits`,
# `churnval.windows`, `churnval.evaluation`). This notebook builds no new
# protocol. It compares the two that exist, and rebuilds each one's
# per-origin breakdown where the saved JSON does not carry enough detail.
#
# **Produces:** a third result isolating the gap alone
# (`reports/results_temporal_nogap.json`), completing the three-row README
# table; the headline chart; a common-origin, common-population comparison
# that isolates training/serving skew in the naive model
# (`naive_serving_skew.png`); and the numbers `README.md` cites.
#
# **Runtime:** several minutes. Two more rolling-origin backtests (`03`'s
# pipeline with `gap_days=0`, and `03`'s own pipeline rebuilt to expose its
# per-origin table) plus one naive-panel fit.
#
# ---
#
# `01` reported ROC AUC of about 0.90 by getting three things wrong at once:
# features that ignore `as_of`, a split that ignores entity identity, and no
# gap. `03` corrected all three and reported about 0.76. The difference is
# real, but two numbers cannot say which mistake carried it. A reader
# cannot tell whether the 7-day gap matters at all, or whether the features
# and the split carry the whole correction. This notebook adds one more data
# point, `03`'s pipeline with the gap set back to zero and everything else
# left correct, so the README table can show what the gap alone is worth.

# %% [markdown]
# ## The two numbers already on disk

# %%
import json
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier

from churnval.config import DEFAULT_GAP_DAYS, DEFAULT_STEP_DAYS, PATHS, SEED
from churnval.evaluation import score
from churnval.features import asof_features
from churnval.io import load_online_retail_ii
from churnval.naive_baseline import GAP_DAYS as NAIVE_GAP_DAYS
from churnval.naive_baseline import HORIZON_DAYS, build_naive_panel, naive_features
from churnval.plotting import PALETTE, set_style
from churnval.splits import build_asof_panel, eligible_customers, rolling_origin_backtest
from churnval.windows import mask_label_events, rolling_origins

PATHS.ensure()
set_style()

naive_result = json.loads((PATHS.reports / "results_naive.json").read_text())
temporal_result = json.loads((PATHS.reports / "results_temporal.json").read_text())
pd.DataFrame([naive_result, temporal_result]).set_index("label")[["roc_auc", "pr_auc", "brier"]]

# %% [markdown]
# These two are not scored on comparable samples. `01`'s number comes from
# one stratified 20% row split of a panel with the entity-overlap leak
# (n=8,570, prevalence 62%). `03`'s comes from pooling six rolling-origin
# backtests (n=25,922, prevalence 59%). Different sampling unit, different
# N, different prevalence. That is worth stating up front, since it is the
# headline comparison.

# %% [markdown]
# ## Isolating the gap
#
# `03`'s pipeline, unchanged, with `gap_days=0` instead of
# `config.DEFAULT_GAP_DAYS`. Everything else (as-of features, the maturity
# check, the expanding window, the backtest) stays as `03` built it. If this
# number lands near `01`'s, the gap is doing most of the correcting. If it
# lands near `03`'s, the features and the split are.
#
# One limitation follows from how `gap_days` is wired. `asof_features`
# filters `invoice_date < as_of` regardless of `gap_days`; the gap only
# moves `Window.label_start` and `label_end`, and through them the
# label-maturity floor. It never moves the feature cutoff. So this ablation
# measures whether shifting the label window's boundary by 7 days changes
# anything. It cannot measure whether a gap protects against feature
# leakage in general, because the features here never reach past `as_of` in
# the first place. The classic zero-gap leak (a feature computed right up to
# the event that encodes the event) needs a feature that does reach past
# `as_of`, which is exactly what `01`'s features do and what the A/B/C
# comparison later in this notebook measures.

# %%
transactions = load_online_retail_ii()
first_as_of = transactions["invoice_date"].min().normalize() + timedelta(days=365)
last_event = transactions["invoice_date"].max()
feature_cols = ["recency_days", "frequency", "monetary"]

origins_nogap = rolling_origins(
    first_as_of, last_event, gap_days=0, horizon_days=HORIZON_DAYS, step_days=DEFAULT_STEP_DAYS
)
panel_nogap = build_asof_panel(transactions, origins_nogap)
per_origin_nogap, predictions_nogap = rolling_origin_backtest(
    panel_nogap, origins_nogap, feature_cols=feature_cols, min_training_origins=3, seed=SEED
)
nogap_result = score(
    predictions_nogap["y_true"], predictions_nogap["y_prob"], label="Temporal split, no gap"
)
result_path = PATHS.reports / "results_temporal_nogap.json"
result_path.write_text(json.dumps(nogap_result.as_row(), indent=2))
nogap_result.as_row()

# %% [markdown]
# Removing the gap also changes which origins are testable. Dropping
# `GAP_DAYS` from 7 to 0 lowers the maturity floor from 97 to 90 days,
# exactly the boundary of one more 30-day step, so this backtest tests
# seven origins where `03`'s tests six. The extra origin is correctly
# matured, but the two backtests are not scored on identical rows, only
# with identical methodology.
#
# `03`'s with-gap backtest is rebuilt here too, rather than only read from
# its pooled JSON, because the comparison below needs its per-origin
# breakdown. That is also a cheap consistency check: it should reproduce
# `03`'s saved result exactly.

# %%
origins_withgap = rolling_origins(
    first_as_of,
    last_event,
    gap_days=DEFAULT_GAP_DAYS,
    horizon_days=HORIZON_DAYS,
    step_days=DEFAULT_STEP_DAYS,
)
panel_withgap = build_asof_panel(transactions, origins_withgap)
per_origin_withgap, predictions_withgap = rolling_origin_backtest(
    panel_withgap, origins_withgap, feature_cols=feature_cols, min_training_origins=3, seed=SEED
)
withgap_check = score(predictions_withgap["y_true"], predictions_withgap["y_prob"], label="check")
assert abs(withgap_check.roc_auc - temporal_result["roc_auc"]) < 1e-9, (
    "recomputed with-gap backtest doesn't match 03's saved result"
)
print(
    f"recomputed with-gap ROC AUC {withgap_check.roc_auc:.6f} matches 03's saved {temporal_result['roc_auc']:.6f}"
)

# %% [markdown]
# ## The headline comparison

# %%
results_table = pd.DataFrame([naive_result, nogap_result.as_row(), temporal_result]).set_index(
    "label"
)
results_table[["n", "roc_auc", "pr_auc", "brier"]]

# %%
bar_labels = [
    "Random split\n(the wrong way)",
    "Temporal split,\nno gap",
    "Temporal split with gap,\nrolling origin",
]

# %% [markdown]
# Every figure the discussion below cites is computed here, once, rather
# than read off the chart by eye.

# %%
roc_auc_gap = naive_result["roc_auc"] - temporal_result["roc_auc"]
pr_auc_gap = naive_result["pr_auc"] - temporal_result["pr_auc"]
brier_relative_increase = (temporal_result["brier"] - naive_result["brier"]) / naive_result["brier"]
decomposition = pd.Series(
    {
        "ROC AUC gap, naive vs. with-gap": roc_auc_gap,
        "PR AUC gap, naive vs. with-gap": pr_auc_gap,
        "Brier relative increase, naive -> with-gap": brier_relative_increase,
    }
)
decomposition

# %%
colours = [PALETTE["both"], PALETTE["neutral"], PALETTE["test"]]
metrics = [("roc_auc", "ROC AUC", True), ("pr_auc", "PR AUC", True), ("brier", "Brier", False)]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (col, title, higher_is_better) in zip(axes, metrics, strict=True):
    values = results_table[col].to_numpy()
    bars = ax.bar(bar_labels, values, color=colours)
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9
        )
    ax.set_title(
        f"{title} ({'higher' if higher_is_better else 'lower'} better)", loc="left", fontsize=10
    )
    ax.tick_params(axis="x", labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

# A bracket spanning bar 1 to bar 3, drawn above both bars rather than on
# top of either one's own value label, so it reads as one shape instead of
# colliding with the numbers already on the chart.
ax0 = axes[0]
ax0.set_ylim(0, 1.04)
bracket_y = 0.97
ax0.plot(
    [0, 0, 2, 2],
    [naive_result["roc_auc"] + 0.02, bracket_y, bracket_y, temporal_result["roc_auc"] + 0.02],
    color=PALETTE["reference_line"],
    lw=1,
)
ax0.text(1, bracket_y + 0.012, f"the gap: {roc_auc_gap:.3f}", ha="center", fontsize=9)
fig.suptitle(
    f"Random split inflates ROC AUC by {roc_auc_gap:.3f}, almost entirely from the features "
    "and the split, not the operational gap",
    fontsize=10.5,
)
fig.tight_layout()
fig.savefig(PATHS.figures / "the_gap.png", dpi=200)
# fig

# %% [markdown]
# ## What the decomposition says
#
# The gap between "the wrong way" and "the honest number" is 0.138 ROC AUC.
# ROC AUC is the number the bracket calls out because it is
# prevalence-invariant, and the naive and corrected panels are scored on
# samples with different prevalence. That is a reason to lead with it, not
# to hide the other two: PR AUC falls by about as much (0.149), and Brier
# gets 56% worse in relative terms.
#
# The middle bar shows where the ROC AUC gap comes from. Fixing the features
# and the split, with the label gap still at zero, already recovers ROC AUC
# to 0.766, within 0.001 of the fully corrected 0.765. Adding the 7-day
# operational gap back costs nothing further. No uncertainty is attached to
# that 0.001 here (a customer-clustered bootstrap would be the right tool),
# but it is small enough that the direction is not in doubt. So `01`'s
# zero-gap mistake was secondary to its feature and split mistakes, which is
# what `01`'s own leakage table predicted. The finding is narrower than "gaps
# don't matter": in this codebase the gap never touches the feature cutoff,
# so this ablation could not have shown a feature-leakage effect even if the
# features had one.

# %% [markdown]
# ## Can the naive model's number be served at all?
#
# An earlier version of this notebook answered that with a "look-forward
# holdout" for each of the three approaches. It had three problems, all of
# which favoured the naive arm:
#
# 1. **The naive holdout was still leaked.** `naive_features` measures
#    recency to the *dataset's own end date*, not to the row's `as_of`. At
#    the naive model's last origin that reference date is past the end of
#    the origin's own label window, so the feature partially encodes whether
#    the label-defining purchase already happened. A holdout *split* does
#    nothing about a feature that ignores `as_of`.
# 2. **The three arms trained on different data.** The naive holdout trained
#    on every earlier origin; the two temporal arms went through
#    `rolling_origin_backtest`, which drops immature origins (ADR-0009).
# 3. **The three arms were not scored on the same date.** The naive and
#    no-gap origins share a `gap_days=0` sequence; the with-gap origins stop
#    one step earlier.
#
# The fix is a narrower, better-posed question. Fix one holdout `as_of` and
# one eligible population, and vary only what is under test: the naive
# model's training, the naive model's *features*, or the correct protocol's
# features.
#
# - **A: naive trained, naive features at scoring.** What `01` would report
#   if pointed at this date. Shown to make the point, not as a legitimate
#   result: it requires knowing the dataset's end date at scoring time.
# - **B: naive trained, as-of features at serving.** The same fitted model
#   as A, scored with `asof_features` computed strictly before the holdout
#   `as_of`, the features a deployment would have. This is the honest number
#   for "what if we shipped the naively trained model", and A minus B is
#   training/serving skew: how much of A's apparent skill came from a
#   feature the model will never see in production.
# - **C: correctly trained, as-of features.** `03`'s own protocol at the
#   same `as_of`, reused from `per_origin_withgap`, not refit.
#
# The common `as_of` is `origins_withgap[-1].as_of`, the latest date the
# correct protocol can test at all. The eligible population is computed
# once at that date and checked against `03`'s own panel.

# %%
common_holdout_window = origins_withgap[-1]
common_holdout_as_of = common_holdout_window.as_of
common_eligible = eligible_customers(transactions, common_holdout_as_of)

withgap_test_customers = set(
    panel_withgap.loc[panel_withgap["as_of"] == common_holdout_as_of, "customer_id"]
)
assert withgap_test_customers == set(common_eligible), (
    "naive and correct eligibility rules disagree at the common holdout as_of"
)

purchasers = set(
    transactions.loc[
        mask_label_events(transactions, common_holdout_window, "invoice_date"), "customer_id"
    ]
)
common_label = pd.Series(
    (~common_eligible.isin(purchasers)).astype(int), index=common_eligible, name="churned"
)
print(
    f"common holdout as_of {common_holdout_as_of:%Y-%m-%d}, "
    f"{len(common_eligible):,} eligible customers, {common_label.mean():.1%} churned"
)

# %% [markdown]
# One model is fit for arms A and B: the naive model, trained the way `01`
# trains it (full-history features, every earlier origin pooled, no
# maturity purge), on origins strictly before the common holdout `as_of`.
# The only thing that differs between A and B is which feature table scores
# it.

# %%
naive_origins = rolling_origins(
    first_as_of,
    last_event,
    gap_days=NAIVE_GAP_DAYS,
    horizon_days=HORIZON_DAYS,
    step_days=DEFAULT_STEP_DAYS,
)
naive_train_origins = [w for w in naive_origins if w.as_of < common_holdout_as_of]
naive_train_panel = build_naive_panel(transactions, naive_train_origins)

naive_model = LGBMClassifier(random_state=SEED, verbosity=-1)
naive_model.fit(naive_train_panel[feature_cols], naive_train_panel["churned"])

naive_scoring_features = naive_features(transactions).loc[common_eligible, feature_cols]
result_a = score(
    common_label.to_numpy(),
    naive_model.predict_proba(naive_scoring_features)[:, 1],
    label="A: naive trained, naive features\n(impossible to serve)",
)

asof_scoring_features = asof_features(transactions, common_holdout_as_of).loc[
    common_eligible, feature_cols
]
result_b = score(
    common_label.to_numpy(),
    naive_model.predict_proba(asof_scoring_features)[:, 1],
    label="B: naive trained, as-of features\n(honest deployment)",
)

arm_c_rows = predictions_withgap[predictions_withgap["as_of"] == common_holdout_as_of]
result_c = score(
    arm_c_rows["y_true"].to_numpy(),
    arm_c_rows["y_prob"].to_numpy(),
    label="C: correctly trained,\nas-of features",
)

arm_table = pd.DataFrame([result_a.as_row(), result_b.as_row(), result_c.as_row()]).set_index(
    "label"
)
arm_table[["n", "roc_auc", "pr_auc", "brier"]]

# %% [markdown]
# `n` for A and B should equal `len(common_eligible)`, and C's `n` should
# match it too given the assertion above. Printed rather than assumed.

# %%
serving_skew = result_a.roc_auc - result_b.roc_auc
naive_vs_correct_honest = result_b.roc_auc - result_c.roc_auc
pd.Series(
    {
        "n, common eligible population": len(common_eligible),
        "training/serving skew (A - B), ROC AUC": serving_skew,
        "B minus C, ROC AUC (negative = naive-trained underperforms, honestly served)": (
            naive_vs_correct_honest
        ),
    }
)

# %%
# Short, dedicated tick labels -- `result_*.label`'s fuller descriptive text
# (used in `arm_table` above) is too wide to sit under three side-by-side
# bars without the neighbouring labels running into each other.
arm_tick_labels = [
    "A\nnaive trained,\nnaive features",
    "B\nnaive trained,\nas-of features",
    "C\ncorrectly trained,\nas-of features",
]
arm_colours = [PALETTE["both"], PALETTE["neutral"], PALETTE["test"]]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
for ax, (col, title, higher_is_better) in zip(axes, metrics, strict=True):
    values = [getattr(result_a, col), getattr(result_b, col), getattr(result_c, col)]
    bars = ax.bar(arm_tick_labels, values, color=arm_colours)
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9
        )
    ax.set_title(
        f"{title} ({'higher' if higher_is_better else 'lower'} better)", loc="left", fontsize=10
    )
    ax.tick_params(axis="x", labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

ax0 = axes[0]
ax0.set_ylim(0, 1.04)
skew_bracket_y = 0.97
ax0.plot(
    [0, 0, 1, 1],
    [result_a.roc_auc + 0.02, skew_bracket_y, skew_bracket_y, result_b.roc_auc + 0.02],
    color=PALETTE["reference_line"],
    lw=1,
)
ax0.text(0.5, skew_bracket_y + 0.012, f"serving skew: {serving_skew:.3f}", ha="center", fontsize=9)
fig.suptitle(
    f"Naive training scores {result_a.roc_auc:.3f} ROC AUC only if it can see the future (A); "
    f"served with the features that actually exist (B), it drops to {result_b.roc_auc:.3f} — "
    f"{'below' if result_b.roc_auc < result_c.roc_auc else 'still above'} the correctly-trained "
    f"model's {result_c.roc_auc:.3f} (C)",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(PATHS.figures / "naive_serving_skew.png", dpi=200)
# fig

# %% [markdown]
# The point is not that a holdout "costs" the naive model more than the
# corrected one. The point is that **the model `01` reports cannot be served
# at all.** Its apparent skill (A) depends on a feature computed from data
# that does not exist yet at scoring time. Feed it the features a deployment
# would have (B), and what is left is a fair comparison to the correctly
# trained model (C) for the first time. A random split does not just inflate
# a number. It reports a number for a model that has no honest feature set
# to be served with.

# %% [markdown]
# ---
# ## Leakage audit
#
# `04` builds no new feature, split or label logic. The no-gap ablation
# reuses `03`'s audited `build_asof_panel` and `rolling_origin_backtest`
# with one parameter changed. The audit surface is small on purpose.
#
# | # | Component | Leak class | Evidence | Severity | Fix |
# |---|---|---|---|---|---|
# | 1 | No-gap panel and backtest | Inherited; audited in `03` | Same functions, same `first_as_of`, `last_event`, `HORIZON_DAYS`, `DEFAULT_STEP_DAYS`, `min_training_origins` and `SEED` as `03`'s call, compared call site to call site. Only `gap_days` differs (0 vs. 7) | n/a | See `03`'s audit for the maturity-check mechanics |
# | 2 | Maturity check at `gap_days=0` | Clean, boundary verified | With no gap a training origin's label matures at `as_of + 90` days, exactly three 30-day steps, at the `label_end <= test.as_of` boundary. Confirmed against the real origins that this admits origins exactly three steps back (seven test origins, vs. six with the 7-day gap pushing the floor to four steps) | n/a | n/a |
# | 3 | Seven tested origins (no gap) against six (with gap) | Disclosed, not a leak | The two arms are not scored on identical rows (30,182 vs. 25,922 predictions). Each number is honestly computed on its own mature backtest, but the 0.001 difference between them is not a paired comparison and should not be read as more precise than it is | Low; affects interpretation | Stated above |
# | 4 | Cross-notebook consistency (`01`, `03`, `04`) | Clean | All three results use `config.SEED` for every fit (ADR-0008) and the same cached Parquet | n/a | n/a |
# | 5 | `naive_features`' `recency_days` reference date at the arm-A/B holdout | Found and fixed; see ADR-0010 | `recency_days` is measured to `transactions["invoice_date"].max()` for every row, so at every origin its reference date is later than the row's `as_of`, and at this holdout later than the end of the origin's own label window. An earlier version of this section used that number as a clean forward holdout. Arm A is now kept but labelled as requiring information unavailable at scoring time | High; a general leak class, not specific to this comparison | Score with `asof_features` instead (arm B), which is what a deployment must do |
# | 6 | Arm A/B/C common `as_of` and population | Clean, verified | All three arms are scored at `common_holdout_as_of = origins_withgap[-1].as_of`; `common_eligible` is asserted equal to the customer set `panel_withgap` produced for that origin, so A, B and C share the same population and ground truth | n/a | n/a |
# | 7 | Arm A/B training set vs. arm C's | Disclosed, not a leak | A and B share one model trained the naive way (every earlier origin pooled, no purge); C comes from the purge-respecting backtest. This difference is the variable under test, "naively trained" vs. "correctly trained", stated rather than left as a confound the way the earlier draft did | n/a | n/a |
#
# One item fixed in a module rather than here: `churnval.windows.assert_no_leak`'s
# docstring referenced a nonexistent `allow_zero_gap` parameter left over
# from an earlier draft. Fixed in `windows.py`; `tests/test_windows.py` still
# passes.

# %% [markdown]
# ---
# ## Closing
#
# **Finding:** the random split overstates ROC AUC by 0.138 (0.903 vs.
# 0.765) and PR AUC by a comparable 0.149; Brier gets 56% worse in relative
# terms under the honest protocol. Almost all of that is the features and
# the split. A no-gap ablation of `03`'s otherwise correct pipeline lands at
# 0.766, within 0.001 of the fully corrected 0.765, though that ablation can
# only test the label window's boundary, since `gap_days` never reaches the
# feature cutoff in this codebase.
#
# The sharper finding came from a wrong first attempt at a holdout
# comparison. `01`'s reported number is not just inflated, it is not
# *servable*. It depends on a feature whose reference date is later than the
# `as_of` of the row it describes, later in fact than the end of that row's
# own label window, which no deployment can have at scoring time. Feeding
# the naively trained model the features a deployment would have (arm B) is
# the honest number for "what if we shipped this", and it is not obviously
# better than the correctly trained model scored the same way (arm C). See
# [ADR-0010](../docs/adr/0010-holdout-comparison-common-origin.md) for why
# the comparison is built this way.
#
# **Next:** `05_calibration` picks up where the ranking metrics leave off.
# ROC AUC and PR AUC say nothing about whether the corrected model's
# probabilities are usable for a retention decision. It fits and compares
# isotonic and Platt calibrators on a slice strictly later than training,
# and reports the expected-value threshold that follows from the calibrated
# probabilities plus stated cost assumptions.
