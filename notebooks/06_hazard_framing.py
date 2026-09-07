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
# # 06 — Hazard framing
#
# **Question this notebook answers:** does *when* a customer churns behave
# differently from *whether*? Does a discrete-time hazard model catch
# anything the binary classifier misses?
#
# **Consumes:** `03`'s pipeline directly (`churnval.features.asof_features`,
# `churnval.splits.eligible_customers`, `churnval.windows.rolling_origins`)
# plus raw transactions. The panel `03`, `04` and `05` share records only
# *whether* a customer purchased within the 90-day label window, not *when*.
# The hazard framing needs the exact date to know which period an event
# falls in, so this notebook goes back to the transaction lines rather than
# reusing that panel's label column.
#
# **Produces:** a discrete-time hazard model, a person-period expansion of
# the same eligible population with three 30-day periods per the 90-day
# horizon, scored with LightGBM (ADR-0008) through the same rolling-origin
# protocol as `03`, and compared against the binary classifier at every
# testable origin. Also a period-by-period hazard and survival curve at the
# same evaluation origin `04` and `05` used, and a retrospective diagnostic
# on what the fixed 90-day window gets wrong: customers the binary rule
# calls churned who return later. That diagnostic needs transaction history
# *after* a window closes, which the evaluation origin barely has, so it
# uses the earliest tested origin instead, disclosed rather than quietly
# swapped in.
#
# **Runtime:** several minutes. One more rolling-origin backtest, over a
# person-period panel roughly 2.5x the row count of `03`'s.
#
# ---
#
# Every notebook so far has asked *whether* a customer churns within a fixed
# 90-day window: one yes/no label, one probability.
# `docs/standards/validation.md` asks for more where "when" matters as well
# as "whether". The fixed window has a specific, nameable blind spot: a
# customer who returns on day 95 is indistinguishable, to the binary label,
# from one who never returns at all. A discrete-time hazard model does not
# remove the window, since the label is still "did they return within some
# horizon", but it estimates risk period by period rather than once, which
# makes that blind spot visible instead of absorbed into one number.

# %% [markdown]
# ## Rebuilding `03`'s with-gap protocol
#
# The same construction `03`, `04` and `05` all use.

# %%
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churnval.calibration import brier_decomposition
from churnval.config import DEFAULT_GAP_DAYS, DEFAULT_STEP_DAYS, PATHS, SEED
from churnval.evaluation import expected_value, score
from churnval.hazard import (
    PERIOD_DAYS,
    build_person_period_panel,
    fit_hazard_at_origin,
    hazard_curve,
    hazard_rolling_origin_backtest,
    survival_from_hazards,
)
from churnval.io import load_online_retail_ii
from churnval.naive_baseline import HORIZON_DAYS
from churnval.plotting import PALETTE, set_style
from churnval.splits import build_asof_panel, rolling_origin_backtest
from churnval.windows import rolling_origins

PATHS.ensure()
set_style()

transactions = load_online_retail_ii()
first_as_of = transactions["invoice_date"].min().normalize() + timedelta(days=365)
last_event = transactions["invoice_date"].max()
feature_cols = ["recency_days", "frequency", "monetary"]

origins_withgap = rolling_origins(
    first_as_of,
    last_event,
    gap_days=DEFAULT_GAP_DAYS,
    horizon_days=HORIZON_DAYS,
    step_days=DEFAULT_STEP_DAYS,
)
panel_withgap = build_asof_panel(transactions, origins_withgap)
n_periods = HORIZON_DAYS // PERIOD_DAYS
print(f"{len(origins_withgap)} origins, {len(panel_withgap):,} panel rows, {n_periods} periods")

# %% [markdown]
# ## The binary classifier, for comparison
#
# `03`'s own backtest, unchanged. This is the baseline the hazard model is
# measured against, not a new number.

# %%
per_origin_binary, predictions_binary = rolling_origin_backtest(
    panel_withgap, origins_withgap, feature_cols=feature_cols, min_training_origins=3, seed=SEED
)
per_origin_binary[["as_of", "n", "prevalence", "roc_auc", "pr_auc", "brier"]]

# %% [markdown]
# ## Building the person-period panel
#
# `build_person_period_panel` goes back to the transactions, not the
# panel's `churned` column, to find for every eligible customer at every
# origin which of the three 30-day periods their first in-window purchase
# falls in, or marks them censored through all three. Features are the
# panel's own as-of-safe values, carried unchanged across a customer's
# period rows within one origin. Nothing new is learned about a customer
# partway through their own label window, so nothing should change.

# %%
person_period_panel = build_person_period_panel(
    transactions, origins_withgap, panel_withgap, feature_cols=feature_cols, time_col="invoice_date"
)
print(
    f"{len(person_period_panel):,} person-period rows "
    f"({len(person_period_panel) / len(panel_withgap):.2f}x the original panel), "
    f"{person_period_panel['event'].mean():.1%} of rows are an event"
)

# %% [markdown]
# The aggregate count hides how much the risk set thins period to period.
# Only customers who survived period 1 contribute a period-2 row at all, and
# only survivors of period 2 contribute a period-3 row, so the training data
# for later periods is a shrinking, self-selected subset by construction.
# That is the standard discrete-time risk set, not a leak (see the audit
# below), but it matters for reading the curves later: a period with fewer
# rows behind it carries a noisier hazard estimate, and any *absolute* drop
# in survival is bounded by how many customers were still at risk to begin
# with.

# %%
person_period_panel.groupby("period").agg(n=("event", "size"), events=("event", "sum"))

# %% [markdown]
# ## The hazard model
#
# `hazard_rolling_origin_backtest` mirrors `rolling_origin_backtest` origin
# for origin, with the same maturity purge (ADR-0009), so the same origins
# should be testable under both. Checked below rather than assumed.

# %%
per_origin_hazard, predictions_hazard = hazard_rolling_origin_backtest(
    person_period_panel,
    panel_withgap,
    origins_withgap,
    feature_cols=feature_cols,
    min_training_origins=3,
    seed=SEED,
)
assert list(per_origin_hazard["as_of"]) == list(per_origin_binary["as_of"]), (
    "hazard and binary backtests tested different origins"
)
per_origin_hazard[["as_of", "n", "prevalence", "roc_auc", "pr_auc", "brier"]]

# %% [markdown]
# ## Binary vs. hazard, origin by origin
#
# `03` already showed the binary classifier's per-origin performance moves
# with prevalence. The question here is whether the hazard model tracks that
# same variation or diverges from it.

# %%
metrics = [("roc_auc", "ROC AUC", True), ("pr_auc", "PR AUC", True), ("brier", "Brier", False)]
model_colours = {"binary classifier": PALETTE["test"], "hazard model": PALETTE["train"]}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, (col, title, higher_is_better) in zip(axes, metrics, strict=True):
    ax.plot(
        per_origin_binary["as_of"],
        per_origin_binary[col],
        marker="o",
        color=model_colours["binary classifier"],
        label="binary classifier",
    )
    ax.plot(
        per_origin_hazard["as_of"],
        per_origin_hazard[col],
        marker="o",
        color=model_colours["hazard model"],
        label="hazard model",
    )
    ax.set_title(
        f"{title} ({'higher' if higher_is_better else 'lower'} better)", loc="left", fontsize=10
    )
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle(
    "The hazard model tracks the binary classifier's per-origin performance closely", fontsize=10.5
)
fig.tight_layout()
fig.savefig(PATHS.figures / "hazard_vs_binary_per_origin.png", dpi=200)
# fig

# %% [markdown]
# ## Pooled headline comparison
#
# The same convention `03` and `04` use: concatenate predictions across
# every tested origin and score once, rather than average per-origin
# metrics.

# %%
binary_pooled = score(
    predictions_binary["y_true"], predictions_binary["y_prob"], label="Binary classifier"
)
hazard_pooled = score(
    predictions_hazard["y_true"], predictions_hazard["y_prob"], label="Hazard model"
)
pooled_table = pd.DataFrame([binary_pooled.as_row(), hazard_pooled.as_row()]).set_index("label")
pooled_table[["n", "roc_auc", "pr_auc", "brier"]]

# %% [markdown]
# ## Where the risk actually is
#
# Aggregate metrics say whether the hazard model separates and calibrates as
# well as the binary one. They say nothing about the thing only the hazard
# model can answer: *when*, within the window, risk is concentrated.
# `fit_hazard_at_origin` gives the model trained for the same evaluation
# origin `04`'s A/B/C comparison and `05`'s calibration slice both used, so
# this curve sits at a familiar reference point.

# %%
evaluation_as_of = per_origin_binary["as_of"].iloc[-1]
evaluation_index = [w.as_of for w in origins_withgap].index(evaluation_as_of)
evaluation_window = origins_withgap[evaluation_index]

hazard_model, hazard_train_windows = fit_hazard_at_origin(
    person_period_panel, origins_withgap, evaluation_index, feature_cols=feature_cols, seed=SEED
)
evaluation_panel = panel_withgap[panel_withgap["as_of"] == evaluation_as_of].reset_index(drop=True)
curve = hazard_curve(hazard_model, evaluation_panel, feature_cols, n_periods)

# fit_hazard_at_origin duplicates hazard_rolling_origin_backtest's purge
# filter rather than sharing it (same deliberate choice
# churnval.splits.fit_at_origin makes) -- checked here, not just in
# tests/test_hazard.py, the same way 05_calibration.py checks
# fit_at_origin against rolling_origin_backtest's own output.
reproduced_survival = survival_from_hazards(curve)
reference_survival = (
    predictions_hazard[predictions_hazard["as_of"] == evaluation_as_of]
    .set_index("customer_id")["y_prob"]
    .reindex(reproduced_survival.index)
)
assert np.allclose(reproduced_survival.to_numpy(), reference_survival.to_numpy()), (
    "fit_hazard_at_origin doesn't reproduce hazard_rolling_origin_backtest's own model"
)
print(
    f"reproduces hazard_rolling_origin_backtest's own {evaluation_as_of.date()} predictions exactly"
)

curve = curve.sort_values(["customer_id", "period"])
curve["survival"] = curve.groupby("customer_id")["hazard"].transform(lambda h: (1 - h).cumprod())
period_summary = curve.groupby("period").agg(
    mean_hazard=("hazard", "mean"), mean_survival=("survival", "mean")
)
period_summary

# %% [markdown]
# Aggregate metrics say the two models perform comparably. They do not say
# whether the two models are *wrong the same way*. Comparing each model's
# mean predicted churn probability against the actual prevalence at this
# origin checks that directly.

# %%
actual_prevalence = evaluation_panel["churned"].mean()
binary_mean_predicted = predictions_binary.loc[
    predictions_binary["as_of"] == evaluation_as_of, "y_prob"
].mean()
hazard_mean_predicted = 1 - period_summary["mean_survival"].iloc[-1]
pd.Series(
    {
        "actual prevalence": actual_prevalence,
        "binary classifier: mean predicted P(churn)": binary_mean_predicted,
        "hazard model: mean predicted P(churn)": hazard_mean_predicted,
    }
)

# %% [markdown]
# The two models miss the base rate in *opposite* directions at this origin.
# The binary classifier overshoots by about 12 points and the hazard model
# undershoots by about 11, while their Brier scores land close together.
# "Brier is symmetric to direction" is a plausible-sounding explanation but
# it is not a check: Brier decomposes into reliability, resolution and
# uncertainty, and a global mean-offset story does not establish that the
# decomposed terms agree. `brier_decomposition`, the tool `05` used for
# exactly this kind of claim, settles it instead of inferring it.

# %%
binary_eval = predictions_binary.loc[predictions_binary["as_of"] == evaluation_as_of]
hazard_eval = predictions_hazard.loc[predictions_hazard["as_of"] == evaluation_as_of]

decomposition_table = pd.DataFrame(
    [
        {
            "model": "binary classifier",
            **brier_decomposition(binary_eval["y_true"], binary_eval["y_prob"]),
        },
        {
            "model": "hazard model",
            **brier_decomposition(hazard_eval["y_true"], hazard_eval["y_prob"]),
        },
    ]
).set_index("model")
decomposition_table

# %% [markdown]
# Resolution is nearly identical between the two models. Reliability is not
# quite the same story: the hazard model's is about 10% lower, a real if
# modest edge that the mean-offset comparison could not have shown, since a
# smaller opposite-direction offset does not by itself imply a
# better-shaped reliability curve. The close Brier scores are genuine
# agreement at the aggregate level, not two errors cancelling, but "the two
# models are equally miscalibrated" overstates it slightly in the hazard
# model's favour. Neither is well calibrated at this origin, which `03`
# already flagged as the backtest's most difficult point.
#
# `docs/standards/validation.md` asks for one more step where a decision
# follows from the prediction. Reusing `05`'s cost assumptions and its
# honestly chosen threshold (ADR-0011) makes this directly comparable to
# that notebook's finding rather than a new set of numbers.

# %%
OFFER_COST = 5.0
SAVED_MARGIN = 40.0
SAVE_RATE = 0.20
break_even_threshold = OFFER_COST / (SAVE_RATE * SAVED_MARGIN)

ev_rows = []
for name, eval_predictions in [("binary classifier", binary_eval), ("hazard model", hazard_eval)]:
    ev_rows.append(
        {
            "model": name,
            "n_targeted": int((eval_predictions["y_prob"] >= break_even_threshold).sum()),
            "expected_value": expected_value(
                eval_predictions["y_true"],
                eval_predictions["y_prob"],
                break_even_threshold,
                offer_cost=OFFER_COST,
                saved_margin=SAVED_MARGIN,
                save_rate=SAVE_RATE,
            ),
        }
    )
ev_table = pd.DataFrame(ev_rows).set_index("model")
ev_table

# %% [markdown]
# The hazard model wins here, by about 4%, while targeting fewer customers
# at the same threshold. That is a small single-origin edge, and the same
# caveat `05`'s matched-budget table raised applies: most of the difference
# comes from *how many* customers each model puts above 0.625, which follows
# from the hazard model's smaller overshoot, not from better targeting. It
# is still the kind of comparison a Brier-only readout would never surface.

# %% [markdown]
# ## From hazard to cumulative hazard to survival
#
# `period_summary`'s two columns are already two views of the same numbers.
# There is a third, standard in survival analysis, that makes the mechanism
# connecting them explicit. The **cumulative hazard** `H(t) = -ln(S(t))`,
# the quantity Nelson-Aalen estimates, accumulates monotonically. Unlike
# survival, which can only fall, cumulative hazard can only rise, and its
# slope between two periods reads directly as how much risk that period
# added. It is computed from the `period_summary` values already on the
# page, a transform of the survival curve rather than a new model output.

# %%
period_summary["cumulative_hazard"] = -np.log(period_summary["mean_survival"])
# Negate the diff *before* filling: the first period's drop is measured from
# survival 1.0, not from a previous row, so filling a NaN that is then
# negated would flip its sign.
period_summary["survival_drop"] = (-period_summary["mean_survival"].diff()).fillna(
    1 - period_summary["mean_survival"].iloc[0]
)
period_summary

# %% [markdown]
# Read the three columns together before plotting them, because two of them
# invite the same wrong reading. `survival_drop` is largest in period 1 and
# smallest in period 3, which looks like "risk is front-loaded". It is not.
# `mean_hazard`, the per-period risk *conditional on still being at risk*,
# is nearly flat across the first two periods and only modestly lower in the
# third. The absolute drop in survival shrinks because the risk set shrinks:
# the same hazard applied to fewer remaining customers removes fewer
# customers. Only the hazard column speaks to whether risk itself changes
# over the window, and it says the window's risk is close to uniform, with a
# mild taper at the end. The chart below plots all three so the distinction
# is visible rather than asserted.

# %%
period_bounds = [
    (
        evaluation_window.label_start + timedelta(days=(p - 1) * PERIOD_DAYS),
        evaluation_window.label_start + timedelta(days=p * PERIOD_DAYS),
    )
    for p in range(1, n_periods + 1)
]
period_labels = [
    f"days {(a - evaluation_window.as_of).days}-{(b - evaluation_window.as_of).days}"
    for a, b in period_bounds
]

hazard_spread = period_summary["mean_hazard"].max() - period_summary["mean_hazard"].min()

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
axes[0].bar(period_labels, period_summary["mean_hazard"], color=PALETTE["train"])
axes[0].set_ylim(0, period_summary["mean_hazard"].max() * 1.25)
axes[0].set_xlabel("period (days after as_of)")
axes[0].set_ylabel("mean predicted hazard")
axes[0].set_title("Hazard: risk in this period, given still at risk", loc="left", fontsize=10)

cumhaz_x = ["as_of", *period_labels]
cumhaz_y = [0.0, *period_summary["cumulative_hazard"]]
axes[1].plot(cumhaz_x, cumhaz_y, marker="o", color=PALETTE["train"])
axes[1].set_xlabel("period (days after as_of)")
axes[1].set_ylabel("cumulative hazard, -ln(survival)")
axes[1].set_title("Cumulative hazard: risk added up so far", loc="left", fontsize=10)

survival_x = ["as_of", *period_labels]
survival_y = [1.0, *period_summary["mean_survival"]]
axes[2].plot(survival_x, survival_y, marker="o", color=PALETTE["train"])
axes[2].set_ylim(0, 1.02)
axes[2].set_xlabel("period (days after as_of)")
axes[2].set_ylabel("mean predicted survival (still active)")
axes[2].set_title("Survival: what's left after that risk", loc="left", fontsize=10)

for ax in axes:
    ax.tick_params(axis="x", labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.suptitle(
    f"Hazard is near-flat across the window (spread {hazard_spread:.3f}) -- survival falls "
    "fastest early only because the risk set is largest early",
    fontsize=10.5,
)
fig.tight_layout()
fig.savefig(PATHS.figures / "hazard_curve.png", dpi=200)
# fig

# %% [markdown]
# The cumulative hazard curve is close to a straight line, which is the same
# statement as "hazard is near-flat" and the cleanest way to see it: a
# constant slope means each period adds the same increment of risk. If risk
# really were front-loaded, that curve would be visibly concave. This is the
# reading a hazard model is built to support, at what point has enough risk
# accumulated to act on this customer, and here the answer is that no
# particular point inside the window stands out. That is a genuine finding
# about this dataset, and `07` shows a contractual dataset where the same
# code produces a very different shape.

# %% [markdown]
# ## What the fixed window gets wrong
#
# `churned = 1` means "no purchase observed within the window", not "gone
# for good". A customer who returns on day 95 is filed identically to one
# who never returns. This section checks retrospectively how often that
# happens: for customers the binary rule calls churned at one origin, does
# the full transaction history show them purchasing again shortly after the
# window closes?
#
# This needs transaction history *after* the window, which a real deployment
# would not have at scoring time. It is a diagnostic on the label's design
# using hindsight this dataset happens to contain, not a feature, a
# prediction, or anything fed to either model above. The evaluation origin
# used elsewhere in this notebook is a bad fit for it: being deliberately
# the latest testable origin, the dataset barely extends past its own window.
# The *earliest* tested origin is used instead, purely because enough
# history exists after its window to make the question answerable.

# %%
earliest_as_of = per_origin_binary["as_of"].iloc[0]
earliest_index = [w.as_of for w in origins_withgap].index(earliest_as_of)
earliest_window = origins_withgap[earliest_index]
slack_days = (last_event - earliest_window.label_end).days
evaluation_slack_days = (last_event - evaluation_window.label_end).days
print(
    f"evaluation origin {evaluation_as_of.date()}: only {evaluation_slack_days} days of "
    f"history past its window -- unusable for this diagnostic\n"
    f"earliest tested origin {earliest_as_of.date()}: {slack_days} days past its window -- used instead"
)

# %%
churned_customers = panel_withgap.loc[
    (panel_withgap["as_of"] == earliest_as_of) & (panel_withgap["churned"] == 1), "customer_id"
]
after_window = transactions[
    (transactions["invoice_date"] >= earliest_window.label_end)
    & transactions["customer_id"].isin(churned_customers)
]
next_purchase = after_window.groupby("customer_id")["invoice_date"].min()
days_to_return = (next_purchase - earliest_window.label_end).dt.days

n_churned = len(churned_customers)
n_returned = len(days_to_return)
print(
    f"{n_churned:,} customers labeled churned at {earliest_as_of.date()}; "
    f"{n_returned:,} ({n_returned / n_churned:.1%}) purchase again within the "
    f"observable {slack_days}-day tail after their window closes"
)

# %% [markdown]
# Two numbers from that distribution, computed rather than read off the
# histogram below: how many returners come back in the first fortnight, and
# how many come back after a further full horizon.

# %%
pd.Series(
    {
        "returners within 14 days of the window closing": int((days_to_return <= 14).sum()),
        "returners after a further 90 days": int((days_to_return > HORIZON_DAYS).sum()),
        "median days to return": float(days_to_return.median()),
    }
)

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(days_to_return, bins=30, color=PALETTE["churned"])
ax.axvline(HORIZON_DAYS, color=PALETTE["reference_line"], lw=1, ls="--")
ax.text(
    HORIZON_DAYS + 3, ax.get_ylim()[1] * 0.95, "another full horizon later", fontsize=8, va="top"
)
ax.set_xlabel("days after the window closed until the next purchase")
ax.set_ylabel("count of customers labeled churned")
ax.set_title(
    f"{n_returned / n_churned:.0%} of 'churned' customers at this origin purchase again anyway",
    loc="left",
    fontsize=10,
)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(PATHS.figures / "late_returners.png", dpi=200)
# fig

# %% [markdown]
# Two things in that histogram, not one. The tallest bar sits at the left
# edge: customers who purchase within days of the window closing, exactly
# the "day 95 versus day 90" blind spot this notebook opened with, now a
# real and sizeable cluster rather than a hypothetical. But returns do not
# decay to nothing after that spike. They dip and then persist for months,
# and the 90-day mark itself is not a turning point in the data.
#
# Both facts say the same thing. The fixed window's cutoff is a modelling
# convenience, not a boundary in customer behaviour. Some "churned"
# customers are on a slower purchase cycle the window is too short to see.
# Note this is right-censored at the observable tail, so the true fraction
# of eventual returners is *at least* the number above, not exactly it. It
# is also one specific cohort, customers scored at the earliest tested
# origin, chosen for its long tail. The rate is not claimed to hold for the
# later, lower-prevalence cohort; what generalises is that the blind spot
# exists structurally wherever a fixed window is used.

# %% [markdown]
# ## What the hazard framing buys
#
# Not better aggregate numbers. Pooled ROC AUC, PR AUC and Brier are all
# within a few thousandths of the binary classifier's, origin by origin as
# well as pooled, and that is the result to expect: both models see the same
# features, and reshaping the same event into person-periods hands the
# hazard model no information the binary classifier lacked.
#
# What it buys is what the binary classifier's single number cannot say. It
# says risk within this window is close to uniform rather than concentrated,
# a claim the binary label cannot express at all and one that turns out to
# be dataset-specific rather than universal. It exposed a real difference in
# *how* the two models are wrong at the hardest regime, overshooting versus
# undershooting the true rate, invisible to a Brier-only comparison. And its
# person-period construction is what made the late-returner check easy to
# frame: "which period would this late purchase have landed in" is a
# question the hazard structure asks naturally, while "was 90 days the right
# cutoff" is a question a single binary label cannot ask of itself.

# %% [markdown]
# ---
# ## Leakage audit
#
# `06` builds one genuinely new label structure, the person-period
# expansion, on top of `03`'s audited features and eligibility. The audit
# surface is that expansion, the hazard backtest's purge, and the
# retrospective diagnostic's use of history no model above sees.
#
# | # | Component | Leak class | Evidence | Severity | Fix |
# |---|---|---|---|---|---|
# | 1 | `asof_features`, `eligible_customers`, `rolling_origins` | Inherited; audited in `03` | Called unchanged. `build_person_period_panel` reads the panel's already as-of-safe feature columns and copies them across period rows without recomputing anything | n/a | n/a |
# | 2 | Features vs. label-window timing | Clean | Each customer's feature values are taken from the panel once per origin and repeated unchanged across their period rows. No feature is a function of anything inside the label window, including the event date | n/a | n/a |
# | 3 | `period` as a model feature: does row truncation leak the outcome? | Clean, reasoned explicitly | Training rows for period `p` are by construction only customers still at risk at the start of period `p`, the standard discrete-time risk set (Singer & Willett, 1993). That is what makes the output a *hazard*. The leak this could be is the model knowing in advance which period a test customer will reach, and it cannot: `hazard_curve` scores every customer at every period from 1 to `n_periods` regardless of their real outcome. `period` only lets the model learn a period-varying baseline rate, a population-level pattern | n/a | n/a |
# | 4 | Maturity purge uses the origin's `label_end`, not a period boundary | Clean, deliberately | A censored customer's period-1 row is not confirmed until the *entire* 90-day window has closed without a purchase. `hazard_rolling_origin_backtest` purges on `w.label_end <= test_window.as_of`, the same full-window boundary `rolling_origin_backtest` uses, not a shorter period-level one that would admit training rows before they were knowable | n/a | n/a |
# | 5 | Same origins testable in both backtests | Clean, but the check is weaker than it looks | The assertion passed. Both purge filters are identical origin-level logic applied to the same origins, so agreement is close to guaranteed by construction rather than independent confirmation. It would catch a typo or off-by-one, not a bug leaving an origin's person-period rows empty without changing which origins are attempted (in practice `LGBMClassifier.fit()` on zero rows raises immediately) | Low; informational | n/a |
# | 6 | `fit_hazard_at_origin`'s duplicated purge filter | Clean, cross-checked twice | Same deliberate duplication `churnval.splits.fit_at_origin` uses. Verified in `tests/test_hazard.py` and directly in this notebook (`reproduced_survival` vs. `reference_survival`, exact match), matching the pattern `05` uses | n/a | n/a |
# | 7 | Retrospective late-returner diagnostic | Clean, isolated from both models | Uses `earliest_as_of` and `earliest_window`, distinct from the evaluation variables used elsewhere. Nothing computed in that section feeds back into the person-period panel, either backtest, or the hazard model, all of which are fully built before it runs. It looks past both models' windows on purpose and says so | n/a | n/a |
# | 8 | Single-origin basis of the miscalibration and hazard-curve findings | Found by the `validation-reviewer` agent, disclosed | Unlike the pooled comparison and the per-origin chart (six origins each), the miscalibration-direction finding, its Brier and EV follow-up, and the period-by-period hazard curve are all computed at one origin. Nothing here confirms the pattern holds at any other tested origin. `05` disclosed the identical limitation for its own single-origin-pair comparison | Medium; informational, does not change the pooled numbers | Not fixable within this dataset's window sizes; `07` should check whether the pattern replicates |

# %% [markdown]
# ---
# ## Closing
#
# **Finding:** a discrete-time hazard model, built on the same as-of-safe
# features and the same rolling-origin protocol `03` validated, matches the
# binary classifier's ranking and calibration almost exactly. Pooled and
# origin by origin, every gap is a few thousandths of ROC AUC, PR AUC or
# Brier. Reframing "whether" as "when" does not buy a better aggregate
# number, and there was no reason to expect it would: both models see the
# same information, organised differently.
#
# What it buys is what a single number cannot show. Within the 90-day
# window, hazard is close to flat: the model finds no period where risk
# concentrates. That is worth stating carefully, because the survival curve
# and the raw drop in survivors both *look* front-loaded, and are not. They
# fall fastest early because the risk set is largest early. The near-straight
# cumulative-hazard line is the honest read.
#
# At this project's hardest-prevalence origin the two models turned out to
# be miscalibrated in opposite directions, comparably in magnitude but not
# identically, worth a modest edge in expected value once priced through a
# targeting decision. And the person-period structure made a real flaw in
# the fixed 90-day window checkable: a large share of customers one origin
# calls churned purchase again anyway, a cluster within days of the window
# closing and a trickle for months after. The window is a modelling
# convenience, not a boundary in customer behaviour.
#
# **Next:** every notebook so far has worked on Online Retail II's
# non-contractual churn, where the event has to be inferred from purchase
# gaps. `07_generalisation` scales the same protocol to KKBox's contractual
# subscription churn, where the event is directly observed, at about 25x the
# row count. It closes with the portable checklist.
