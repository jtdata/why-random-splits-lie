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
# # 05 — Calibration
#
# **Question this notebook answers:** are the corrected model's predicted
# probabilities usable for a retention decision, or only well ranked?
#
# **Consumes:** `03`'s pipeline directly (`churnval.splits`,
# `churnval.windows`, `churnval.features`, `churnval.io`), not `04`'s saved
# JSON. Calibration needs a genuine three-way temporal split (train the
# classifier, fit the calibrator, evaluate) that a pooled ranking number
# does not carry. Online Retail II's with-gap protocol has six tested
# origins after `03`'s maturity purge (ADR-0009). This notebook uses the
# mature origins through the fifth to train the classifier, the fifth origin
# itself to fit each calibrator, and the sixth (the origin `04`'s A/B/C
# comparison used) to evaluate, so no origin plays two roles.
#
# **Produces:** reliability curves and a Brier decomposition for the
# uncalibrated model against Platt and isotonic calibration; a choice
# between the two with a stated reason; and an expected-value comparison
# under explicit retention-economics assumptions, at a threshold derived
# from those assumptions rather than searched for.
#
# **Runtime:** a few minutes. One classifier fit, two cheap calibrator fits,
# and a threshold sweep over already-computed probabilities.
#
# ---
#
# `04` reported ROC AUC and PR AUC and stopped there: a ranking metric,
# twice. `docs/standards/validation.md` is explicit that this is not enough,
# and the reason is worth stating rather than citing. A ranking metric is
# invariant to any monotonic transformation of the scores. Multiply every
# predicted probability by 0.5 and ROC AUC does not move at all, but every
# threshold rule built on those probabilities now targets a different set of
# customers. A model that separates churners from active customers well can
# still be systematically overconfident or timid, and a retention programme
# acts on the miscalibrated number, not the ranking. This notebook checks
# whether that gap exists here and what it costs in a targeting decision.

# %% [markdown]
# ## Rebuilding `03`'s with-gap protocol
#
# The same construction `03` and `04` both use. Rebuilt from the modules,
# not imported from either notebook, since notebooks do not import each
# other.

# %%
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churnval.calibration import brier_decomposition, fit_isotonic, fit_platt, reliability_table
from churnval.config import DEFAULT_GAP_DAYS, DEFAULT_STEP_DAYS, PATHS, SEED
from churnval.evaluation import expected_value, score, sweep_expected_value
from churnval.io import load_online_retail_ii
from churnval.naive_baseline import HORIZON_DAYS
from churnval.plotting import PALETTE, set_style
from churnval.splits import (
    build_asof_panel,
    fit_at_origin,
    rolling_origin_backtest,
)
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
print(f"{len(origins_withgap)} origins, {len(panel_withgap):,} panel rows")

# %% [markdown]
# ## Choosing the calibration and evaluation origins
#
# `rolling_origin_backtest` identifies which origins are testable under the
# maturity purge: six, for this protocol. The second-to-last becomes the
# calibration slice and the last the evaluation slice, both taken from this
# table rather than recomputed by hand, so "which six origins are testable"
# has one source of truth here.

# %%
per_origin_withgap, predictions_withgap = rolling_origin_backtest(
    panel_withgap, origins_withgap, feature_cols=feature_cols, min_training_origins=3, seed=SEED
)
calibration_as_of = per_origin_withgap["as_of"].iloc[-2]
evaluation_as_of = per_origin_withgap["as_of"].iloc[-1]
origin_as_ofs = [w.as_of for w in origins_withgap]
calibration_index = origin_as_ofs.index(calibration_as_of)
evaluation_index = origin_as_ofs.index(evaluation_as_of)
print(
    f"{len(per_origin_withgap)} testable origins: "
    f"{[d.date() for d in per_origin_withgap['as_of']]}\n"
    f"calibration slice: {calibration_as_of.date()} (index {calibration_index})\n"
    f"evaluation slice:  {evaluation_as_of.date()} (index {evaluation_index}, "
    "the same origin 04's A/B/C comparison used)"
)

# %% [markdown]
# ## Training the classifier once, through the calibration origin
#
# `fit_at_origin` gives the exact model `rolling_origin_backtest` would use
# to test the calibration origin: trained on every mature origin strictly
# before it, same purge rule, same seed. That model is used twice, scored on
# the calibration origin's own rows and then, **without retraining**, on the
# evaluation origin's rows. Not retraining through the calibration origin is
# deliberate. Folding it into training would leave nothing temporally later
# than training to fit a calibrator on, which is the leak
# `docs/standards/validation.md` rules out. The cost is one origin's worth of
# training data relative to what `04`'s arm C had for the same evaluation
# origin, so ranking performance here will not reproduce `04`'s numbers
# exactly.
#
# Before trusting it, the model is checked against the backtest's own
# predictions for the calibration origin. Same purge, same data, same seed,
# so it should reproduce them exactly.

# %%
model, mature_train_windows = fit_at_origin(
    panel_withgap, origins_withgap, calibration_index, feature_cols=feature_cols, seed=SEED
)
print(
    f"trained on {len(mature_train_windows)} mature origins: "
    f"{[w.as_of.date() for w in mature_train_windows]}"
)

calibration_rows = panel_withgap[panel_withgap["as_of"] == calibration_as_of].sort_values(
    "customer_id"
)
reproduced = model.predict_proba(calibration_rows[feature_cols])[:, 1]
reference = (
    predictions_withgap[predictions_withgap["as_of"] == calibration_as_of]
    .sort_values("customer_id")["y_prob"]
    .to_numpy()
)
assert np.allclose(reproduced, reference), (
    "fit_at_origin doesn't reproduce the backtest's own model"
)
print(f"reproduces rolling_origin_backtest's own {calibration_as_of.date()} predictions exactly")

# %% [markdown]
# ## The three slices
#
# `raw_prob` on the calibration origin is what each calibrator fits against.
# `raw_prob` on the evaluation origin, and its two calibrated versions, are
# what gets compared. Nothing from the evaluation origin is touched until
# the comparison itself.

# %%
evaluation_rows = panel_withgap[panel_withgap["as_of"] == evaluation_as_of].sort_values(
    "customer_id"
)

calibration_raw_prob = reproduced
calibration_y_true = calibration_rows["churned"].to_numpy()

evaluation_raw_prob = model.predict_proba(evaluation_rows[feature_cols])[:, 1]
evaluation_y_true = evaluation_rows["churned"].to_numpy()

print(
    f"calibration slice: n={len(calibration_y_true):,}, "
    f"churn rate {calibration_y_true.mean():.1%}\n"
    f"evaluation slice:  n={len(evaluation_y_true):,}, "
    f"churn rate {evaluation_y_true.mean():.1%}"
)

# %% [markdown]
# The two slices are 30 days apart, not two independent draws of customers.
# Worth measuring rather than leaving implicit, since it bears on what the
# comparison can claim.

# %%
calibration_customers = set(calibration_rows["customer_id"])
evaluation_customers = set(evaluation_rows["customer_id"])
overlap = calibration_customers & evaluation_customers
print(
    f"{len(overlap):,} of {len(evaluation_customers):,} evaluation customers "
    f"({len(overlap) / len(evaluation_customers):.1%}) were also in the calibration slice"
)

# %% [markdown]
# That overlap is nearly total. Most of the evaluation slice is the same
# people the calibrator was fit on, a month later. That is not a temporal
# leak in the sense `03` settled (every row's features are still computed
# strictly before its own `as_of`), but it narrows what the numbers below
# claim: the calibration holds for *this population a month later*, not that
# it generalises to customers the calibrator has never seen. A
# new-customer evaluation would need an origin pair with less eligibility
# overlap, which this dataset's six-origin backtest does not offer.
#
# ## Fitting the calibrators
#
# Both fit on the calibration slice only, never on training data and never
# on the evaluation slice. Platt scaling (`churnval.calibration.fit_platt`)
# is a single-feature logistic regression: it can only stretch or compress
# the probability scale monotonically, correcting systematic over- or
# under-confidence but not a non-monotonic pattern. Isotonic regression
# (`fit_isotonic`) fits an arbitrary monotonic step function, more flexible
# but hungrier for data. With a calibration slice this size, data scarcity
# is unlikely to decide it either way, which the comparison below checks.

# %%
platt_model = fit_platt(calibration_raw_prob, calibration_y_true)
isotonic_model = fit_isotonic(calibration_raw_prob, calibration_y_true)

evaluation_platt_prob = platt_model.predict_proba(evaluation_raw_prob.reshape(-1, 1))[:, 1]
evaluation_isotonic_prob = isotonic_model.predict(evaluation_raw_prob)

variants = {
    "uncalibrated": evaluation_raw_prob,
    "Platt": evaluation_platt_prob,
    "isotonic": evaluation_isotonic_prob,
}
variant_colours = {
    "uncalibrated": PALETTE["neutral"],
    "Platt": PALETTE["train"],
    "isotonic": PALETTE["test"],
}

# %% [markdown]
# ## Reliability curves
#
# Each point is one bin of the evaluation slice: mean predicted probability
# on the x axis, observed churn rate among that bin's rows on the y axis. A
# perfectly calibrated model sits on the diagonal in every bin. A point
# below the diagonal means the model predicted more churn than actually
# happened in that bin, which is overconfidence.

# %%
fig, ax = plt.subplots(figsize=(7, 6.5))
ax.plot([0, 1], [0, 1], color=PALETTE["reference_line"], lw=1, ls="--", label="perfect calibration")
for name, prob in variants.items():
    table = reliability_table(evaluation_y_true, prob, n_bins=10)
    ax.plot(
        table["mean_predicted"],
        table["observed_rate"],
        marker="o",
        color=variant_colours[name],
        label=name,
    )
ax.set_xlabel("mean predicted probability (evaluation slice, per bin)")
ax.set_ylabel("observed churn rate (evaluation slice, per bin)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title(
    "Reliability: how far each calibration variant's predictions sit from the diagonal",
    loc="left",
    fontsize=10,
)
ax.legend(frameon=False, fontsize=9, loc="upper left")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(PATHS.figures / "reliability_curves.png", dpi=200)
# fig

# %% [markdown]
# The bin-level numbers behind that curve, rather than left to be read off
# the chart. The signed gap is `mean_predicted - observed_rate`: positive
# means the model over-predicts churn in that bin.

# %%
reliability_detail = reliability_table(evaluation_y_true, evaluation_raw_prob, n_bins=10).assign(
    signed_gap=lambda t: t["mean_predicted"] - t["observed_rate"]
)
reliability_detail

# %% [markdown]
# ## Brier decomposition
#
# `churnval.calibration.brier_decomposition` (Murphy, 1973) splits the
# binned Brier score into reliability (lower is better, the distance from
# the diagonal above), resolution (higher is better, how much the model's
# bins separate from the base rate), and uncertainty (fixed by the
# evaluation slice's own prevalence, identical across all three variants).
# The raw per-instance Brier score from `churnval.evaluation.score` is shown
# alongside; see `brier_decomposition`'s docstring for why the two are not
# expected to match exactly.

# %%
decomposition_rows = []
for name, prob in variants.items():
    decomp = brier_decomposition(evaluation_y_true, prob, n_bins=10)
    raw_result = score(evaluation_y_true, prob, label=name)
    decomposition_rows.append({"variant": name, "brier_raw": raw_result.brier, **decomp})
decomposition_table = pd.DataFrame(decomposition_rows).set_index("variant")
decomposition_table

# %% [markdown]
# ## Choosing a calibrator
#
# The winner is whichever has the lower reliability term, computed rather
# than assumed. The loser is not discarded: both feed the expected-value
# comparison below, since a calibrator that looks slightly worse on
# reliability can still change the decision, and that is a separate
# question from which one is better calibrated.

# %%
chosen = decomposition_table["reliability"].idxmin()
print(
    f"lower reliability term: {chosen} "
    f"({decomposition_table.loc[chosen, 'reliability']:.5f} vs. "
    f"{decomposition_table.drop(chosen)['reliability'].min():.5f} for the other calibrator)"
)

# %% [markdown]
# `n_bins=10` is a common default (`sklearn.calibration.calibration_curve`
# uses it too), not a value tuned to produce this result. Checked by
# recomputing the reliability term across several bin counts.

# %%
bin_sensitivity = pd.DataFrame(
    {
        n_bins: {
            name: brier_decomposition(evaluation_y_true, prob, n_bins=n_bins)["reliability"]
            for name, prob in variants.items()
        }
        for n_bins in (5, 10, 15, 20, 30)
    }
).T
bin_sensitivity.index.name = "n_bins"
bin_sensitivity["winner"] = bin_sensitivity[["uncalibrated", "Platt", "isotonic"]].idxmin(axis=1)
bin_sensitivity

# %% [markdown]
# ## What a retention decision costs
#
# A ranking or calibration metric does not say what to *do*. Turning a
# probability into a decision needs a threshold, and the threshold that
# maximises expected value depends on three cost assumptions this dataset
# cannot supply. Online Retail II has no recorded retention campaign to fit
# them from, so they are illustrative:
#
# - **`offer_cost = $5`**, a discount code or retention email, spent on
#   everyone above the threshold regardless of outcome.
# - **`saved_margin = $40`**, near-term margin retained when an offer
#   actually prevents a churn.
# - **`save_rate = 0.20`**, the fraction of genuinely churning targeted
#   customers the offer saves. `churnval.evaluation.expected_value`'s
#   docstring calls this "the number nobody measures and everybody assumes".
#   It is assumed here too, at a deliberately conservative value.
#
# The same three assumptions apply to all three probability variants. Only
# the probabilities used to decide who gets targeted change.

# %%
OFFER_COST = 5.0
SAVED_MARGIN = 40.0
SAVE_RATE = 0.20

thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
sweeps = {
    name: sweep_expected_value(
        evaluation_y_true,
        prob,
        thresholds,
        offer_cost=OFFER_COST,
        saved_margin=SAVED_MARGIN,
        save_rate=SAVE_RATE,
    )
    for name, prob in variants.items()
}

fig, ax = plt.subplots(figsize=(8, 6))
for name, sweep in sweeps.items():
    ax.plot(sweep["threshold"], sweep["expected_value"], color=variant_colours[name], label=name)
ax.set_xlabel("targeting threshold (predicted churn probability)")
ax.set_ylabel(
    f"expected value ($, offer_cost=${OFFER_COST:.0f}, saved_margin=${SAVED_MARGIN:.0f}, save_rate={SAVE_RATE:.0%})"
)
ax.set_title(
    "Expected value at every threshold\n(descriptive only -- the reported number comes from a fixed threshold below)",
    loc="left",
    fontsize=10,
)
ax.legend(frameon=False, fontsize=9)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(PATHS.figures / "ev_threshold_sweep.png", dpi=200)
# fig

# %% [markdown]
# ## A tempting read of that chart, and why it is wrong
#
# It is tempting to read the EV-optimal decision off the chart by taking
# each curve's own peak, and doing that would say the uncalibrated model
# wins. That is the wrong number to report, and an adversarial review of
# this notebook is what caught it. Each curve's peak is chosen by searching
# thresholds *against the evaluation slice's own labels* and reporting the
# best one found: the same "pick the best of N options on the test set,
# quote that as the result" bias a random train/test split produces.
# `docs/standards/validation.md` asks for expected value "at the chosen
# threshold", which only means something if the threshold is chosen before
# looking at the outcomes being scored.

# %%
peeked_optimum = {
    name: sweep.loc[sweep["expected_value"].idxmax(), ["threshold", "expected_value"]].to_dict()
    for name, sweep in sweeps.items()
}
pd.DataFrame(peeked_optimum).T

# %% [markdown]
# ## The threshold, chosen honestly
#
# The stated costs imply a threshold on their own, with no peeking at any
# outcome. Contacting a customer with true churn probability `p` has
# expected benefit `p * save_rate * saved_margin` and a fixed cost
# `offer_cost`, so the break-even rule is to target when
# `p > offer_cost / (save_rate * saved_margin)`, the standard Bayes decision
# threshold. That is computable from the three assumptions alone, before the
# evaluation slice's labels are touched.
#
# This is also exactly where calibration should matter. The rule is a claim
# about what `p` *means*, not about how customers are ordered. A model whose
# probabilities do not mean what they say can rank customers perfectly and
# still get this rule wrong.

# %%
break_even_threshold = OFFER_COST / (SAVE_RATE * SAVED_MARGIN)
print(f"break-even threshold: {break_even_threshold:.3f}")

fixed_threshold_rows = []
for name, prob in variants.items():
    ev = expected_value(
        evaluation_y_true,
        prob,
        break_even_threshold,
        offer_cost=OFFER_COST,
        saved_margin=SAVED_MARGIN,
        save_rate=SAVE_RATE,
    )
    fixed_threshold_rows.append(
        {
            "variant": name,
            "n_targeted": int((prob >= break_even_threshold).sum()),
            "expected_value": ev,
        }
    )
fixed_threshold_table = pd.DataFrame(fixed_threshold_rows).set_index("variant")
fixed_threshold_table

# %% [markdown]
# ## Does the decision change?
#
# It does, and by a lot, in the opposite direction from the peeked-at
# chart. At the same cost-derived threshold the uncalibrated model targets
# far more customers than either calibrated version and earns far less. Its
# raw probabilities are systematically overconfident through the middle of
# the range (the signed gaps in `reliability_detail` above are positive
# across the middle bins), so a cutoff at 0.625 catches customers whose true
# churn probability is nowhere near 62.5%, and the offer is spent on them
# for nothing.
#
# This is not in tension with the ranking check below. It is the same fact
# from the other side. Platt scaling is strictly monotonic
# (`sigmoid(coef * raw_prob + intercept)` with a fitted `coef > 0`, checked
# directly rather than observed on one sample), so it reorders nothing and
# earns no ranking advantage: the top-`N` customers by raw probability and
# by Platt probability are identical for any `N`. But a *fixed threshold
# value* is not a fixed `N`. Rescaling the axis changes how many customers
# clear a fixed cutoff, and that is the whole mechanism behind the table
# above.

# %%
rank_raw = evaluation_raw_prob.argsort().argsort()
rank_platt = evaluation_platt_prob.argsort().argsort()
rank_isotonic = evaluation_isotonic_prob.argsort().argsort()

top_n = fixed_threshold_table.loc["uncalibrated", "n_targeted"]
raw_top_k = set(np.argsort(-evaluation_raw_prob)[:top_n])
platt_top_k = set(np.argsort(-evaluation_platt_prob)[:top_n])
pd.Series(
    {
        "Platt coefficient (must be > 0 for strict monotonicity)": float(platt_model.coef_[0, 0]),
        "raw vs. Platt: identical customer ranking, every N": bool((rank_raw == rank_platt).all()),
        "raw vs. isotonic: identical customer ranking": bool((rank_raw == rank_isotonic).all()),
        "raw vs. Platt: same customers in uncalibrated's own top-N": raw_top_k == platt_top_k,
    }
)

# %% [markdown]
# ### Separating the two effects: EV at a matched budget
#
# The table above compares three variants that each target a different
# number of customers, so the EV differences mix two things: who gets
# picked, and how many. Holding the budget fixed separates them. Targeting
# the top `N` customers by each variant's own score, for the same `N`,
# removes the threshold effect entirely and leaves only ranking. If
# calibration buys nothing for ranking, as the monotonicity check above
# says, these curves should sit on top of each other for Platt and
# uncalibrated.

# %%
budget_rows = []
for budget in (500, 1000, 1500, 2000, 2500, 3000):
    for name, prob in variants.items():
        chosen_idx = np.argsort(-prob)[:budget]
        churners = int(evaluation_y_true[chosen_idx].sum())
        budget_rows.append(
            {
                "budget (customers targeted)": budget,
                "variant": name,
                "expected_value": churners * SAVE_RATE * SAVED_MARGIN - budget * OFFER_COST,
            }
        )
budget_table = (
    pd.DataFrame(budget_rows)
    .pivot_table(index="budget (customers targeted)", columns="variant", values="expected_value")
    .loc[:, ["uncalibrated", "Platt", "isotonic"]]
)
budget_table

# %% [markdown]
# At a matched budget the uncalibrated and Platt columns are identical, as
# strict monotonicity requires, and isotonic differs only slightly, from
# ties in its step function reshuffling customers within a step. So the
# entire EV advantage in the fixed-threshold table comes from *how many*
# customers each variant targets, not from targeting better ones. That is
# the honest statement of what calibration bought here: it did not improve
# the model, it made a cost-derived threshold mean what it claims to mean.
# It is also a warning about the headline. If a retention programme sets its
# budget by headcount rather than by probability cutoff, calibration buys
# nothing at all.

# %%
results_path = PATHS.reports / "results_calibration.json"
results_path.write_text(
    pd.concat([decomposition_table.add_prefix("brier_"), fixed_threshold_table], axis=1).to_json(
        orient="index", indent=2
    )
)
ev_spread = (
    fixed_threshold_table["expected_value"].max() - fixed_threshold_table["expected_value"].min()
)
print(
    f"expected-value spread across variants at the honestly-chosen threshold: ${ev_spread:,.0f} "
    f"({fixed_threshold_table['expected_value'].idxmax()} highest, "
    f"{fixed_threshold_table['expected_value'].idxmin()} lowest)"
)

# %% [markdown]
# ---
# ## Leakage audit
#
# Covers this notebook's new surface: the three-way temporal split,
# `fit_at_origin`, and `churnval.calibration`. `05` builds no new feature,
# split or label logic; `build_asof_panel`, `eligible_customers` and
# `asof_features` are `03`'s, unchanged and audited there.
#
# | # | Component | Leak class | Evidence | Severity | Fix |
# |---|---|---|---|---|---|
# | 1 | `build_asof_panel`, `eligible_customers`, `asof_features` | Inherited; audited in `03` | Called unchanged, same functions, same `feature_cols` | n/a | n/a |
# | 2 | Train / calibrate / evaluate ordering | Clean for per-row temporal safety; population overlap disclosed | The classifier trains only on origins whose label window closes before the calibration origin's `as_of`; no origin plays two roles; every row's features are computed strictly before its own `as_of`. But the calibration and evaluation slices share 97.1% of customers (4,198 of 4,323, computed above), so the numbers show the calibration holds for the same population a month later, not that it generalises to new customers | Medium; narrows the claim rather than invalidating it | Disclosed above; a new-customer evaluation needs an origin pair this six-origin backtest does not offer |
# | 3 | Calibrator fit isolation | Clean | `fit_platt` and `fit_isotonic` are called only with the calibration origin's own rows. `evaluation_y_true` is never passed to a `.fit()` call anywhere in this notebook, only to `score`, `brier_decomposition`, `sweep_expected_value` and `expected_value`, none of which fit anything | n/a | n/a |
# | 4 | Label maturity, evaluation origin | Clean, verified numerically | The evaluation origin's label window is `[2011-09-04, 2011-12-03)`; the dataset's last event is `2011-12-09 12:50`, six days after it closes. Computed from `transactions["invoice_date"].max()`, not assumed from `rolling_origins`' internal guarantee | n/a | n/a |
# | 5 | `fit_at_origin`'s duplicated purge filter | Clean; the duplication is a disclosed maintenance risk | The one-line purge filter is copied from `rolling_origin_backtest` rather than shared, so the two could drift. The notebook's cross-check is a real test, not a vacuous one: a LightGBM model refit on even a slightly different training set gives detectably different output, so `np.allclose` passing is strong evidence the two rules agree | Low | Also covered by `tests/test_splits.py::test_fit_at_origin_matches_rolling_origin_backtests_own_purge_and_predictions` |
# | 6 | Bin count (`n_bins=10`) | Clean, robustness verified | Recomputed at `n_bins` in {5, 10, 15, 20, 30} (`bin_sensitivity` above). Both calibrators beat uncalibrated at every bin count tested, so the choice is not an artefact of the default | n/a | n/a |
# | 7 | Isotonic's coarse output | Disclosed, not a leak | Isotonic's step-function form produces far fewer distinct values than the raw probabilities on a calibration slice this size, which is why its ranking, unlike Platt's, is not identical to the uncalibrated model's | Low | Stated above |
# | 8 | EV threshold chosen by argmax over the evaluation slice's own labels | Found by the `validation-reviewer` agent, fixed | An earlier version picked each variant's threshold by searching the sweep for the highest expected value on the evaluation slice itself. That is in-sample selection bias, best-of-19 grid points instead of best-of-one-split. Fixed by deriving the threshold from the stated costs before touching `evaluation_y_true`; see [ADR-0011](../docs/adr/0011-calibration-split-and-honest-ev-threshold.md). The peeked-at and honest numbers disagree sharply | High; the peeked-at numbers said calibration made the decision worse, and the honest numbers say the opposite | Fixed here; `sweep_expected_value` is unchanged and still used for the descriptive chart |
# | 9 | The comparison rests on one origin pair | Disclosed, not fixable in this dataset | Unlike `03`/`04`'s six-origin ranking metrics, the calibration and EV numbers come from exactly one calibration/evaluation pair. Five origins are needed for training, leaving one pair to spare from the six the purge allows | Medium; no variance estimate across regimes for the calibration finding | Not fixable within Online Retail II's window sizes; `07`'s KKBox scale-up has more origins |

# %% [markdown]
# ---
# ## Closing
#
# **Finding:** `03`'s corrected model is meaningfully miscalibrated out of
# the box. Its reliability term is roughly four to five times worse than
# either Platt or isotonic calibration, stable across bin counts, and
# visible directly in the reliability curve as systematic overconfidence
# through the middle of the probability range. Platt wins the calibrator
# comparison on reliability and is the cheaper, more data-efficient choice.
#
# That gap matters for a decision, not just for a metric. At the threshold
# the stated retention economics imply, decided from the costs alone and
# never by searching evaluation outcomes, the uncalibrated model's expected
# value trails Platt's by $357 and isotonic's by $615. An earlier,
# uncorrected version of this analysis said the opposite by picking each
# threshold from the evaluation slice's own labels; catching that
# (ADR-0011) is as much this notebook's finding as the calibration gap.
#
# One qualification the matched-budget table makes explicit: the entire EV
# advantage comes from calibration changing *how many* customers clear a
# fixed cutoff, not from targeting better ones. Ranking and calibration are
# genuinely separate properties, and calibration only pays when the decision
# rule is a probability threshold. Under a fixed headcount budget it pays
# nothing.
#
# **Next:** `06_hazard_framing` asks a different question of the same
# corrected protocol. Not just *whether* a customer churns but *when*,
# fitting a discrete-time hazard model alongside the binary classifier and
# comparing what each one gets wrong.
