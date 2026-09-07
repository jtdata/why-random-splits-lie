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
# # 02 Leakage diagnosis
#
# **Question this notebook answers:** why is `01_naive_baseline`'s ROC AUC of
# about 0.90 wrong, feature by feature?
#
# **Consumes:** the naive panel and split built in `01_naive_baseline`,
# rebuilt here from `churnval.naive_baseline` and `churnval.asof_preview`.
# Both modules allow reuse through this notebook; the "do not reuse" rule
# starts at `03_temporal_protocol`, which builds the corrected pipeline from
# scratch.
#
# **Produces:** a leakage table covering each of the three naive features, an
# adversarial-validation result, and ablation results, as reusable functions
# in `churnval.leakage` rather than one-off notebook code.
#
# **Runtime:** a few minutes, on `01`'s cached Parquet.
#
# ---
#
# `01_naive_baseline` showed *that* the number is inflated and named three
# mechanisms. This notebook builds the general-purpose tools that `01` left
# as one-off checks, and uses them to ask a sharper question: which
# diagnostic catches this leak, and which one looks clean while missing it?
#
# The second half of that question matters in practice. Plain adversarial
# validation, a classifier trained to tell train rows from test rows, comes
# back at chance on this panel. A reader who runs that recipe and stops
# there would conclude the split is fine. The reason it comes back clean is
# not subtle once stated: a random split produces train and test sets with
# the same distribution *by construction*, whatever the features are.
# Adversarial validation detects distribution shift between two groups. It
# cannot detect entity overlap or features that read the future, because
# neither of those shifts the feature distribution between a random train
# set and a random test set. It has to be pointed at two groups that would
# actually differ if the leak were present.

# %% [markdown]
# ## Rebuild the panel and split
#
# Identical construction to `01_naive_baseline`: same origins, same naive
# (`as_of`-blind) features, same `GAP_DAYS = 0`, same seed. Rebuilding it
# rather than loading a saved object keeps this notebook runnable on its own
# from a fresh kernel, and guarantees the split below is bit-for-bit the one
# `01` reported on.

# %%
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from churnval.asof_preview import asof_recency_frequency_monetary
from churnval.config import DEFAULT_STEP_DAYS, PATHS, SEED
from churnval.io import load_online_retail_ii
from churnval.leakage import ablation_auc, adversarial_validation_auc, single_feature_auc
from churnval.naive_baseline import GAP_DAYS, HORIZON_DAYS, build_naive_panel
from churnval.plotting import PALETTE, set_style
from churnval.windows import rolling_origins

PATHS.ensure()
set_style()
transactions = load_online_retail_ii()
first_as_of = transactions["invoice_date"].min().normalize() + timedelta(days=365)
last_event = transactions["invoice_date"].max()
origins = rolling_origins(
    first_as_of,
    last_event,
    gap_days=GAP_DAYS,
    horizon_days=HORIZON_DAYS,
    step_days=DEFAULT_STEP_DAYS,
)
panel = build_naive_panel(transactions, origins)

X = panel[["recency_days", "frequency", "monetary"]]
y = panel["churned"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"{len(panel):,} rows, {len(X_train):,} train / {len(X_test):,} test")

# %% [markdown]
# ## Adversarial validation, the standard way
#
# Train a classifier to tell train rows from test rows, using the same three
# features the naive model uses. If it can, train and test differ
# systematically and something is wrong with the split. If it cannot, the
# usual reading is "the split is fine".

# %%
auc_rows_split, importance_rows_split = adversarial_validation_auc(X_train, X_test, seed=SEED)
print(f"train-vs-test adversarial AUC: {auc_rows_split:.3f}")
importance_rows_split

# %% [markdown]
# Barely above chance, as it has to be. A stratified random split draws
# train and test from the same rows, so there is no distribution shift for
# the classifier to find. This says nothing about whether the features are
# safe or whether the same customers sit on both sides. Adversarial
# validation is the right tool for a different problem: checking whether a
# *temporal* test set looks like the training data it follows. On a random
# split it is guaranteed to pass.

# %% [markdown]
# ## Adversarial validation, pointed at what actually differs
#
# The leak in `01` is that the features a row carries are not the features
# that row would have had at its own `as_of`. So compare those two things
# directly: for the *same* (customer, `as_of`) rows, can a classifier tell
# the naive feature values from the as-of-correct ones?
# `asof_recency_frequency_monetary` (`churnval.asof_preview`) recomputes
# them. If the two are separable, the model in `01` was trained on inputs it
# would never see in production. This is the same training/serving skew
# that `04_the_gap` later measures on the model's output.
#
# One detail needs care. `X_naive_side` and `X_asof_side` are *matched*: row
# `i` of each is the same (customer, `as_of`) occasion computed two ways. If
# the classifier's own internal train/test split puts a customer's naive row
# in its training fold and that same customer's as-of row in its test fold,
# it can partly recognise the customer (their general spending scale carries
# across both versions) instead of the naive-vs-as-of difference, and the
# AUC inflates. `adversarial_validation_auc` takes a `groups` argument for
# this. Passing `customer_id` keeps both rows of a customer on the same side
# of the internal split.

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

feature_cols = ["recency_days", "frequency", "monetary"]
X_naive_side = comparison[[f"{c}_naive" for c in feature_cols]].rename(
    columns=lambda c: c.removesuffix("_naive")
)
X_asof_side = comparison[[f"{c}_asof" for c in feature_cols]].rename(
    columns=lambda c: c.removesuffix("_asof")
)
auc_feature_diff, importance_feature_diff = adversarial_validation_auc(
    X_naive_side, X_asof_side, seed=SEED, groups=comparison["customer_id"]
)
print(f"naive-vs-as-of-correct adversarial AUC (grouped by customer): {auc_feature_diff:.3f}")
importance_feature_diff

# %% [markdown]
# Clearly separable, even with customer identity controlled for. Same three
# features, same classifier, same procedure. The only thing that changed is
# which two groups are being compared. The internal split's grouping is
# worth showing rather than asserting:

# %%
auc_feature_diff_ungrouped, _ = adversarial_validation_auc(X_naive_side, X_asof_side, seed=SEED)
print(f"same comparison, ungrouped internal split: {auc_feature_diff_ungrouped:.3f}")
print(f"same comparison, grouped by customer:      {auc_feature_diff:.3f}")

# %% [markdown]
# Ungrouped overstates it by about 0.07. Both numbers say the same
# qualitative thing; only the grouped one is an honest estimate of how much.
# The diagnostic for a leaky split has its own split inside it, and that
# split needs the same entity discipline as the one being diagnosed.

# %%
fig, ax = plt.subplots(figsize=(7, 3.5))
labels = ["train rows\nvs. test rows", "naive features\nvs. as-of-correct"]
values = [auc_rows_split, auc_feature_diff]
colours = [PALETTE["neutral"], PALETTE["both"]]
bars = ax.bar(labels, values, color=colours)
ax.axhline(0.5, color=PALETTE["reference_line"], lw=1, ls="--")
ax.text(-0.55, 0.52, "chance", va="bottom", fontsize=8)
for bar, v in zip(bars, values, strict=True):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        v + 0.02,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
ax.set_ylim(0, 1.05)
ax.set_ylabel("adversarial-validation ROC AUC")
ax.set_title("The standard adversarial-validation recipe misses this leak entirely", loc="left")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(PATHS.figures / "leakage_adversarial_validation.png", dpi=200)
# fig

# %% [markdown]
# ## How much does each feature differ from its as-of-correct value?
#
# The adversarial result says the naive and as-of-correct feature sets are
# separable overall. Per feature: how often does the naive value differ at
# all, and by how much when it does?

# %%
delta_rows = []
for col in feature_cols:
    delta = comparison[f"{col}_naive"] - comparison[f"{col}_asof"]
    delta_rows.append(
        {
            "feature": col,
            "pct rows differing": (delta != 0).mean(),
            "median delta (naive - as-of)": delta.median(),
            "mean delta (naive - as-of)": delta.mean(),
        }
    )
delta_table = pd.DataFrame(delta_rows).set_index("feature")
delta_table

# %% [markdown]
# `recency_days` differs on almost every row, because it is measured to the
# dataset's end for every row regardless of `as_of`. `frequency` and
# `monetary` differ only on the roughly 59% of rows where the customer has at
# least one transaction between `as_of` and the dataset's end. Less
# pervasive, but when it happens the size of the difference is large:
# hundreds of pounds and multiple orders.

# %% [markdown]
# ## Ablation results
#
# `ablation_auc` (`churnval.leakage`) refits with each feature removed in
# turn, on the same train/test split `01` reported on.

# %%
ablation = ablation_auc(X_train, y_train, X_test, y_test, seed=SEED)
ablation

# %%
fig, ax = plt.subplots(figsize=(8, 3.2))
order = ablation.sort_values()
ax.barh(order.index, order.to_numpy(), color=PALETTE["active"])
for i, v in enumerate(order.to_numpy()):
    ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
ax.set_xlabel("ROC AUC on the held-out test rows")
ax.set_title(
    "No single feature explains the inflation,\ndropping any one still leaves it above 0.87",
    loc="left",
)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(PATHS.figures / "leakage_ablation.png", dpi=200)
# fig

# %% [markdown]
# A caveat on what ablation can show here. Every subset is evaluated on the
# same entangled split, so the split leak (mistake #2) is present in every
# bar. Ablation can rank the features by how much each one contributes on
# top of that; it cannot separate the feature leak from the split leak. That
# separation needs both things varied at once, which `03` and `04` do.
#
# ### Is that ordering real, or one lucky split?
#
# The ablation above ran once, on `01`'s exact train/test split. Before
# trusting the ordering (`recency_days` costs the most to remove, `monetary`
# the least), check it against five independent resamples of the same panel:
# same features, same labels, a different random 80/20 draw each time.

# %%
resampled_costs = []
for resample_seed in range(SEED, SEED + 5):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=resample_seed, stratify=y
    )
    result = ablation_auc(X_tr, y_tr, X_te, y_te, seed=SEED)
    cost = result["all features"] - result.drop("all features")
    cost.index = cost.index.str.removeprefix("drop ")
    resampled_costs.append(cost)
resampled_costs = pd.DataFrame(resampled_costs)
resampled_costs.index.name = "resample"
resampled_costs.agg(["mean", "std", "min", "max"])

# %% [markdown]
# The ranges do not overlap. In every resample `recency_days` costs more
# than `frequency`, which costs more than `monetary`. The ordering is a
# property of the panel, not of which 20% of rows happened to land in test.

# %% [markdown]
# ## The leakage table
#
# Everything above, per feature, in one place: how strong the feature looks
# alone (`single_feature_auc`, over the full panel of 42,846 rows), how much
# of the model's score it is responsible for on `01`'s held-out test rows
# (ablation cost, 8,570 rows), and how often and how much it differs from
# its honestly computed value (full panel again). The three AUC columns are
# not computed on the same rows, which is why the column names say so.

# %%
single_auc = single_feature_auc(X, y)
ablation_cost = ablation["all features"] - ablation.drop("all features")
ablation_cost.index = ablation_cost.index.str.removeprefix("drop ")

leakage_table = pd.DataFrame(
    {
        "single-feature AUC (full panel)": single_auc,
        "ablation cost, AUC (test split only)": ablation_cost,
        "pct rows differing from as-of-correct (full panel)": delta_table["pct rows differing"],
        "median delta (naive - as-of)": delta_table["median delta (naive - as-of)"],
    }
).sort_values("ablation cost, AUC (test split only)", ascending=False)

leakage_table_path = PATHS.reports / "leakage_table_naive.csv"
leakage_table.to_csv(leakage_table_path)
leakage_table

# %% [markdown]
# The columns disagree about which feature matters most, and the
# disagreement is informative. `frequency` has the highest single-feature
# AUC: alone, it ranks customers best. But `recency_days` costs the most to
# remove from the full model, differs from its honest value on almost every
# row, and is what the naive-vs-as-of classifier leans on hardest. Read
# together: `frequency` is a strong standalone ranker that is largely
# redundant once the other two are present, while `recency_days` carries
# information the others do not, which also makes it the single biggest
# as-of leak in this panel.

# %% [markdown]
# ---
# ## Leakage audit
#
# Covers this notebook's own diagnostics. `01`'s audit of the underlying
# panel and split stands as written and is not repeated.
#
# | # | Component | Leak class | Evidence | Severity | Fix |
# |---|---|---|---|---|---|
# | 1 | Panel and split (inherited from `01`) | Already audited | `01_naive_baseline`'s audit covers the three intentional mistakes, the same-customer label-agreement finding, and the `io.py` fix | n/a | See `01` |
# | 2 | `adversarial_validation_auc(X_train, X_test)` | Clean result, wrong tool | AUC 0.504 (computed above). A random split has no train/test distribution shift to detect, so this diagnostic cannot see either of `01`'s leaks by construction | Informational | None needed; the point of the first section is that this recipe has to be pointed at the right two groups |
# | 3 | Internal split of `adversarial_validation_auc(X_naive_side, X_asof_side)` | Found and fixed while building this notebook | An earlier version called it without `groups`. Because the two sides are matched pairs, about 42% of occasions had one version in the internal training fold and the other in its test fold (checked by inspecting the split indices). The grouped-vs-ungrouped cell above shows both numbers, 0.786 vs. 0.853: still clearly separable either way, but the ungrouped number overstated it by 0.067 | Medium; did not change the qualitative finding, but the number was wrong | `adversarial_validation_auc` gained a `groups` parameter (`GroupShuffleSplit` when provided); regression-tested in `tests/test_leakage.py` |
# | 4 | Absolute AUC values in the ablation table | Needs a reading caveat | `ablation_auc`'s baseline (0.903) and per-drop values are fit on the same entangled split `01` reported on, and inherit all three of its mistakes. They are not estimates of how good any feature subset really is. Only the relative ordering, confirmed stable across five resamples, is the valid takeaway | Medium; a reader could mistake 0.872 for a corrected number | Honest absolute numbers are `03`'s and `04`'s job |
# | 5 | `ablation_auc`'s four refits sharing one split | Clean | All four models (baseline plus three drop-one variants) are fit on the same training rows and scored on the same held-out rows. A different split per variant would confound the feature's effect with split variance, so sharing the split is what makes the comparison fair | n/a | n/a |

# %% [markdown]
# ---
# ## Closing
#
# **Finding:** the 0.90 from `01` is wrong for reasons that show up
# differently depending on which tool is pointed at them. Ablation says no
# single feature explains it: dropping the most expensive one,
# `recency_days`, still leaves ROC AUC above 0.87. Adversarial validation
# says something sharper, but only when asked the right question. Pointed at
# train rows vs. test rows it reports 0.504 and would tell a reader the split
# is fine, because a random split has no distribution shift to find. Pointed
# at the same rows' naive vs. as-of-correct feature values it reports 0.786
# (grouped by customer; an ungrouped version overstated it at 0.853). The
# leakage table (`reports/leakage_table_naive.csv`) puts each feature's
# univariate strength, marginal contribution, and honest-vs-naive divergence
# side by side. `frequency` ranks best alone but is largely redundant once
# the others are present. `recency_days` is the smaller standalone signal
# that carries the most information the other two lack, which is also why
# it is the biggest as-of leak in the panel.
#
# **Next:** `03_temporal_protocol` builds the pipeline this notebook only
# diagnosed: features computed strictly before each row's own `as_of` (the
# tested version, not the `asof_preview` stand-in used here), a real gap,
# and a rolling-origin backtest. `04_the_gap` then reports the honest
# performance next to `01`'s inflated one and the size of the difference.
