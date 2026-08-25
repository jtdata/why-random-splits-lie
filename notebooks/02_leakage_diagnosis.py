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
# # 02 — Leakage diagnosis
#
# **Question this notebook answers:** why is `01_naive_baseline`'s ROC AUC ≈
# 0.90 wrong, feature by feature?
#
# **Consumes:** the naive panel and split built in `01_naive_baseline` —
# rebuilt here from `churnval.naive_baseline` and `churnval.asof_preview`,
# which both say reuse is fine through this notebook (the "don't reuse" rule
# starts at `03_temporal_protocol`, which builds the corrected pipeline from
# scratch instead).
#
# **Produces:** a leakage table covering each of the three naive features;
# an adversarial-validation result; ablation results — as reusable functions
# in `churnval.leakage`, not one-off notebook code, per `01`'s closing note.
#
# **Runtime:** a few minutes — reuses `01`'s cached Parquet, no re-fetch.
#
# ---
#
# `01_naive_baseline` already showed *that* the ~0.90 number is inflated and
# named three mechanisms. This notebook is not a repeat of that — it builds
# the general-purpose tools `01` deliberately left as one-off checks
# (`assert_no_dominant_single_feature` was the exception, already reusable),
# and it uses them to ask a sharper question: which diagnostic actually
# catches this leak, and which one looks clean while missing it entirely?
#
# That second question matters concretely here. Plain adversarial validation
# — train a classifier to tell train rows from test rows — comes back
# looking clean on this panel (AUC ≈ 0.50, shown in `01`'s audit), because
# the leak isn't distributional drift between train and test: it's features
# that don't change with `as_of`, joined onto a split that doesn't respect
# entity identity. A reader who runs the standard adversarial-validation
# recipe and stops there would conclude the split is fine. This notebook
# shows why that conclusion is wrong, and what adversarial validation has to
# be pointed at instead to actually catch it.

# %% [markdown]
# ## Rebuild the panel and split
#
# Identical construction to `01_naive_baseline`: same origins, same naive
# (`as_of`-blind) features, same `GAP_DAYS = 0`, same seed. Rebuilding it
# rather than importing a saved object keeps this notebook runnable on its
# own from a fresh kernel, and guarantees the split below is bit-for-bit the
# one `01` reported ROC AUC ≈ 0.90 on.

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
# Train a classifier to tell train rows from test rows apart, using the same
# three features the naive model uses. If it can, train and test differ
# systematically — a warning sign. If it can't, the usual reading is "the
# split is fine."

# %%
auc_rows_split, importance_rows_split = adversarial_validation_auc(X_train, X_test, seed=SEED)
print(f"train-vs-test adversarial AUC: {auc_rows_split:.3f}")
importance_rows_split

# %% [markdown]
# That reads as clean — barely above chance. It is not evidence the split is
# safe. `naive_features` computes the same value for a customer regardless
# of which `as_of` a row is attached to, so a row that ends up in train and
# a row that ends up in test for the *same customer* have near-identical
# feature values. The classifier above has nothing to key on, not because
# there's no leak, but because the leak doesn't move the feature
# distribution between train and test — it duplicates it.

# %% [markdown]
# ## Adversarial validation, pointed at the actual difference
#
# A sharper question: for the *same* (customer, `as_of`) rows, how different
# are the naive features from what they'd be if computed strictly before
# `as_of`? `asof_recency_frequency_monetary` (`churnval.asof_preview`, built
# in `01`) recomputes them; if a classifier can tell "naive-computed" values
# apart from "as-of-correct" values for the same rows, that's the leak,
# directly.
#
# One thing has to be handled carefully here that the first comparison
# didn't need: `X_naive_side` and `X_asof_side` are *matched* — row `i` of
# each is the same (customer, `as_of`) occasion, just computed two ways. If
# `adversarial_validation_auc`'s own internal train/test split is allowed to
# put one customer's naive row in its training fold and that *same*
# customer's as-of row in its test fold, the classifier can partly recognize
# the customer (their general spending scale carries across both versions)
# rather than purely the naive-vs-as-of difference — inflating the AUC.
# `adversarial_validation_auc` takes an optional `groups` argument for
# exactly this: pass `customer_id` and both rows of a customer always land
# on the same side of the internal split.

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
# Still unambiguously separable, even with customer identity controlled
# for. The same three features, the same classifier, the same evaluation
# procedure — the only thing that changed is *what two groups are being
# compared*. Adversarial validation is a real diagnostic; it just has to be
# pointed at the two things that might actually differ, not at whichever two
# groups a checklist says to compare, and its own internal split needs the
# same entity-grouping discipline as the split being diagnosed.
#
# That last point is worth showing, not just asserting — the difference it
# makes here:

# %%
auc_feature_diff_ungrouped, _ = adversarial_validation_auc(X_naive_side, X_asof_side, seed=SEED)
print(f"same comparison, ungrouped internal split: {auc_feature_diff_ungrouped:.3f}")
print(f"same comparison, grouped by customer:      {auc_feature_diff:.3f}")

# %% [markdown]
# Ungrouped overstates it by about 0.07 — not because the leak is smaller
# than it looks, but because without grouping, the classifier's own internal
# split can put one customer's naive row in its training fold and that same
# customer's as-of row in its test fold, letting it partly recognize the
# customer (their general spending scale carries across both versions of
# their features) rather than purely the naive-vs-as-of difference. Both
# numbers say the same qualitative thing; only the grouped one is an honest
# estimate of *how much*.

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
# ## How much does each feature actually differ from its as-of-correct value?
#
# The adversarial-validation result above says the naive and as-of-correct
# feature sets are separable overall. Per feature: how often does the naive
# value even differ, and by how much when it does?

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
# `recency_days` differs on almost every row — it's measured to the
# dataset's own end for every row regardless of `as_of`, so unless a
# customer's last-ever purchase happens to fall exactly on their as-of-safe
# last purchase, the two disagree. `frequency` and `monetary` differ only
# on the ~59% of rows where a transaction the naive version wrongly counts
# actually exists between `as_of` and the dataset's end — less pervasive,
# but the size of the effect when it happens is large (hundreds of pounds,
# multiple orders).

# %% [markdown]
# ## Ablation results
#
# `ablation_auc` (`churnval.leakage`) refits with each feature removed in
# turn, on the same train/test split `01` reported ROC AUC ≈ 0.90 on.

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
    "No single feature explains the inflation —\ndropping any one still leaves it above 0.87",
    loc="left",
)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(PATHS.figures / "leakage_ablation.png", dpi=200)
# fig

# %% [markdown]
# ### Is that ordering real, or one lucky split?
#
# `ablation_auc` above ran once, on `01`'s exact train/test split. Before
# trusting the ordering (`recency_days` costs the most, `monetary` the
# least), check it against five independent resamples of the same panel —
# same features, same labels, different random 80/20 draws each time.

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
# The ranges don't overlap at all: every one of the five resamples has
# `recency_days` costing more than `frequency`, which costs more than
# `monetary`. The ordering isn't a property of `01`'s particular split — it
# holds regardless of which 20% of rows happen to land in test.

# %% [markdown]
# ## The leakage table
#
# Everything above, per feature, in one place: how strong the feature looks
# alone (`single_feature_auc`, computed over the full panel — 42,846 rows,
# every origin), how much of the model's score it alone is responsible for
# on `01`'s held-out test rows (ablation cost — baseline AUC minus the AUC
# with that feature dropped, 8,570 rows), and how often and how much it
# actually differs from its honestly-computed value (full panel again). The
# three AUC-bearing columns are not computed on the same rows — noted here
# rather than left for a reader to assume otherwise.

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
# The columns don't all agree on which feature matters most, and that
# disagreement is informative rather than a problem. `frequency` has the
# highest *single-feature* AUC — alone, it ranks customers best. But
# `recency_days` costs the most to remove from the full model, differs from
# its honest value on almost every row (not just the ~59% where
# `frequency`/`monetary` differ), and is what the naive-vs-as-of-correct
# adversarial classifier leans on hardest. Read together: `frequency` is a
# strong standalone ranker that's partly redundant with the other two once
# they're all in the model, while `recency_days` carries information the
# other features don't — which also makes it the single biggest as-of leak
# in this panel, not merely the strongest feature.

# %% [markdown]
# ---
# ## Leakage audit
#
# Run against this notebook's own diagnostics (not a repeat of `01`'s audit
# of the underlying panel and split, which stands as written), following the
# `leakage-audit` skill.
#
# | # | Component | Leak class | Evidence | Severity | Fix |
# |---|---|---|---|---|---|
# | 1 | Panel and split (inherited from `01`) | Already audited | `01_naive_baseline`'s "Leakage audit" section covers the three intentional mistakes, the same-customer label-agreement finding, and the `io.py` fix. Not repeated here | n/a | n/a — see `01` |
# | 2 | `adversarial_validation_auc(X_train, X_test)` | Checked — **clean result, but a misleading diagnostic** | AUC 0.504 (computed above). `naive_features` is `as_of`-invariant, so a customer's train-partition and test-partition rows carry near-identical feature values regardless of the real entity-overlap leak established in `01` — this diagnostic cannot see that leak by construction, not because it isn't there | n/a (informational — the whole point of this notebook's first section) | n/a — `02` exists to show adversarial validation needs to be paired with the right two groups, not to fix this call |
# | 3 | `adversarial_validation_auc(X_naive_side, X_asof_side)`'s internal split | **Found and fixed during this notebook's own construction** | An earlier version of this cell called `adversarial_validation_auc` without `groups`. Because `X_naive_side`/`X_asof_side` are matched pairs (row `i` of each is the same (customer, `as_of`) occasion), the classifier's own internal split let ~42% of occasions land with one version in its training fold and the other in its test fold — verified directly by inspecting the split indices. The "ungrouped vs. grouped" cell above reproduces both numbers (0.853 vs. 0.786): still clearly separable either way, but the ungrouped number overstated it by 0.067 | Medium — did not change the qualitative finding, but the specific number was wrong | Fixed: `adversarial_validation_auc` gained an optional `groups` parameter (`GroupShuffleSplit` when provided), used for the main naive-vs-as-of result with `customer_id`; regression-tested in `tests/test_leakage.py` |
# | 4 | Absolute AUC values in the ablation table | Checked — **needs a reading caveat, not a fix** | `ablation_auc`'s baseline (0.903) and per-drop values are fit on the exact same entangled train/test split `01` reported ROC AUC ≈ 0.90 on — they inherit all three of that split's mistakes (`as_of`-blind features, entity overlap across train/test, `GAP_DAYS = 0`), not just the two most visible in this notebook's own charts. They are not clean estimates of how good any of these feature subsets "really" are; only the *relative* ordering (which feature costs the most to remove, confirmed stable across five resamples above) is the valid takeaway here | Medium — a reader could mistake 0.872 for a corrected number | n/a for this notebook — `03_temporal_protocol`/`04_the_gap` are where an honest absolute number gets reported |
# | 5 | `ablation_auc`'s four refits sharing one split | Checked — **clean** | All four models (baseline + three drop-one variants) are fit on the same `X_train`/`y_train` and scored on the same held-out `X_test`/`y_test`; using a different split per variant would confound the feature's effect with split variance, so sharing the split is what makes the comparison fair, not a shortcut | n/a | n/a |
#
# **Answering the three questions this audit was asked to settle:**
#
# - **Does the matched-pair structure of the naive-vs-as-of comparison
#   invalidate its adversarial AUC?** It would have, uncorrected — row 3
#   above is exactly that, caught by checking the internal split's index
#   overlap rather than assumed away. Grouped, the comparison is honest and
#   the finding survives (0.786, still clearly separable).
# - **Does `ablation_auc`'s split reuse across its four refits cause
#   leakage?** No — row 5. Reusing the split is required for a fair
#   comparison, not a source of leakage.
# - **Does reusing `01`'s exact panel and split cause a new problem here?**
#   Not a leakage problem, but a reporting one — row 4. This notebook's
#   absolute AUC numbers are diagnostic, not honest performance estimates;
#   `03`/`04` are where the corrected numbers belong.

# %% [markdown]
# ---
# ## Closing
#
# **Finding:** the ~0.90 number from `01` is wrong for reasons that show up
# differently depending on which tool is pointed at them. Ablation says no
# single feature explains it — dropping the most expensive one
# (`recency_days`, confirmed the biggest cost across five independent
# resamples) still leaves ROC AUC above 0.87. Adversarial validation says
# something sharper, and only if it's asked the right question: pointed at
# train rows vs. test rows, it reports a clean 0.504 and would tell a reader
# the split is fine — wrong, because `naive_features` doesn't vary with
# `as_of`, so the leak never shows up as a train/test distributional
# difference. Pointed instead at the same rows' naive-computed vs.
# as-of-correct feature values, it reports 0.786 (customer-grouped — an
# earlier, ungrouped version of this exact notebook overstated it at 0.853,
# caught and fixed during its own construction, not after the fact). The
# leakage table (`reports/leakage_table_naive.csv`) puts every feature's
# univariate strength, marginal contribution, and honest-vs-naive divergence
# in one place: `frequency` ranks best alone but is largely redundant once
# the others are present; `recency_days` is the smaller standalone signal
# that turns out to carry the most information the other two don't have —
# which is also why it's the biggest as-of leak in the panel.
#
# **Next:** `03_temporal_protocol` builds the pipeline this notebook only
# diagnosed — features computed strictly before each row's own `as_of` (not
# the `asof_preview` stand-in used here for comparison), a split grouped by
# `customer_id`, and `config.DEFAULT_GAP_DAYS` instead of zero. `04_the_gap`
# then reports the number this repo has been building toward: the honest
# performance next to `01`'s inflated one, and the size of the difference
# between them.
