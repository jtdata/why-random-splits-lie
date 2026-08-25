# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 00 — Problem definition
#
# **Question this notebook answers:** what exactly are we predicting, over what
# windows, and what counts as an event?
#
# **Consumes:** nothing. This notebook is argument, not computation.
#
# **Produces:** the timeline diagram in `reports/figures/timeline.png`, and the
# window constants that every later notebook imports from `churnval.config`.
#
# **Runtime:** under a minute.
#
# ---
#
# Almost every churn project goes wrong here rather than in the modelling. Four
# things have to be decided before a single feature is computed, and each one
# has a wrong answer that produces a good-looking number:
#
# 1. **What is the event?** The date something *happened*, or the date it was
#    *recorded*? These differ, and the recorded date is contaminated by the
#    process that recorded it.
# 2. **When do we score?** The as-of instant. Features may use data strictly
#    before it — not up to and including it.
# 3. **How long until we can act?** The gap. If it takes a week to score,
#    decide and make an offer, a model trained with no gap is answering a
#    question nobody can act on.
# 4. **Over what window does the outcome count?** The horizon. Entities whose
#    horizon has not closed are censored, not negatives.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from churnval.config import DEFAULT_GAP_DAYS, DEFAULT_HORIZON_DAYS, PATHS
from churnval.windows import Window

PATHS.ensure()

window = Window(
    as_of=pd.Timestamp("2017-01-01"),
    gap_days=DEFAULT_GAP_DAYS,
    horizon_days=DEFAULT_HORIZON_DAYS,
)
print(window.describe())

# %% [markdown]
# ## The diagram
#
# If this cannot be drawn, the problem is not yet defined. Everything
# downstream — the split, the as-of aggregation, the label — is a restatement
# of this picture.

# %%
fig, ax = plt.subplots(figsize=(9, 2.2))
start = window.as_of - pd.Timedelta(days=90)

bars = [
    ("feature window", start, window.feature_end, "#4B44A6"),
    ("gap", window.feature_end, window.label_start, "#B0B7C0"),
    ("label window", window.label_start, window.label_end, "#A26A05"),
]
for i, (name, x0, x1, colour) in enumerate(bars):
    ax.barh(0, (x1 - x0).days, left=(x0 - start).days, height=0.4, color=colour)
    ax.text((x0 - start).days + (x1 - x0).days / 2, 0.35, name, ha="center", fontsize=9)

ax.axvline((window.as_of - start).days, color="#141D26", lw=1.2)
ax.text((window.as_of - start).days, -0.42, "as-of", ha="center", fontsize=9)
ax.set_yticks([])
ax.set_xlabel("days")
ax.set_title(
    f"Scoring occasion: {DEFAULT_GAP_DAYS}-day gap, {DEFAULT_HORIZON_DAYS}-day horizon",
    loc="left",
)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(PATHS.figures / "timeline.png", dpi=200)
fig

# %% [markdown]
# ## Closing
#
# **Finding:** the problem is defined as — given everything known strictly
# before the as-of instant, will this entity churn during the 30-day window
# that opens 7 days later?
#
# **Next:** `01_naive_baseline` deliberately violates this definition, in the
# way most tutorials do, and reports the resulting number.
