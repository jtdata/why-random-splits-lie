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
# # 00 Problem definition
#
# **Question this notebook answers:** What exactly are we predicting, over what
# windows, and what counts as an event?
#
# **Consumes:** Nothing. This notebook is argument, not computation.
#
# **Produces:** The timeline diagram in `reports/figures/timeline.png`, a
# companion random-split illustration in
# `reports/figures/random_split_illustration.png`, and the window constants
# that every later notebook imports from `churnval.config`.
#
# **Runtime:** Under a minute.
#
# ---
#
# Most churn projects go wrong before any modelling starts. Four things have to
# be pinned down before the first feature is computed, and each has a tempting
# wrong answer that produces a good-looking number:
#
# 1. **What is the event?** The date something *happened*, or the date it was
#    *recorded*? These can differ by days or weeks, and the recorded date
#    carries the delays and habits of whoever recorded it.
# 2. **When do we score?** The **as-of** instant. Features may use data
#    strictly before it, never data from the instant itself.
# 3. **How long until we can act?** The gap. If it takes a week to score,
#    decide and make an offer, a model trained with no gap answers a question
#    nobody can act on.
# 4. **Over what window does the outcome count?** The horizon. Entities whose
#    horizon has not closed yet are censored, not negatives.

# %% jupyter={"source_hidden": true}
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churnval.config import DEFAULT_GAP_DAYS, DEFAULT_HORIZON_DAYS, PATHS, SEED
from churnval.plotting import PALETTE
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
# Everything downstream of this notebook, the split, the as-of aggregation
# and the label, is a restatement of this one picture.

# %% jupyter={"source_hidden": true}
fig, ax = plt.subplots(figsize=(9, 2.2))
start = window.as_of - timedelta(days=90)

bars = [
    ("feature window", start, window.feature_end, "#4B44A6"),
    ("gap", window.feature_end, window.label_start, "#B0B7C0"),
    ("label window", window.label_start, window.label_end, "#A26A05"),
]
for name, x0, x1, colour in bars:
    ax.barh(0, (x1 - x0).days, left=(x0 - start).days, height=0.4, color=colour)
    ax.text(
        (x0 - start).days + (x1 - x0).days / 2,
        0.26,
        name,
        ha="center",
        va="bottom",
        fontsize=9,
    )

# ymin/ymax are axes fractions: clip the marker to the bar band so it
# does not strike through the "as-of" label below it.
ax.axvline((window.as_of - start).days, color="#141D26", lw=1.2, ymin=0.30, ymax=0.72)
ax.text((window.as_of - start).days, -0.28, "as-of", ha="center", va="top", fontsize=9)

# Fixed limits rather than autoscale: the labels sit above the bars and the
# as-of marker below them, so the axes need explicit headroom or matplotlib
# packs them against the title.
ax.set_ylim(-0.7, 0.8)
ax.set_yticks([])
ax.set_xlabel("days")
ax.set_title(
    f"Scoring occasion: {DEFAULT_GAP_DAYS}-day gap, {DEFAULT_HORIZON_DAYS}-day horizon",
    loc="left",
    pad=14,
)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(PATHS.figures / "timeline.png", dpi=200)
# fig

# %% [markdown]
# ## For comparison: a random split on the same timeline
#
# The diagram above is one correctly designed scoring occasion: a clean
# boundary at as-of, and a label window that opens only after it. Most
# tutorials never draw that boundary. They shuffle a table of rows and put
# 80% in training and 20% in test, with no regard for where in time each row
# sits. The chart below scatters synthetic customer observations across the
# same span and splits them 80/20 at random. `01_naive_baseline` builds
# exactly this kind of split on real data.

# %% jupyter={"source_hidden": true}
rng = np.random.default_rng(SEED)
n_points = 120
event_days = rng.uniform(0, (window.label_end - start).days, size=n_points)
is_train = rng.random(n_points) >= 0.2  # 80/20, same split fraction 01_naive_baseline uses

fig2, ax2 = plt.subplots(figsize=(9, 1.8))
ax2.scatter(
    event_days[is_train],
    [0.12] * is_train.sum(),
    marker="|",
    s=220,
    color=PALETTE["train"],
    label=f"train ({is_train.sum()})",
)
ax2.scatter(
    event_days[~is_train],
    [-0.12] * (~is_train).sum(),
    marker="|",
    s=220,
    color=PALETTE["test"],
    label=f"test ({(~is_train).sum()})",
)
ax2.set_ylim(-0.7, 0.8)
ax2.set_yticks([])
ax2.set_xlabel("days")
ax2.set_title(
    "A random 80/20 split scatters both classes across the entire timeline, no boundary to check",
    loc="left",
    pad=14,
)
ax2.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
for side in ("top", "right", "left"):
    ax2.spines[side].set_visible(False)
fig2.tight_layout()
fig2.savefig(PATHS.figures / "random_split_illustration.png", dpi=200)
# fig2

# %% [markdown]
# ## Closing
#
# **Finding:** The problem is defined as: given everything known strictly
# before the as-of instant, will this entity churn during the 30-day window
# that opens 7 days later? Online Retail II will need a longer window than
# that default, for reasons `01_naive_baseline` gives, but the shape of the
# question is the same.
#
# **Next:** `01_naive_baseline` deliberately violates this definition, in the
# way most tutorials do, and reports the resulting number.

# %% jupyter={"source_hidden": true}
