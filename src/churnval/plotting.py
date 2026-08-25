"""One place to set matplotlib style for anything published from this repo.

Per the notebook standard, no chart that ends up in `reports/figures/` uses
matplotlib's defaults. `set_style()` is called once, before the first figure
in a notebook.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

#: Shared category colours. Kept here, not re-picked per chart, so a reader
#: comparing two figures in the same report can rely on the colour meaning
#: the same thing in both -- and so changing a colour once (here) changes it
#: everywhere, across every notebook, on the next run.
#:
#: Colour roles are kept semantically separate on purpose:
#:
#: * ``active`` / ``churned`` -- the churn *class label* only. Never reused
#:   for anything else, so "rust" always means "churned" and nothing else.
#: * ``train`` / ``test`` / ``both`` -- *split membership* only (which side
#:   of a train/test split, or which evaluation protocol, a row or entity
#:   belongs to). Deliberately a different set of hues from the class-label
#:   colours above, so a reader never reads "test" as "churned" or "train"
#:   as "active".
#: * ``neutral`` / ``reference_line`` -- axis furniture: diagonal
#:   references, "no-signal" baselines, chance lines.
PALETTE = {
    "active": "#4B44A6",
    "churned": "#B0432E",
    "train": "#1B7A72",
    "test": "#D98E04",
    "both": "#D6349C",
    "neutral": "#B0B7C0",
    "reference_line": "#141D26",
}


def set_style() -> None:
    """Apply the shared rcParams. Safe to call more than once.

    No gridlines: every chart in this repo labels its bars/points directly
    or uses axis ticks that are legible without them, and a bare background
    keeps the colour roles above (the actual signal) from competing with a
    grid for attention.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#3A3F4B",
            "axes.grid": False,
            "axes.axisbelow": True,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )
