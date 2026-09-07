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
#: * ``emphasis`` -- highlighting *one bar or point among otherwise-neutral
#:   ones* (e.g. the 30-day plan in a `payment_plan_days` histogram, or one
#:   origin called out on a per-origin chart). Not a class label and not a
#:   split label -- never reused for "churned"/"active" or "train"/"test",
#:   which already own their own colours above. Validated with
#:   `dataviz/scripts/validate_palette.js` against the categorical set
#:   (`active`/`churned`/`train`/`test`/`both`) before adding: CVD and
#:   normal-vision separation both pass against every one of them.
PALETTE = {
    "active": "#4B44A6",
    "churned": "#B0432E",
    "train": "#1B7A72",
    "test": "#D98E04",
    "both": "#D6349C",
    "neutral": "#B0B7C0",
    "reference_line": "#141D26",
    "emphasis": "#1F6FEB",
}

#: One-hue, monotone-lightness ramp for *ordinal* dimensions -- a discrete,
#: ordered sequence where position carries meaning (e.g. successive `as_of`
#: scoring dates), as distinct from the categorical roles above (identity,
#: order-independent). Ten steps (250-700) from the palette skill's
#: validated default sequential-blue ramp, already checked against the 2:1
#: light-surface floor at the light end -- not eyeballed. Index 0 is the
#: earliest/lightest step; -1 is the latest/darkest.
ORDINAL_BLUE_10 = [
    "#86B6EF",
    "#6DA7EC",
    "#5598E7",
    "#3987E5",
    "#2A78D6",
    "#256ABF",
    "#1C5CAB",
    "#184F95",
    "#104281",
    "#0D366B",
]


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
