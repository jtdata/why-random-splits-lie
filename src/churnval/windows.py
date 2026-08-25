"""Window arithmetic for temporal validation.

This module is small and heavily tested on purpose. Every silent failure mode
of a churn pipeline lives here: an off-by-one that lets the label window touch
the feature window, a gap that is documented but not enforced, an origin that
runs past the end of the data and quietly produces a truncated label.

Convention used throughout, and enforced by the assertions below:

    ... feature data ...  |  gap  |  label window  |
    (strictly before      as_of   label_start      label_end
     as_of)

* Feature aggregations use events with timestamp **strictly less than**
  ``as_of``. The as-of instant itself is excluded, because a record written at
  the same instant may or may not have been visible.
* The label window is **half-open**: ``label_start <= t < label_end``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


@dataclass(frozen=True)
class Window:
    """One scoring occasion: when we score, and what we are predicting.

    Args:
        as_of: features may use only data strictly before this instant.
        gap_days: operational lead time between scoring and the label window.
        horizon_days: length of the label window.
    """

    as_of: pd.Timestamp
    gap_days: int
    horizon_days: int

    def __post_init__(self) -> None:
        if self.gap_days < 0:
            raise ValueError("gap_days must be >= 0")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be > 0")
        if not isinstance(self.as_of, pd.Timestamp):
            raise TypeError("as_of must be a pandas Timestamp")

    @property
    def label_start(self) -> pd.Timestamp:
        return self.as_of + timedelta(days=self.gap_days)

    @property
    def label_end(self) -> pd.Timestamp:
        return self.label_start + timedelta(days=self.horizon_days)

    @property
    def feature_end(self) -> pd.Timestamp:
        """Exclusive upper bound for feature events."""
        return self.as_of

    def describe(self) -> str:
        return (
            f"features < {self.as_of:%Y-%m-%d} | gap {self.gap_days}d | "
            f"label [{self.label_start:%Y-%m-%d}, {self.label_end:%Y-%m-%d})"
        )


def assert_no_leak(window: Window) -> None:
    """Fail loudly if the feature and label windows can touch.

    A gap of zero is permitted only because notebook 01 needs to demonstrate
    what it costs — but it is never silent.
    """
    if window.label_start < window.feature_end:
        raise ValueError(f"label window starts before features end: {window.describe()}")
    if window.gap_days == 0:
        raise ValueError(
            "gap_days=0 lets features computed at the as-of instant encode the "
            "event. Pass gap_days=0 only via allow_zero_gap=True in the caller "
            "that is deliberately demonstrating the failure."
        )


def rolling_origins(
    first_as_of: pd.Timestamp,
    last_event: pd.Timestamp,
    *,
    gap_days: int,
    horizon_days: int,
    step_days: int,
) -> list[Window]:
    """Generate scoring occasions spaced ``step_days`` apart.

    Stops before any window whose label period would extend past ``last_event``.
    A truncated label window silently mislabels unresolved entities as
    non-churners, which is the single most common way a backtest flatters
    itself, so it is excluded rather than clipped.

    Args:
        first_as_of: as-of date of the first origin.
        last_event: the latest timestamp present in the data.
        gap_days: operational lead time.
        horizon_days: label window length.
        step_days: spacing between origins.

    Returns:
        Windows in chronological order. Empty if no complete window fits.
    """
    if step_days <= 0:
        raise ValueError("step_days must be > 0")

    windows: list[Window] = []
    as_of = first_as_of
    while True:
        w = Window(as_of=as_of, gap_days=gap_days, horizon_days=horizon_days)
        if w.label_end > last_event:
            break
        windows.append(w)
        as_of = as_of + timedelta(days=step_days)
    return windows


def mask_feature_events(events: pd.DataFrame, window: Window, time_col: str) -> pd.Series:
    """Boolean mask selecting events usable for features at this window.

    Strictly before ``as_of``. Kept as a function rather than an inline
    comparison so that the rule lives in exactly one place and can be tested.
    """
    return events[time_col] < window.feature_end


def mask_label_events(events: pd.DataFrame, window: Window, time_col: str) -> pd.Series:
    """Boolean mask selecting events that fall inside the label window."""
    t = events[time_col]
    return (t >= window.label_start) & (t < window.label_end)
