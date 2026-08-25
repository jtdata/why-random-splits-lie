"""Tests for the window arithmetic.

These exist because every assertion here corresponds to a way a churn backtest
can be silently wrong. A failure in this file is a leakage bug, not a style
complaint.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churnval.windows import (
    Window,
    assert_no_leak,
    mask_feature_events,
    mask_label_events,
    rolling_origins,
)

TS = pd.Timestamp


def test_label_window_starts_after_the_gap():
    w = Window(as_of=TS("2017-01-01"), gap_days=7, horizon_days=30)
    assert w.label_start == TS("2017-01-08")
    assert w.label_end == TS("2017-02-07")


def test_feature_bound_is_exclusive_of_the_as_of_instant():
    w = Window(as_of=TS("2017-01-01"), gap_days=7, horizon_days=30)
    events = pd.DataFrame({"t": [TS("2016-12-31"), TS("2017-01-01"), TS("2017-01-02")]})
    assert mask_feature_events(events, w, "t").tolist() == [True, False, False]


def test_label_window_is_half_open():
    w = Window(as_of=TS("2017-01-01"), gap_days=7, horizon_days=30)
    events = pd.DataFrame({"t": [TS("2017-01-07"), TS("2017-01-08"), TS("2017-02-07")]})
    assert mask_label_events(events, w, "t").tolist() == [False, True, False]


def test_feature_and_label_masks_never_overlap():
    w = Window(as_of=TS("2017-01-01"), gap_days=7, horizon_days=30)
    days = pd.date_range("2016-12-01", "2017-03-01", freq="D")
    events = pd.DataFrame({"t": days})
    overlap = mask_feature_events(events, w, "t") & mask_label_events(events, w, "t")
    assert not overlap.any()


def test_zero_gap_is_rejected_unless_deliberate():
    w = Window(as_of=TS("2017-01-01"), gap_days=0, horizon_days=30)
    with pytest.raises(ValueError, match="gap_days=0"):
        assert_no_leak(w)


def test_rolling_origins_excludes_truncated_label_windows():
    origins = rolling_origins(
        TS("2017-01-01"),
        last_event=TS("2017-04-01"),
        gap_days=7,
        horizon_days=30,
        step_days=30,
    )
    # 2017-03-02 is excluded: its label window would end 2017-04-08, past the data.
    assert [w.as_of for w in origins] == [TS("2017-01-01"), TS("2017-01-31")]
    assert all(w.label_end <= TS("2017-04-01") for w in origins)


def test_rolling_origins_is_empty_when_nothing_fits():
    origins = rolling_origins(
        TS("2017-01-01"),
        last_event=TS("2017-01-15"),
        gap_days=7,
        horizon_days=30,
        step_days=30,
    )
    assert origins == []


@pytest.mark.parametrize(("gap", "horizon"), [(-1, 30), (7, 0), (7, -5)])
def test_invalid_windows_are_rejected(gap, horizon):
    with pytest.raises(ValueError):
        Window(as_of=TS("2017-01-01"), gap_days=gap, horizon_days=horizon)
