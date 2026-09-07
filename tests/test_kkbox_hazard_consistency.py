"""Regression test: the hazard model's "event" must agree with the binary label.

`build_person_period_panel` was originally called on unfiltered transactions
in `07_generalisation`, which would let a cancellation-only transaction count
as "the event" (a return) -- backwards from `build_kkbox_asof_panel`'s own
label rule, where a cancellation inside the label window is *stronger*
evidence of churn, not a renewal. The fix (filter to `is_cancel = 0`
transactions before calling `build_person_period_panel`) lives in the
notebook, not in `churnval.hazard` -- nothing in that module knows about
`is_cancel` at all, so nothing there enforces the fix. This test exists
because that leaves the notebook's own discipline as the only thing
preventing a regression (found under adversarial review: unlike the
`eligibility_gap_days` fix, this one had no test anywhere).
"""

from __future__ import annotations

import pandas as pd

from churnval.hazard import build_person_period_panel
from churnval.kkbox_features import FEATURE_COLS, build_kkbox_asof_panel
from churnval.windows import Window

AS_OF = pd.Timestamp("2016-06-01")
WINDOW = Window(as_of=AS_OF, gap_days=7, horizon_days=30)


def _tx(customer_id, transaction_date, membership_expire_date, is_cancel, **overrides):
    row = {
        "customer_id": customer_id,
        "transaction_date": pd.Timestamp(transaction_date),
        "membership_expire_date": pd.Timestamp(membership_expire_date),
        "actual_amount_paid": 149,
        "is_auto_renew": 1,
        "is_cancel": is_cancel,
    }
    row.update(overrides)
    return row


def _hazard_panel(transactions: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    renewal_transactions = transactions.loc[transactions["is_cancel"] == 0]
    return build_person_period_panel(
        renewal_transactions, [WINDOW], panel, FEATURE_COLS, time_col="transaction_date"
    )


def test_cancellation_only_in_window_is_censored_not_an_event():
    # cust1's only pre-as_of transaction is a real renewal (eligible); its
    # only label-window transaction is a cancellation. The binary label
    # must call this churned=1 (no renewal observed); the hazard panel must
    # agree by showing every one of cust1's person-period rows censored
    # (event=0) -- a cancellation must not be mistaken for "the event."
    transactions = pd.DataFrame(
        [
            _tx("cust1", "2016-01-01", "2016-05-01", 0, actual_amount_paid=99, is_auto_renew=0),
            _tx("cust1", "2016-06-20", "2016-06-20", 1),  # cancellation, inside the label window
        ]
    )
    panel = build_kkbox_asof_panel(transactions, [WINDOW])
    assert panel.set_index("customer_id").loc["cust1", "churned"] == 1

    hazard_panel = _hazard_panel(transactions, panel)
    cust1_rows = hazard_panel.loc[hazard_panel["customer_id"] == "cust1"]
    assert not cust1_rows.empty
    assert (cust1_rows["event"] == 0).all()


def test_genuine_renewal_in_window_is_an_event_not_censored():
    # The other direction: a real (is_cancel=0) transaction inside the label
    # window must produce churned=0 from the binary label and event=1 at the
    # matching period from the hazard panel -- confirms the fix doesn't
    # over-correct by also hiding genuine renewals.
    transactions = pd.DataFrame(
        [
            _tx("cust1", "2016-01-01", "2015-01-31", 0),
            _tx("cust1", "2016-05-06", "2016-06-05", 0),
            _tx("cust1", "2016-06-15", "2016-07-15", 0),  # renewal inside the label window
        ]
    )
    panel = build_kkbox_asof_panel(transactions, [WINDOW])
    assert panel.set_index("customer_id").loc["cust1", "churned"] == 0

    hazard_panel = _hazard_panel(transactions, panel)
    cust1_rows = hazard_panel.loc[hazard_panel["customer_id"] == "cust1"]
    assert cust1_rows["event"].sum() == 1
    assert cust1_rows.loc[cust1_rows["event"] == 1, "period"].item() == 1
