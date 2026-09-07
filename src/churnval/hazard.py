"""Discrete-time hazard framing for `06_hazard_framing`.

Every other notebook in this project asks *whether* a customer churns within
a fixed window -- one label, one probability. This module asks the same
underlying question a different way: period by period, what is the risk of
returning, given no return yet? Multiplying `(1 - hazard)` across periods
gives a survival probability directly comparable to the binary label
(`churned = 1` is exactly "no return observed by the end of the window", the
same event a survival curve's endpoint describes) -- but the person-period
form also answers a question the binary label cannot: which period the risk
is concentrated in, not just whether it clears a threshold by the end.

Standard discrete-time survival analysis (Singer & Willett, 1993): reshape
each entity's follow-up into one row per period it was still at risk, fit an
ordinary binary classifier on the expanded rows (period included as a
feature so it can learn a period-varying baseline hazard), and reconstruct
survival by multiplying `(1 - predicted hazard)` across periods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from churnval.evaluation import score
from churnval.windows import Window, mask_label_events

#: Width of one hazard period, in days. 30 days -- matching
#: `config.DEFAULT_STEP_DAYS` and this dataset's already-established
#: monthly-ish cadence (ADR-0007) -- rather than a finer grain: Online
#: Retail II's RFM features carry no signal at a finer time resolution than
#: that, so a shorter period would multiply the row count (and the training
#: cost) without adding information the features can actually distinguish.
#: See ADR-0012.
PERIOD_DAYS = 30


def first_event_period(
    transactions: pd.DataFrame, window: Window, *, time_col: str, period_days: int = PERIOD_DAYS
) -> pd.Series:
    """Which period, if any, each customer's first in-window event falls in.

    Args:
        transactions: cleaned transaction lines.
        window: the scoring occasion whose label window is being discretized.
        time_col: name of the event-timestamp column in `transactions` --
            no default, deliberately (matches `mask_feature_events`/
            `mask_label_events`'s own no-default convention in
            `churnval.windows`), so a caller can't silently reuse Online
            Retail II's `"invoice_date"` against a differently-shaped
            dataset by omission.
        period_days: width of one period. `window.horizon_days` must be an
            exact multiple of it.

    Returns:
        Series indexed by `customer_id`, 1-indexed period number, for
        customers with at least one event in the label window. Customers
        with no event in the window are absent -- they are censored, not
        assigned a period, which is the point of a survival label rather
        than a binary one.

    Raises:
        ValueError: `window.horizon_days` is not a multiple of `period_days`.
    """
    if window.horizon_days % period_days != 0:
        raise ValueError(
            f"horizon_days ({window.horizon_days}) must be a multiple of "
            f"period_days ({period_days})"
        )
    in_window = mask_label_events(transactions, window, time_col)
    first_event = transactions.loc[in_window].groupby("customer_id")[time_col].min()
    days_into_window = (first_event - window.label_start).dt.days
    return (days_into_window // period_days + 1).rename("event_period")


def build_person_period_panel(
    transactions: pd.DataFrame,
    origins: list[Window],
    panel: pd.DataFrame,
    feature_cols: list[str],
    *,
    time_col: str,
    period_days: int = PERIOD_DAYS,
) -> pd.DataFrame:
    """Expand `03`'s one-row-per-(customer, as_of) panel into person-periods.

    For each origin, each customer eligible at that origin gets one row per
    period they were still at risk: every period up to and including the one
    their first in-window event falls in (`event=1` there, `event=0`
    before it), or every period in the window if no event ever falls in it
    (`event=0` throughout -- censored, not a positive/negative pair the way
    the binary label treats it).

    Args:
        transactions: cleaned transaction lines.
        origins: scoring occasions, typically from
            `churnval.windows.rolling_origins`.
        panel: output of a `build_asof_panel`-shaped function for the same
            `origins` -- supplies the eligible population and its features.
        feature_cols: columns of `panel` to carry onto every person-period
            row (unchanged across periods within one origin -- nothing new
            is learned about a customer partway through their own label
            window).
        time_col: name of the event-timestamp column in `transactions`,
            forwarded to `first_event_period` -- no default, same reasoning
            as there.
        period_days: width of one period.

    Returns:
        One row per (customer_id, as_of, period): `customer_id`, `as_of`,
        `period`, `event`, plus `feature_cols`.
    """
    n_periods = origins[0].horizon_days // period_days
    rows = []
    for window in origins:
        event_period = first_event_period(
            transactions, window, time_col=time_col, period_days=period_days
        )
        origin_panel = panel.loc[panel["as_of"] == window.as_of].set_index("customer_id")

        for customer_id, row in origin_panel.iterrows():
            has_event = customer_id in event_period.index
            observed_period = int(event_period[customer_id]) if has_event else n_periods
            for period in range(1, observed_period + 1):
                rows.append(
                    {
                        "customer_id": customer_id,
                        "as_of": window.as_of,
                        "period": period,
                        "event": int(has_event and period == observed_period),
                        **{col: row[col] for col in feature_cols},
                    }
                )
    return pd.DataFrame(rows)


def survival_from_hazards(hazard_frame: pd.DataFrame) -> pd.Series:
    """Multiply `(1 - hazard)` across periods, per customer, in period order.

    Args:
        hazard_frame: columns `customer_id`, `period`, `hazard` -- one row
            per (customer, period) with that period's predicted hazard.

    Returns:
        Series indexed by `customer_id`, named `survival`: the probability
        of no event through the last period present for that customer.
    """
    ordered = hazard_frame.sort_values(["customer_id", "period"])
    return (1 - ordered["hazard"]).groupby(ordered["customer_id"]).prod().rename("survival")


def hazard_curve(
    model: LGBMClassifier,
    panel_slice: pd.DataFrame,
    feature_cols: list[str],
    n_periods: int,
) -> pd.DataFrame:
    """Score every period for every customer -- the hazard curve before it's collapsed to survival.

    Unlike `build_person_period_panel`'s training rows (truncated at the
    observed event or censoring), every customer here is scored at every
    period from 1 to `n_periods` -- at prediction time there is no observed
    outcome to truncate on, only the periods a real deployment would need a
    hazard estimate for.

    Args:
        model: fitted on `feature_cols + ["period"]` -> `event`.
        panel_slice: one row per customer, with `customer_id` and
            `feature_cols` -- typically `panel` filtered to one `as_of`.
        feature_cols: feature columns the model was trained on.
        n_periods: number of periods to score per customer.

    Returns:
        One row per (customer_id, period): `customer_id`, `period`,
        `hazard` -- the predicted probability of the event in that period,
        given no event before it.
    """
    expanded = panel_slice.loc[panel_slice.index.repeat(n_periods)].reset_index(drop=True)
    expanded["period"] = np.tile(np.arange(1, n_periods + 1), len(panel_slice))
    expanded["hazard"] = model.predict_proba(expanded[[*feature_cols, "period"]])[:, 1]
    return expanded[["customer_id", "period", "hazard"]]


def predict_survival(
    model: LGBMClassifier,
    panel_slice: pd.DataFrame,
    feature_cols: list[str],
    n_periods: int,
) -> pd.Series:
    """`hazard_curve`, reduced to one survival probability per customer.

    Returns:
        Series indexed by `customer_id`, named `survival`:
        `churnval.evaluation.score`-compatible predicted `P(churned)`, since
        "churned" and "no event observed by the end of the window" are the
        same event.
    """
    return survival_from_hazards(hazard_curve(model, panel_slice, feature_cols, n_periods))


def hazard_rolling_origin_backtest(
    person_period_panel: pd.DataFrame,
    panel: pd.DataFrame,
    origins: list[Window],
    *,
    feature_cols: list[str],
    min_training_origins: int,
    seed: int,
    period_days: int = PERIOD_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The hazard-model analogue of `churnval.splits.rolling_origin_backtest`.

    Same purge rule (a training origin is only used once its label window
    has closed before the test origin's `as_of` -- ADR-0009), applied to the
    person-period rows instead of one-row-per-customer rows, so the two
    backtests are comparable origin for origin. The purge filter is
    duplicated from `rolling_origin_backtest` rather than shared, the same
    deliberate choice `churnval.splits.fit_at_origin` makes and for the same
    reason (see that function's docstring).

    Args:
        person_period_panel: output of `build_person_period_panel`.
        panel: output of `churnval.splits.build_asof_panel` for the same
            `origins` -- supplies the test population, its features, and the
            ground-truth `churned` label to score against.
        origins: the same origins both panels were built from, in order.
        feature_cols: columns to use as model features (`"period"` is added
            automatically).
        min_training_origins: see `rolling_origin_backtest`.
        seed: passed to every model fit.
        period_days: width of one period, must match what
            `person_period_panel` was built with.

    Returns:
        `(per_origin, predictions)`, same shape as
        `rolling_origin_backtest`'s: `per_origin` has `as_of`, `n`,
        `prevalence`, `roc_auc`, `pr_auc`, `brier`; `predictions` has
        `as_of`, `customer_id`, `y_true`, `y_prob` (the survival
        probability).
    """
    n_periods = origins[0].horizon_days // period_days
    model_feature_cols = [*feature_cols, "period"]

    per_origin_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for i in range(min_training_origins, len(origins)):
        test_window = origins[i]
        mature_train_windows = [w for w in origins[:i] if w.label_end <= test_window.as_of]
        if not mature_train_windows:
            continue

        train_as_of = [w.as_of for w in mature_train_windows]
        train = person_period_panel[person_period_panel["as_of"].isin(train_as_of)]

        model = LGBMClassifier(random_state=seed, verbosity=-1)
        model.fit(train[model_feature_cols], train["event"])

        test_panel = panel.loc[panel["as_of"] == test_window.as_of].reset_index(drop=True)
        survival = predict_survival(model, test_panel, feature_cols, n_periods)
        y_true = test_panel.set_index("customer_id")["churned"].reindex(survival.index)

        result = score(y_true.to_numpy(), survival.to_numpy(), label=str(test_window.as_of.date()))
        per_origin_rows.append(
            {
                "as_of": test_window.as_of,
                "n": result.n,
                "prevalence": result.prevalence,
                "roc_auc": result.roc_auc,
                "pr_auc": result.pr_auc,
                "brier": result.brier,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "as_of": test_window.as_of,
                    "customer_id": survival.index.to_numpy(),
                    "y_true": y_true.to_numpy(),
                    "y_prob": survival.to_numpy(),
                }
            )
        )

    per_origin = pd.DataFrame(per_origin_rows)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame(columns=["as_of", "customer_id", "y_true", "y_prob"])
    )
    return per_origin, predictions


def fit_hazard_at_origin(
    person_period_panel: pd.DataFrame,
    origins: list[Window],
    target_index: int,
    *,
    feature_cols: list[str],
    seed: int,
) -> tuple[LGBMClassifier, list[Window]]:
    """The hazard-model analogue of `churnval.splits.fit_at_origin`.

    Gives the fitted model `hazard_rolling_origin_backtest` would use to
    test `origins[target_index]`, for a caller that needs the model itself
    -- to plot its per-period hazard curve, for instance, not just its
    aggregated survival prediction. Same purge rule, same deliberate
    duplication of it (rather than a shared refactor of
    `hazard_rolling_origin_backtest`) for the same reason
    `churnval.splits.fit_at_origin` gives.

    Args:
        person_period_panel: output of `build_person_period_panel`.
        origins: the same origins `person_period_panel` was built from, in
            order.
        target_index: index into `origins` of the origin to train for.
        feature_cols: columns of `person_period_panel` to use as model
            features (`"period"` is added automatically).
        seed: passed to the model fit.

    Returns:
        `(model, mature_train_windows)`: the fitted model, and which
        origins it trained on.

    Raises:
        ValueError: no origin before `origins[target_index]` has a label
            window that closes before `origins[target_index].as_of`.
    """
    target_window = origins[target_index]
    mature_train_windows = [w for w in origins[:target_index] if w.label_end <= target_window.as_of]
    if not mature_train_windows:
        raise ValueError(
            f"no mature training origin for origins[{target_index}] ({target_window.as_of})"
        )

    train_as_of = [w.as_of for w in mature_train_windows]
    train = person_period_panel[person_period_panel["as_of"].isin(train_as_of)]
    model = LGBMClassifier(random_state=seed, verbosity=-1)
    model.fit(train[[*feature_cols, "period"]], train["event"])
    return model, mature_train_windows
