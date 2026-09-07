"""The corrected panel and rolling-origin backtest for `03_temporal_protocol`.

Builds the customer x as_of panel the honest way -- as-of-safe features
(`churnval.features`), a real gap, and a real forward-looking label -- and
evaluates it the honest way: retrain at each origin on everything strictly
before it, test on that origin, repeat. Not shared code with
`naive_baseline.py` (whose docstring says nothing in it is reused past
notebook 02): `eligible_customers` here is independently written even though
its logic happens to already be temporally safe there, so this module's
dependency boundary stays unambiguous for later readers.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
from lightgbm import LGBMClassifier

from churnval.evaluation import score
from churnval.features import asof_features
from churnval.windows import Window, mask_label_events

#: A customer needs at least one purchase in this many days before an as_of
#: to be scored at that origin -- the maturity check. Same window ADR-0007
#: established for this dataset's naive panel; the reasoning (a real recent
#: purchase to build features from) applies identically here.
ELIGIBILITY_LOOKBACK_DAYS = 365


def eligible_customers(
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int = ELIGIBILITY_LOOKBACK_DAYS,
) -> pd.Index:
    """Customers with at least one purchase in `lookback_days` before as_of.

    Args:
        transactions: cleaned transaction lines.
        as_of: the scoring occasion. The lookback window is
            `[as_of - lookback_days, as_of)`.
        lookback_days: width of the lookback window.

    Returns:
        Distinct customer IDs eligible at this origin, named `customer_id`
        so it survives a `.loc[]` selection without corrupting the target
        frame's index name.
    """
    window_start = as_of - timedelta(days=lookback_days)
    recent = transactions[
        (transactions["invoice_date"] >= window_start) & (transactions["invoice_date"] < as_of)
    ]
    return pd.Index(recent["customer_id"].unique(), name="customer_id")


def build_asof_panel(transactions: pd.DataFrame, origins: list[Window]) -> pd.DataFrame:
    """One row per (customer, as_of): the corrected panel.

    For each origin: restrict to eligible customers, compute each one's
    features strictly before that origin's `as_of` (`asof_features` --
    recomputed fresh per origin, unlike the naive panel's reused
    full-history values), and set the label from
    `churnval.windows.mask_label_events` on that origin's label window,
    which opens `window.gap_days` after `as_of`.

    Args:
        transactions: cleaned transaction lines.
        origins: scoring occasions, typically from
            `churnval.windows.rolling_origins`.

    Returns:
        One row per (customer_id, as_of): customer_id, as_of, recency_days,
        frequency, monetary, churned.
    """
    rows = []
    for window in origins:
        eligible = eligible_customers(transactions, window.as_of)
        features = asof_features(transactions, window.as_of).loc[eligible]

        in_window = mask_label_events(transactions, window, "invoice_date")
        purchasers = set(transactions.loc[in_window, "customer_id"])

        panel = features.copy()
        panel["as_of"] = window.as_of
        panel["churned"] = (~panel.index.isin(purchasers)).astype(int)
        rows.append(panel.reset_index())
    return pd.concat(rows, ignore_index=True)


def rolling_origin_backtest(
    panel: pd.DataFrame,
    origins: list[Window],
    *,
    feature_cols: list[str],
    min_training_origins: int,
    seed: int,
    max_lookback_days: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrain forward at each origin; test only on strictly later data.

    At each origin from `min_training_origins` onward: fit on every panel
    row whose `as_of` is one of the *earlier* origins (an expanding window,
    not a fixed lookback -- see ADR-0009 for why), then score on that
    origin's own rows. A customer appearing in both an earlier training
    origin and the current test origin is not a leak here the way it was in
    `01_naive_baseline` -- what matters is that every training row's
    features and label come from strictly before the test origin's `as_of`,
    which `build_asof_panel` already guarantees per row; a real customer
    persisting across origins is what a real deployment looks like.

    A training origin is only used if its *label window has fully closed*
    before the test origin's `as_of` -- not merely if it comes earlier.
    Pooling every prior origin regardless of label maturity would train on
    rows whose true label (a 90-day-out outcome) had not actually happened
    yet as of the simulated "now" being backtested; that label is real and
    correctly computed from this offline dataset's full history, but it
    would not have been *knowable* at that point in a live deployment.
    Because `HORIZON_DAYS` and `GAP_DAYS` together span more than three
    origin-steps for this dataset, this excludes the three most recent
    origins before every test origin, regardless of how many origins have
    accumulated by then.

    Args:
        panel: output of `build_asof_panel`.
        origins: the same origins `panel` was built from, in order.
        feature_cols: columns of `panel` to use as model features.
        min_training_origins: how many of the earliest origins to reserve
            as training-only before the first origin is even considered for
            testing. This is a floor, not a guarantee -- an origin is only
            actually tested once at least one *mature* training origin
            exists for it (see above); with this dataset's window sizes,
            that takes more origins than `min_training_origins` alone would
            suggest, and any origin still short a mature training set is
            skipped rather than trained on immature labels.
        seed: passed to every model fit.
        max_lookback_days: if given, additionally restrict training origins
            to those within this many days of the test origin's `as_of` --
            a *sliding* window on top of the maturity filter, rather than
            the default *expanding* one (every mature origin, however old).
            A training origin still has to be mature regardless of this
            argument; a short enough `max_lookback_days` can therefore
            leave a test origin with no usable training data at all (e.g.
            90 days is shorter than this dataset's own maturity delay, so
            no origin ever has both a mature *and* within-90-days training
            origin -- every test origin gets skipped). See ADR-0009.

    Returns:
        `(per_origin, predictions)`:
        - `per_origin`: one row per tested origin -- `as_of`, `n`,
          `prevalence`, `roc_auc`, `pr_auc`, `brier`. Empty (zero rows) if
          `max_lookback_days` leaves every origin without mature,
          in-window training data.
        - `predictions`: one row per test prediction across every tested
          origin -- `as_of`, `customer_id`, `y_true`, `y_prob`. Concatenate
          and score this directly for one pooled, honest headline number
          rather than averaging the per-origin metrics. Empty if
          `per_origin` is.
    """
    per_origin_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for i in range(min_training_origins, len(origins)):
        test_window = origins[i]
        mature_train_windows = [w for w in origins[:i] if w.label_end <= test_window.as_of]
        if max_lookback_days is not None:
            cutoff = test_window.as_of - timedelta(days=max_lookback_days)
            mature_train_windows = [w for w in mature_train_windows if w.as_of >= cutoff]
        if not mature_train_windows:
            continue

        train_as_of = [w.as_of for w in mature_train_windows]
        test_as_of = test_window.as_of

        train = panel[panel["as_of"].isin(train_as_of)]
        test = panel[panel["as_of"] == test_as_of]

        model = LGBMClassifier(random_state=seed, verbosity=-1)
        model.fit(train[feature_cols], train["churned"])
        y_prob = model.predict_proba(test[feature_cols])[:, 1]

        result = score(test["churned"], y_prob, label=str(test_as_of.date()))
        per_origin_rows.append(
            {
                "as_of": test_as_of,
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
                    "as_of": test_as_of,
                    "customer_id": test["customer_id"].to_numpy(),
                    "y_true": test["churned"].to_numpy(),
                    "y_prob": y_prob,
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


def fit_at_origin(
    panel: pd.DataFrame,
    origins: list[Window],
    target_index: int,
    *,
    feature_cols: list[str],
    seed: int,
) -> tuple[LGBMClassifier, list[Window]]:
    """The fitted model `rolling_origin_backtest` would use to test `origins[target_index]`.

    Factors out that function's purge-then-fit step for a caller that needs
    the fitted *model itself*, not just its predictions -- `05_calibration`'s
    calibration slice, for one, which needs to score the same origin with a
    model it can also reuse afterwards on a later origin.
    `rolling_origin_backtest` is left untouched rather than rewritten to call
    this helper internally: it is already tested and ADR-documented
    (ADR-0009), and the one-line purge filter duplicated here is simple
    enough that the duplication is lower-risk than touching that function's
    internals for this.

    Args:
        panel: output of `build_asof_panel`.
        origins: the same origins `panel` was built from, in order.
        target_index: index into `origins` of the origin to train for.
        feature_cols: columns of `panel` to use as model features.
        seed: passed to the model fit.

    Returns:
        `(model, mature_train_windows)`: the fitted model, and which
        origins it trained on, so a caller can log or verify the selection.

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
    train = panel[panel["as_of"].isin(train_as_of)]
    model = LGBMClassifier(random_state=seed, verbosity=-1)
    model.fit(train[feature_cols], train["churned"])
    return model, mature_train_windows


def frozen_model_backtest(
    panel: pd.DataFrame,
    origins: list[Window],
    *,
    feature_cols: list[str],
    min_training_origins: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train once, at the earliest origin with mature training data; never retrain.

    Answers a different question than `rolling_origin_backtest`: not "does
    the correct protocol perform honestly," but "how much is retraining at
    every origin actually buying." One model is fit at the first origin
    index (from `min_training_origins` onward) that has any mature training
    data at all -- the same origin `rolling_origin_backtest` would first
    test, and on the same training rows it would use there -- then that one
    model, un-retrained, scores every origin from that point through the
    last one. If performance holds up as well as the retrained version
    across `origins_since_training`, retraining that often isn't earning
    its cost; if it decays, that decay is the argument for retraining.

    Args:
        panel: output of `build_asof_panel`.
        origins: the same origins `panel` was built from, in order.
        feature_cols: columns of `panel` to use as model features.
        min_training_origins: same meaning as in `rolling_origin_backtest`
            -- how many of the earliest origins to reserve before even
            looking for the first origin with mature training data.
        seed: passed to the single model fit.

    Returns:
        `(per_origin, predictions)`:
        - `per_origin`: one row per scored origin -- `as_of`,
          `origins_since_training` (0 at the freeze point itself, since
          that origin is also scored by the frozen model), `n`,
          `prevalence`, `roc_auc`, `pr_auc`, `brier`.
        - `predictions`: one row per prediction, same columns as
          `rolling_origin_backtest`'s.

    Raises:
        ValueError: no origin in `origins` has any mature training data,
            so there is nothing to freeze.
    """
    train_origin_index: int | None = None
    train_windows: list[Window] = []
    for i in range(min_training_origins, len(origins)):
        mature = [w for w in origins[:i] if w.label_end <= origins[i].as_of]
        if mature:
            train_origin_index = i
            train_windows = mature
            break
    if train_origin_index is None:
        raise ValueError("no origin in `origins` has any mature training data to freeze on")

    train_as_of = [w.as_of for w in train_windows]
    train = panel[panel["as_of"].isin(train_as_of)]
    model = LGBMClassifier(random_state=seed, verbosity=-1)
    model.fit(train[feature_cols], train["churned"])

    per_origin_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for i in range(train_origin_index, len(origins)):
        test_window = origins[i]
        test = panel[panel["as_of"] == test_window.as_of]
        y_prob = model.predict_proba(test[feature_cols])[:, 1]

        result = score(test["churned"], y_prob, label=str(test_window.as_of.date()))
        per_origin_rows.append(
            {
                "as_of": test_window.as_of,
                "origins_since_training": i - train_origin_index,
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
                    "customer_id": test["customer_id"].to_numpy(),
                    "y_true": test["churned"].to_numpy(),
                    "y_prob": y_prob,
                }
            )
        )

    per_origin = pd.DataFrame(per_origin_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return per_origin, predictions
