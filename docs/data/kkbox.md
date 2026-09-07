# KKBox WSDM Churn Prediction Challenge

- **Source:** https://www.kaggle.com/c/kkbox-churn-prediction-challenge
  (Kaggle, WSDM 2018)
- **Licence / terms:** Kaggle competition data, subject to the competition's
  own rules (acceptance required before download). This repo does not
  redistribute the data — fetched manually per `churnval fetch kkbox`'s
  instructions (`kaggle competitions download`), which require accepting
  those rules first.
- **Retrieved:** 2026-08-27
- **Files used:** `transactions.csv` (1.73 GB, 21,547,746 rows) and
  `members_v3.csv` (427 MB, ~6.77M rows — inspected, then deliberately not
  used; see Known quirks). `train.csv`/`train_v2.csv` (the competition's own
  official `is_churn` labels) are used only as an informal, heavily-caveated
  cross-check in `07_generalisation`, never as ground truth for this
  project's own numbers. `transactions_v2.csv` (a smaller supplementary file
  for the competition's second phase) is not used — `transactions.csv` alone
  spans 26 months and 21.5M rows, ample for this protocol's needs.
- **Redistributed in this repo:** No — fetched manually into
  `data/raw/kkbox/`

## Shape

Figures are for the frame returned by
`churnval.kkbox_io.load_kkbox_transactions`, i.e. **after** cleaning (see
Known quirks).

| Table | Rows | Columns | Bytes (parquet) |
|---|---|---|---|
| transactions (cleaned) | 21,537,701 | 9 | 1,088,697,717 |

Raw row count: 21,547,746. Net rows dropped by cleaning: 10,045 (0.05%) —
far fewer than the sum of the three individual drop-rule counts below,
because the rule sets overlap (a corrupt sentinel date is often also an
internally-inconsistent one). Distinct `customer_id` (renamed from `msno`)
**after cleaning**: 2,362,901 — 725 fewer than the raw file's 2,363,626
distinct `msno` values (below), because a small number of customers'
transaction rows are entirely dropped by cleaning (e.g. a customer whose
only row has a corrupt sentinel expiry date).

## Temporal coverage

- `transaction_date` is the date a subscription transaction was recorded —
  the event date. `membership_expire_date` is a second, forward-looking date
  on the same row: the membership's expiry as of that transaction, which is
  what this project's eligibility rule and label construction are built
  around (see [ADR-0013](../adr/0013-kkbox-eligibility-and-label-definition.md)).
- `transaction_date` min: 2015-01-01. Max: 2017-02-28. Span: ~26 months.
- Timezone: not stated by the source; both date columns are integer
  `YYYYMMDD` with no time-of-day component, parsed here to midnight-anchored
  timestamps and treated as naive throughout, same convention as Online
  Retail II's `invoice_date`.

## Label

This dataset has no per-row churn label — like Online Retail II, the label
is constructed downstream, and like Online Retail II, the construction here
is *not* identical to any single "official" definition:

- **KKBox's own competition rule** (used to build `train.csv`/`train_v2.csv`,
  not reproduced by this project): a member has churned if they fail to
  make a new valid subscription within 30 days of *their own*
  `membership_expire_date`. Anchored per-customer, no shared scoring date.
- **This project's label** (`churnval.kkbox_features.build_kkbox_asof_panel`,
  `07_generalisation`): `churned = 0` iff a non-cancellation transaction
  (`is_cancel = 0`) lands in `[as_of + gap_days, as_of + gap_days +
  horizon_days)` for a shared `as_of` per origin — the same
  `Window`/`rolling_origins` structure `03`'s Online Retail II protocol
  uses. A cancellation transaction inside that window counts as *stronger*
  evidence of churn, not a renewal. See ADR-0013 for the full reconciliation
  argument and why this project doesn't reproduce KKBox's own rule
  byte-for-byte.
- Eligibility (who gets scored at all) is expiry-based: a customer's current
  membership must expire in `[as_of - 45 days, as_of + gap_days)` — see
  ADR-0013. Customers outside that window are excluded, not labelled
  negative.

## Known quirks

- **`is_cancel = 1` rows are real, informative subscription events, not
  noise.** Unlike Online Retail II's cancelled invoices (dropped outright
  by `churnval.io._clean`), a KKBox cancellation transaction is kept through
  cleaning — it feeds `n_cancellations_lookback`
  (`churnval.kkbox_features`) and is excluded only at label-construction
  time, where it deliberately does *not* count as a renewal.
- **3,339 exact full-row duplicates**, dropped. No explanation offered by
  the source; treated the same way Online Retail II's ~26K duplicate lines
  are — not deduplicated by choice, dropped because they are literally
  identical rows, not merely similar ones.
- **1,778 rows carry a corrupt sentinel `membership_expire_date`** (e.g.
  `1970-01-01`), dropped — these would otherwise produce absurd
  `days_until_expiry` outliers (a membership "expiring" 45+ years in the
  past).
- **6,460 rows have `membership_expire_date < transaction_date` and
  `is_cancel = 0`** — an active-subscription transaction claiming to expire
  before it happened, internally inconsistent and not explainable as a
  legitimate cancellation the way an `is_cancel = 1` row with the same date
  ordering is (of which there are many more — kept, per above). Dropped.
- **255,595 `(customer_id, transaction_date)` pairs have more than one
  transaction row**, surviving cleaning untouched (they aren't exact
  duplicates — see above — just two distinct rows dated the same day), and
  94.6% of those groups disagree on `membership_expire_date` between the
  tied rows. This isn't cosmetic: found under adversarial review,
  `churnval.kkbox_features._current_membership_state`'s `ASOF JOIN` gives
  no tie-break guarantee for same-date rows, so an earlier version of this
  pipeline picked a different one on every run, changing eligibility and
  labels, not just a feature value. Fixed with an explicit, deterministic
  tie-break (`_dedupe_for_asof_join`: largest `membership_expire_date`
  wins) before the join — see ADR-0015's addendum.
- **9.5% of transactions have `actual_amount_paid = 0`**, and most of these
  are *not* cancellations — plausibly free trials or fully-discounted
  renewals. Kept as a known quirk, not corrected; no evidence they are
  invalid rows.
- **4.0% of transactions have `payment_plan_days = 0`**, and 851,535 of
  those have `is_cancel = 0` with a normal-looking future expiry date —
  looks like a legitimate promotional/adjustment pattern (a fee waiver on
  an otherwise-real transaction), not garbage. Kept.
- **`payment_plan_days` is not uniformly 30 days.** 88% of transactions are
  on a 30-day plan, 3.6% on 31-day, 2.7% on a 7-day trial, with a real tail
  at 90/100/180/195/410 days. This is why
  `churnval.kkbox_features.ELIGIBILITY_LOOKBACK_DAYS` (45 days) is wider
  than KKBox's own 30-day grace convention — see ADR-0013.
- **`members_v3.csv` is not used anywhere in this project.** Its `bd` (age)
  column is outside the plausible `[1, 100]` range for 67% of rows (min
  observed: -7168; max: 2016 — clearly not birth years or ages), and
  `gender` is null in 65% of rows. Measured directly, not estimated. The
  corrected protocol doesn't need demographics on top of transaction-derived
  features anyway (`03` already proved this for Online Retail II's RFM
  features), so no loader for this file exists in `src/churnval/`.
- **2,363,626 distinct customers ever appear in the raw `transactions.csv`**
  (2,362,901 after cleaning, above), but far fewer are *eligible* at any
  single origin (200K-424K, per ADR-0013) —
  most customers' subscriptions simply aren't due for a renewal decision at
  a given `as_of`, the contractual analogue of Online Retail II's "28% of
  customers have exactly one invoice" observation.

## Fitness for this project

Yes, and for a complementary reason to Online Retail II's: KKBox's
`membership_expire_date` is an explicit, per-row forward-looking event —
this dataset's churn signal is observed structure (a subscription lapsing),
not inferred from a purchase gap the way Online Retail II's is. Pairing the
two datasets (per [ADR-0003](../adr/0003-dataset-selection.md)) tests
whether this project's temporal-validation protocol generalizes across that
exact difference — contractual vs. non-contractual churn — not just across
two arbitrary datasets. See ADR-0013/0014/0015 for the KKBox-specific
eligibility, module-reuse, and scale decisions this required.
