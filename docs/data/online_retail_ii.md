# Online Retail II

- **Source:** https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
  (UCI Machine Learning Repository)
- **Licence / terms:** CC BY 4.0. Redistribution with attribution is
  permitted by the licence; this repo does not redistribute the data anyway —
  it is fetched by `uv run churnval fetch retail`.
- **Retrieved:** 2026-08-25
- **Files used:** `online_retail_II.xlsx` (43.5 MB), two sheets — `Year
  2009-2010` and `Year 2010-2011`
- **Redistributed in this repo:** No — fetched by `churnval fetch retail`
  into `data/raw/online_retail_ii/`

## Shape

Figures are for the frame returned by `churnval.io.load_online_retail_ii`,
i.e. **after** cleaning (dropped: no customer ID, cancellations, non-positive
quantity or price, administrative line items — see Known quirks).

| Table | Rows | Columns | Bytes (parquet) |
|---|---|---|---|
| transactions (cleaned) | 802,651 | 9 | 6,402,954 |

Distinct customers: 5,852. Distinct invoices: 36,594. Countries: 41.

## Temporal coverage

- `invoice_date` is the transaction timestamp — the event date, not a
  separately recorded date. There is no distinct "recorded date" in this
  dataset.
- Min: 2009-12-01 07:45:00. Max: 2011-12-09 12:50:00. Span: just over 24
  months.
- Timezone: not stated by the source. Assumed to be the retailer's local time
  (UK) and treated as naive (tz-unaware) throughout this repo — no
  cross-timezone joins are performed against it.

## Label

This dataset has no recorded churn label — it is a raw transaction log. The
label is constructed downstream:

- `01_naive_baseline` (`churnval.naive_baseline.build_naive_panel`, via
  `churnval.windows.mask_label_events`): at each of several `as_of` scoring
  occasions, a customer is "churned" if they make no purchase during the
  `HORIZON_DAYS`-wide window that opens after `as_of` (with `GAP_DAYS = 0`
  between them — see [ADR-0007](../adr/0007-online-retail-ii-naive-panel-definition.md)
  for why `HORIZON_DAYS = 90` for this dataset). This label is genuinely
  forward-looking; the notebook's intentional mistakes are in the features
  and the split built around it, not in what "churn" means here.
- `01_naive_baseline` also keeps a second, deliberately more broken
  construction, `naive_baseline.tautological_label`, purely as a closing
  demonstration: churn defined as "no purchase in the final `HORIZON_DAYS` of
  the *whole dataset*," anchored to the same reference date as the recency
  feature — making the two identical by construction (ROC AUC ≈ 1.0). This
  is flagged in the notebook as a bug, not a result, and is not part of the
  panel above.
- The corrected, as-of label definition is built in `03_temporal_protocol`
  and will be recorded here once that notebook is written.
- Customers whose outcome window has not closed are excluded, not labelled
  negative: `churnval.windows.rolling_origins` drops any origin whose label
  window would extend past the dataset's last event, so no scoring occasion
  in the panel has a truncated or still-open label window.

## Known quirks

- **Administrative line items share the transaction log with real
  purchases.** `StockCode` values `POST`/`DOT` (postage), `M` (manual),
  `C2` (carriage), `BANK CHARGES`, `ADJUST`/`ADJUST2`, `D` (discount), and
  `TEST001`/`TEST002` (literal test rows) are not products — they were
  found by inspecting the description text behind every non-numeric stock
  code in the cleaned data. `_clean` now drops them
  (`churnval.io.EXCLUDED_STOCK_CODES`); before this fix, 25 customers had
  their *entire* purchase history made of nothing but these rows (one
  customer's `monetary` feature was ~$13,916 built from two "Manual"
  adjustment lines alone) — their RFM features and naive-panel labels were
  computed from fees, not behaviour. Caught in adversarial review of
  `01_naive_baseline`, not by inspection beforehand; regression-tested in
  `tests/test_io.py::test_clean_drops_administrative_line_items`. **Not
  caught by this filter:** a small number of real-looking numeric SKUs are
  also fees in disguise (e.g. `23444` "Next Day Carriage", `23574`
  "PACKING CHARGE") — these were not enumerated and remain in the cleaned
  data; revisit if a future notebook's per-customer monetary total looks
  implausible.
- **26,055 fully duplicate transaction lines** (same invoice, stock code,
  quantity, price, customer, timestamp) after cleaning. The source
  documentation for this dataset does not explain these; they may be
  legitimate repeated scans of the same item within an order or genuine
  double-entries. Not deduplicated here — no decision has been made yet
  about whether frequency/monetary features should treat them as one line
  or two. Whoever builds `features.py` in `03_temporal_protocol` should
  decide explicitly and record it (ADR if it changes a reported number).
- **`stock_code` is mixed-type across the two sheets** — mostly numeric-
  looking, but with alphanumeric codes like `"79323P"` and administrative
  codes (e.g. postage, discounts) in both sheets. `_clean` casts it to `str`
  unconditionally; failing to do so breaks the Parquet write outright
  (`pyarrow.lib.ArrowInvalid`), which is how this was caught — see
  `tests/test_io.py::test_clean_stock_code_is_string_even_when_mixed_with_numeric_codes`.
- **1,618 of 5,852 customers (28%) have exactly one invoice** in the cleaned
  data. A single-invoice customer has no purchase-gap history at all, which
  matters for any feature (or hazard model, in `06_hazard_framing`) that
  relies on inter-purchase intervals.
- **Country is self-reported at checkout**, not verified; treat cross-country
  comparisons as indicative only.
- **No separate `Description` nulls survive cleaning** (0 after dropping
  rows with no customer ID) — earlier UCI releases of this dataset are known
  to have unlabelled stock codes with blank descriptions; that turned out not
  to affect the customer-level features used here since `description` is not
  used as a feature.

## Fitness for this project

Yes, with the caveat already recorded in
[ADR-0003](../adr/0003-dataset-selection.md): the event (`invoice_date`) is a
real transaction timestamp, so features can be snapshotted strictly before an
as-of date and a temporal split is possible. What this dataset does **not**
give you is an explicit churn event — "churn" for a non-contractual retailer
is inferred from a purchase gap, which is a definitional choice (the
`HORIZON_DAYS` window) rather than an observed fact the way a KKBox
subscription cancellation is. That is the point of pairing this dataset with
KKBox rather than using it alone.
