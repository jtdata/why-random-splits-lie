# Architecture Decision Records

Decisions that closed off a real alternative, recorded when they were made.

Use the `adr` skill to add one. Never edit an accepted ADR's substance —
supersede it with a new record and link both ways.

| # | Decision | Status | Date |
|---|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted | 2026-08-25 |
| [0002](0002-repo-per-project.md) | One git repo per portfolio project, standards vendored into each | Accepted | 2026-08-25 |
| [0003](0003-dataset-selection.md) | Use KKBox and Online Retail II; reject the Telco churn dataset | Accepted | 2026-08-25 |
| [0004](0004-duckdb-over-spark.md) | DuckDB over Parquet instead of Spark for local scale | Accepted | 2026-08-25 |
| [0005](0005-notebooks-narrate-src-implements.md) | Notebooks narrate, `src/` implements, jupytext pairs them | Accepted | 2026-08-25 |
| [0006](0006-mlflow-local-file-store.md) | MLflow local file store during development | Accepted | 2026-08-25 |
| [0007](0007-online-retail-ii-naive-panel-definition.md) | Online Retail II naive panel: 90-day horizon, 365-day >=1-purchase eligibility | Accepted | 2026-08-25 |
| [0008](0008-lightgbm-shared-model-family.md) | LightGBM is the single model family reused across 01, 03, and 04 | Accepted | 2026-08-25 |
| [0009](0009-rolling-origin-backtest-window-and-purge.md) | Rolling-origin backtest: expanding training window with a label-maturity purge | Accepted | 2026-08-25 |
| [0010](0010-holdout-comparison-common-origin.md) | Naive-vs-correct holdout comparison: one common origin/population, isolating training/serving skew | Accepted | 2026-08-26 |
| [0011](0011-calibration-split-and-honest-ev-threshold.md) | Calibration: three-way temporal split, and EV threshold derived from costs, never chosen by peeking at evaluation outcomes | Accepted | 2026-08-26 |
| [0012](0012-hazard-period-width.md) | Discrete-time hazard model uses 30-day periods (3 per 90-day horizon) | Accepted | 2026-08-26 |
| [0013](0013-kkbox-eligibility-and-label-definition.md) | KKBox eligibility is expiry-based; no HORIZON_DAYS/GAP_DAYS override; label doesn't reproduce KKBox's own churn definition | Accepted | 2026-08-28 |
| [0014](0014-kkbox-module-reuse-strategy.md) | KKBox reuses backtest/evaluation/calibration/hazard engines unchanged, rebuilds panel construction fresh, generalizes one hazard function in place | Accepted | 2026-08-28 |
| [0015](0015-duckdb-for-kkbox-scale.md) | KKBox's read/clean/panel-construction layer uses DuckDB, not pandas | Accepted | 2026-08-28 |
