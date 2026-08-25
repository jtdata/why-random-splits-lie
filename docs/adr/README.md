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
