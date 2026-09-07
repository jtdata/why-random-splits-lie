# ADR-0004: We use DuckDB over Parquet instead of Spark for local processing

- **Status:** Accepted
- **Date:** 2026-08-25
- **Deciders:** JT Moeller

## Context

The author's professional background is Databricks and PySpark, so Spark is the
reflexive choice. The largest table in this project is KKBox `user_logs` at tens
of gigabytes; the tables actually used for the core argument are a few
gigabytes. The machine is a single Windows laptop with 32 GB of RAM, and the
project must run entirely locally.

## Decision

Store everything as Parquet and query it with DuckDB. Use SQL for set-based
work (as-of joins, windowed aggregations, label construction) and pandas only
for row-wise logic that SQL would obscure.

## Alternatives considered

| Option | Why it was plausible | Why it was rejected |
|---|---|---|
| PySpark local mode | Familiar; the code would port to Databricks unchanged | JVM startup, shuffle overhead and configuration cost on a single machine buy nothing at this scale; a reviewer cloning the repo has to install a JVM to run it |
| pandas alone | Simplest; no new dependency | The larger joins exceed comfortable memory and force manual chunking, which is exactly the kind of incidental complexity that obscures an argument |
| Polars | Fast, modern, good Windows story | Genuinely viable. DuckDB chosen because the as-of and windowed logic reads better as SQL, and SQL is more legible to the reviewers this repo targets |

## Consequences

**Good:** no cluster, no JVM, single `uv sync` to reproduce; larger-than-memory
joins work out of core; the as-of join logic is legible as SQL, which is itself
part of the teaching value.

**Bad:** the code does not port to a Spark cluster unchanged, and DuckDB SQL has
dialect quirks that differ from Spark SQL.

**Revisit if:** a later project in the portfolio genuinely needs distributed
compute, in which case that project makes its own decision rather than
retrofitting this one.
