# ADR-0006: MLflow tracks to a local file store during development

- **Status:** Accepted
- **Date:** 2026-08-25
- **Deciders:** JT Moeller

## Context

This project produces many runs that differ only in validation configuration —
split strategy, gap length, origin date, calibrator. Comparing them by hand is
error-prone, and the comparison *is* the result, so it has to be auditable.

A persistent MLflow server on the household TrueNAS box is planned for the
portfolio, but standing it up is a separate piece of work and this project must
not be blocked on it.

## Decision

Track every run to a local file store at `./mlruns` via
`MLFLOW_TRACKING_URI=file:./mlruns`. Log the validation configuration as
parameters, every metric in the reporting set, and the reliability curve and
gap chart as artefacts. `mlruns/` is gitignored.

## Alternatives considered

| Option | Why it was plausible | Why it was rejected |
|---|---|---|
| MLflow server on TrueNAS now | Persistent, always on, screenshots well, mirrors the Databricks workspace | Blocks the project on infrastructure work; a reviewer cloning the repo cannot reach it, so results would not be reproducible from the repo alone |
| A CSV of results | No dependency at all | Loses artefacts and parameter provenance; the comparison table is the deliverable and needs to be regenerable |
| Weights & Biases | Better UI, hosted, free tier | Sends run data to a third party, and adds an account requirement for anyone reproducing the work |

## Consequences

**Good:** zero setup, works offline, reproducible from a clone, and the run
comparison that forms the headline table is generated rather than typed.

**Bad:** runs are not shared across machines and are lost if the working
directory is deleted. Nothing in the repo depends on them surviving.

**Revisit if:** a second machine (the TrueNAS box) starts producing runs for
this project, at which point a shared tracking server becomes worth the setup.
