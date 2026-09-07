# churn-temporal-validation

## What this project is

A controlled demonstration that random train/test splits inflate offline churn
model performance, and a worked protocol for validating properly. The
deliverable is an argument with receipts, not a model.

The headline result is a single number: **the gap between offline performance
under a random split and realised forward performance.** Everything else in the
repo exists to make that number credible.

Read `docs/timeline.md` for the notebook sequence and `docs/adr/README.md` for
decisions already settled.

## Standards

@docs/standards/coding.md
@docs/standards/notebooks.md
@docs/standards/validation.md

## The rule that governs this repo

Every model in this repo is validated by time, with an explicit gap between the
feature as-of date and the label window, and evaluated with a rolling-origin
backtest. The one exception is `01_naive_baseline`, which does it wrong **on
purpose**: that notebook must carry a prominent warning banner in its opening
markdown cell so it can never be mistaken for the recommended approach.

## Layout

```
src/churnval/      importable logic: config, io, features, splits, evaluation
notebooks/         numbered narrative, jupytext-paired
docs/adr/          decision records
docs/standards/    the rules this repo follows (published with the repo)
data/              gitignored; fetched by `churnval fetch`
reports/figures/   exported charts referenced by the essay
```

## Commands

```powershell
uv sync                      # create/refresh the environment
uv run churnval fetch retail # download a dataset into data/raw
uv run pytest                # tests
uv run ruff check --fix .    # lint
uv run ruff format .         # format
uv run jupytext --sync notebooks/*.ipynb
.\make.ps1 nb                # execute all notebooks in order into reports/
.\make.ps1 site              # render the executed notebooks to reports/_site
.\make.ps1 publish           # push reports/_site to the gh-pages branch
```

The published site is https://jtdata.github.io/why-random-splits-lie/. It is
built locally, never in CI: the notebooks cannot run without `data/`, which is
gitignored by design, and `07` needs Kaggle credentials plus about half an
hour. So `nb`, then `site`, then `publish`, in that order, and only when the
outputs are worth republishing.

## Working notes for Claude

- Do not add a dependency without saying what it replaces and why the stdlib or
  an existing dep will not do. Record anything non-obvious as an ADR.
- Do not put modelling logic in a notebook. Put it in `src/churnval/` and
  import it. If you are unsure which module, ask.
- When a validation choice is made, run the `leakage-audit` skill against it
  before moving on.
- Treat `data/` as read-only once fetched. Derived artefacts go to
  `data/interim/` and `data/processed/`, never over the raw files.
- No results are ever hardcoded into prose. If prose needs a number, the
  notebook computes it and the prose is generated or copied from the output.
