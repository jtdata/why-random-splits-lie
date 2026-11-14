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
scripts/           build_site.py renders reports/executed into the case-study site
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
.\make.ps1 site              # render the executed notebooks to reports/_site (local preview)
.\make.ps1 results           # push reports/executed to the `executed` branch
.\make.ps1 publish           # manual fallback: push reports/_site to gh-pages
```

The published site is https://jtdata.github.io/why-random-splits-lie/. GitHub
Actions renders and publishes it (`.github/workflows/site.yml`) on every push
to `main` or `executed`. CI renders, it never executes: the notebooks cannot
run without `data/`, which is gitignored by design, and `07` needs Kaggle
credentials plus about half an hour. See ADR-0016.

- A change to prose, the README, the checklist, an ADR or the site builder:
  push `main` from any machine. The site rebuilds from the outputs already on
  `executed`.
- A change to results: on a machine that has `data/`, run `nb`, commit the
  refreshed `reports/results_*.json` and `reports/figures/` to `main`, then
  run `results`. Push `main` first so the numbers and the outputs it is
  rendered with come from the same run.

The site is a case study, not a notebook dump, and `scripts/build_site.py` owns
that presentation: the landing page's numbers are read from
`reports/results_*.json`, the notebooks are grouped into chapters by the
`CHAPTERS` list, and code cells are folded by `scripts/site_template/report`.
A new notebook must be added to a chapter there. The author card comes from
`AUTHOR` in the same file; a link with an empty URL is not shown.

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
