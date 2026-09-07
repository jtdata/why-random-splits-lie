# Coding standards

These apply to every project in the portfolio. They are deliberately short,
a standard nobody reads is decoration.

## Environment

- Python 3.12. One `uv`-managed environment per project, never shared.
- Dependencies declared in `pyproject.toml` and locked in `uv.lock`, which is
  committed. `uv sync` is the only supported way to build the environment.
- A new dependency needs a reason. If it replaces something already present,
  remove the old one in the same commit.

## Layout

- All importable code under `src/<package>/`. The package is installed into the
  environment in editable mode, so notebooks and tests import it the same way a
  user would, no `sys.path` manipulation anywhere.
- Modules are small and named for what they do: `io.py`, `features.py`,
  `splits.py`, `evaluation.py`, `calibration.py`.
- One config module (`config.py`) owns every path and constant. Nothing else
  constructs a path. No absolute paths in any file.

## Style

- `ruff` for lint and format, line length 100, config in `pyproject.toml`.
  Formatting is not a matter of opinion; run the formatter.
- Type hints on every public function signature. Not required inside function
  bodies.
- Docstrings on every public function, and the docstring says *why the function
  exists*, not what each line does. Where a function implements a method with a
  source, cite it.
- Names describe the thing, not its type. `features_asof` not `df2`.
- No mutable default arguments, no bare `except`, no `print` in `src/`, use
  `logging`.

## Data handling

- `data/raw/` is immutable once written. Nothing writes back to it.
- Intermediate artefacts go to `data/interim/`, publishable ones to
  `data/processed/`, always Parquet.
- Anything larger than comfortable in pandas goes through DuckDB over Parquet.
  SQL for set-based work, Python for row-wise logic that SQL would obscure.
- Every dataset gets a dataset card in `docs/data/` recording source, licence,
  retrieval date, row count, and known quirks.

## Randomness and reproducibility

- One seed constant in `config.py`. Every stochastic call takes it explicitly.
  No reliance on global numpy state.
- Any script or notebook that produces a number in a report writes that number
  to a file under `reports/`, so prose is never hand-transcribed.

## Testing

- `pytest`. Tests cover the parts where being wrong would be silent: split
  boundaries, as-of feature computation, label construction, metric
  implementations.
- Every leakage bug found gets a regression test before it gets a fix.
- Tests do not require the real datasets. Use small synthetic fixtures with
  known answers.

## Git

- `main` is always runnable. Work on short-lived branches when a change spans
  more than one sitting.
- Commit messages: imperative subject under 72 characters, then a body
  explaining *why*. `fix: exclude label window from feature aggregation` beats
  `update features`.
- One logical change per commit. A commit that touches a notebook, a module and
  the README because they belong to one change is fine; a commit that does
  three unrelated things is not.
- Never commit data, credentials, or notebook outputs.
