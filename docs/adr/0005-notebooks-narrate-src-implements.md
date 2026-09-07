# ADR-0005: Notebooks narrate, `src/` implements, and jupytext pairs them

- **Status:** Accepted
- **Date:** 2026-08-25
- **Deciders:** JT Moeller

## Context

The deliverable is an argument a reader can follow, so notebooks are the right
presentation format. But notebooks are hostile to the things this project also
needs: reviewable diffs, unit tests around split and as-of logic, and CI that
proves the whole thing still runs.

The correctness of this project depends almost entirely on code that is easy to
get subtly wrong and impossible to eyeball: as-of feature computation, gap
enforcement, label windows. That code must be tested.

## Decision

Notebooks orchestrate and narrate; all logic lives in `src/churnval/` and is
imported. The package is installed in editable mode, so notebooks import it the
way a user would, no `sys.path` manipulation.

Every notebook is paired with a `.py:percent` file via jupytext, and diffs are
reviewed on the `.py`. `nbstripout` runs as a git filter so outputs never enter
version control; executed copies with outputs are produced separately by
papermill for publication.

## Alternatives considered

| Option | Why it was plausible | Why it was rejected |
|---|---|---|
| Everything in notebooks | Fastest to write; matches the Databricks habit | The split and as-of logic cannot be unit tested; notebook diffs are unreviewable; a leakage bug would be invisible |
| Everything in scripts, notebooks only for charts | Cleanest engineering | Loses the narrative, which is the deliverable — this repo is read more than it is run |
| Notebooks committed with outputs | A reader sees results without running anything | Enormous diffs, merge conflicts on every run, and a standing risk of committing data values inside outputs |
| `nbdev` | Solves the notebook/module split properly | Imposes a whole framework and directory convention on a project whose audience will not be reading it as a library |

## Consequences

**Good:** the risky logic is tested; diffs are readable; the repo stays small;
CI can execute notebooks on sampled data to prove reproducibility.

**Bad:** two files per notebook to keep in sync, and `jupytext --sync` becomes a
step that will occasionally be forgotten.

**Revisit if:** the sync step is forgotten often enough to cause a real
divergence, at which point add a pre-commit hook to enforce it.
