# Notebook standards

A notebook here is a published argument. A reader should be able to read it
without running it, and run it without reading it.

## Structure

- Named `NN_short_name.ipynb`, numbered in execution order.
- **Opens** with a markdown cell stating: the question this notebook answers,
  what it consumes, what it produces, and approximate runtime.
- **Closes** with a markdown cell stating the finding in plain language and what
  the next notebook does with it.
- Markdown precedes every code cell and explains *why* the cell exists.
- Runs top to bottom in a fresh kernel. If it does not, it is broken regardless
  of the outputs currently displayed.

## Division of labour

Notebooks orchestrate and narrate. They do not implement. A cell moves into
`src/` when any of these is true:

- it is longer than about 25 lines
- it would be reused in another notebook
- it encodes a decision that ought to be tested
- it would be hard to read as part of an argument

## Mechanics

- Paired to `.py:percent` via jupytext (`formats = "ipynb,py:percent"` is set in
  `pyproject.toml`). Run `jupytext --sync` before committing. Review diffs on
  the `.py` file.
- `nbstripout` installed as a git filter, so outputs never enter version
  control. Install it once per clone: `uv run nbstripout --install`.
- Executed copies with outputs are produced by `papermill` into
  `reports/executed/` (gitignored) and rendered to HTML for publication.

## Charts

- Every chart has axis labels with units, and a title that states the finding
  rather than naming the variables. "Random split overstates AUC by 0.11" beats
  "AUC by split type".
- Charts are written to `reports/figures/` as PNG **and** referenced from the
  notebook, so the essay can reuse them without re-running anything.
- No default matplotlib styling for anything published. Set the style once in
  `src/<pkg>/plotting.py`.

## Prohibited

- `df.head()` presented as the answer to a question.
- Absolute paths.
- Credentials, tokens, or personally identifying values.
- Cells commented out "for later". Delete them; git remembers.
- A number in a markdown cell that no visible code cell produced.
