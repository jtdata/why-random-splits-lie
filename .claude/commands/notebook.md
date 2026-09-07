---
description: Work through one notebook in the project sequence, following the project's build loop with checkpoints
argument-hint: "notebook number, e.g. 01"
---

Work on notebook **$ARGUMENTS** from this project's sequence.

## Before anything else

Read, in this order, and do not skip any of them:

- `CLAUDE.md` (this project) and the standards it imports
- `docs/timeline.md` — find the row for notebook $ARGUMENTS: the question it
  answers, what it consumes, what it produces
- `docs/adr/README.md` — what is already decided. Do not relitigate an accepted
  ADR; propose superseding it if you disagree
- The previous notebook in the sequence, so you inherit its outputs and its
  closing finding
- Any module in `src/churnval/` you are about to change

Then state, in under ten lines: the question this notebook answers, what it
consumes, what it produces, which modules you expect to add or change, and
anything in the timeline row you think is wrong.

**Stop there and wait for my confirmation.** Do not write code yet.

## Then, once I confirm

Work in this order, and pause after each numbered step so I can look:

1. **Frame it.** Write only the notebook's opening markdown cell — question,
   inputs, outputs, approximate runtime — into
   `notebooks/$ARGUMENTS_*.py` (the jupytext `.py`, which I will convert).
   If notebook $ARGUMENTS is the deliberately-wrong one, the banner stating so
   goes here and must be impossible to miss.

2. **Implement in `src/`, not the notebook.** Name the module before you write
   into it. The notebook imports and narrates; anything longer than ~25 lines,
   reusable, or encoding a decision belongs in a module. Write the test in
   `tests/` in the same step, with hand-computed expected values — not values
   read back from your own implementation.

3. **Write the notebook body.** Markdown before every code cell, explaining
   *why* the cell exists. Charts get axis labels with units and a title that
   states the finding. Every number that appears in prose is computed in a
   visible cell and written to `reports/`.

4. **Audit.** Run the `leakage-audit` skill against whatever this notebook
   builds — features, split, or label. Record the findings in a markdown cell,
   including the checks that came back clean. If the notebook is the
   deliberately-wrong one, the audit output *is* the point: keep it.

5. **Adversarial review.** Launch the `validation-reviewer` agent against the
   notebook and any module you changed. Report its findings verbatim before
   you act on them. If you disagree with a finding, say so and why — do not
   silently drop it.

6. **Record decisions.** For anything you chose over a plausible alternative,
   use the `adr` skill. Dataset handling, label definition, gap length, model
   family, calibrator — all qualify. Say "no ADR needed" explicitly if none do.

7. **Close the loop.** Write the closing markdown cell: the finding in plain
   language and what the next notebook does with it. Then run:

   ```
   uv run jupytext --sync notebooks/
   uv run ruff check --fix . && uv run ruff format .
   uv run pytest
   ```

   Report the results and **stop**. I review and commit — you do not.

## Rules that override anything above

- No results hardcoded into prose. If you do not have a number yet, write
  UNKNOWN.
- No new dependency without saying what it replaces and why what is already
  installed will not do.
- `data/raw/` is read-only.
- If the timeline row and what the data actually supports disagree, stop and
  tell me. Do not quietly redefine the notebook to fit what is convenient.
