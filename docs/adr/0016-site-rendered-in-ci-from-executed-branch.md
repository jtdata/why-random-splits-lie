# ADR-0016: The site is rendered in CI from an `executed` branch; notebooks are still never executed in CI

- **Status:** Accepted
- **Date:** 2026-11-13
- **Deciders:** JT Moeller

## Context

The published site is rendered from the executed notebooks in
`reports/executed/`. Those are gitignored, because `docs/standards/coding.md`
and `docs/standards/notebooks.md` keep notebook outputs out of version
control, so they existed on exactly one machine: the one that last ran
`.\make.ps1 nb`. The site could therefore only be rebuilt and published from
that machine, with `.\make.ps1 site` and `.\make.ps1 publish`.

That coupled two things that change at very different rates. The results
change rarely: a full run needs `data/`, and `07` needs Kaggle credentials
and about half an hour. The presentation changes often: the landing page, the
README it is built from, the checklist, the decision records, the builder's
layout. A wording fix made on a second machine could be pushed to `main` but
could not reach the site until someone went back to the first machine.

It also meant two machines could publish different renders of the same
source, since each would build with whatever environment it had.

## Decision

1. The executed notebooks are published to an orphan branch, `executed`,
   that holds only those files, a `SOURCE_COMMIT` marker and a copy of the
   workflow file. `main` still never contains an output. `.\make.ps1 results`
   pushes the branch. The workflow copy is there because GitHub only runs a
   workflow for a push when the file exists on the pushed branch; the job
   itself always builds from `main`.
2. Before they are pushed, `scripts/export_executed.py` removes the run
   metadata papermill writes (it records absolute input and output paths)
   and stops if a home-directory path remains anywhere in a notebook.
3. A GitHub Actions workflow (`.github/workflows/site.yml`) runs on every
   push to `main` or `executed`. It checks out `main`, checks out `executed`
   into `reports/executed/`, runs `scripts/build_site.py` with the dev
   dependency group at the versions in `uv.lock`, and commits the result to
   `gh-pages`.
4. CI renders. It does not execute a notebook, fetch data, or hold a
   credential beyond the workflow's own repository token.

`.\make.ps1 site` remains for local preview and `.\make.ps1 publish` remains
as a manual fallback.

## Alternatives considered

- **Execute the notebooks in CI.** Rejected. It needs the datasets, which are
  not redistributable and are not in the repo, a Kaggle credential stored as a
  secret, and half an hour per run. It would also make the published KKBox
  figures move on every push, because that notebook is not bit-reproducible
  (`07`'s leakage audit, row 11), while the README quotes one run.
- **Commit outputs to `main`.** Rejected. It reverses the nbstripout rule,
  makes every notebook diff unreadable, and puts megabytes of base64 images
  into the history of the branch people read.
- **Build and publish by hand from each machine.** Rejected. It needs the
  outputs copied between machines by some other route, a publish script for
  each platform, and it leaves the two machines free to publish different
  renders.
- **Recover the outputs from the published HTML.** Rejected as a source of
  truth. It works for a preview but makes `gh-pages` its own input.

## Consequences

- A presentation change reaches the site from any clone with push access, with
  no local build.
- The outputs are public as notebooks on the `executed` branch. They were
  already public as HTML on `gh-pages`, so nothing new is disclosed, and the
  export step guards the one thing the notebooks carried that the HTML did
  not.
- The landing page's numbers come from `reports/results_*.json` on `main`,
  and the notebook pages come from `executed`. After a new run the two must
  be pushed together, `main` first, or the site briefly shows numbers from
  one run beside outputs from another. `SOURCE_COMMIT` records which commit
  the outputs were produced from, so a mismatch can be seen.
- The site now depends on GitHub Actions being enabled with write permission
  for the workflow token. If it is not, `.\make.ps1 publish` still works.
- `gh-pages` commits are authored by the Actions bot and name both the source
  commit and the outputs commit they were rendered from.
