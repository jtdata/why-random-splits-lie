# ADR-0002: Each portfolio project is its own git repository, with standards vendored into it

- **Status:** Accepted
- **Date:** 2026-08-25
- **Deciders:** JT Moeller

## Context

The portfolio comprises three independent projects across three tracks (energy
M&V, OT/ICS security, DS methodology) targeting different audiences. They share
engineering and validation standards but share no code.

A reviewer typically arrives at one project from a link in an application or an
essay, and should land on that project's README, not on a folder listing.

The standards have to be readable by someone who clones a single repo. A
standards file that lives only in the local workspace folder would be invisible
to that reader.

## Decision

Each project is a separate git repository under `portfolio/projects/`, pushed
independently and public on its own. The workspace root `portfolio/` is a plain
folder, not a repository.

Standards live in each project's `docs/standards/` and ship with the repo. The
first project is the origin copy; later projects copy it and record intentional
divergence as an ADR in the new project.

The workspace root `CLAUDE.md` holds only the working agreement between the
author and Claude Code: private, machine-specific, and inherited by every
project because Claude Code walks up the directory tree.

## Alternatives considered

| Option | Why it was plausible | Why it was rejected |
|---|---|---|
| Portfolio monorepo | Standards defined once, no drift; one push | Visitors land on a folder listing rather than a project; three unrelated tracks share one history and issue tracker; a reviewer clones 3 projects to read 1 |
| Per-project repos + a standards repo as a git submodule | Rigorous single source of truth | Submodules are a well-known friction point for readers and for CI; a cloned repo with an uninitialised submodule has no standards at all |
| Standards only in the workspace root | No duplication | Invisible to anyone who clones the repo; defeats the purpose of publishing them |

## Consequences

**Good:** each repo is self-contained and independently reviewable; a clone
carries its own rules; project histories stay clean and topical.

**Bad:** standards can drift between projects, and a fix has to be copied
forward manually.

**Revisit if:** a third project needs the same standards fix propagated, which
is the point at which manual copying stops being cheaper than a submodule.
