# ADR-0001: We record architecture decisions in this repository

- **Status:** Accepted
- **Date:** 2026-08-25
- **Deciders:** JT Moeller

## Context

This repository is a portfolio artefact. Its audience is a technical reviewer
deciding whether the author exercises judgement, not only whether the code
runs. Judgement is invisible in a finished codebase: the reader sees what was
chosen and never sees what was rejected or why.

The project also spans months of intermittent work alongside coursework, so the
author's own memory of a decision's reasoning will degrade.

## Decision

We keep numbered Architecture Decision Records in `docs/adr/`, one per decision
that closed off a plausible alternative. Each record states the context, the
decision, the alternatives rejected with reasons, and the condition under which
the decision should be revisited.

## Alternatives considered

| Option | Why it was plausible | Why it was rejected |
|---|---|---|
| Comments in code | Zero overhead; lives next to what it explains | Cannot hold a rejected alternative; disappears when the code is refactored |
| A single DECISIONS.md | Simpler than many files | Becomes an unordered log; no way to supersede cleanly; grows unreadable |
| Commit messages only | Already required; naturally timestamped | Not discoverable by a reader browsing the repo; buried in history |
| Nothing | Fastest | The reasoning is the portfolio value; leaving it out defeats the purpose |

## Consequences

**Good:** the reasoning is legible to a reviewer without a conversation; the
author can re-enter the project after weeks away; superseded decisions leave a
visible trail.

**Bad:** a per-decision writing cost, and a temptation to record trivia, which
would dilute the log.

**Revisit if:** the log exceeds roughly 25 records for a single project, at
which point it needs categorising rather than abandoning.
