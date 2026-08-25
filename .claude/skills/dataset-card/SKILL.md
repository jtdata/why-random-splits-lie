---
name: dataset-card
description: Create or update a dataset card in docs/data/ recording a dataset's source, licence, retrieval date, shape, temporal coverage, and known quirks. Use whenever a new dataset is downloaded, when a dataset's schema is first explored, or when the user asks what is in a dataset.
allowed-tools: Read, Write, Edit, Glob, Bash
---

# Dataset cards

Every dataset that enters `data/raw/` gets a card in `docs/data/<name>.md`
before it is used for anything. The card is what lets a reader trust a result
without having the data.

Compute the numbers — never estimate them. If a figure has not been computed,
write UNKNOWN.

## Template

```markdown
# <Dataset name>

- **Source:** <URL>
- **Licence / terms:** <exact terms; state whether redistribution is permitted>
- **Retrieved:** YYYY-MM-DD
- **Files used:** <filenames and sizes>
- **Redistributed in this repo:** No — fetched by `churnval fetch <name>`

## Shape

| Table | Rows | Columns | Bytes (parquet) |
|---|---|---|---|

## Temporal coverage

- Date column(s) and what each one means (event date vs record date)
- Min and max dates, and any gaps
- Timezone, and whether it is stated or assumed

## Label

How the outcome is defined in this dataset, including the exact window, and
what happens to entities whose window has not closed.

## Known quirks

Anything that would produce a wrong result if missed — duplicated keys,
backfilled corrections, encoding oddities, sentinel values, class balance.

## Fitness for this project

One paragraph: does this dataset support an as-of feature computation and a
temporal split? If not, say so plainly and do not use it.
```

## Rules

- The "Fitness for this project" section is mandatory and is the reason the card
  exists. A dataset that fails it gets an ADR explaining the rejection.
- Record the licence terms verbatim where redistribution is concerned. Never
  commit the data itself.
