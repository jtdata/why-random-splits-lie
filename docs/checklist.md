# A portable checklist for validating a churn model

This distills what `notebooks/00`-`07` found, built twice, once on
non-contractual retail churn (Online Retail II), once on contractual
subscription churn (KKBox, ~25x the row count), into something you can use
on a churn dataset this repo has never seen. Each item names the failure it
catches and, where it matters, what changes between a small dataset and a
large one. It assumes nothing about this repo except what's stated inline.

## 1. Define the event before you define the label

Write down, in one sentence: is churn something that *happens* (a
subscription lapses, a cancellation is recorded) or something you have to
*infer* from an absence (no purchase in N days)? These are different
problems wearing the same name.

- **Observed-event churn** (contractual: subscriptions, memberships): you
  have an explicit date the relationship ends or is due to renew. Your
  label can be built directly from that date plus a horizon.
- **Inferred churn** (non-contractual: retail, usage-based products): there
  is no event to point to. "Churned" is a modeling choice (how many days of
  silence counts?), and that choice needs its own justification, not just a
  round number.

If you can't say which one your problem is, you can't design the next four
steps correctly. This determines your eligibility rule, your label, and
whether a hazard model is answering a genuinely different question or a
redundant one.

## 2. Draw the timeline before writing code

Four points, always in this order: **feature as-of date** → **gap** →
**label window start** → **label window end**. If you can't draw this for
your problem, the problem isn't defined yet, go back to step 1.

- The **gap** is your real operational lead time: how long between scoring
  a customer and being able to act on it (write an email, place a call,
  ship an offer). Zero gap means your features and your label window touch,
  and a feature computed at the instant of the event can encode the event.
- **On contractual data, check what the gap does to your label, not just to
  your feature cutoff.** If eligibility means "due for a renewal decision
  around now," then customers who renew *during* the gap have their next
  decision pushed toward the far edge of the label window, and whether they
  land inside it is driven by plan length rather than by any intent to
  leave. This project measured a large share of one panel affected that way
  (`07`). It is not a leak and not necessarily the wrong choice, but it
  means changing the gap changes *who counts as a churner*, so a
  "gap vs. no gap" ablation on such a dataset is not measuring only the
  gap. Measure the prevalence of both arms before comparing their scores.
- The **label window** length is a real modeling choice with a real
  alternative, not a default. State why this window, not another, and
  what would make you reconsider it (`ADR-0007`/`ADR-0013` in this repo are
  worked examples, one dataset needed a value 3x the project's own
  default, one needed no override at all, and knowing *why* is the point,
  not the specific numbers).

## 3. Make eligibility a real rule, checked at real scale

"Which entities get scored at this origin" is not a formality, get it
wrong and you either waste computation scoring people who were never
candidates, or silently exclude the people the label is actually about.

- **Non-contractual:** typically activity-based, some recent signal of
  engagement (a purchase, a login) in a lookback window.
- **Contractual:** typically state-based, whose current membership/contract
  is due for a decision *around this origin*, not "has ever transacted."
  An activity-based rule ported directly to a contractual dataset can
  silently score a population 5-10x larger than the real one, most of whom
  aren't due for a decision for months (measured directly in this repo:
  1.05M-1.74M customers under an activity rule vs. 200K-424K under an
  expiry-based one, same dataset, same origins).
- Whichever rule you pick, **measure the population size it produces**
  against the real data before committing to it. A rule that looks
  reasonable on paper can be wildly wrong in practice, and the only way to
  know is to run it.

## 4. Snapshot features strictly before the as-of date, mechanically, not just conceptually

For every feature, ask: at the as-of instant, would this value actually
have been populated? Then check it, don't just reason about it:

- **Reference-date check.** For any "days since X" or "days until Y"
  feature, extract the reference date the feature is measured *to*, per
  row, and assert it is `<= as_of`. A feature whose reference date is
  frozen to a dataset's global end date (or any other fixed anchor reused
  across rows with different as-of dates) partially or wholly encodes the
  label window's own outcome. This repo found and demonstrated exactly
  this bug (`06`'s naive-baseline audit, and the leakage-audit skill this
  project uses now runs this check by default).
- **Aggregation window check.** Any count, sum, or "most recent" feature
  needs its own window boundary stated and enforced, not "all history,"
  unless that's a deliberate, disclosed choice.
- At scale, this check needs to be a query, not a manual read-through, a
  vectorized SQL pass across every origin at once with an explicit
  `< as_of` boundary in the join condition, checked once as a unit test
  against a small hand-built fixture, not eyeballed against millions of
  rows.

## 5. Use a rolling-origin backtest, with a real purge, not a single split

One split is one draw from a distribution and tells you nothing about
variance across regimes. A rolling-origin backtest (several origins,
retraining forward at each) does, but only if training data is *actually*
knowable at each origin:

- A training origin's label isn't settled until its own label window has
  fully closed. Pooling every earlier origin regardless of whether its
  label had matured yet is a leak this project found live, mid-build, in
  its own "correct" notebook (`ADR-0009`), it is not a hypothetical.
- The purge window is a direct, computable consequence of `gap + horizon`
  vs. the spacing between origins, not a number to guess: work out how
  many origins get excluded before writing the backtest loop, and check
  the real code excludes exactly that many.
- **At small scale**, a Python loop over origins, each doing a pandas
  filter+groupby, is fine (this project's first dataset: ~800K rows, ~10
  origins, seconds per pass).
- **At large scale**, that same loop does not scale linearly, it scales
  with `origins x rows`, and an inequality join across the same axes can
  blow up in ways a bounded time window doesn't obviously fix (measured
  directly: an inequality join with a 500-day bound did not finish in
  several minutes against 21.5M rows; the purpose-built "nearest prior row
  per entity" operation most SQL engines provide finished the same
  computation in under a minute). If your backtest is taking longer than
  you can iterate on, that's a real signal to change the *algorithm*, not
  just add compute.

## 6. Report a ranking metric and a calibration metric together, always

ROC AUC (or PR AUC) alone answers "does this model separate the classes."
It says nothing about whether the predicted probability of 0.7 means
anything like a 70% chance. Two models can rank identically and be
calibrated completely differently, or vice versa.

- Fit any calibrator (Platt, isotonic) on a slice **strictly later** than
  the classifier's own training data, never on training folds. This
  project's own three-way split (train / calibrate / evaluate, each a
  distinct later slice) is the minimal structure that avoids the
  calibrator memorizing the classifier's overconfidence rather than
  correcting it.
- Report the *decomposed* calibration metric (reliability, resolution,
  uncertainty), not just an aggregate Brier score. Two models can land at
  the same Brier score for genuinely different reasons (opposite-direction
  miscalibration of similar magnitude looks identical to "both well
  calibrated" if you only look at the aggregate number).

## 7. If a decision follows from the prediction, price the decision honestly

A ranking or calibration metric doesn't say what to *do*. If there's a
threshold-based action downstream (contact this customer, don't contact
that one), that threshold needs its own honest derivation:

- **Derive the threshold from the stated costs, before looking at
  outcomes**, `offer_cost / (save_rate * saved_margin)`, the standard
  break-even rule, never by sweeping a grid of thresholds against the
  evaluation outcomes and reporting whichever one wins. This project built
  exactly that mistake once, reported a headline number from it, and only
  caught it under adversarial review: the biased version said the wrong
  model made the better business decision, and the honest version said the
  opposite (`ADR-0011`). If a review process is going to catch this, catch
  it before the number goes in a report, not after.
- State the cost assumptions plainly, including which ones are invented
  for illustration versus grounded in real data. Nobody actually measures
  the "does the intervention work" number (`save_rate`), say so, don't
  hide the assumption inside a clean-looking number.

## 8. If "when" matters as well as "whether," build a hazard model, and check what it's actually for

A binary "churned within N days" label has a specific, nameable blind
spot: a customer who returns on day N+5 is filed identically to one who
never returns. A discrete-time hazard model (person-period expansion, a
hazard estimate per period, aggregated to a survival probability) doesn't
remove the window, but it can expose that blind spot directly, and it
answers a question the binary label structurally cannot ("where within the
window is risk concentrated").

- Don't expect better aggregate ranking/calibration numbers from switching
  to a hazard framing, reshaping the same underlying event into
  person-periods doesn't hand the model new information. The value is in
  what the binary label can't say at all, not in beating it on its own
  terms.
- **The period width is not portable across datasets or horizons.** A
  period count tuned for one horizon can silently collapse to a single
  period (i.e., no hazard resolution at all) on a shorter horizon,
  rederive it, don't reuse an absolute day count.
- **The hazard model's notion of "event" has to match the binary label's
  notion of "success," explicitly.** If your event log contains rows that
  represent a negative outcome dressed as an event (a cancellation
  transaction, a downgrade, a support ticket that precedes a churn), a
  hazard model built on "any row in the log" will silently disagree with a
  binary label that specifically excludes those rows, this project caught
  exactly this mismatch while building its second dataset (a cancellation
  transaction would have counted as "the customer returned," the opposite
  of what the label meant), and it would not have surfaced on the first
  dataset, where that category of row was already dropped in cleaning for
  an unrelated reason. Check this explicitly; don't assume a working
  binary label implies a working hazard label built from the same events.

## 9. Retrospective diagnostics need enough future data to be honest about their own limits

A "how many customers labeled churned actually come back later" check
needs transaction history well past the window you're diagnosing, and the
amount of history available shrinks for your most recent, often most
policy-relevant, origins. State the observation window's own length
alongside the number it produced (this project's version: "at least X%
return within the observable N-day tail," not "X% return," because the
true number is right-censored by how much history exists past that
origin).

## 10. Before scaling to a new dataset, separate what's genuinely reusable from what only looks reusable

Not everything in a working pipeline transfers to a new dataset, and not
everything needs to be rebuilt from scratch either. Check each function
individually against three questions, not one blanket policy:

- **Does it only touch generic column names** (an entity ID, a score date, a
  label, caller-supplied feature names) **or does it hardcode a
  dataset-specific one?** The former is free to reuse; the latter needs a
  parameter added (if it's a single hardcoded name in an otherwise-generic
  function) or a fresh implementation (if the *logic*, not just a name,
  differs, a different eligibility criterion, a different feature set).
- **Confirm genericity empirically, not by inspection alone** if your
  test suite already exercises the "generic" functions with a
  deliberately non-domain-specific fixture column name, that's stronger
  evidence than reading the function and assuming.
- **When you do touch a shared, already-audited function**, grep for every
  call site first, make the smallest change that generalizes it, and
  re-run whatever already depended on it, a full re-execution with a
  before/after diff of every numeric output, not just "the tests still
  pass." This project did exactly that for one function shared across two
  datasets and confirmed, not assumed, zero behavioral change to the
  already-published result that depended on it.

## 11. The gap between "the wrong way" and "the honest number" is not always where you expect it

On the non-contractual dataset, the *features and the split* carried
almost all of the inflation and the operational gap contributed nothing
measurable on top. That result is narrower than it sounds, and the
narrowing is the lesson. This project's `gap_days` moves the label
window's boundary, never the feature cutoff, so that ablation could not
have detected a gap effect on feature leakage even if one existed.

The same ablation does **not** transfer to the contractual dataset, for
the reason step 2 gives: there, changing the gap changes which customers
count as churners, so the two arms score differently-labelled panels and
the comparison answers a different question. This project ran it on both
datasets and can only report the decomposition for one.

Two things to take from that. Decompose your own pipeline piece by piece
rather than assuming which piece does the damage, and before comparing
two ablation arms, check they are still scoring the same problem. Report
the honest number even when, especially when, it complicates the headline
you expected to write.
