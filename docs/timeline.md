# Notebook sequence

The argument, in order. Each notebook answers one question and hands one thing
to the next.

| # | Notebook | Question it answers | Produces |
|---|---|---|---|
| 00 | `00_problem_definition` | What exactly are we predicting, and over what windows? | The timeline diagram; window constants in `config.py`; the label specification |
| 01 | `01_naive_baseline` | What does the conventional approach report? | An inflated offline metric, produced faithfully and flagged as wrong |
| 02 | `02_leakage_diagnosis` | Why is that number wrong, feature by feature? | A leakage table; adversarial-validation AUC; ablation results |
| 03 | `03_temporal_protocol` | What does the correct design look like? | As-of features, gapped split, rolling-origin backtest |
| 04 | `04_the_gap` | How big is the lie? | The headline chart and the README results table |
| 05 | `05_calibration` | Are the probabilities usable for a decision? | Reliability curves, Brier decomposition, isotonic vs Platt, EV-optimal threshold |
| 06 | `06_hazard_framing` | Does "when" behave differently from "whether"? | Discrete-time hazard model compared against the binary classifier |
| 07 | `07_generalisation` | Does the same failure appear in contractual churn, at scale? | The KKBox replication of `00`-`06`'s protocol; the portable checklist |

## Working order

Build on **Online Retail II first**, it is small enough to run the entire
sequence end to end in an afternoon, which gets the argument working before the
data gets big. Then scale notebooks 00–06 to KKBox in notebook 07, keeping
Online Retail II as the completed non-contractual reference.

## Definition of done

- [ ] All notebooks run top to bottom from a fresh kernel
- [ ] README results table populated from `reports/` output, not typed
- [ ] `leakage-audit` run against the final feature set, findings recorded
- [ ] Every ADR-worthy decision made along the way is recorded
- [ ] The portable checklist stands alone and is readable without the repo
