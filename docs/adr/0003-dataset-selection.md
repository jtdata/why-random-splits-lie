# ADR-0003: We use KKBox and Online Retail II, and reject the Telco churn dataset

- **Status:** Accepted
- **Date:** 2026-08-25
- **Deciders:** JT Moeller

## Context

This project's entire argument depends on comparing a random split against a
temporal split. That comparison requires panel data with **real event
timestamps**, a time axis to split on, features that can be snapshot as-of
a date, and labels with a definable window.

The two datasets most commonly reached for in churn tutorials do not have this
property, and this was verified before any code was written.

## Decision

Use the **KKBox WSDM Churn Challenge** data as the main body (contractual
subscription churn, with transaction dates, membership expiry, renewals and
cancellations), and **Online Retail II** from UCI as a second panel
(non-contractual churn, where the event is latent and must be inferred).

Reject the IBM Telco churn dataset and Santander Customer Transaction
Prediction for this project.

## Alternatives considered

| Option | Why it was plausible | Why it was rejected |
|---|---|---|
| IBM Telco churn | The default churn dataset; familiar to every reviewer; small and fast | A single cross-sectional snapshot: a tenure column and a binary label, no event dates. There is no time axis to split on. Manufacturing one would be obvious to any reviewer who knows the file, and would invalidate the whole demonstration |
| Santander Customer Transaction Prediction | Large, well known, clean | Fully anonymized with no temporal structure at all |
| Santander Product Recommendation | Genuine monthly customer×product panel with `fecha_dato` | Viable, and kept as a fallback. Label must be constructed from product drops, adding a definitional argument that distracts from the main one |
| Olist e-commerce | Fully timestamped, small, fast to iterate on | Marketplace repeat-purchase framing rather than churn; kept as a prototyping set |
| Synthetic data with injected leakage | Perfect ground truth; no download | A demonstration on synthetic data proves the simulation, not the phenomenon. The claim is about real pipelines |

## Consequences

**Good:** both contractual and non-contractual churn are covered, which turns a
case study into a method. Ground truth about *when* events happened is real.

**Bad:** KKBox's `user_logs` table is tens of gigabytes; the transactions and
members tables alone are used for the core argument. Kaggle competition data
requires accepting the competition rules and an API token.

**Revisit if:** the KKBox data becomes unavailable from Kaggle, in which case
Santander Product Recommendation becomes the main body.
