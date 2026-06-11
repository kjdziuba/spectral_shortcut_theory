# Geometry of Neural Network Loss Surfaces via Random Matrix Theory — Pennington & Bahri (2017)

**Link**: http://proceedings.mlr.press/v70/pennington17a.html
**Read on**: 2026-MM-DD
**Time spent**: ___ hours
**Effort**: __

## Why this paper fifth

Closes the loop on Hessian theory. RMT framework that complements Karakida — gives spectral density predictions and rigorously characterizes the eigenvalue distribution.

## What to focus on while reading

1. **The RMT setup** — Gaussian assumptions, random matrix products
2. **Marchenko-Pastur-like spectral densities** — what shape do they predict?
3. **The role of network depth and parameter-to-data ratio**

## 3-line summary
_Fill in after reading._

## Main theorem(s)
- _Spectral density of Hessian under their assumptions_
- _Conditions for the appearance of negative eigenvalues_

## Proof technique
Random matrix theory: free probability, Stieltjes transforms.

## What we borrow
- Justification for treating $H_{\phi\phi}$ as having a structured spectrum
- The RMT framework if we need tighter bounds
- The relationship between energy and spectrum (their main insight)

## What we extend
- Pennington-Bahri: monolithic networks
- Us: explicitly multi-block structure with capacity asymmetry

## Limitations / open questions
- Their analysis is for specific architectures (deep linear or simple MLP). ViT is more complex.
- Free probability is heavy machinery — we may only need the conclusions, not the techniques.

## My questions for next session
_List._

## After this paper

Phase 0 is done. You've got the five foundations. Next session: Phase 1 (Math Setup) starts — we sit together and write Section 3 of the paper.
