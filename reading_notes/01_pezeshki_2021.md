# Gradient Starvation: A Learning Proclivity in Neural Networks — Pezeshki et al. (2021)

**Link**: https://arxiv.org/abs/2011.09468
**Read on**: 2026-MM-DD
**Time spent**: ___ hours
**Effort**: __

## Why this paper first

This paper is the half of Theorem 2 that says: once the spatial module starts winning, the gradient signal to the spectral module **vanishes exponentially**. Without it our spurious-stationary-point argument doesn't close.

## What to focus on while reading

1. **Section 2 (Setup)** — read carefully. NTRF (Neural Tangent Random Feature) matrix is their key object. Understand it.
2. **Section 3 (Main Result)** — the gradient starvation theorem. Make sure you can state it cleanly.
3. **Section 4 (Mechanism)** — they explain WHY cross-entropy specifically causes this. Critical for our argument.
4. **Skim proofs** in appendix on first pass. Come back if needed.

## 3-line summary
_Fill in after reading._

## Main theorem(s)
_Restate in our notation. Use $\nabla_\theta L$, etc._

## Proof technique
_What tools? NTK regime? Eigenvalue decomposition? Continuous-time SGD?_

## What we borrow
- The cross-entropy residual decay argument (`r → 0` exponentially as predictions saturate)
- The "dominant principal components capture the signal" framing
- Their notion of "feature" as direction in tangent kernel eigenspace

## What we extend
- Pezeshki: one network, features compete internally
- Us: TWO networks (compositional), with architectural asymmetry forcing one to dominate
- Pezeshki: feature-level starvation
- Us: module-level starvation

## Limitations / open questions
- They use NTK assumptions (infinite width). Does our finite spectral module break this?
- They don't address compositional architectures explicitly.
- What is their exact statement of "weak feature"? Can our spectral module be cast as a weak feature?

## My questions for next session
_List confusing points here._
