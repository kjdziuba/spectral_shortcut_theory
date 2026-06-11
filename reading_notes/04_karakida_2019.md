# Universal Statistics of Fisher Information in Deep Neural Networks: Mean Field Approach — Karakida et al. (2019)

**Link**: https://arxiv.org/abs/1806.01316
**Read on**: 2026-MM-DD
**Time spent**: ___ hours
**Effort**: __

## Why this paper fourth

Provides the **quantitative scaling laws** for FIM/Hessian eigenvalues in terms of width and depth. This is what makes Theorem 1's bound concrete instead of hand-wavy.

## What to focus on while reading

1. **The mean-field setup** — they assume random initialization, take width → ∞
2. **Their closed-form expressions for $\lambda_{\max}$ of FIM** — this is what we use
3. **The relationship FIM ≈ Hessian near minima** — they justify using FIM as a proxy

## 3-line summary
_Fill in after reading._

## Main theorem(s)
- _$\E[\lambda_{\max}(F)] = ?$ (their main result)_
- _Variance bounds_
- _Scaling with width / depth_

## Proof technique
Mean-field theory: replace finite-width averages with infinite-width expectations under random init.

## What we borrow
- The scaling result: $\lambda_{\max}(F_{\phi\phi}) \propto C_g$ for our spatial module
- Justification for using FIM as Hessian proxy
- The mean-field methodology if we want to do similar analysis for our spectral module

## What we extend
- Karakida: single network
- Us: compositional pipeline, apply their result to ONE block, derive what changes for the other block

## Limitations / open questions
- Mean-field assumes Gaussian init. Our pretrained models violate this.
- Result is asymptotic in width. Our ViT is finite (though large).
- We need to be careful what "C_g" means in their framework vs ours.

## My questions for next session
_List._
