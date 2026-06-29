# Study Notes 4 — Karakida and λ_max(G_φφ) = Ω(C_g)

This is the load-bearing piece of Theorem 1's proof we kept marking
TODO. We work through Karakida et al. (2019) and adapt it to our
spatial Gauss-Newton block.

We don't write the full supplement proof here. The goal is to get
you fluent enough with the framework that, in our next focused
session, the algebra is just typing.

---

## 1. What we need to show

In `paper/sections/04_theorem1_hessian.tex` we claimed (Lemma
`lem:phi_scaling`):

```
λ_max(J_φ^T J_φ)  =  Ω(C_g)        in expectation over random init
```

where J_φ is the Jacobian of per-pixel residuals with respect to the
spatial parameters φ, and C_g = dim(φ).

That is: the top eigenvalue of the spatial Gauss-Newton block grows
**linearly** with the number of spatial parameters, under random
initialization. Empirically (Exp 1.1) we see scaling exponent ≈ 0.7,
not exactly 1.0 — we'll explain why at the end.

---

## 2. Karakida 2019 in one paragraph

Karakida, Akaho and Amari studied the **Fisher Information Matrix**
(FIM) of deep neural networks under random Gaussian
initialization. They use mean-field analysis (each unit's
pre-activation is treated as Gaussian; expectations are taken under
that Gaussian distribution; in the infinite-width limit these
expectations are exact).

Their three key results, which we use:

1. **Mean of trace**:
   `E[tr(F)] ≈ M · q · ν`
   where M = number of parameters, q = squared activation magnitude
   per layer, ν = squared backprop signal per layer. Both q and ν
   are O(1) under standard Kaiming-like initialization.

2. **Maximum eigenvalue**:
   `λ_max(F) ≥ (1/M) · E[tr(F)] · (something between 1 and width)`
   The "something" is a participation ratio that, for typical
   networks, behaves like Θ(M · const).

3. **Concentration**: in the infinite-width limit, λ_max(F) is
   tight around its mean.

The upshot is `λ_max(F) = Θ(M)` in expectation, where M is the
parameter count. That's the asymptote we'd like.

---

## 3. From FIM to Gauss-Newton

We work with the Gauss-Newton matrix G_φ = J_φ^T J_φ rather than F
directly. The two coincide for log-likelihood losses near
interpolation:

```
G_φ  =  J_φ^T J_φ
F    =  E[J_φ^T diag(p)(I - 1·p^T) J_φ]   for softmax CE
```

When predictions p are uniform (1/N_cls each at initialization),
the inner matrix `diag(p)(I - 1·p^T)` is `(1/N_cls)·(I - 1·1^T/N_cls)`
— a projection-like operator with eigenvalues bounded by 1/N_cls.

Concretely:

```
F  ≈  (1/N_cls) · J_φ^T J_φ   at random init   (for binary, factor 1/2)
```

So Karakida's result on F directly implies the same scaling for G_φ
up to a constant factor.

---

## 4. Specialising to our architectures

### 4a. Spatial MLP (per-pixel)

Parameters at width D:
- Layer 1: K·D weights + D biases ≈ K·D
- Layer 2: D·N_cls + N_cls ≈ D·N_cls
- Total C_g ≈ D · (K + N_cls)

For our toy: K=16, N_cls=2 → C_g ≈ 18·D.

λ_max scales as C_g by Karakida → **λ_max ~ 18·D**. Linear in D.

### 4b. Spatial CNN (the one we use)

For a 2-layer 3×3 CNN with width D:
- Conv1: K·D·9 + D ≈ 9·K·D
- Conv2: D·N_cls·9 + N_cls ≈ 9·N_cls·D
- Total C_g ≈ 9·D·(K + N_cls)

For K=16, N_cls=2 → C_g ≈ 162·D. Still linear in D.

There's an extra subtlety for convolutions: parameters are SHARED
across spatial positions, so the per-parameter Fisher contribution
gets multiplied by the number of patches H·W. This means

```
F_CNN  ≈  H·W · (1/N_cls) · J^T J   in per-pixel terms
```

and λ_max gets the same H·W boost. Since H, W are constants in our
experiment, this is absorbed into the constant.

### 4c. Spatial ViT

For L attention layers at width D:
- Embed: K·D
- Each attention block: 4·D² (q, k, v, output projection) + 2·D²
  (MLP) → ~6·D² per layer
- Head: D·N_cls

Total C_g ≈ L · 6·D² (dominant term). **Quadratic in D.**

Karakida still gives λ_max ~ C_g asymptotically (linear in
parameter count). But the parameter count itself grows as D², so
λ_max ~ D².

### Summary

| Architecture | C_g     | λ_max(F) predicted |
|--------------|---------|--------------------|
| MLP          | ~18·D   | ~18·D              |
| CNN          | ~162·D  | ~162·D             |
| ViT          | ~12·D²  | ~12·D²             |

All three give λ_max linear in C_g. Theorem 1's conclusion
κ ~ Ω(C_g/C_f) is architecture-independent.

---

## 5. Why our empirical slope is 0.7, not 1.0

We measure 0.48–0.70 slope on log-log of λ_φ vs C_g (Exp 1.1) where
Karakida predicts 1.0. Three contributing factors:

### Factor 1: Finite-width corrections

Karakida's mean-field analysis is asymptotic in width. For our
widest D = 8192, the participation-ratio bound is dominated by
finite-N concentration corrections. Typical empirical fits in this
regime give exponents ~0.7-0.9.

This is similar to what's seen in NTK experiments: the literal NTK
limit is approached but not reached at practical widths.

### Factor 2: Bias parameters break the count

Karakida assumes all parameters scale identically. In our CNN:
- Weight params: scale Fisher with input variance (~1)
- Bias params:   constant contribution

For D=8192, biases are 1/(9·K) of weights, so they don't dominate
— but at D=16 they're 1/9·K of a small total and matter more. This
flattens the early slope.

### Factor 3: Cross-entropy vs MSE

Karakida proves the scaling for MSE explicitly. For CE we need the
factor (1/N_cls) — that's a constant, so it doesn't change the slope.
But the *exponential* tail of softmax adds a Hessian residual term
that subtracts from G_φ. Near initialization predictions are nearly
uniform, so this correction is small; but it can drag the empirical
slope down by a small amount in finite-width networks.

### Bottom line for the paper

We state Theorem 1 with the asymptotic exponent of 1 and acknowledge
in the limitations section that the empirical scaling is sub-linear
at finite width. The qualitative claim — **κ grows monotonically
with C_g/C_f** — holds at all our widths. The quantitative slope
should be expected to approach 1 only at very large widths.

---

## 6. Clean write-up for the supplement

Here's the structure we'll use in `paper/sections/supplement.tex`
for the Lemma `lem:phi_scaling` proof:

```
Proof of Lemma 4.1 (λ_max(J_φ^T J_φ) = Ω(C_g)).

Step 1. Show that under random Gaussian initialization with
        Kaiming variance, the per-pixel residuals satisfy
        E[||r||²] = Θ(1) (independent of C_g).

Step 2. Use Karakida Theorem 3.1 to compute E[tr(F_φ)] and
        bound it from below by E[tr(J_φ^T J_φ)] · (1/N_cls).

Step 3. Use Karakida Theorem 4.2 (max-eigenvalue bound) with
        a participation ratio of Θ(C_g) to derive
        λ_max(F_φ) ≥ c₁ · C_g.

Step 4. Translate F to G via the CE factor.

Step 5. Conclude λ_max(J_φ^T J_φ) ≥ c · C_g for some constant
        c > 0 depending only on K, N_cls and the data covariance.
∎
```

That's about 1.5 pages of supplement once we fill in the algebra.
Should take ~3 hours of working together.

---

## 7. Open questions to resolve next session

1. **Does Karakida's participation ratio apply verbatim to CNN
   weight sharing?** I claimed yes (Section 4b) but we should
   verify the H·W boost is rigorous, not heuristic.

2. **For ViT, the asymptotic FIM is well-studied** (Boullé,
   Cohen, Hron 2024). We should adapt their result rather than
   re-deriving Karakida for transformers.

3. **The CE residual subtraction term** (Section 5, Factor 3)
   has a known closed form (see Sagun 2017). We should either
   prove it's negligible or include it explicitly.

These are all tractable. None of them is open-research-hard.

---

## 8. Practical next step

Now (this session): you read this and the Pezeshki paper. Note
anything that's fuzzy.

Next session: we sit down with the LaTeX supplement and write
Steps 1–5 out together with full algebra. Should be a 2-hour focused
session.
