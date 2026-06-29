# Study Notes 3 — Block Hessian Algebra (Theorem 1, Part 1)

These notes work through the algebra behind Theorem 1's bound on the
condition number κ(G) of the joint Gauss-Newton matrix. We don't prove
the full theorem here — we set up the block structure, define the
Schur complement, and derive the two pieces (upper bound on λ_max,
lower bound on λ_min) that we'll combine.

---

## 1. Recap: what we want to prove

Theorem 1 (informal): under our assumptions,

```
κ(G) = λ_max(G) / λ_min(G)  ≥  c · (C_g / C_f)
```

To get this lower bound on the ratio, we need:
- a **lower bound** on λ_max(G) — show it's at least as big as something proportional to C_g
- an **upper bound** on λ_min(G) — show it's at most as big as something proportional to C_f

Then the ratio is at least Ω(C_g / C_f).

---

## 2. Setting up the block structure

Recall: G is the Gauss-Newton matrix of the loss with respect to both
parameter blocks θ (spectral, dim C_f) and φ (spatial, dim C_g). So G
has total dimension (C_f + C_g) × (C_f + C_g).

The block decomposition is:

```
G = [ G_θθ   G_θφ ]
    [ G_φθ   G_φφ ]
```

where each block is:
- `G_θθ` : C_f × C_f, curvature wrt spectral params
- `G_φφ` : C_g × C_g, curvature wrt spatial params
- `G_θφ = G_φθᵀ` : C_f × C_g, cross-coupling between modules

This is a symmetric block matrix because G is symmetric positive
semidefinite (Gauss-Newton always is).

---

## 3. The Schur complement — your new best friend

The Schur complement is a way to "fold" one block of a matrix into
another. Given the block structure above, the **Schur complement of
G_φφ in G** is defined as:

```
S = G_θθ  −  G_θφ · G_φφ⁻¹ · G_φθ
```

It's the upper-left block "corrected" by the cross terms. Geometrically:
S is the effective curvature in the θ subspace after the φ subspace
has been optimally adjusted to compensate.

**Key fact #1 (Schur):** if G is positive definite, then

```
λ_min(G)  ≤  λ_min(S)
```

This is what we want — it lets us upper-bound λ_min(G) using S.

**Key fact #2 (positive semidefinite):** S itself is positive semidefinite.

These two facts are what make the bound work. They are standard;
see Bhatia *Matrix Analysis*, Chapter III, Section 1.

---

## 4. Lower bound on λ_max(G)

This is the easier half. We use **Cauchy interlacing**, which says:
eigenvalues of a principal submatrix are bounded by eigenvalues of
the parent matrix.

For our block matrix:

```
λ_max(G)  ≥  λ_max(G_φφ)
```

That is, the largest eigenvalue of the full Gauss-Newton matrix is at
least as big as the largest eigenvalue of the spatial block (we just
need to feed it the eigenvector of G_φφ extended with zeros for the
spectral block).

By Karakida et al. (2019), under mean-field assumptions on the spatial
module:

```
λ_max(G_φφ)  =  Ω(C_g)
```

So:

```
λ_max(G)  =  Ω(C_g)
```

Done. Spatial module's parameter count drives the upper-tail of the
spectrum.

---

## 5. Upper bound on λ_min(G)

Apply Key fact #1 (Schur):

```
λ_min(G)  ≤  λ_min(S)  =  λ_min( G_θθ  −  G_θφ G_φφ⁻¹ G_φθ )
```

Now we need to bound λ_min(S). The Schur complement S is a C_f × C_f
matrix. Its eigenvalues are constrained by:
- the bottleneck dimension K (the spectral feature dimension before
  feeding g_φ) — this caps the rank of the spectral pathway
- the input covariance Σ
- the parameter count C_f

For our linear spectral reduction, we can write G_θθ explicitly in
terms of the input covariance:

```
G_θθ  =  Σ  ⊗  Aᵀ A
```

where ⊗ is the Kronecker product and A is an effective "downstream
classifier" matrix that depends on g_φ's current state. Eigenvalues
of a Kronecker product are products of eigenvalues:

```
λ_min(G_θθ)  =  λ_min(Σ) · λ_min(Aᵀ A)
```

The second factor is bounded by the bottleneck dimension K times some
classifier-dependent constant. The first factor is bounded by
sub-Gaussian Assumption ass:subg.

Putting this together — and using Schur — we get:

```
λ_min(G)  =  O( C_f / (S · K) )
```

So the **spectral parameter count drives the lower-tail of the spectrum**.

(This is the part of the proof that takes the most algebraic work. We'll
write it out fully in Phase 2 when we tackle the supplement proofs.
For now, the key point is that λ_min scales with C_f, not with C_g.)

---

## 6. Combining

```
κ(G)  =  λ_max(G) / λ_min(G)  ≥  Ω(C_g) / O(C_f / (S K))  =  Ω(C_g · S K / C_f)
```

Under our assumption `f_θ` linear: C_f = S · K. So `S · K / C_f = 1`, and
the bound simplifies to:

```
κ(G)  ≥  Ω(C_g / C_f)
```

There's a small subtlety about whether the simplification gives us a
factor of 1 or a logarithmic correction; we'll resolve that in Phase 2.
For now the qualitative scaling is what matters: **κ grows linearly
with the ratio of module capacities**.

---

## 7. What we still need to fill in (the Phase 2 TODOs in the LaTeX)

- A clean derivation of Karakida's λ_max scaling specialised to our
  setting (CNN / MLP / ViT) — Karakida proves the asymptote; we want
  the explicit constant.
- The Kronecker product expansion for G_θθ written out step by step.
- A sharp upper bound on λ_min(S) including the dependence on the
  input covariance condition number.
- Empirical comparison: at our experimental D values, how loose is
  the bound vs the measured κ? (Exp 1.1 tells us the bound predicts
  slope 1 but we measure 0.7 — explain the gap.)

These are the entries marked `\TODO{Phase 2}` in section 04 of the
paper.

---

## 8. One-paragraph version for non-mathematician readers

The total parameter capacity is split between two modules. The
optimization landscape is a high-dimensional bowl, and the bowl is
stretched: very steep along the directions corresponding to the
high-capacity spatial parameters, very shallow along the directions
corresponding to the low-capacity spectral parameters. The ratio of
steepness to shallowness — the **condition number** — grows in
proportion to the ratio of the two capacity counts. This is what
forces the timescale separation that Theorem 2 then turns into
gradient starvation.
