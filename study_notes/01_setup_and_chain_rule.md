# Study Notes 1 — Pipeline Setup, Chain Rule, Jacobian, Hessian

These are the explanations from our working sessions, formatted for later reference. Convert to PDF when needed (e.g., `pandoc 01_setup_and_chain_rule.md -o 01_setup.pdf`).

---

## 1. What we're modeling

A generic two-module pipeline:

1. **Input** X — a hyperspectral image
2. **Spectral reduction** f_θ — compresses each pixel's spectrum to a feature vector
3. **Spatial model** g_φ — looks at the feature image and produces predictions

The math is independent of spectroscopy specifics; the canonical example is FTIR/QCL tissue classification.

---

## 2. The input

For each pixel (h, w) in an H × W image, we measure a spectrum across S wavenumbers. Default for the theorem: **single channel** (C=1) for clarity. The C=3 case (raw + d1 + d2) is a trivial extension — just a larger flattened vector.

Per pixel:

```
x_{h,w} ∈ R^S
```

For our spectroscopy data: S = 314 wavenumbers covering 1000–1800 cm⁻¹.

---

## 3. The spectral reduction

f_θ is applied **independently to every pixel**:

```
f_θ : R^S → R^K       (K = 128 in our case)
```

**Critical assumption for the theorem**: f_θ is **linear** in θ. Write θ as a matrix W ∈ R^(S × K):

```
f_θ(x_{h,w}) = Wᵀ · x_{h,w}
```

Parameter count: C_f = S · K = 314 · 128 = 40,192.

**Why linear?**
- Clean Jacobian: ∂Z/∂θ = X
- Hessian block H_θθ is computable from data covariance + g_φ's backward pass
- Proofs close in closed form

**Nonlinear extension**: handled empirically (Experiment 1.5 in our plan) and discussed as future work. Standard ML theory practice: prove cleanest case rigorously, show empirically that it extends.

---

## 4. The spatial model

```
g_φ : R^(K × H × W) → R^(N_cls × H × W)
```

Parameter count C_g (e.g., 13M for our ViT).

**The architecture doesn't matter for the theorem**. We assume only:
- C_g parameters
- twice-differentiable in φ
- **capacity gap**: C_g / C_f ≫ 1

Architectures we will test empirically: linear, CNN, ViT. The theorem applies to all three (with different constants in the bound).

---

## 5. The loss — and why cross-entropy specifically matters

Per-pixel cross-entropy:

```
L(θ, φ) = (1 / HW) · Σ Σ CE( g_φ(f_θ(X))[:, h, w] , y_{h,w} )
```

**Why CE matters for Theorem 2**: cross-entropy has the property that the gradient with respect to the logits is **prediction minus target**:

```
∂L/∂ŷ = softmax(ŷ) - y_onehot
```

When predictions become confident in the correct class, this residual goes to **zero exponentially fast** (because softmax saturates). That's the engine of gradient starvation.

**Other losses**:
- MSE: gradient ∝ residual but decays linearly (not exponentially). Starvation weaker but still present.
- Focal loss: even faster saturation than CE. Theorem holds more strongly.
- Hinge: gradient = 0 once margin satisfied. Sharper version of starvation.
- Label smoothing: slows saturation but doesn't eliminate it.

**The principle**: theorem applies to any loss with a vanishing-residual property. CE is the canonical case (used in 95% of classification problems).

---

## 6. Notation reference card (BOOKMARK)

```
Symbol     Meaning                              Dimensions
─────────────────────────────────────────────────────────────
X          Input image (full)                   R^(H × W × S)
x_{h,w}    Input at pixel (h, w)                R^S
y_{h,w}    True class at pixel (h, w)           {1, ..., N_cls}
ŷ          Predictions (logits)                 R^(N_cls × H × W)
Z          Feature map (after f_θ)              R^(K × H × W)

θ          Spectral module parameters           dim C_f
φ          Spatial module parameters            dim C_g
W          Spectral weight matrix (= θ)         R^(S × K)

f_θ        Spectral reduction                   R^S → R^K
g_φ        Spatial model                        R^(K × H × W) → R^(N_cls × H × W)
L          Loss (cross-entropy)                 scalar

C_f        # spectral parameters                = S · K  (e.g. 40,192)
C_g        # spatial parameters                 (e.g. 13 million)

∇_θ L      Gradient wrt spectral params         vector of dim C_f
∇_φ L      Gradient wrt spatial params          vector of dim C_g
EGR        Effective Gradient Ratio             ‖∇_θ L‖ / ‖∇_φ L‖

H          Joint Hessian (second derivative)    (C_f+C_g) × (C_f+C_g)
H_θθ       Hessian block wrt θ                  C_f × C_f
H_φφ       Hessian block wrt φ                  C_g × C_g
H_θφ       Cross-block                          C_f × C_g
```

**Mnemonic**: θ has "f" subscripts (f for *first* — closest to input, front of pipeline). φ has "g" subscripts.

---

## 7. The chain rule — the heart of the paper

We can't compute ∂L/∂θ directly because L doesn't depend on θ directly — only through Z, which feeds into ŷ, which goes into L. The chain rule chains these together:

```
∂L/∂θ  =  (∂L/∂ŷ)  ·  (∂ŷ/∂Z)  ·  (∂Z/∂θ)
            ▲           ▲           ▲
            │           │           │
          residual   spatial      spectral
        (CE error) Jacobian      input
                  (depends on φ) (= X, fixed)
```

**Read it like a relay race**: each runner carries the gradient signal one step backward toward θ.

**Why each term matters**:

- **∂L/∂ŷ** — starts the race. The error signal. If predictions are perfect, this is zero and **no signal reaches anywhere**.
- **∂ŷ/∂Z** — middle runner. Spatial model's Jacobian — tells the spectral module "this is what I currently want from you." Depends on current φ. **This is the moving target.**
- **∂Z/∂θ** — final handoff. Since Z = WᵀX, this is just X. Static, doesn't change during training.

**Central insight of the paper**: even if Jacobian (term 2) is huge and input (term 3) is huge, if **term 1 (the residual) is zero, the entire product is zero**. The starvation kills the signal at the first stage. The other two stages can't compensate.

---

## 8. Toy example — seeing the Jacobian concretely

To make the abstract notation concrete, here is a tiny version of the pipeline:

- 1 pixel
- 3-dimensional spectrum
- Reduce to 2 features
- 2 classes
- Both modules linear

```
INPUT          x = [0.5, 0.8, 0.3] ∈ R³

SPECTRAL       z = Wᵀx,   W ∈ R^(3×2)
REDUCTION      θ = W,  C_f = 6 params

"SPATIAL"      ŷ = Vᵀz,   V ∈ R^(2×2)
MODEL          φ = V,  C_g = 4 params

LOSS           L = CE(ŷ, y),    y = 0 (cancer)
```

Pick weights at some training step:

```
W = [[ 0.2, -0.1],         V = [[1.0, -0.5],
     [ 0.5,  0.3],              [0.8,  0.2]]
     [ 0.1,  0.4]]
```

Forward pass:

```
z = Wᵀx = [0.2·0.5 + 0.5·0.8 + 0.1·0.3,    = [0.53,
           -0.1·0.5 + 0.3·0.8 + 0.4·0.3]      0.31]

ŷ = Vᵀz = [1.0·0.53 + 0.8·0.31,            = [ 0.778,
           -0.5·0.53 + 0.2·0.31]              -0.203]

softmax(ŷ) = [0.728, 0.272]   ← model 72.8% confident in cancer
```

### The Jacobian ∂ŷ/∂Z

Since ŷ = Vᵀz is linear, **the Jacobian is just V**:

```
∂ŷ/∂z = V = [[1.0, -0.5],
             [0.8,  0.2]]
```

What this matrix means:
- V[0, 0] = 1.0 → feature 1 strongly *increases* cancer logit
- V[0, 1] = -0.5 → feature 1 *decreases* normal logit
- etc.

**Critical**: when we train, V changes. So this Jacobian changes every step. **The moving target.**

In the REAL pipeline, the Jacobian is NOT one matrix — it's the chain-rule product through all 12 ViT layers + decoder. We never write it out explicitly; backprop computes it implicitly.

### The residual ∂L/∂ŷ

For softmax-CE:

```
∂L/∂ŷ = softmax(ŷ) - y_onehot
      = [0.728, 0.272] - [1, 0]        ← truth = cancer = [1, 0]
      = [-0.272, 0.272]
```

Magnitude 0.272. If training continues and the model becomes confident:

```
softmax(ŷ) → [1, 0]
∂L/∂ŷ → [0, 0]    ← STARVATION
```

When this term goes to zero, the entire product is zero. **θ stops updating.**

### The gradient ∇_θ L

Combining via the chain rule, the gradient with respect to the spectral weight matrix is the outer product:

```
∇_W L = x ⊗ (V · ∂L/∂ŷ)

      = [0.5, 0.8, 0.3]ᵀ ⊗ [-0.408, -0.163]

      = [[-0.204, -0.082],
         [-0.326, -0.131],
         [-0.122, -0.049]]
```

If ∂L/∂ŷ → 0, every entry of this matrix → 0. W stops updating.

---

## 9. The Hessian — using a 2D quadratic example

The Hessian is the second derivative — how the gradient itself changes.

Forget our pipeline for a moment. Imagine a 2D quadratic loss:

```
L(θ, φ) = 0.01·θ²  +  100·φ²
```

This is a stretched bowl — much steeper in φ direction than θ direction.

### Gradients (first derivatives)

```
∂L/∂θ = 0.02·θ        ← small slope per unit θ
∂L/∂φ = 200·φ         ← large slope per unit φ
```

### Hessian (second derivatives)

```
∂²L/∂θ² = 0.02        ← how does ∂L/∂θ change with θ? Slowly.
∂²L/∂φ² = 200         ← how does ∂L/∂φ change with φ? Fast.
∂²L/∂θ∂φ = 0          ← independent in this example.

H = [[0.02,    0],
     [   0,  200]]
```

### Eigenvalues

For a diagonal matrix, eigenvalues are the diagonal entries:

```
λ₁ = 0.02     ← θ direction
λ₂ = 200      ← φ direction
```

### Condition number

```
κ(H) = λ_max / λ_min = 200 / 0.02 = 10,000
```

### Training behavior: same η, very different speeds

Pick learning rate η = 0.001. Starting point (θ, φ) = (1, 1). One SGD step:

**φ direction (steep)**:
```
φ_new = 1 - 0.001 · 200 · 1 = 0.8       ← 20% reduction
```

**θ direction (flat)**:
```
θ_new = 1 - 0.001 · 0.02 · 1 = 0.99998  ← 0.002% reduction
```

**φ moved 10,000× faster than θ. That ratio equals the condition number.**

### The cliff

Can't just increase η to make θ move faster:

```
With η = 0.1:
φ_new = 1 - 0.1 · 200 · 1 = -19      ← overshoots, diverges
```

η is bounded by ~2/λ_max. With that bound, λ_min direction crawls. Forced timescale separation.

---

## 10. Bringing this back to our paper

In our pipeline, instead of 1 θ and 1 φ parameter, we have:

```
C_f ≈ 40,000     (θ — spectral)
C_g ≈ 13,000,000 (φ — spatial)
```

The Hessian is block-structured:

```
H = ⎡ H_θθ   H_θφ ⎤    
    ⎣ H_φθ   H_φφ ⎦

  • H_θθ is C_f × C_f          (small block, ~40k × 40k)
  • H_φφ is C_g × C_g          (huge block, ~13M × 13M)
```

**Theorem 1 (informal)**: the eigenvalues of H_φφ are *huge* (scale with C_g), the eigenvalues of H_θθ are *small* (scale with C_f and bottleneck), so:

```
κ(H) ≥ C_g / C_f ≈ 325×    (in our spectroscopy case)
```

That's a *much* bigger condition number than the 10,000 of our toy quadratic. The timescale separation is severe.

**Our experiments measure these eigenvalues directly via power iteration.**

---

## 11. What our experiments are doing

The theorems make quantitative predictions; experiments verify them.

```
Experiment 1.1 (Capacity ablation):
  Measure λ_max(H_φφ) and λ_max(H_θθ) at initialization.
  Predict: ratio scales as C_g / C_f.
  Tool: power iteration via PyHessian.

Experiment 1.2 (Two-timescale dynamics):
  Log ‖∇_θ L‖ and ‖∇_φ L‖ every epoch during training.
  Predict: ‖∇_φ‖ peaks early, saturates loss; ‖∇_θ‖ collapses to ~0.
  Tool: gradient norm logging.

Experiment 1.3 (Equal-information experiment):
  Construct synthetic data where spectral and spatial features carry
  exactly equal information about the label. Show end-to-end joint
  training still picks spatial.
  Tool: controlled data generation + 5 training conditions.

Experiment 1.4 (EGR as predictor):
  Train 20 model variants. Compute EGR over time.
  Predict: early-epoch EGR predicts final test F1.
  Tool: correlation analysis.

Experiment 1.5 (Spatial architecture ablation, ADDED):
  Repeat key experiments with three spatial models: linear, CNN, ViT.
  Predict: same gradient starvation pattern across all three.
  Tool: same as above, swap g_φ.
```

So our experiments measure:
- **Eigenvalues** (= curvature, = the speeds Theorem 1 talks about)
- **Gradient norms** (= step sizes, = starvation Theorem 2 talks about)
- **EGR** (= ratio of step sizes, = our diagnostic)

The theorems predict relationships; experiments verify them.
