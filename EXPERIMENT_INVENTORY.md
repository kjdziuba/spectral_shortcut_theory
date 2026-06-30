# Experiment Inventory

Status of every experiment in PLAN.md, what it validates, and what
still needs to run. Maintain after each session so you can always
recover state with one read.

Last updated: 2026-06-30

---

## Completed

### Experiment 1.1 — Capacity ablation
**Validates Theorem 1** (κ growth from capacity asymmetry).

| Version | What | Status | Headline |
|---------|------|--------|----------|
| v1      | D 16–512, 1 seed       | done | slope 0.48; small range too noisy |
| v2      | D 16–8192, 5 seeds     | **canonical** | slope 0.70; κ from 40 → 2,407; λ_θ flat at 0.02 |

**Artifacts**: `code/experiments/exp1_1.py`, `results/exp1_1_*.csv`,
`results/exp1_1_paired_eigenvalues.png` (headline figure).

---

### Experiment 1.2 — Two-timescale training dynamics
**Validates Theorem 2** (EGR collapse from joint training).

| Version | What | Status | Headline |
|---------|------|--------|----------|
| v1 | D {32,128,512,2048}, Adam, SpatialMLP, noise=0.10 | done | striking test-accuracy collapse at D=2048, but SpatialMLP has no spatial pathway |
| v2 | SGD + frozen baseline, SpatialMLP, noise=0.05 | done | identical joint vs frozen — confirmed MLP can't host a spatial shortcut |
| v3 | SpatialCNN, Adam, noise=0.10 | **canonical** | EGR collapse depth monotonic with width; frozen > joint at every D |
| v4 | CNN + ViT universality sweep | done | CNN clean; ViT enters memorization regime (separate failure mode) |

**Artifacts**: `code/experiments/exp1_2v4.py`, `results/exp1_2v3_*` and
`results/exp1_2v4_*`. Headline: `exp1_2v3_cnn_egr_overlay.png`.

---

### Experiment 1.3 — Equal-information killer test
**Validates the spectral-shortcut conjecture** (joint training prefers spatial even at equal info).

| Version | What | Status | Headline |
|---------|------|--------|----------|
| v1 (broken) | 3 calibrations, original data (spatial only in labels) | done | spatial_only at chance for all modes — design flaw |
| v1 (fixed)  | 3 calibrations, data with spatial signal in orthogonal X direction | **canonical** | Bayes mode wins; joint collapses to 0.48 (below chance, below either single-modality baseline 0.53); frozen 0.61 |

**Artifacts**: `code/experiments/exp1_3.py`, `results/exp1_3v1_*`.
Headline: `exp1_3v1_summary.png`.

---

### Experiment 1.4 — EGR as a diagnostic
**Validates Section 6's EGR proposition** (within-capacity correlation real).

| Version | What | Status | Headline |
|---------|------|--------|----------|
| v1 | 24 runs across capacity, early-window EGR | done | r = -0.56 NEGATIVE — early window too soon to see collapse |
| v2 | same runs, multiple EGR windows | done | egr_depth pooled r = -0.78; but 77% of that is capacity-confounded (partial r = -0.17) |
| fw | 144 runs at D=256 FIXED, varying noise × lr × wd × 6 seeds | **canonical** | within-capacity r(egr_min, acc) = +0.344; survives every adversarial control; ROC-AUC 0.61 (real but modest) |

**Artifacts**: `code/experiments/exp1_4.py`, `code/experiments/exp1_4_fw.py`,
`results/exp1_4_*` and `results/exp1_4_fw_*`. Headline: `exp1_4_correlation.png`.

---

## Pending — required for the paper

### Exp 1.5 — Smallest capacity ratio where pathology disappears
**Sweep C_g / C_f from 1 to 1000 at fixed architecture.** Identify the
threshold below which joint training reaches frozen-level
performance. This bounds the regime where our prescription matters.

- estimated runs: 8 ratios × 2 conditions × 5 seeds = 80
- estimated wall time: ~40 min
- priority: medium (strengthens generality claim)

### Exp 1.6 — Spectral Decoupling comparison
**Pezeshki's L2-on-logits regularizer as an alternative to freezing.**
Train under three conditions at fixed CNN: joint, frozen, joint + spectral decoupling.
- estimated runs: 3 conditions × 5 noise levels × 5 seeds = 75
- estimated wall time: ~40 min
- priority: high (preempts a likely reviewer ask)

### Exp 1.7 — Real-data EGR trajectories
**Compute EGR for the 5 production variants in the spectroscopy companion paper.**
No retraining — reuse the existing checkpoints and rerun training-mode
forward+backward to log gradient norms.
- estimated runs: 5 variants × 5 folds = 25 retraining runs
- estimated wall time: ~3 hours (real data, full pipeline)
- priority: high (this is Section 8's empirical link)

### Exp 1.8 — Hessian eigenvalues on real data
**Power iteration on the actual FTIR/QCL spectroscopy ViT.** Validate
that κ in the wild matches our synthetic prediction at C_g/C_f ≈ 325.
- estimated runs: 2 variants (learned-linear, frozen-pretrained), 3 checkpoints each = 6
- estimated wall time: ~30 min once we have the env set up
- priority: medium (nice-to-have empirical anchor)

### Exp 1.9 — Loss-variant robustness
**Re-run Exp 1.2 v3 with focal loss and label-smoothed CE.**
- estimated runs: 3 losses × 3 widths × 3 seeds × 2 conditions = 54
- estimated wall time: ~30 min
- priority: low (we already cite Pezeshki's robustness)

---

## NOT required for the paper

These are nice-to-haves that we explicitly decided to skip:

- **Optimizer ablation in detail** — partially done (Adam vs SGD in v2 vs v3); supplementary mention suffices
- **NTK measurements** — would tighten Pezeshki connection but not load-bearing
- **3D CNN spatial architecture** — too much work for marginal universality claim; ViT covers the "modern arch" check

---

## Compute summary

| Pending | Wall time | Cumulative |
|---------|-----------|------------|
| Exp 1.5 | 40 min    | 40 min     |
| Exp 1.6 | 40 min    | 1 h 20 min |
| Exp 1.7 | 3 h       | 4 h 20 min |
| Exp 1.8 | 30 min    | 4 h 50 min |
| Exp 1.9 | 30 min    | 5 h 20 min |

Synthetic experiments can run on RTX 5000 Ada locally without queueing.
Real-data (1.7, 1.8) needs the FTIR/QCL pipeline — same env as the
spectral_tokenization repo.

---

## Theory work (no experiments needed)

- **Lemma 4.1** — partially done. 5-step proof drafted in
  `paper/sections/supplement.tex`. Karakida derivation in
  `study_notes/04_karakida_derivation.md`.
- **Lemma 4.2** — done (Cauchy interlacing, 1 paragraph).
- **Lemma 4.3** — TODO. Schur complement bound for λ_min, with
  Kronecker product expansion of J_θ^T J_θ. ~1 hour focused session.
- **Theorem 1 combine step** — TODO. ~30 min once Lemma 4.3 is in.
- **Theorem 2 full proof** — TODO. Borkar + Pezeshki combined. The
  hardest piece. ~3 hours of focused session work.

---

## Order I recommend

If we have the energy to keep pushing:

1. **Lemma 4.3 + combine** — finishes Theorem 1's proof. Half-day session.
2. **Exp 1.6** (Spectral Decoupling comparison) — reviewers will ask.
3. **Exp 1.5** (smallest capacity ratio) — bounds the prescription.
4. **Theorem 2 proof** — the big math session.
5. **Exp 1.7 + 1.8** (real-data link) — Section 8's empirical anchor.
6. **Exp 1.9** (loss variants) — supplement robustness only.

If we have to stop after one more session it should be **Lemma 4.3 + combine**, because that closes a load-bearing piece of the main result.
