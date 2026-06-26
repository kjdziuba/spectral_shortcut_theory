# Progress Log

Running session-by-session log. Newest entries at the top.

---

## 2026-06-26 (late+) — Experiment 1.2 v4: CNN extended + ViT universality

**Done**:
- Added SpatialViT to `synthetic/models.py` (2-layer pre-LN ViT, ReLU
  activation for Hessian compatibility).
- Wrote `experiments/exp1_2v4.py` running 42 conditions (CNN widths
  {16,64,256,1024} + ViT widths {64,128,256}, joint vs frozen, 3 seeds).
- Drafted Section 1 (Introduction) properly.
- Drafted Section 2 (Related Work) with all 6 subsections.

**Findings (42-run sweep)**:

CNN sweep:
- EGR collapse depth scales monotonically with capacity (the headline
  Theorem 2 result):
    D=16   drops then recovers to ~0.4
    D=64   stable around ~0.25
    D=256  drops to ~0.10
    D=1024 drops to ~0.08
- Joint training final test loss diverges at high capacity:
    D=1024 joint test loss = 2.3, frozen test loss = 1.2
- Test accuracy gap exists but is small (~2-3 points) — joint overfits.

ViT sweep:
- Train loss reaches 0.01-0.03 at all widths — full memorization regime.
- Test loss is catastrophic (2.5-5.0) — joint training fails harder than CNN.
- EGR pattern INVERTED: rises with capacity rather than falling.
  This is the memorization regime where both gradient norms approach
  noise floor. Theorem 2's prediction holds in spirit (joint training
  fails) but the EGR observable loses signal.
- Frozen variants still beat joint at every width.

**Decisions**:
- For the paper: lead with CNN as the primary demonstration of Theorem 2.
  Note ViT as a different failure mode (memorization regime) where
  the EGR observable becomes uninformative but the overall prescription
  (freeze) still holds.
- Will mention in discussion that EGR diagnostic needs to be used
  before full memorization; otherwise the ratio loses meaning.

**Next session**:
- Move to Experiment 1.3 (equal-information killer test).
- Design the data calibration carefully (Shannon I(y; spectral) =
  I(y; spatial) via signal-strength tuning).

---

## 2026-06-26 (late) — Experiment 1.2 dynamics, three iterations

**v1 (Adam, SpatialMLP, noise=0.1):**
- Trained joint at D ∈ {32, 128, 512, 2048}, 3 seeds each, 150 epochs.
- Observed striking test-accuracy COLLAPSE at D=2048: peaks at ~70%
  early, then decays to ~50% by epoch 150 while train loss continues to
  drop. Two seeds at D=512 also collapse, one holds.
- EGR did not show dramatic decay (Adam's per-param normalization
  masks the natural gradient asymmetry).

**v2 (SGD+momentum, SpatialMLP, noise=0.05) — diagnostic miss:**
- Joint vs frozen converged to nearly identical accuracies (~75%).
- The collapse from v1 disappeared.
- ROOT CAUSE: SpatialMLP has zero spatial receptive field (per-pixel
  MLP). It has no spatial pathway for a "spatial shortcut" to exploit.
  The whole framing of "joint training discovers spatial shortcuts"
  cannot be demonstrated without 2D spatial mixing in g_phi.

**v3 (Adam, SpatialCNN with 2x 3x3 convs, noise=0.1) — the right setup:**
- Trained joint AND frozen at D ∈ {16, 64, 256}, 3 seeds each.
- Frozen beats joint at every capacity; gap widens with D.
  D=256: joint acc 54%, frozen acc 57%; joint test loss 1.05, frozen 0.80.
- Joint test accuracy at D=256 peaks ~70% then COLLAPSES to ~52%.
  Train loss meanwhile drops to 0.29 — classic shortcut overfit.
- EGR trajectories show the predicted capacity-monotonic collapse:
    D=16:  drops then recovers to ~0.4
    D=64:  drops to ~0.25 and plateaus
    D=256: drops to ~0.2 then keeps decaying to ~0.1
  This matches Proposition prop:egr_monotonic.

**Headline plots ready for paper drafts:**
- exp1_2v3_cnn_joint_vs_frozen.png — joint vs frozen by capacity
- exp1_2v3_cnn_egr_overlay.png    — capacity-monotonic EGR collapse
- exp1_2v3_cnn_joint_D256.png     — 4-panel full mechanism at D=256

**Decisions**:
- SpatialMLP retained as "no-spatial-mixing baseline" for ablations.
- SpatialCNN is the canonical architecture for the dynamics figures.
- Adam is the realistic optimizer for the literature contrast;
  SGD ablation can stay in supplement.
- noise=0.1 is the "interesting" regime (CE doesn't fully saturate,
  shortcuts emerge); we'll also report noise=0.05 in supplement to
  show the regime where joint training "looks fine."

**Next session**:
- Push SpatialCNN widths up to D=1024 to test whether the gap
  continues widening, or saturates somewhere.
- Add SpatialViT for the third architecture in our universality story.

---

## 2026-06-26 (eve) — Experiment 1.1 scaled (5 seeds, D up to 8192)

**Done**:
- Extended widths to D ∈ {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}.
- 5 seeds per D (5×10 = 50 measurements).
- Added paired eigenvalues plot (λ_θ and λ_φ on same axes).

**Findings**:
- λ_θ stays flat at ~0.02 across the full D range — confirms the spectral
  block does not grow because C_f is fixed. The pathology is entirely on
  the spatial side.
- λ_φ grows 74× (0.62 → 46) as C_g grows 510×.
- κ grows from 40 → 2,407 as C_g/C_f goes from 0.3 → 152.
- Fit slope is now 0.70 (up from 0.48 with the small range).
- At C_g/C_f ≈ 152 (close to real architecture's 325), κ ~ 2,400 —
  loss landscape is severely ill-conditioned.

**Decisions**:
- The "spectral block flat" finding is a HEADLINE result. Add paired
  plot as the primary figure for Theorem 1 in the paper.
- Will further push D to test asymptotic slope (whether it reaches 1)
  but the qualitative claim is firmly established.

**Next session**:
- Run Experiment 1.2 (training dynamics) — should give cleaner result.

---

## 2026-06-26 (PM) — Sections 4–6 drafted + Experiment 1.1 first run

**Done**:
- Drafted Section 4 (Theorem 1: Hessian Capacity Bound) — formal
  statement, 3 supporting lemmas (Karakida scaling, Cauchy interlacing,
  Schur complement), proof skeleton with TODO markers for Phase 2.
- Drafted Section 5 (Theorem 2: Two-Timescale Spurious Convergence) —
  formal statement, 3 supporting lemmas (timescale gap, fast manifold,
  residual decay), proof sketch combining Borkar + Pezeshki.
- Drafted Section 6 (EGR) — initial-value proposition, monotonicity
  proposition, threshold criterion, implementation reference.
- Workflow built infrastructure in parallel: synthetic/data.py,
  synthetic/models.py, hessian/eigenvalues.py, egr/callback.py,
  experiments/exp1_1.py, experiments/exp1_2.py. Smoke test passed.
- Fixed signature mismatch in exp1_1.py (top_eigenvalue_block API).
- **Ran Experiment 1.1 (capacity ablation)** on synthetic data:
    D ∈ {16, 32, 64, 128, 256, 512}, C_f = 1024, C_g ∈ [306, 9730]
    κ ranges from 23 to 464.

**Findings**:
- Direction confirmed: λ_φ grows with C_g, κ grows with C_g/C_f.
- Scaling exponent α ≈ 0.48 (predicted 1 from Karakida asymptote).
- κ already exceeds 400 at modest C_g/C_f ≈ 10 — landscape is
  pathological even far from the real-architecture regime
  (C_g/C_f ≈ 325).
- Need: scale D up to 2048+, average over seeds, run Exp 1.2 dynamics.

**Decisions**:
- Theorem 1 statement will be reframed as "monotonic growth with
  measurable exponent" rather than tight Ω(C_g/C_f).
- Cite Karakida for asymptote, document empirical finite-size α.
- Add capacity ablation up to D=8192 as next step.

**Next session**:
- Run Experiment 1.2 (dynamics) — should give clean qualitative result.
- Scale Experiment 1.1 to larger D + 5-seed averaging.
- User reads Pezeshki when ready.

---

## 2026-06-26 (AM) — Section 3 (Setup) drafted

**Done**:
- Walked through model setup, chain rule, Jacobian, Hessian conceptually
  - Simplified to C=1 (single-channel) for clean theory; channel extension trivial
  - Confirmed scope: linear `f_θ` for proofs, nonlinear empirically
  - Confirmed CE-specific role: vanishing-residual property
  - Concrete toy example (3-dim spectrum, 2 features, 2 classes) showing
    Jacobian = V in linear case
  - 2D quadratic example showing condition number → timescale gap
- Added Experiment 1.5 (Spatial architecture ablation: linear vs CNN vs ViT)
  - Important because CNN-based pipelines report most spectral-collapse issues
- Saved full conversational explanations to `study_notes/01_setup_and_chain_rule.md`
  - Convertible to PDF for offline study
- Wrote Section 3 (`paper/sections/03_setup.tex`) in formal LaTeX
  - Four assumptions stated (linearity, regularity, sub-Gaussian, capacity gap)
  - Loss formula
  - Chain-rule decomposition with named factors
  - Block Hessian structure
  - EGR formal definition
  - Notation conventions

**Decisions made**:
- Default C=1 in theory; real-data experiments use d2 (matches field standard)
- Run ablations later with raw, d1, d2 as robustness check
- Run BOTH linear and nonlinear (small MLP) in synthetic experiments
- Add CNN to spatial-architecture ablation (Experiment 1.5)

**Next session**:
- Begin Phase 0 (Track A reading): Pezeshki 2021
- In parallel: I start drafting Section 2 Related Work — gradient starvation subsection

**Open questions**:
- None yet.

---

## 2026-06-08 — Project initialized

**Done**:
- Created directory structure at `/home/u37314kd/Projects/spectral_shortcut_theory/`
- Wrote `README.md`, `PLAN.md` (8-phase plan), this `PROGRESS.md`
- Set up LaTeX paper skeleton with section stubs
- Created reading note templates for 5 foundational papers
- Initialized git repository

**Next session**:
- Begin Phase 0, Track A: Start reading Pezeshki et al. (2021) "Gradient Starvation"
- Fill in `reading_notes/01_pezeshki_2021.md` as you read
- In parallel, I'll start drafting Section 2 (Related Work) based on what you note

**Open questions**:
- None yet.

**Decisions log**:
- Project lives separate from `spectral_tokenization/` to keep theory work clean
- LaTeX-first for proofs; markdown drafts in `proofs/` for early iteration
- EGR diagnostic will be released as standalone PyPI package
- Target ICML/NeurIPS theory track first, AISTATS as backup
