# Progress Log

Running session-by-session log. Newest entries at the top.

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
