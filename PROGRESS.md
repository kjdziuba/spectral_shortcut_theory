# Progress Log

Running session-by-session log. Newest entries at the top.

---

## 2026-06-26 — Section 3 (Setup) drafted

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
