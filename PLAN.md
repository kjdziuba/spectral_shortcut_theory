# Project Plan — Spectral Shortcut Theorem Paper

**Start date**: 2026-06-08
**Target submission**: ~2026-12 (6 months part-time)

---

## Working in parallel: Learning + Writing

We work two tracks simultaneously:

- **Track A (Learn)**: You read the foundational papers and do reading notes. This builds the literature review.
- **Track B (Build)**: We start writing the model setup, drafting proofs, and building synthetic experiment infrastructure.

Sessions alternate. After each reading you do, we use the new knowledge immediately by writing the matching section.

---

## Phase 0 — Foundations (Weeks 1–2)

### Track A: Reading
- W1 Day 1–3: Pezeshki et al. (2021) "Gradient Starvation" — most critical, read first
- W1 Day 4–5: Sagun et al. (2017) "Empirical Analysis of the Hessian"
- W2 Day 1–2: Heusel et al. (2017) "TTUR for GANs" — your template for Theorem 2
- W2 Day 3–4: Karakida et al. (2019) "Universal Statistics of Fisher Information"
- W2 Day 5: Pennington & Bahri (2017) "Geometry of Neural Network Loss Surfaces"

Each paper gets a `reading_notes/<paper_slug>.md` file with:
- 3-line summary
- Main theorem(s) restated in our notation
- What we borrow from this paper
- What we extend or differ from

### Track B: Build (in parallel)
- Set up LaTeX project, write abstract draft (v0 — will be rewritten)
- Write Section 2 (Related Work) as we read — turn reading notes into prose
- Start `code/synthetic/` data generator skeleton

**Deliverable end of Phase 0**: 5 reading notes + draft related work section + working synthetic data generator.

---

## Phase 1 — Mathematical Setup (Weeks 3–4)

### Track A: Reading
- Borkar Chapter 6 (two-timescale stochastic approximation) — your bible for Theorem 2
- Bhatia "Matrix Analysis" — Schur complement chapter

### Track B: Build
- W3: Write Section 3 (Setup & Preliminaries) — compositional model, loss, Hessian decomposition, EGR definition
- W3: State all assumptions (Lipschitz, smoothness, sub-Gaussian inputs)
- W4: Draft `proofs/thm1_hessian.md` outline — what we want to prove, with proof sketch
- W4: Draft `proofs/thm2_two_timescale.md` outline

**Deliverable end of Phase 1**: Methods/preliminaries section complete. Both theorems formally stated. Proof skeletons drafted.

---

## Phase 2 — Theorem 1 (Weeks 5–7)

Hessian Condition Number Bound: κ(G) ~ Ω(C_g / C_f)

### W5: Block Gauss-Newton derivation
- Define GN matrix for compositional model
- Express in block form
- Identify what we need to bound

### W6: λ_max bound via Karakida
- Apply Karakida's FIM scaling results
- Show λ_max(G) ≥ λ_max(J_φ^T J_φ) ~ Ω(C_g)
- Verify all assumptions hold

### W7: λ_min bound via Schur complement
- Schur complement of dominant block
- Bound by spectral bottleneck dimension and C_f
- Combine into condition number theorem
- Write clean proof in `paper/sections/04_theorem1.tex`

**Deliverable end of Phase 2**: Theorem 1 with full proof, ~5 pages of paper.

---

## Phase 3 — Theorem 2 (Weeks 8–10)

Two-Timescale Spurious Convergence

### W8: SGD → ODE limit
- Continuous-time gradient flow for both modules
- Identify slow-fast structure from Hessian eigenvalues
- Define ε = λ_min(H_θθ) / λ_max(H_φφ) ≈ C_f / C_g

### W9: Borkar's singular perturbation
- Apply two-timescale theorem
- Derive φ*(θ) equilibrium manifold
- Show ‖∇_φ L‖ → 0 on fast timescale at φ*(θ_0)

### W10: Gradient starvation cascade
- Apply Pezeshki-style cross-entropy decay
- Show residual r → 0 exponentially
- ‖∇_θ L‖ = ‖J_θ^T r‖ → 0 before θ converges
- Combine into spurious stationary point theorem
- Write clean proof in `paper/sections/05_theorem2.tex`

**Deliverable end of Phase 3**: Theorem 2 with full proof, ~5 pages of paper.

---

## Phase 4 — EGR Diagnostic (Weeks 11–12)

### W11: Formalization
- Define EGR(t) = ‖∇_θ L‖_t / ‖∇_φ L‖_t
- Prove EGR tracks the timescale ratio
- State threshold criterion: EGR < τ → gradient starvation onset
- Write `paper/sections/06_egr.tex`

### W12: Implementation
- PyTorch training callback in `code/egr/`
- Tested on toy network
- Documentation + example usage

**Deliverable end of Phase 4**: EGR section + working PyTorch package.

---

## Phase 5 — Synthetic Experiments (Weeks 13–15)

### W13: Build infrastructure
- Synthetic spectral-spatial data generator
- Configurable: capacity ratio, signal strength α/β, noise level
- Save under `code/synthetic/`

### W14: Experiment 1.1 — Capacity ablation (validates Theorem 1)
- Vary C_g/C_f ∈ {1, 10, 100, 1000}
- Compute κ(H) via PyHessian power iteration
- Plot log-log: predict slope ≈ 1

### W14: Experiment 1.2 — Two-timescale dynamics (validates Theorem 2)
- Fixed C_g/C_f = 100
- Track loss, ‖∇_θ‖, ‖∇_φ‖, EGR per epoch
- Predict: ‖∇_φ‖ drops → loss residual → 0 → ‖∇_θ‖ collapses

### W15: Experiment 1.3 — Equal-information experiment (the killer)
- Construct problem where I(y; h_s) = I(y; h_l) exactly
- Five conditions: joint, frozen random, frozen PCA, frozen pretrained, spectral-only
- Predict: joint collapses to spatial-only despite equal info

### W15: Experiment 1.4 — EGR as predictor
- 20 model variants, varying hyperparameters
- Correlate early-epoch EGR with final test F1
- Predict: strong correlation, threshold predicts failure

### W15: Experiment 1.5 — Spatial architecture ablation
- Repeat key experiments with three spatial models: linear, CNN, ViT
- Predict: gradient starvation pattern emerges across all three
- Important because CNN-based pipelines (still common in spectroscopy) have the most reported "spectral collapse" issues
- Strengthens universality claim significantly

**Deliverable end of Phase 5**: 4 main figures + experiment section draft + universality ablation.

---

## Phase 6 — Real-Data Validation (Weeks 16–17)

### W16: EGR on existing models
- Compute EGR trajectories for 5 variants from sibling spectroscopy project
- Plot in single figure

### W17: Hessian eigenvalue tracking
- Power iteration on H_θθ vs H_φφ for learned linear and fine-tuned pretrained
- Show capacity-ratio scaling holds in the wild

**Deliverable end of Phase 6**: 1 supplementary figure + brief real-data section.

---

## Phase 7 — Drafting (Weeks 18–22)

### W18: Introduction + Related Work
### W19: Setup + Theorem 1 polish
### W20: Theorem 2 + EGR polish
### W21: Experiments + Discussion
### W22: Abstract, polish, supplement, figure captions

**Deliverable end of Phase 7**: Full draft ready for review.

---

## Phase 8 — Internal Review & Submission (Weeks 23–24)

### W23: Send to advisor + 2 trusted reviewers (one ML theorist, one spectroscopist)
### W24: Revise, submit

---

## Risk register (branches)

| Risk | Probability | Mitigation |
|---|---|---|
| Hessian bound is loose empirically | Medium-high | Reframe as lower bound, add empirical κ measurements |
| Two-timescale assumptions don't hold | Medium | Switch to discrete-time Konda-Tsitsiklis analysis |
| EGR is too noisy | Low-medium | Moving average smoothing, report smoothed-EGR |
| Competing paper appears | Low | Position as spectroscopy-specific complement |
| Theorem 2 proof requires new lemmas | Medium | Weaken to ε-stationary point with ε bounded by EGR |
| Reviewers reject as "domain-restricted" | Medium at ML venues | Resubmit to TMI/MIA with biomedical framing |

---

## Working session protocol

Each session ends with:
1. A concrete artifact (proven lemma, written paragraph, code experiment)
2. An entry in `PROGRESS.md` (what we did, what's next)
3. Git commit

Between sessions:
- You read/learn (Track A)
- I keep context, ready to pick up

---

## Definition of done for the paper

- [ ] Theorem 1 stated and proved
- [ ] Theorem 2 stated and proved
- [ ] EGR formally defined with monotonicity property
- [ ] 4 synthetic experiments verifying predictions
- [ ] Real-data sanity check
- [ ] EGR PyPI package released
- [ ] Internal review complete
- [ ] Submitted to ICML or NeurIPS
